# SPDX-License-Identifier: Apache-2.0
"""Mmap-backed per-expert slice reader for MoE banks.

Follows the _SafeTensorMMap pattern from qwen4_exp (mmap + MADV_RANDOM)
but slices a stacked bank (E, O, I) per expert id.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import bisect
import ctypes
import fcntl
import json
import math
import logging
import mmap
import os
import re
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Dict, NamedTuple, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

_PAGE_SIZE = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 4096

# macOS kernel readahead (fcntl F_RDADVISE / struct radvisory): an async hint
# that pulls a file range into the page cache without copying into userspace
# — the zero-copy alternative to the warmer's discarded preads (ds4's
# metal_graph_stream_readahead). Best-effort: any failure is silent.
# F_RDADVISE is 44 on Darwin (not exported by Python's fcntl module).
_F_RDADVISE = getattr(fcntl, "F_RDADVISE", 44 if sys.platform == "darwin" else None)
_RADVISORY = struct.Struct("=qi4x")  # off_t ra_offset; int ra_count; + tail pad

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mlock.restype = ctypes.c_int


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_pyapi = ctypes.pythonapi
_pyapi.PyObject_GetBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_pyapi.PyObject_GetBuffer.restype = ctypes.c_int
_pyapi.PyBuffer_Release.argtypes = [ctypes.POINTER(_PyBuffer)]

_MLOCK_FAILED_LOGGED = False


def _mlock_range(mm: mmap.mmap, offset: int, length: int) -> bool:
    """mlock a page-aligned range of an existing file mapping. Zero-copy:
    the locked pages are the file cache pages themselves. The mapping's
    base address is obtained via PyObject_GetBuffer (the mmap is
    read-only, so from_buffer's writable requirement does not apply)."""
    global _MLOCK_FAILED_LOGGED
    try:
        view = _PyBuffer()
        if _pyapi.PyObject_GetBuffer(mm, ctypes.byref(view), 0) != 0 or not view.buf:
            return False
        try:
            base = view.buf
            start = (offset // _PAGE_SIZE) * _PAGE_SIZE
            end = min(view.len, ((offset + length + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE)
            if end <= start:
                return False
            rc = _libc.mlock(ctypes.c_void_p(base + start), ctypes.c_size_t(end - start))
            if rc != 0 and not _MLOCK_FAILED_LOGGED:
                import errno

                _MLOCK_FAILED_LOGGED = True
                logger.warning(
                    "mlock failed (errno=%s) — pinned-expert mode disabled for new pins",
                    errno.errorcode.get(ctypes.get_errno(), ctypes.get_errno()),
                )
            return rc == 0
        finally:
            _pyapi.PyBuffer_Release(ctypes.byref(view))
    except Exception as e:
        if not _MLOCK_FAILED_LOGGED:
            _MLOCK_FAILED_LOGGED = True
            logger.warning("mlock unavailable: %s", e)
        return False


_DTYPE_MAP: dict[str, tuple[np.dtype, int]] = {
    "BF16": (np.dtype("<u2"), 2),
    "F16": (np.dtype("<f2"), 2),
    "F32": (np.dtype("<f4"), 4),
    "U32": (np.dtype("<u4"), 4),
    "U8": (np.dtype("u1"), 1),
    "I32": (np.dtype("<i4"), 4),
    "I64": (np.dtype("<i8"), 8),
    "F8_E4M3": (np.dtype("u1"), 1),
}


def _np_to_mx(key: str, np_view: np.ndarray, dtype_str: str) -> mx.array:
    """Promote an expert's np.ndarray slice to the MLX representation."""
    if dtype_str == "BF16":
        # bf16 is stored as raw uint16 bits — reinterpret directly. This
        # matches mx.load's native handling exactly (the old
        # shift->f32->astype roundtrip flushed bf16 subnormals to zero via
        # Metal FTZ) and is ~9x faster on 4 MB slices: no numpy shift, half
        # the copy bytes, no GPU conversion kernel.
        return mx.array(np_view).view(mx.bfloat16)
    if dtype_str == "F8_E4M3":
        return mx.from_fp8(mx.array(np_view), dtype=mx.bfloat16)
    return mx.array(np_view)


# Queue depth for the per-run preadvs issued inside one read_expert_into call
# (Fase K F6, port of faseJ 0a4d3c7). This is the depth that matters: the
# coalesced bank read replaced the old one-job-per-run pool.map with a
# sequential loop, which dropped effective depth to 1 and cost ~45% of decode
# throughput on qwen4_exp. Decode demand is sparse (a handful of scattered
# experts per layer), so it has no readahead to fall back on and depth is all
# it has — prefill, which asks for hundreds of contiguous experts, already
# reaches ~2.6 GB/s against this device's ~2.8 GB/s ceiling even serially.
#
# Default 16: measured on qwen (2k prompt, 48 decode, single request), two
# rounds, 2 and 3 reps per arm. QD=1 is the old sequential behaviour.
# 16 is the peak AND by far the steadiest arm; 32 regresses (oversubscription
# past the device's useful queue depth). +55% over depth 1, all runs
# token-ID bit-exact.
_RUN_IO_QD = max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_RUN_QD", "16")))

# Fase 2: demand-read telemetry, collected ONLY under the profiling env
# (the bench sets OMLX_EXPERT_STREAMING_PROFILE=1). Zero cost otherwise:
# one module-global guard read per read_expert_into call.
_PROFILE_READS = os.environ.get("OMLX_EXPERT_STREAMING_PROFILE", "") == "1"
_READ_STATS: dict = {}


def read_stats() -> dict | None:
    """Fase 2 demand-read telemetry snapshot (None when unarmed)."""
    if not _PROFILE_READS:
        return None
    lat = sorted(_READ_STATS.get("lat_us", []))
    sizes = sorted(_READ_STATS.get("run_sizes", []))

    def pct(vals: list, p: float) -> int | None:
        if not vals:
            return None
        return int(vals[min(len(vals) - 1, int(len(vals) * p))])

    run_mean = int(sum(sizes) / len(sizes)) if sizes else None
    return {
        "calls": _READ_STATS.get("calls", 0),
        "runs": _READ_STATS.get("runs", 0),
        "bytes": _READ_STATS.get("bytes", 0),
        "lat_us_p50": pct(lat, 0.5),
        "lat_us_p95": pct(lat, 0.95),
        "run_size_mean": run_mean,
        "run_size_max": max(sizes) if sizes else None,
        "peak_inflight": _READ_STATS.get("peak_inflight", 0),
    }

# NOTE: the singleton and the accessor must not share a name — a module-level
# def _run_io_pool would rebind the very global the accessor is meant to
# populate, and it would then return itself instead of an executor.
_RUN_IO_POOL_SINGLETON: ThreadPoolExecutor | None = None
_run_io_pool_lock = threading.Lock()


def _run_io_pool() -> ThreadPoolExecutor:
    """Threads for the per-run preadvs of read_expert_into calls.

    Deliberately NOT the caller's pool. read_expert_into is dispatched on
    _EXPERT_IO_POOL (16 workers) via the layer-context prefetch and the
    union path's pool.map, so submitting to that same bounded pool and then
    waiting would deadlock as soon as every worker was a parent blocked on a
    queued child. This pool is separate and its tasks never submit anywhere,
    so it always drains.

    The SINGLETON is bounded at _RUN_IO_QD workers PROCESS-WIDE: every
    concurrent parent shares the same 16 run-read workers (K10 — the old
    docstring claimed one call was capped but N parents stacked N*QD depth;
    the executor caps the process, which is why QD32 measured slower: 32
    workers oversubscribe the device's useful queue depth, they do not
    multiply it). A single call keeps at most _RUN_IO_QD reads in flight
    because its planning window is sized to the pool (K11); the bound also
    keeps the transient buffer memory bounded.
    """
    global _RUN_IO_POOL_SINGLETON
    if _RUN_IO_POOL_SINGLETON is None:
        with _run_io_pool_lock:
            if _RUN_IO_POOL_SINGLETON is None:
                _RUN_IO_POOL_SINGLETON = ThreadPoolExecutor(
                    max_workers=_RUN_IO_QD,
                    thread_name_prefix="omlx-expert-run",
                )
    return _RUN_IO_POOL_SINGLETON


def segment_runs(
    eids_sorted: list[int],
    *,
    same: Any | None = None,
    merge_gap: int = 0,
    max_run: int | None = None,
) -> list[tuple[int, int]]:
    """Split ascending expert ids into (first, count) runs (Fase K K2).

    ONE shared segmentation for the demand path (_group_runs), the stash
    planner, the advisor and read_expert_into, so the four callers can never
    diverge again: a run groups CONSECUTIVE ids while ``same(first, nxt)``
    holds (reader identity for tier-aware paths; tier match for the demand
    fallback).

    With ``merge_gap > 0`` a run may BRIDGE a hole of up to merge_gap
    missing ids when the next demanded id still satisfies ``same`` — the
    hole rows are read with the run but the caller scatters only the
    demanded ids (gap rows never enter the output, the LRU or any promote).
    ``max_run`` bounds the run length (a bridge clamps to the cap).
    """
    if not eids_sorted:
        return []
    same = same if same is not None else (lambda a, b: True)
    limit = max_run if max_run is not None and max_run > 0 else None
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(eids_sorted)
    while i < n:
        first = eids_sorted[i]
        count = 1
        j = i + 1
        while j < n:
            if limit is not None and count >= limit:
                break
            nxt = eids_sorted[j]
            gap = nxt - (first + count)
            if gap == 0 and same(first, nxt):
                count += 1
                j += 1
                continue
            if merge_gap > 0 and 1 <= gap <= merge_gap and same(first, nxt):
                add = gap + 1
                if limit is not None and count + add > limit:
                    add = max(1, limit - count)
                count += add
                # Consume nxt only when the bridge fully covers it; a
                # clamped bridge leaves nxt for the NEXT run (the old
                # demand planner advanced by covered row count, and the
                # max_run clamp must not swallow demanded ids).
                if add == gap + 1:
                    j += 1
                continue
            break
        runs.append((first, count))
        i = j
    return runs

class _ReadParams(NamedTuple):
    """Immutable, precomputed per-key read parameters.

    The safetensors header is fixed after __init__, so once derived a key's
    params never need re-validation on the hot path (no per-call shape/dtype
    re-derivation, no repeated size-mismatch arithmetic).
    """

    shape: tuple[int, ...]
    dtype_str: str
    np_dtype: np.dtype
    item: int
    num_experts: int
    expert_bytes: int
    tensor_abs_off: int

    @property
    def per_shape(self) -> tuple[int, ...]:
        return self.shape[1:] if len(self.shape) > 1 else self.shape


class _ShardReader:
    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        self._rp: dict[str, _ReadParams] = {}
        hsize = struct.unpack("<Q", self._file.read(8))[0]
        self.header: dict = json.loads(self._file.read(hsize))
        self.data_start = 8 + hsize
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            self._mmap.madvise(mmap.MADV_RANDOM)  # type: ignore[attr-defined]
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._mmap is not None:
                self._mmap.close()
        except Exception:
            pass
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass
        self._mmap = None  # type: ignore[assignment]
        self._file = None  # type: ignore[assignment]

    def _rp_for(self, key: str) -> _ReadParams:
        """Precompute (and cache) the immutable per-key read parameters.

        The safetensors header is fixed after __init__, so once derived a key's
        params never need re-validation on the hot path (no per-call shape/dtype
        re-derivation, no repeated size-mismatch arithmetic). Raises ValueError
        on a header size mismatch — computed once, then cached so it stays stable.
        """
        rp = self._rp.get(key)
        if rp is not None:
            return rp
        entry = self.header[key]
        shape = tuple(entry["shape"])
        dtype_str = str(entry["dtype"])
        np_dtype, item = _DTYPE_MAP.get(dtype_str, (None, None))  # type: ignore[assignment]
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype {dtype_str} for {key}")
        start, end = entry["data_offsets"]
        n_elements = 1
        for d in shape:
            n_elements *= int(d)
        expected_bytes = n_elements * int(item)
        if (end - start) != expected_bytes:
            raise ValueError(f"Header size mismatch for {key}: {end-start} vs {expected_bytes}")
        num_experts = shape[0] if shape else 1
        expert_bytes = (end - start) // num_experts
        if expert_bytes % item != 0:
            # Slices would misalign against the dtype; surface it rather than
            # produce a silently-corrupt view.
            raise ValueError(
                f"Unaligned expert slice for {key}: expert_bytes={expert_bytes} "
                f"not divisible by itemsize {item}"
            )
        rp = _ReadParams(
            shape=shape,
            dtype_str=dtype_str,
            np_dtype=np_dtype,
            item=item,
            num_experts=num_experts,
            expert_bytes=expert_bytes,
            tensor_abs_off=self.data_start + int(start),
        )
        self._rp[key] = rp
        return rp

    def _read_into(self, abs_off: int, out: np.ndarray) -> None:
        """Zero-copy read of out.nbytes bytes at abs_off into the writable
        uint8 buffer out.

        os.preadv writes straight into the buffer — no intermediate heap copy
        (the old path did pread -> bytes -> bytearray -> view, a double copy).
        Raises OSError on a short read or IO error. Falls back to os.pread +
        copy when preadv is unavailable; short reads still surface as OSError
        so the caller can fail-high.
        """
        n = out.nbytes
        fd = self._file.fileno()
        try:
            got = os.preadv(fd, [memoryview(out)], abs_off)
        except (AttributeError, OSError):
            data = os.pread(fd, n, abs_off)
            if len(data) != n:
                raise OSError(f"Short read at {abs_off}: {len(data)} of {n}") from None
            out[:] = np.frombuffer(data, dtype=np.uint8)
            return
        if got != n:
            raise OSError(f"Short read (preadv) at {abs_off}: {got} of {n}") from None

    def expert_slice(self, key: str, expert_id: int) -> np.ndarray:
        rp = self._rp_for(key)
        off = rp.tensor_abs_off + expert_id * rp.expert_bytes
        # Single zero-copy preadv into a writable buffer; the typed view is a
        # zero-copy reinterpret (no bytearray double-copy as the old path had).
        out = np.empty(rp.expert_bytes, dtype=np.uint8)
        self._read_into(off, out)
        return np.frombuffer(out, dtype=rp.np_dtype).reshape(rp.per_shape)

    def expert_byte_range(self, key: str, expert_id: int) -> Tuple[int, int]:
        """Absolute file offsets (start, end) of one expert's slice."""
        rp = self._rp_for(key)
        off = rp.tensor_abs_off + expert_id * rp.expert_bytes
        return off, off + rp.expert_bytes

    def advise_range(self, offset: int, length: int) -> bool:
        """Kernel readahead hint (F_RDADVISE) for one file range.

        Tells macOS to start pulling [offset, offset+length) into the page
        cache asynchronously — no userspace copy, no buffer, nothing to
        free. Used to overlap the NVMe fetch of a predicted demand set with
        GPU compute. Best-effort: returns False on any failure.
        """
        if _F_RDADVISE is None or length <= 0:
            return False
        try:
            fd = self._file.fileno()
            end = offset + length
            pos = offset
            while pos < end:
                chunk = min(end - pos, 0x7FFFFFFF)
                fcntl.fcntl(fd, _F_RDADVISE, _RADVISORY.pack(pos, chunk))
                pos += chunk
            return True
        except Exception:
            return False

    def expert_run(self, key: str, first_id: int, count: int) -> list:
        """One preadv covering experts [first_id, first_id+count), sliced per expert.

        Row-major stacked banks give consecutive ids contiguous offsets, so a
        run reads back as one sequential transfer — fewer syscalls and larger
        requests than per-expert preads (matters most when the demand set is
        dense, i.e. long-prompt prefill). The single buffer stays alive via the
        returned per-expert views (zero-copy, no bytearray double-copy).
        """
        rp = self._rp_for(key)
        count = max(1, min(int(count), rp.num_experts - first_id))
        off = rp.tensor_abs_off + first_id * rp.expert_bytes
        buf = np.empty(rp.expert_bytes * count, dtype=np.uint8)
        self._read_into(off, buf)
        per = rp.per_shape
        per_elements = rp.expert_bytes // rp.item
        return [
            np.frombuffer(buf, dtype=rp.np_dtype, count=per_elements, offset=i * rp.expert_bytes).reshape(per)
            for i in range(count)
        ]
    def pin_expert(self, key: str, expert_id: int) -> int:
        """mlock the page-aligned file range of one expert slice.

        Returns the locked byte count (page-rounded) or 0 on failure. The
        locked pages are the file-cache pages themselves — no copy, no
        committed anonymous memory (wired, though: it cannot be evicted).
        """
        off, end = self.expert_byte_range(key, expert_id)
        length = end - off
        ok = _mlock_range(self._mmap, off, length)
        if not ok:
            return 0
        start = (off // _PAGE_SIZE) * _PAGE_SIZE
        end_pg = min(len(self._mmap), ((end + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE)
        return end_pg - start


_COLD_BANK_MARKERS = (
    ".switch_mlp.gate_proj.",
    ".switch_mlp.up_proj.",
    ".switch_mlp.down_proj.",
    ".switch_mlp.gate_up_proj.",
)


# Fase I6: the env override is the bench/developer opt-in — an EMPTY env
# means "no opinion" (None) so the runtime contract stays "unset = uniform
# tier"; the settings key (expert_streaming_hot_fraction) is the per-model UI.
HOT_FRACTION_ENV: str | None = os.environ.get("OMLX_EXPERT_STREAMING_HOT_FRACTION", "") or None
HOT_FRACTION_DEFAULT = float(HOT_FRACTION_ENV or 0.25)


def load_hot_set_from_profile(
    profile_path: str | Path,
    hot_fraction: float,
    num_experts: int | None = None,
) -> dict[str, set]:
    """HOBBIT hot set (Fase I6) from a learned pin profile.

    The profile's `freq` maps layer -> [[expert, count], ...]; the top
    ceil(fraction * experts) by count per layer keep the ORIGINAL packing
    while the rest read the cold tier. Keys are the backing's bare layer
    keys ("layer_<i>"); MTP stages are absent from profiles (uniform cold
    there). Missing profile → empty dict (split stays off, uniform I5).

    ``num_experts`` (I6 fix) is the REAL per-layer expert width from the
    model estimate; the fraction's denominator must be it, not the number
    of recorded profile entries — the profile keep cap (and old profiles)
    truncate the record list, which made ceil(0.25 * 64) == 16 an
    arbitrary id-prefix selection on a 288-expert model. The hot count is
    still clamped to the available records (a sparse profile cannot elect
    experts it never observed)."""
    try:
        data = json.loads(Path(profile_path).read_text())
        freq = data.get("freq") or {}
        if not freq or hot_fraction <= 0.0:
            return {}
        hot: dict[str, set] = {}
        for layer_key, pairs in freq.items():
            counts = [(int(e), int(c)) for e, c in pairs]
            if not counts:
                continue
            width = num_experts if num_experts and num_experts > 0 else len(counts)
            n_hot = max(1, math.ceil(hot_fraction * width))
            n_hot = min(n_hot, len(counts))  # cannot elect unseen experts
            top = sorted(counts, key=lambda kv: (-kv[1], kv[0]))[:n_hot]
            hot[f"layer_{int(layer_key)}"] = {e for e, _ in top}
        return hot
    except Exception:
        return {}


def cold_tier_status(model_path: str | Path) -> tuple[bool, str]:
    """Is a complete cold tier present for *model_path*?

    Complete = every switch_mlp bank weight key of the checkpoint exists in
    some expert_cold/ shard header (partial tiers are rejected: the runtime
    uniform-packing assumption would silently break)."""
    model_path = Path(model_path)
    cold_dir = model_path / "expert_cold"
    if not cold_dir.is_dir():
        return False, "expert_cold/ missing"
    index = model_path / "model.safetensors.index.json"
    if not index.is_file():
        return False, "no model.safetensors.index.json"
    try:
        weight_map = json.loads(index.read_text()).get("weight_map") or {}
    except Exception as e:
        return False, f"unreadable index: {e}"
    needed = {
        key
        for key in weight_map
        if key.endswith(".weight") and any(m in key for m in _COLD_BANK_MARKERS)
    }
    if not needed:
        return False, "no expert banks in the checkpoint"
    have: set[str] = set()
    for shard in cold_dir.glob("*.safetensors"):
        try:
            have.update(_read_header_keys(shard))
        except Exception:
            continue
    missing = needed - have
    if missing:
        return False, f"{len(missing)} bank key(s) missing from expert_cold/"
    return True, f"complete ({len(needed)} banks)"


def _read_header_keys(path: Path) -> set[str]:
    with path.open("rb") as f:
        hsize = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hsize))
    return {k for k in hdr if k != "__metadata__"}


class ExpertBackingStore:
    """Open shard readers and serve per-expert mx.arrays on demand."""

    def __init__(
        self,
        model_path: str | Path,
        extra_roots: list[str | Path] | None = None,
        cold_root: str | Path | None = None,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        # Optional stripe roots: per-shard files are resolved primary-first;
        # roots listed here win for shards mirrored onto them (see
        # _resolve_file) — used to stripe a big MoE across two SSDs.
        self._extra_roots = [Path(r).expanduser().resolve() for r in (extra_roots or [])]
        # Cold precision tier (Fase I5): when set, expert-bank keys that
        # exist under <model>/expert_cold/ resolve there FIRST — the whole
        # runtime (slices, runs, pins, readahead, dtypes) reads the cold
        # packing uniformly. Same filenames/key names, lower bit width.
        self.cold_root = Path(cold_root).expanduser().resolve() if cold_root else None
        self._cold_readers: Dict[str, _ShardReader] = {}
        self._cold_key_to_reader: Dict[str, _ShardReader] = {}
        # HOBBIT hot/cold split (Fase I6): {stacked_key_prefix -> set(expert_id)}
        # of experts served from the ORIGINAL (higher-precision) shards while
        # the rest read expert_cold/. Empty/absent = uniform tier (I5, every
        # expert cold). Keyed by the bank prefix (…switch_mlp.<proj>) with a
        # per-layer fallback key ("layer_<i>") for profile sources that only
        # know the layer.
        self._hot_experts: Dict[str, set] = {}
        self._readers: Dict[str, _ShardReader] = {}
        self._key_to_reader: Dict[str, _ShardReader] = {}
        self._key_to_shape_dtype: Dict[str, Tuple[Tuple[int, ...], str]] = {}
        weight_map = self._load_weight_map()
        # Pre-open readers for files that contain expert banks
        needed_files = set(weight_map.values())
        # We will lazily open per key, but also keep weight_map for lookup
        self._weight_map = weight_map
        # Build header cache lazily
        self._header_cache: Dict[str, dict] = {}
        # mlock pin tracking (expert_streaming pin mode)
        self._pinned: set = set()
        self.pinned_bytes = 0

    def _roots(self) -> list[Path]:
        return [self.model_path, *self._extra_roots]

    def _resolve_file(self, fname: str) -> Path | None:
        """Resolve a shard filename across roots (extra roots win: mirrored
        shards on the stripe SSD are the ones we want served from there)."""
        for root in reversed(self._roots()):
            p = root / fname
            if p.is_file():
                return p
        return None

    def _load_weight_map(self) -> Dict[str, str]:
        idx = self.model_path / "model.safetensors.index.json"
        if idx.is_file():
            try:
                return json.loads(idx.read_text()).get("weight_map") or {}
            except Exception:
                return {}
        # fallback: single shard case — enumerate headers
        wm: Dict[str, str] = {}
        for shard in self.model_path.glob("*.safetensors"):
            hdr = self._header_for_file(shard)
            for k in hdr.keys():
                if k == "__metadata__":
                    continue
                wm[k] = shard.name
        return wm

    def _header_for_file(self, path: Path) -> dict:
        key = str(path)
        if key in self._header_cache:
            return self._header_cache[key]
        try:
            with path.open("rb") as f:
                hsize = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hsize))
                self._header_cache[key] = hdr
                return hdr
        except Exception:
            return {}

    def set_hot_experts(self, hot: Dict[str, set] | None) -> None:
        """Declare the HOBBIT hot set: experts that keep the ORIGINAL packing
        while the cold tier is active. Keys are stacked-bank prefixes (as in
        the weight map, e.g. "…switch_mlp.gate_proj") and/or bare layer keys
        ("layer_12"); values are expert-id sets. None/empty = uniform cold."""
        self._hot_experts = {
            str(k): {int(e) for e in v} for k, v in (hot or {}).items() if v
        }

    def _hot_key_for(self, key: str, expert_id: int | None) -> str | None:
        """Hot-set key matching *key*/*expert_id*, or None when not hot."""
        if not self._hot_experts or expert_id is None:
            return None
        # bank prefix: strip the trailing ".weight"/".scales"/".biases"
        prefix = key
        for suffix in (".biases", ".scales", ".weight"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
                break
        if prefix in self._hot_experts and int(expert_id) in self._hot_experts[prefix]:
            return prefix
        # layer_<i> fallback: any prefix containing ".layers.<i>."
        m = re.search(r"\.layers\.(\d+)\.", key)
        if m:
            lk = f"layer_{m.group(1)}"
            if lk in self._hot_experts and int(expert_id) in self._hot_experts[lk]:
                return lk
        return None

    def _cold_reader_for_key(
        self, key: str, expert_id: int | None = None
    ) -> _ShardReader | None:
        """Cold-tier reader for *key*, or None (not in the cold root).
        HOBBIT hot experts (I6) never resolve cold — they stay on the
        original packing, so the demand path must fetch them from the
        source shards and compute at the source bits."""
        if self.cold_root is None:
            return None
        if self._hot_key_for(key, expert_id) is not None:
            return None
        if key in self._cold_key_to_reader:
            return self._cold_key_to_reader[key]
        fname = self._weight_map.get(key)
        if fname is None:
            return None
        cold_path = self.cold_root / fname
        if not cold_path.is_file():
            return None
        reader = self._cold_readers.get(str(cold_path))
        if reader is None:
            try:
                reader = _ShardReader(cold_path)
            except Exception:
                return None
            self._cold_readers[str(cold_path)] = reader
        if key not in reader.header:
            return None
        self._cold_key_to_reader[key] = reader
        return reader

    def _reader_for_key(self, key: str, expert_id: int | None = None) -> _ShardReader:
        # expert_id routes the HOBBIT split (I6): a hot expert resolves the
        # ORIGINAL shard even when a cold copy exists; everyone else cold.
        # The no-id lookup stays tier-blind (cold-first) for dtype/metadata
        # probes — never fall into it from the id-aware path when the id is
        # hot, or a hot expert would land on the cold packing.
        if expert_id is not None:
            k2 = (key, int(expert_id))
            if k2 in self._key_to_reader:
                return self._key_to_reader[k2]
            cold = self._cold_reader_for_key(key, int(expert_id))
            if cold is not None:
                self._key_to_reader[k2] = cold
                return cold
            reader = self._reader_for_key_source(key)
            self._key_to_reader[k2] = reader
            return reader
        if key in self._key_to_reader:
            return self._key_to_reader[key]
        cold = self._cold_reader_for_key(key)
        if cold is not None:
            return cold
        return self._reader_for_key_source(key)

    def _reader_for_key_source(self, key: str) -> _ShardReader:
        """Resolve the ORIGINAL shard for *key* — never the cold tier."""
        if ("src", key) in self._key_to_reader:
            return self._key_to_reader[("src", key)]
        fname = self._weight_map.get(key)
        if fname is None:
            # key may be a stacked name not in weight_map for sharded raw experts case;
            # try to find file containing key by scanning headers
            for root in self._roots():
                for shard in root.glob("*.safetensors"):
                    hdr = self._header_for_file(shard)
                    if key in hdr:
                        fname = shard.name
                        break
                if fname is not None:
                    break
        if fname is None:
            raise KeyError(f"Expert key {key!r} not in weight_map and not found in any shard")
        fpath = self._resolve_file(fname)
        if fpath is None:
            raise FileNotFoundError(f"Shard {fname!r} not found in any root of {self.model_path}")
        reader = self._readers.get(str(fpath))
        if reader is None:
            reader = _ShardReader(fpath)
            self._readers[str(fpath)] = reader
        self._key_to_reader[("src", key)] = reader
        return reader

    def cold_quant_params(self, key: str) -> Tuple[int, int] | None:
        """(bits, group_size) of the cold packing for *key*, from the cold
        shard's __metadata__ — None when the cold tier is not active for it."""
        cold = self._cold_reader_for_key(key)
        if cold is None:
            return None
        meta = cold.header.get("__metadata__") or {}
        try:
            return int(meta["omlx_cold_bits"]), int(meta["omlx_cold_group_size"])
        except (KeyError, TypeError, ValueError):
            return None

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        # try cached headers
        for root in self._roots():
            for shard in root.glob("*.safetensors"):
                hdr = self._header_for_file(shard)
                if key in hdr:
                    return tuple(hdr[key]["shape"])
        raise KeyError(key)

    def tensor_dtype(self, key: str) -> str | None:
        """Safetensors dtype string for *key* (e.g. "U32", "BF16"), or None."""
        try:
            reader = self._reader_for_key(key)
            return str(reader.header[key]["dtype"])
        except Exception:
            return None

    def expert_bytes(self, key: str) -> int:
        """Bytes of one expert slice for a stacked (E, ...) key, or 0."""
        try:
            entry = self._reader_for_key(key).header[key]
            shape = entry["shape"]
            num_experts = shape[0] if shape else 1
            start, end = entry["data_offsets"]
            return (end - start) // num_experts
        except Exception:
            return 0

    def load_expert(self, key: str, expert_id: int) -> mx.array:
        np_view = self.load_expert_slice(key, expert_id)
        return _np_to_mx(key, np_view, self._reader_for_key(key, expert_id).header[key]["dtype"])

    def load_expert_slice(self, key: str, expert_id: int) -> np.ndarray:
        """Return a fresh np.ndarray copy of one expert's slice (mmap-backed).

        Use this when the caller (e.g. the async prefetcher) does not want
        MLX ops allocated off the inference thread — they stay on the caller
        thread as plain numpy buffers and the inference thread promotes them
        to mx.array at use time, avoiding cross-thread stream errors.
        """
        reader = self._reader_for_key(key, expert_id)
        return reader.expert_slice(key, expert_id)

    def load_expert_run(self, key: str, first_id: int, count: int) -> list[np.ndarray]:
        """Read *count* consecutive experts starting at *first_id* in one pread.

        Adjacent expert ids occupy contiguous byte ranges in a row-major
        stacked bank, so a run of ids collapses into a single sequential
        read instead of *count* separate ones. Returns per-expert numpy
        views into the run buffer (views are safe: promotion copies).
        """
        reader = self._reader_for_key(key, first_id)
        return reader.expert_run(key, first_id, count)

    def read_expert_into(
        self,
        components: list[tuple[str, list[int]]],
        outs: list[np.ndarray],
        *,
        merge_gap: int = 0,
    ) -> bool:
        """Coalesced zero-copy read of several (key, expert-ids) components.

        For each (key, eids) in components this resolves the backing reader
        per expert (tier-aware) and issues batched preadv calls covering every
        requested expert id (row-major banks make consecutive ids one
        contiguous range), writing the raw bytes into outs[i] — a caller-owned
        writable uint8 buffer of shape (len(eids), per_expert_bytes). This
        replaces the per-expert load_expert_slice loop used by the
        demand-assembly miss path: one reader resolution per component and one
        syscall per run instead of one resolution + one read per expert.

        Fase K tier contract (HOBBIT/I6): a component's expert ids must all
        resolve to the SAME reader (same tier packing) — hot and cold copies
        of one key can have different per-expert byte sizes, so a mixed
        component cannot share an output buffer layout. The caller splits
        demand by tier before calling; if a mixed component slips through
        this returns False and the caller falls back to per-expert loads.

        The per-run preadv calls of a single component are issued
        concurrently on _run_io_pool (see _RUN_IO_QD), in batches, so a
        fragmented demand set keeps the device's queue depth above 1.

        Returns True on success, False if any component could not be served
        (caller must fall back to load_expert_slice). Bytes are written in
        expert-id order so outs[i][j] is expert eids[j].
        """
        _read_t0 = time.perf_counter_ns() if _PROFILE_READS else None
        if len(components) != len(outs):
            return False
        for (key, eids), out in zip(components, outs):
            if out.dtype != np.uint8 or out.ndim != 2:
                return False
            n = len(eids)
            # Fase K: resolve the reader per expert (tier-aware) and require
            # ONE reader for the component — mixed tiers cannot share the
            # uniform output layout.
            try:
                readers = [self._reader_for_key(key, int(e)) for e in eids]
            except Exception:
                return False
            if n == 0:
                continue
            reader = readers[0]
            if any(r is not reader for r in readers[1:]):
                return False
            try:
                rp = reader._rp_for(key)
            except Exception:
                return False
            if out.shape[0] != n or out.shape[1] != rp.expert_bytes:
                return False
            if any(eid < 0 or eid >= rp.num_experts for eid in eids):
                return False
            # Read contiguous runs separately. This keeps sparse demand from
            # over-reading the gap between the first and last expert. Fase K
            # K5: with merge_gap > 0 a run may BRIDGE holes of up to
            # merge_gap missing ids — the hole rows are read with the run
            # but the scatter below writes ONLY the demanded ids, so gap
            # bytes can never enter the output (or the LRU). Same shared
            # segmentation as the demand planner and the stash.
            order = sorted(range(n), key=lambda j: eids[j])
            sorted_ids = [eids[j] for j in order]
            segments = segment_runs(sorted_ids, merge_gap=merge_gap)
            # Map each run back to its order slice: every demanded id inside
            # [first, first+count) belongs to this run in id order. The
            # scatter base for a demanded id is (eid - first) rows into the
            # buffer — bridge rows shift the demanded ids away from the
            # leading positions, so the base can never be the index of the
            # order slice.
            runs: list[tuple[int, int, int, list[int]]] = []
            # (abs_off, first_id, count, order slice)
            for first, count in segments:
                lo = bisect.bisect_left(sorted_ids, first)
                hi = bisect.bisect_left(sorted_ids, first + count)
                off = rp.tensor_abs_off + first * rp.expert_bytes
                runs.append((off, first, count, order[lo:hi]))

            # Issue each batch's preadvs concurrently. One at a time they are
            # a chain of blocking syscalls, which pins the device's I/O queue
            # depth at 1: the device idles for a full round trip between runs.
            # Prefill largely hides this — its runs are long and contiguous, so
            # kernel readahead covers it and it already reaches ~2.6 GB/s
            # against this device's ~2.8 GB/s ceiling. Decode has no such luck:
            # its demand is a handful of scattered experts per projection and
            # depth is all it has (measured: depth 1 gave 0.46 GiB/s and 1.86
            # tok/s against the baseline's ~0.9 GiB/s and 3+ tok/s).
            #
            # Batched rather than all-at-once so peak transient memory stays
            # at _RUN_IO_QD run buffers instead of one per run — a fully
            # fragmented component is one run per expert, which would
            # otherwise double the component's footprint.
            def scatter_one(off, first, js, buf) -> None:
                # The row base for a demanded id is (eid - first) — never
                # the slice index, which is what would corrupt every id
                # behind a bridge. Rows are disjoint per descriptor, so the
                # byte content of out never depends on completion order.
                for j in js:
                    base = (eids[j] - first) * rp.expert_bytes
                    out[j, :] = buf[base : base + rp.expert_bytes]

            if len(runs) == 1:
                off, first, count, js = runs[0]
                buf = np.empty(count * rp.expert_bytes, dtype=np.uint8)
                try:
                    reader._read_into(off, buf)
                except Exception:
                    return False
                scatter_one(off, first, js, buf)
            else:
                # Sliding window (K11): keep up to _RUN_IO_QD reads in
                # flight continuously. The old drain-all-per-batch loop
                # emptied the queue at every batch boundary — a sawtooth
                # that let the device idle even though demand was waiting.
                # The window pops the OLDEST descriptor (submission order)
                # so out bytes stay deterministic descriptor by descriptor.
                io_exec = _run_io_pool()
                window: list = []  # (off, first, js, buf, future)
                ok = True
                for idx, (off, first, count, js) in enumerate(runs):
                    buf = np.empty(count * rp.expert_bytes, dtype=np.uint8)
                    window.append(
                        (off, first, js, buf, io_exec.submit(reader._read_into, off, buf))
                    )
                    if len(window) >= _RUN_IO_QD or idx == len(runs) - 1:
                        wo, wfirst, wjs, wbuf, wfut = window.pop(0)
                        try:
                            wfut.result()
                        except Exception:
                            ok = False
                        if ok:
                            scatter_one(wo, wfirst, wjs, wbuf)
                    if not ok:
                        # Drain the rest so no exception is left unretrieved
                        # and no read keeps writing into a buffer the caller
                        # has already been told to discard.
                        for _o, _f, _j, _b, fut in window:
                            try:
                                fut.result()
                            except Exception:
                                pass
                        return False
                # The window may still hold descriptors submitted on the
                # final iterations (when the run count is below the window
                # size); drain them in submission order.
                for wo, wfirst, wjs, wbuf, wfut in window:
                    try:
                        wfut.result()
                    except Exception:
                        ok = False
                    if ok:
                        scatter_one(wo, wfirst, wjs, wbuf)
                if not ok:
                    return False
            if _PROFILE_READS:
                # Fase 2 demand-read telemetry (bench-only cost).
                _READ_STATS["calls"] = _READ_STATS.get("calls", 0) + 1
                _READ_STATS["runs"] = _READ_STATS.get("runs", 0) + len(runs)
                _READ_STATS["bytes"] = _READ_STATS.get("bytes", 0) + sum(
                    cnt * rp.expert_bytes for (_o, _f, cnt, _js) in runs
                )
                _READ_STATS.setdefault("run_sizes", []).extend(
                    cnt for (_o, _f, cnt, _js) in runs
                )
                _READ_STATS.setdefault("lat_us", []).append(
                    (time.perf_counter_ns() - _read_t0) // 1000
                )
                _READ_STATS["peak_inflight"] = max(
                    _READ_STATS.get("peak_inflight", 0),
                    min(_RUN_IO_QD, len(runs)),
                )
        return True

    def advise_expert_run(
        self, key: str, first_id: int, count: int
    ) -> tuple[bool, int, int]:
        """Kernel readahead of experts [first_id, first_id+count) of one bank.

        Row-major stacked banks make a run of ids one contiguous byte range,
        so the whole run collapses into a single F_RDADVISE — the zero-copy
        readahead counterpart of load_expert_run. Under the HOBBIT split
        (I6) the run may straddle the hot/cold boundary; a run reads ONE
        reader (the one its first id resolves), so advise breaks at tier
        boundaries exactly like the demand path's _group_runs.

        Returns (ok, bytes_advised, tier_segments): bytes_advised is the
        total file range covered by the accepted advisories (Fase 2
        telemetry — the caller accumulates it, so advised_bytes stops being
        a permanent zero in the bench output), tier_segments counts the
        reader groups the run needed. ok is False when the platform lacks
        F_RDADVISE or nothing resolved.
        """
        try:
            if count <= 0:
                return (False, 0, 0)
            ids = [first_id + i for i in range(count)]
            readers = [self._reader_for_key(key, eid) for eid in ids]
            ok = False
            total_bytes = 0
            segments = 0
            # group consecutive ids sharing the same reader (tier boundary)
            i = 0
            while i < len(ids):
                j = i
                reader = readers[i]
                while j + 1 < len(ids) and readers[j + 1] is reader:
                    j += 1
                start, _ = reader.expert_byte_range(key, ids[i])
                _, end = reader.expert_byte_range(key, ids[j])
                if end > start:
                    ok = reader.advise_range(start, end - start) or ok
                    total_bytes += end - start
                    segments += 1
                i = j + 1
            return (ok, total_bytes, segments)
        except Exception:
            return (False, 0, 0)

    def pin_expert(self, key: str, expert_id: int) -> int:
        """mlock one expert's file range across the resolved shard.

        Returns locked bytes (0 on failure). Duplicate pins of the same
        (key, expert) are tracked and skipped."""
        reader = self._reader_for_key(key, expert_id)
        pkey = (str(reader.path), key, expert_id)
        if pkey in self._pinned:
            return 0
        locked = reader.pin_expert(key, expert_id)
        if locked > 0:
            self._pinned.add(pkey)
            self.pinned_bytes += locked
        return locked

    @property
    def pinned_count(self) -> int:
        return len(self._pinned)

    def close(self) -> None:
        # Fase K K1: drain the speculation workers before the readers die —
        # a live stash future would read from closed files or, worse, write
        # stale bytes into a ring past its owning engine's lifetime.
        spec = getattr(self, "spec_state", None)
        if spec is not None:
            try:
                spec.close()
            except Exception:
                pass
            try:
                self.spec_state = None  # type: ignore[attr-defined]
            except Exception:
                pass
        for readers in (self._readers, self._cold_readers):
            for r in list(readers.values()):
                try:
                    r.close()
                except Exception:
                    pass
            readers.clear()
        self._key_to_reader.clear()
