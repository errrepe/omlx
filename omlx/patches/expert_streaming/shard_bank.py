# SPDX-License-Identifier: Apache-2.0
"""Mmap-backed per-expert slice reader for MoE banks.

Follows the _SafeTensorMMap pattern from qwen4_exp (mmap + MADV_RANDOM)
but slices a stacked bank (E, O, I) per expert id.
"""

from __future__ import annotations

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
from pathlib import Path
from typing import Dict, Tuple

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


class _ShardReader:
    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
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

    def expert_slice(self, key: str, expert_id: int) -> np.ndarray:
        entry = self.header[key]
        shape = tuple(entry["shape"])
        dtype_str = str(entry["dtype"])
        start, end = entry["data_offsets"]
        np_dtype, item = _DTYPE_MAP.get(dtype_str, (None, None))  # type: ignore[assignment]
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype {dtype_str} for {key}")
        # Build view over the whole stacked tensor, then index expert_id on axis 0
        # For ranks >2, the view is (E, ...) — slicing axis 0 is contiguous for that expert
        # because safetensors is row-major: expert slice offset = expert_id * stride
        n_elements = 1
        for d in shape:
            n_elements *= int(d)
        expected_bytes = n_elements * int(item)
        if (end - start) != expected_bytes:
            raise ValueError(f"Header size mismatch for {key}: {end-start} vs {expected_bytes}")
        # One large os.pread instead of mmap page faults. With MADV_RANDOM the
        # cold-fault path tops out around 0.2-0.4 GB/s (16K faults, no
        # readahead), while a single pread of the contiguous slice sustains
        # several GB/s on the same NVMe — a ~10-30x difference per bundle.
        # expert_id indexes axis 0 of a row-major tensor, so the slice is one
        # contiguous byte range: expert_bytes = (end-start) / E.
        num_experts = shape[0] if shape else 1
        expert_bytes = (end - start) // num_experts
        fd = self._file.fileno()
        buf = bytearray(os.pread(fd, expert_bytes, self.data_start + int(start) + expert_id * expert_bytes))
        if len(buf) != expert_bytes:
            raise OSError(f"Short read for {key}[{expert_id}]: {len(buf)} of {expert_bytes}")
        return np.frombuffer(buf, dtype=np_dtype).reshape(shape[1:] if len(shape) > 1 else shape)

    def expert_byte_range(self, key: str, expert_id: int) -> Tuple[int, int]:
        """Absolute file offsets (start, end) of one expert's slice."""
        entry = self.header[key]
        start, end = entry["data_offsets"]
        shape = entry["shape"]
        num_experts = shape[0] if shape else 1
        expert_bytes = (end - start) // num_experts
        off = self.data_start + int(start) + expert_id * expert_bytes
        return off, off + expert_bytes

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
        """One pread covering experts [first_id, first_id+count), sliced per expert.

        Row-major stacked banks give consecutive ids contiguous offsets, so a
        run reads back as one sequential transfer — fewer syscalls and larger
        requests than per-expert preads (matters most when the demand set is
        dense, i.e. long-prompt prefill).
        """
        entry = self.header[key]
        shape = tuple(entry["shape"])
        dtype_str = str(entry["dtype"])
        start, end = entry["data_offsets"]
        np_dtype, item = _DTYPE_MAP.get(dtype_str, (None, None))  # type: ignore[assignment]
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype {dtype_str} for {key}")
        num_experts = shape[0] if shape else 1
        count = max(1, min(int(count), num_experts - first_id))
        expert_bytes = (end - start) // num_experts
        off = self.data_start + int(start) + first_id * expert_bytes
        buf = os.pread(self._file.fileno(), expert_bytes * count, off)
        if len(buf) != expert_bytes * count:
            raise OSError(f"Short run read for {key}[{first_id}+{count}]: {len(buf)}")
        per_shape = shape[1:] if len(shape) > 1 else shape
        n_elements = 1
        for d in shape:
            n_elements *= int(d)
        n_elements //= num_experts
        return [
            np.frombuffer(buf, dtype=np_dtype, count=n_elements, offset=i * expert_bytes).reshape(per_shape)
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

    def advise_expert_run(self, key: str, first_id: int, count: int) -> bool:
        """Kernel readahead of experts [first_id, first_id+count) of one bank.

        Row-major stacked banks make a run of ids one contiguous byte range,
        so the whole run collapses into a single F_RDADVISE — the zero-copy
        readahead counterpart of load_expert_run. Under the HOBBIT split
        (I6) the run may straddle the hot/cold boundary; a run reads ONE
        reader (the one its first id resolves), so advise breaks at tier
        boundaries exactly like the demand path's _group_runs. Returns
        False when the platform lacks F_RDADVISE or nothing resolved.
        """
        try:
            if count <= 0:
                return False
            ids = [first_id + i for i in range(count)]
            readers = [self._reader_for_key(key, eid) for eid in ids]
            ok = False
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
                i = j + 1
            return ok
        except Exception:
            return False

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
        for r in list(self._readers.values()):
            try:
                r.close()
            except Exception:
                pass
        self._readers.clear()
        self._key_to_reader.clear()
