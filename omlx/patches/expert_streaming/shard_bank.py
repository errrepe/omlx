# SPDX-License-Identifier: Apache-2.0
"""Mmap-backed per-expert slice reader for MoE banks.

Follows the _SafeTensorMMap pattern from qwen4_exp (mmap + MADV_RANDOM)
but slices a stacked bank (E, O, I) per expert id.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from pathlib import Path
from typing import Dict, Tuple

import mlx.core as mx
import numpy as np


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
        u32 = np_view.astype(np.uint32) << np.uint32(16)
        f32 = u32.view(np.float32)
        return mx.array(f32).astype(mx.bfloat16)
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


class ExpertBackingStore:
    """Open shard readers and serve per-expert mx.arrays on demand."""

    def __init__(self, model_path: str | Path, extra_roots: list[str | Path] | None = None):
        self.model_path = Path(model_path).expanduser().resolve()
        # Optional stripe roots: per-shard files are resolved primary-first;
        # roots listed here win for shards mirrored onto them (see
        # _resolve_file) — used to stripe a big MoE across two SSDs.
        self._extra_roots = [Path(r).expanduser().resolve() for r in (extra_roots or [])]
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

    def _reader_for_key(self, key: str) -> _ShardReader:
        if key in self._key_to_reader:
            return self._key_to_reader[key]
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
        self._key_to_reader[key] = reader
        return reader

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

    def load_expert(self, key: str, expert_id: int) -> mx.array:
        np_view = self.load_expert_slice(key, expert_id)
        return _np_to_mx(key, np_view, self._reader_for_key(key).header[key]["dtype"])

    def load_expert_slice(self, key: str, expert_id: int) -> np.ndarray:
        """Return a fresh np.ndarray copy of one expert's slice (mmap-backed).

        Use this when the caller (e.g. the async prefetcher) does not want
        MLX ops allocated off the inference thread — they stay on the caller
        thread as plain numpy buffers and the inference thread promotes them
        to mx.array at use time, avoiding cross-thread stream errors.
        """
        reader = self._reader_for_key(key)
        return reader.expert_slice(key, expert_id)

    def close(self) -> None:
        for r in list(self._readers.values()):
            try:
                r.close()
            except Exception:
                pass
        self._readers.clear()
        self._key_to_reader.clear()
