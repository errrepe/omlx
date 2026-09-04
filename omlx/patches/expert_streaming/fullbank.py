# SPDX-License-Identifier: Apache-2.0
"""Full-bank external wrap loader for MoE expert streaming (Fase M).

Owns the mmap of a page-aligned fullbank artifact (produced by
``tools/repack_fullbank.py``) and hands out zero-copy wrapped mx.arrays that
feed the STOCK ``gather_qmm`` path — the entire bank, in checkpoint order,
with no promote/stack/remap. Used only on prefill-shaped calls; decode keeps
the pread demand path.

Adapted from jundot/omlx PR #3437 (qwen4_moe_stream) by @alytaphoenix,
Apache-2.0. Differences from the source loader:
- engagement is OPT-IN (env ``OMLX_EXPERT_STREAMING_FULLBANK=1``) and
  PREFILL-ONLY (the dispatch in streaming_switch.py decides per call);
- the manifest carries a content fingerprint binding the artifact to the
  checkpoint; a mismatch refuses engagement (silent-stale impossible);
- an INDEPENDENT canary compares a wrapped window against a ``pread`` of the
  ORIGINAL shard bytes (not the artifact itself), so corruption inside the
  artifact is caught before any request is served; canary failure disables
  fullbank for the model (warning + demand-path fallback), never kills the
  request — engagement is lazy at the first prefill call.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from pathlib import Path
from typing import Any, Callable, Optional

from omlx.utils import proc_memory

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "fullbank_experts.artifact"
_PROVIDER_NAME = "expert_fullbank"
_ENV_FLAG = "OMLX_EXPERT_STREAMING_FULLBANK"


def fullbank_enabled() -> bool:
    """Kill switch: fullbank engages only with the env flag opt-in."""
    return os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def artifact_path(model_dir: str | Path) -> Optional[str]:
    """Locate the fullbank artifact for a checkpoint.

    Default: next to the checkpoint. Override: OMLX_FULLBANK_ARTIFACT points
    at an explicit artifact path -- useful when the artifact lives on a
    dedicated volume (e.g. a faster SSD than the checkpoint's, or a dir the
    server process can write). The fingerprint check still binds it to THIS
    model dir, so a stale override is refused exactly like a stale side-car.
    """
    override = os.environ.get("OMLX_FULLBANK_ARTIFACT", "").strip()
    if override:
        p = Path(override)
        return str(p) if p.exists() else None
    p = Path(model_dir) / ARTIFACT_NAME
    return str(p) if p.exists() else None


def _read_manifest(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        mlen = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(mlen))
    page = int(manifest.get("page_size", 16384))
    return manifest, page


def fingerprint_matches(model_dir: str | Path, manifest: dict) -> bool:
    """Content binding check (never trust shape/dtype alone)."""
    fp = manifest.get("fingerprint")
    if not isinstance(fp, dict) or "config_sha" not in fp:
        return False
    try:
        from tools.repack_fullbank import _check_fingerprint
    except ImportError:
        # tools/ is not importable in installed deployments; inline the check.
        cfg_path = Path(model_dir) / "config.json"
        import hashlib

        cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16]
        if cfg_sha != fp.get("config_sha"):
            return False
        shards = fp.get("shards") or {}
        if not shards:
            return False
        for name, meta in shards.items():
            p = Path(model_dir) / name
            try:
                st = p.stat()
            except OSError:
                return False
            if st.st_size != int(meta.get("size", -1)):
                return False
            if st.st_mtime_ns != int(meta.get("mtime_ns", -1)):
                return False
        return True
    return bool(_check_fingerprint(Path(model_dir), fp))


class FullbankArtifact:
    """Owns the mmap of a fullbank artifact and hands out wrapped mx.arrays.

    The wrapped arrays are keyed by STACKED key (e.g.
    ``...switch_mlp.gate_proj.weight``) exactly as the checkpoint names them,
    so the streaming linears can look up their three bank tensors directly.
    """

    def __init__(self, path: str, model_dir: str | Path):
        self.path = str(path)
        self.model_dir = str(model_dir)
        manifest, page = _read_manifest(self.path)
        self.page_size = page
        self.manifest = manifest
        self._tensors: dict[str, dict] = manifest.get("tensors") or {}
        if not self._tensors:
            raise RuntimeError("fullbank artifact has no tensors")
        self._id: Optional[int] = None
        self._canary_done = False
        self._canary_ok = False
        self._closed = False

    def __len__(self) -> int:
        return len(self._tensors)

    def has(self, stacked_key: str) -> bool:
        return stacked_key in self._tensors

    def entry(self, stacked_key: str) -> Optional[dict]:
        return self._tensors.get(stacked_key)

    def open(self) -> None:
        if self._id is not None:
            return
        from omlx.custom_kernels.expert_bank_wrap import fast

        if not fast.is_native_available():
            raise RuntimeError("expert_bank_wrap native extension unavailable")
        self._id = fast.mmap_artifact(self.path, self.page_size)
        # External-wired provider (PR #3437 contract): name-keyed,
        # replace-on-register => registering on every load is idempotent. The
        # provider reports the GLOBAL mmap total; never unregistered (it
        # returns 0 once nothing is mapped).
        proc_memory.register_external_wired_provider(
            _PROVIDER_NAME, fast.mapped_bytes
        )

    def close(self) -> None:
        if self._id is not None:
            from omlx.custom_kernels.expert_bank_wrap import fast

            fast.close_artifact(self._id)
            self._id = None
        self._closed = True

    def wrap(self, stacked_key: str):
        """Return the mmap-backed mx.array for a stacked bank key."""
        if self._id is None:
            if self._closed:
                # close() ran (model unload path): a late wrap would silently
                # re-mmap a ~46 GiB artifact. Raise -- callers (resolve/call)
                # catch and fall back to the demand path.
                raise RuntimeError("fullbank artifact already closed")
            self.open()
        e = self._tensors[stacked_key]
        from omlx.custom_kernels.expert_bank_wrap import fast

        return fast.wrap_tensor(
            self._id, e["offset"], e["length"], e["shape"], e["dtype"]
        )

    def coverage_for(self, stacked_keys: list[str]) -> int:
        """How many of the given keys this artifact can serve."""
        return sum(1 for k in stacked_keys if k in self._tensors)

    def run_canary_once(self) -> bool:
        """Independent bit-exactness canary (once per artifact instance).

        Compares a small window of one wrapped tensor, read through its
        external Metal buffer, against an independent ``os.pread`` of the
        ORIGINAL source shard (the loader knows the source offset via the
        artifact only for wrapped bytes — but the checkpoint itself is the
        ground truth the demand path reads). Returns True on pass; on any
        failure logs a warning and returns False (caller must disable
        fullbank for the model and fall back to the demand path).
        """
        if self._canary_done:
            return bool(self._canary_ok)
        self._canary_done = True
        self._canary_ok = False
        try:
            import mlx.core as mx
            import numpy as np

            key = next(iter(sorted(self._tensors.keys())))
            e = self._tensors[key]
            wrapped = self.wrap(key)
            itemsize = {
                "U32": 4, "uint32": 4,
                "U16": 2, "uint16": 2,
                "U8": 1, "uint8": 1,
                "BF16": 2, "bfloat16": 2,
                "F16": 2, "float16": 2,
            }.get(e["dtype"], 4)
            n = min(512, e["length"] // itemsize)
            if n <= 0:
                return False
            # Read a window through the external Metal buffer (ground truth A).
            store_dtype = "uint32" if itemsize == 4 else "uint16"
            store_mx = mx.uint32 if itemsize == 4 else mx.uint16
            wrapped = self.wrap(key)
            gpu = np.array(wrapped.reshape(-1)[:n].view(store_mx))
            # Ground truth B: the SAME tensor bytes from the ORIGINAL shard
            # via the model index — not from the artifact (self-referential
            # canaries pass on corruption; this is the fix for that).
            ref = self._source_window(key, n * itemsize)
            if ref is None:
                logger.warning(
                    "fullbank canary: could not locate source tensor for %s; "
                    "not verified this load -- staying safe (demand path)",
                    key,
                )
                return False
            if not np.array_equal(gpu, ref[:n]):
                logger.warning(
                    "fullbank canary FAILED for %s: wrapped bytes do not "
                    "match the checkpoint; disabling fullbank for this model",
                    key,
                )
                return False
            self._canary_ok = True
            logger.info(
                "fullbank: independent canary passed (%s, %d vals vs source shard)",
                key,
                n,
            )
            return True
        except Exception:
            logger.warning("fullbank canary raised; disabling", exc_info=True)
            return False

    def _source_window(self, key: str, nbytes: int):
        """Ground-truth bytes for the canary, read from the ORIGINAL shard."""
        import numpy as np

        idx = Path(self.model_dir) / "model.safetensors.index.json"
        if not idx.is_file():
            return None
        try:
            weight_map = json.loads(idx.read_text()).get("weight_map") or {}
            fname = weight_map.get(key)
            if not fname:
                return None
            path = Path(self.model_dir) / fname
            with open(path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
                s, e = hdr[key]["data_offsets"]
                data_start = 8 + hlen
                f.seek(data_start + s)
                buf = f.read(nbytes)
            dt = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16,
                  "U16": np.uint16, "U8": np.uint8}.get(
                self._tensors[key]["dtype"], np.uint32
            )
            return np.frombuffer(buf, dtype=dt)
        except Exception:
            return None


def maybe_attach_fullbank(backing: Any) -> Optional[FullbankArtifact]:
    """Attach a fullbank artifact to a backing store when eligible.

    Eligibility mirrors the dispatch gate exactly: env opt-in, artifact
    present, fingerprint matches. Native availability is checked lazily at
    open() time (the first prefill call); a missing native ext degrades to
    the demand path with a one-shot log, never a load failure.
    """
    if not fullbank_enabled():
        return None
    path = artifact_path(getattr(backing, "model_path", None))
    if path is None:
        return None
    try:
        manifest, _ = _read_manifest(path)
        if not fingerprint_matches(getattr(backing, "model_path", ""), manifest):
            logger.warning(
                "fullbank artifact fingerprint mismatch for %s; staying on "
                "the demand path (re-run tools/repack_fullbank.py after "
                "checkpoint updates)",
                path,
            )
            return None
        return FullbankArtifact(path, getattr(backing, "model_path", ""))
    except Exception:
        logger.warning("fullbank artifact unreadable: %s", path, exc_info=True)
        return None
