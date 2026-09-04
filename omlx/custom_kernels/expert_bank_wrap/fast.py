# SPDX-License-Identifier: Apache-2.0
"""Native wrappers for the full-bank external wrap (Fase M).

Thin, ABI-guarded surface over the ``_ext`` nanobind module. If the native
extension is absent or built against a mismatched nanobind ABI, every entry
point degrades gracefully: :func:`is_native_available` returns ``False`` and
the streaming stack falls back to the demand (pread) path — no streaming
fullbank. This mirrors the glm_moe_dsa / qwen35_prefill pattern.

Adapted from jundot/omlx PR #3437 (qwen4_moe_stream) by @alytaphoenix,
Apache-2.0.
"""

from __future__ import annotations

import logging
from pathlib import Path

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

try:
    from . import _ext  # type: ignore
except Exception as exc:  # noqa: BLE001
    _ext = None
    _IMPORT_ERROR: Exception | None = exc
    if any(Path(__file__).parent.glob("_ext*.so")):
        logger.warning(
            "expert_bank_wrap native extension is present but failed to load; "
            "fullbank expert streaming disabled: %s",
            exc,
        )
else:
    _IMPORT_ERROR = None

_ABI_OK: bool | None = None


def _verify_abi() -> bool:
    """One-shot nanobind-ABI canary: pass an mx.array through the native
    boundary. A mismatched nanobind isolates the ``mlx`` NB_DOMAIN and this
    raises, so we disable the native path instead of risking a hard crash."""
    global _ABI_OK
    if _ABI_OK is not None:
        return _ABI_OK
    if _ext is None:
        _ABI_OK = False
        return False
    try:
        import mlx.core as mx

        probe = mx.zeros((3,), dtype=mx.uint32)
        _ABI_OK = int(_ext.abi_probe(probe)) == 3
    except Exception as exc:  # noqa: BLE001
        logger.warning("expert_bank_wrap ABI probe failed; disabling: %s", exc)
        _ABI_OK = False
    return _ABI_OK


def is_native_available() -> bool:
    return _ext is not None and _verify_abi()


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def mmap_artifact(path: str, page_size: int) -> int:
    """mmap a page-aligned fullbank artifact; returns an opaque id."""
    if not is_native_available():
        raise RuntimeError("expert_bank_wrap native extension unavailable")
    return int(_ext.mmap_artifact(str(path), int(page_size)))


def close_artifact(artifact_id: int) -> None:
    if _ext is not None:
        _ext.close_artifact(int(artifact_id))


def mapped_bytes() -> int:
    """Total bytes currently mmap'd across all live artifacts (worst-case
    external-wired figure for the registry in omlx.utils.proc_memory)."""
    if _ext is None:
        return 0
    return int(_ext.mapped_bytes())


def wrap_tensor(artifact_id: int, offset: int, length: int, shape, dtype: str):
    """Return an mx.array viewing a page-aligned tensor region of the artifact,
    backed by external mmap memory (no copy)."""
    if not is_native_available():
        raise RuntimeError("expert_bank_wrap native extension unavailable")
    return _ext.wrap_tensor(int(artifact_id), int(offset), int(length),
                            [int(d) for d in shape], str(dtype))
