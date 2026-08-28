# SPDX-License-Identifier: Apache-2.0
"""Adaptive top-k truncation for MoE routing (opt-in quality/speed knob).

Ports the cumulative-mass idea from macqwen-releases (FlashNext,
MIT licensed): after the router's top-k selection, keep the smallest
score-descending prefix whose cumulative relative mass reaches
``threshold``. Dropped slots are padded with the top expert at score 0
(the duplicate collapses in the streaming plan, so no extra expert I/O)
and the kept scores are renormalized to the ORIGINAL total top-k mass
(blend 1.0), preserving activation magnitude.

Bit-exactness contract: ``threshold`` None or >= 1.0 bypasses everything
— the stock routing body runs untouched. Only 0 < threshold < 1.0
engages the approximation, and it changes outputs by design.

Applies to:
  * qwen4_exp (inherited ``Qwen3_5MoeSparseMoeBlock`` from installed
    mlx_vlm.models.qwen3_5_moe) — monkey-patched, mirroring the
    qwen35_moe_router.py convention;
  * glm5_next — a direct hook in the vendored ``Glm5NextMoE.__call__``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_THRESHOLD: float | None = None
_MEAN_KEEPS: float | None = None

_MIN_THRESHOLD = 0.05
_MAX_THRESHOLD = 1.0


def configure(threshold: float | None) -> None:
    """Set the active routing threshold (None or >= 1.0 = exact)."""
    global _THRESHOLD
    if threshold is None:
        _THRESHOLD = None
        return
    t = float(threshold)
    if not (_MIN_THRESHOLD <= t <= _MAX_THRESHOLD):
        raise ValueError(
            f"top-k threshold must be in [{_MIN_THRESHOLD}, 1.0], got {t}"
        )
    _THRESHOLD = None if t >= 1.0 else t
    if _THRESHOLD is not None:
        logger.info("Adaptive top-k truncation active: threshold=%.2f", _THRESHOLD)


def configure_from_settings(settings: Any | None) -> float | None:
    """Resolve the threshold from ModelSettings with an env fallback."""
    t = getattr(settings, "expert_streaming_topk_threshold", None) if settings else None
    if t is None:
        env = os.environ.get("OMLX_MOE_TOPK_THRESHOLD", "")
        if env.strip():
            try:
                t = float(env)
            except ValueError:
                logger.warning("Invalid OMLX_MOE_TOPK_THRESHOLD=%r; ignoring", env)
    configure(t)
    return _THRESHOLD


def current_threshold() -> float | None:
    return _THRESHOLD


def mean_keeps() -> float | None:
    """Mean kept experts per routed row of the last truncation pass (None
    when truncation is inactive or has not run)."""
    return _MEAN_KEEPS


def truncate_topk_mass(inds, scores, threshold: float, return_keeps: bool = False):
    """Truncate a top-k routing by cumulative relative mass.

    ``inds``: [..., k] expert ids; ``scores``: [..., k] routing scores
    (any positive scaling — relative mass is used). Returns
    (inds, scores) with the same shapes: kept experts in score-descending
    order, dropped slots holding the top expert at score 0, kept scores
    renormalized to the original total top-k mass.
    """
    global _MEAN_KEEPS
    total = mx.sum(scores, axis=-1, keepdims=True)
    rel = scores / mx.maximum(total, 1e-30)
    order = mx.argsort(-rel, axis=-1)
    s_sorted = mx.take_along_axis(rel, order, axis=-1)
    i_sorted = mx.take_along_axis(inds, order, axis=-1)
    # mass accumulated BEFORE the current expert — keep while it is still
    # below the threshold (the first expert always keeps)
    cum_before = mx.cumsum(s_sorted, axis=-1) - s_sorted
    keep = (cum_before < threshold).astype(scores.dtype)
    first = i_sorted[..., :1]
    i_pad = mx.where(keep > 0, i_sorted, first)
    s_pad = s_sorted * keep
    denom = mx.maximum(mx.sum(s_pad, axis=-1, keepdims=True), 1e-30)
    s_final = (s_pad / denom) * total
    if return_keeps:
        keeps = mx.sum(keep, axis=-1).mean().item()
        _MEAN_KEEPS = keeps
        return i_pad.astype(inds.dtype), s_final.astype(scores.dtype), keeps
    _MEAN_KEEPS = None
    return i_pad.astype(inds.dtype), s_final.astype(scores.dtype)


def apply_qwen35_moe_topk_patch() -> bool:
    """Engage truncation for Qwen3.5/3.6/qwen4_exp sparse MoE blocks.

    Wraps ``Qwen3_5MoeSparseMoeBlock.__call__`` in the installed mlx_vlm
    (qwen3_5_moe.language, shared by the vendored qwen4_exp). When the
    active threshold is exact (None/1.0) the wrapped call — stock or the
    fused-router patch — runs untouched.
    """
    try:
        from mlx_vlm.models.qwen3_5_moe import language as q35
    except ImportError:
        return False
    cls = getattr(q35, "Qwen3_5MoeSparseMoeBlock", None)
    if cls is None or getattr(cls, "_omlx_topk_truncate", False):
        return cls is not None

    orig_call = cls.__call__

    def patched_call(self, x, target_verify: bool = False):
        thr = current_threshold()
        if thr is None or thr >= 1.0:
            return orig_call(self, x, target_verify=target_verify)
        try:
            gates = q35._target_verify_linear(self.gate, x, target_verify)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)
            inds, scores = truncate_topk_mass(inds, scores, thr)
            y = q35._target_verify_switch_glu(self.switch_mlp, x, inds, target_verify)
            y = (y * scores[..., None]).sum(axis=-2)
            shared_y = self.shared_expert(x, target_verify)
            shared_y = (
                mx.sigmoid(q35._target_verify_linear(self.shared_expert_gate, x, target_verify))
                * shared_y
            )
            return y + shared_y
        except Exception:
            logger.warning("adaptive top-k routing failed; stock fallback", exc_info=True)
            return orig_call(self, x, target_verify=target_verify)

    cls.__call__ = patched_call
    cls._omlx_topk_truncate = True
    logger.info("Qwen3.5/qwen4_exp adaptive top-k patch applied")
    return True
