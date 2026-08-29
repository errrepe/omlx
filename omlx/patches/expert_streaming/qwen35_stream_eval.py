# SPDX-License-Identifier: Apache-2.0
"""Per-layer eval boundary for the installed qwen3_5_moe decoder (streaming).

The streaming converter flags every converted decoder layer with
``_stream_eval``. The vendored GLM-5.3 decoder honors the flag inline
(``Glm5NextDecoderLayer.__call__``); the installed ``mlx_vlm``
``Qwen3_5MoeDecoderLayer`` does not, so on qwen4_exp the flag is inert and a
long prefill chunk accumulates one streaming mini-bank per layer in the lazy
graph until the chunk-end eval (~17 MB/token measured, intra-chunk pool
peaks ~29 GiB). This wraps the decoder with the DeepSeek/GLM boundary:
evaluate the layer output as soon as the layer returns and trim the
allocator cache so the retained pool cannot grow past the layer's working
set and evict the OS page cache the streaming path depends on (the Fase G
post-mortem's 341 s/8k case).

Prefill-shaped calls only (``x.shape[1] > 1``; batch decode is [B, 1, H]):
decode graphs are small and 48 forced syncs/token would erode the QD16 win,
and MTP verify passes (``target_verify``) stay lazy for the same reason.
Output is bit-identical — ``mx.eval`` only materializes what the next layer
reads anyway; ``mx.clear_cache`` frees cached buffers back to Metal.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Default ON: prefill is where the lazy-graph accumulation hurts and the
# boundary is bit-exact. Disable with OMLX_EXPERT_STREAMING_PER_LAYER_EVAL=0
# or the per-model setting expert_streaming_per_layer_eval=false.
_PER_LAYER_EVAL_DEFAULT = (
    os.environ.get("OMLX_EXPERT_STREAMING_PER_LAYER_EVAL", "1") != "0"
)
_per_layer_eval_enabled = _PER_LAYER_EVAL_DEFAULT

_APPLIED_FLAG = "_omlx_stream_eval_wrapped"


def configure_from_settings(value: Any) -> bool:
    """Resolve the knob (``None`` = env / built-in default) and store the
    effective flag the wrapper reads per call. Returns the effective value."""
    global _per_layer_eval_enabled
    _per_layer_eval_enabled = (
        _PER_LAYER_EVAL_DEFAULT if value is None else bool(value)
    )
    return _per_layer_eval_enabled


def _wrap_call(orig_call: Any) -> Any:
    def call(self, x, *args, **kwargs):
        out = orig_call(self, x, *args, **kwargs)
        if (
            _per_layer_eval_enabled
            and getattr(self, "_stream_eval", False)
            and not kwargs.get("target_verify", False)
            and x.ndim >= 2
            and x.shape[1] > 1
        ):
            mx.eval(out)
            mx.clear_cache()
        return out

    return call


def apply_qwen35_moe_stream_eval() -> bool:
    """Wrap ``Qwen3_5MoeDecoderLayer.__call__`` with the streaming eval
    boundary. Idempotent (the class flag short-circuits re-wrapping); the
    per-call behavior follows the current module flag so a settings change
    that reloads the engine takes effect without a process restart. Returns
    True when the installed mlx_vlm exposes the target class."""
    try:
        from mlx_vlm.models.qwen3_5_moe import language as q35
    except ImportError:
        return False
    cls = getattr(q35, "Qwen3_5MoeDecoderLayer", None)
    if cls is None or getattr(cls, _APPLIED_FLAG, False):
        return cls is not None

    cls.__call__ = _wrap_call(cls.__call__)
    cls._omlx_stream_eval_wrapped = True
    logger.info("Qwen3.5/qwen4_exp streaming per-layer eval boundary installed")
    return True
