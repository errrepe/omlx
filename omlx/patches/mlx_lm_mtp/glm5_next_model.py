# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (glm5_next, mlx-vlm vendored) native Lightning MTP head.

JANG-MTP checkpoints (GLM-5.3-Flash-JANG-MTP) declare
``num_nextn_predict_layers: 1`` and ship the draft as one extra decoder
layer (index ``num_hidden_layers``) with the DeepSeek-V3 nextn layout:
RMSNorm'd trunk hidden + next-token embedding (``hnorm``/``enorm``)
concatenated, projected back to ``hidden_size`` (``eh_proj``), run
through one full sparse-attention + 288-expert MoE decoder layer, and
read out through ``shared_head.norm`` + the shared ``lm_head``. That is
the same head shape as GLM-5.2's glm_moe_dsa nextn layer — this patch
mirrors ``glm_moe_dsa_model`` onto the vendored mlx-vlm module.

The compat patch (``mlx_vlm_glm5_next_compat``) registers
``mlx_vlm.models.glm5_next``; this patch sits on top and adds:

- ``Glm5NextMTPBlock``: enorm/hnorm/eh_proj fusion + a full
  ``Glm5NextDecoderLayer`` (synthesized sparse config entry).
- ``LanguageModel.__init__`` wrap: attach ``self.mtp`` when the load-time
  MTP flag is active; stamp the chain/depth/head-prenorm markers that
  ``batch_generator._resolve_mtp_chain_depth`` reads.
- ``LanguageModel.__call__`` wrap: ``return_hidden=True`` returns the
  pre-norm trunk hidden (the chain applies the final norm via
  ``_HEAD_HIDDEN_POST_NORM``; ``hnorm`` re-normalises inside the block).
- ``mtp_forward`` / ``make_mtp_cache``: the chain-cycle contract.
- ``mtp_partial_rollback``: CacheList(KVCache + PoolingCache) trim for
  the DSA layers and ArraysCache rollback for the linear-attention
  trunk layers — the batched-verify semantics the vendored README
  deferred are exactly this shim.

Apply order: the compat patch must register the module first; this
patch's ``apply()`` no-ops when ``mlx_vlm.models.glm5_next`` is absent.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _is_our_method(cls: Any, attr: str, marker: str) -> bool:
    existing = cls.__dict__.get(attr)
    return getattr(existing, marker, False)


def apply() -> bool:
    """Apply the glm5_next MTP patches when the vendored module is present."""
    glm = sys.modules.get("mlx_vlm.models.glm5_next")
    if glm is None or not hasattr(glm, "LanguageModel"):
        logger.debug(
            "glm5_next module not registered; skipping MTP patch (expected "
            "for non-GLM-5.3 models)"
        )
        return False

    _patch_language_model(glm)
    if not hasattr(glm.LanguageModel, "_omlx_mtp_patched"):
        glm.LanguageModel._omlx_mtp_patched = "patch"
        logger.info("GLM-5.3 (glm5_next) MTP model patch applied")
    return True


# ---------------------------------------------------------------------------
# Module-path counterpart of the wrapper sanitize's weight-key special map.
# ---------------------------------------------------------------------------

_MTP_QUANT_SPECIAL = {
    "eh_proj": "eh_proj",
    "enorm": "enorm",
    "hnorm": "hnorm",
    "shared_head.norm": "norm",
}


def remap_mtp_quant_overrides(
    params: dict[str, Any], n_main: int, n_mtp: int
) -> None:
    """Copy per-module quantization overrides to the runtime MTP paths.

    ``sanitize`` renames the nextn layer's config keys from
    ``model.layers.<n_main + i>.*`` to ``mtp.<i>.*`` but per-module
    quantization lookups key on runtime module paths, so the copies must
    ride along (same trick as glm_moe_dsa's ``_remap_mtp_quant_overrides``;
    mutating ``params`` in place is enough — it is the same dict the
    loader closes over).
    """
    quant = params.get("quantization") if isinstance(params, dict) else None
    if not isinstance(quant, dict):
        return
    for k, v in list(quant.items()):
        for i in range(n_mtp):
            prefix = f"model.layers.{n_main + i}."
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix) :]
            if rest.startswith(("shared_head.head", "embed_tokens")):
                nk = None
            elif rest in _MTP_QUANT_SPECIAL:
                nk = f"mtp.{i}.{_MTP_QUANT_SPECIAL[rest]}"
            else:
                nk = f"mtp.{i}.block.{rest}"
            if nk is not None and nk not in quant:
                quant[nk] = v
            break


def _mtp_layer_config(args: Any):
    """Synthesize the draft layer's config entry.

    The checkpoint's ``layer_types`` / ``mlp_layer_types`` lists span the
    trunk layers only; the JANG draft is a sparse-attention + MoE layer
    (DSA indexer weights + switch_mlp banks present). Extend both lists
    by one sparse entry on a copy so ``Glm5NextDecoderLayer`` can
    construct at index ``num_hidden_layers``.
    """
    import copy

    cfg = copy.copy(args)
    n = args.num_hidden_layers
    layer_types = list(getattr(args, "layer_types", None) or [])
    mlp_types = list(getattr(args, "mlp_layer_types", None) or [])
    while len(layer_types) < n + 1:
        layer_types.append("deepseek_sparse_attention")
    while len(mlp_types) < n + 1:
        mlp_types.append("sparse")
    cfg.layer_types = layer_types
    cfg.mlp_layer_types = mlp_types
    return cfg


def _patch_language_model(glm: Any) -> None:
    cls = glm.LanguageModel
    init_wrapped = getattr(cls, "_omlx_mtp_init_wrapped", False)
    call_owned = _is_our_method(cls, "__call__", "_omlx_mtp_call_marker")
    if init_wrapped and call_owned:
        return

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, args, config=None):
        original_init(self, args, config)
        n_mtp = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
        from . import is_mtp_active

        mtp_decode_enabled = bool(n_mtp > 0 and is_mtp_active())
        self._omlx_mtp_decode_enabled = mtp_decode_enabled
        if mtp_decode_enabled:
            self.mtp = [_make_mtp_block(glm, _mtp_layer_config(args), args)]
            from . import get_mtp_depth

            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = get_mtp_depth()
            self._omlx_mtp_head_clone = False
            self._omlx_mtp_head_prenorm = True
            # Marginal-cost prior for the adaptive depth controller: the
            # draft's own MoE pulls a near-disjoint expert set per extra
            # verify row (same regime as GLM-5.2's 8-of-256 routing).
            self._omlx_mtp_marginal_ms = 35.0
            quant = getattr(args, "quantization", None)
            if isinstance(quant, dict):
                remap_mtp_quant_overrides(quant, int(args.num_hidden_layers), n_mtp)

    def __call__(
        self,
        inputs=None,
        inputs_embeds=None,
        cache=None,
        mask=None,
        **kwargs,
    ):
        return_hidden = bool(kwargs.pop("return_hidden", False))
        if not return_hidden:
            return original_call(
                self,
                inputs,
                inputs_embeds=inputs_embeds,
                cache=cache,
                mask=mask,
                **kwargs,
            )
        # return_hidden path: one forward, both products. The stock call
        # projects the post-norm hidden and discards the pre-norm variant;
        # the chain needs the pre-norm hidden (it applies the final norm
        # itself via _HEAD_HIDDEN_POST_NORM), so project the logits tail
        # from the same single backbone pass here.
        if inputs is None:
            inputs = kwargs.get("input_ids")
        out = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        nlk = kwargs.get("num_logits_to_keep", 0)
        out_tail = out[:, -nlk:, :] if nlk else out
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out_tail)
        else:
            from mlx_vlm.models.glm5_next.language import linear_forward

            logits = linear_forward(self.lm_head, out_tail)
        try:
            from mlx_vlm.models.base import LanguageModelOutput
        except ImportError:
            from ..base import LanguageModelOutput

        return LanguageModelOutput(logits=logits, hidden_states=out)

    def get_mtp_module(self):
        return getattr(self, "mtp", None)

    def make_mtp_cache(self):
        """One CacheList(KVCache + PoolingCache) per draft block.

        Mirrors ``LanguageModel.make_cache``'s sparse-attention entry so
        the DSA indexer's pooling window advances identically in the
        draft.
        """
        from mlx_lm.models.cache import PoolingCache

        blocks = getattr(self, "mtp", None) or []
        caches = []
        for block in blocks:
            from mlx_vlm.models.cache import KVCache, CacheList

            caches.append(
                CacheList(
                    KVCache(),
                    PoolingCache(block.block.self_attn.indexer.index_kpool),
                )
            )
        return caches

    def mtp_forward(
        self,
        h,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        """Run the draft block(s) + shared_head norm + shared lm_head.

        ``h`` is the trunk's pre-norm hidden for the history fold (the
        chain applies the trunk norm before calling when needed) or the
        head's own raw output for chained steps — ``hnorm`` inside the
        block normalises either. ``return_hidden`` returns the block's
        raw residual output (the next chain step's ``h``).
        """
        import mlx.core as mx  # noqa: F401  (block __call__ imports its own)

        if cache is None:
            cache = [None] * len(self.mtp)

        last_block = None
        for i, block in enumerate(self.mtp):
            layer_cache = cache[i] if i < len(cache) else None
            h = block(h, self.model.embed_tokens, input_ids, layer_cache)
            last_block = block

        logits_source = h
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:, :]
        normed = last_block.norm(logits_source)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            from mlx_vlm.models.glm5_next.language import linear_forward

            logits = linear_forward(self.lm_head, normed)
        if return_hidden:
            return logits, h
        return logits

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Trim the verify window back to ``accepted`` drafts on every layer.

        Sparse layers are CacheList(KVCache + PoolingCache): both expose
        ``trim`` and PoolingCache carries the cross-boundary rollback
        buffer from the DeepSeek-V4 patch set. Linear-attention layers
        are ArraysCache with a ``rollback_state`` snapshot (populated by
        the patched ``Glm5NextLinearAttention.__call__`` under verify —
        see the linear-attention snapshot hook in this package's
        ``glm5_next_linear`` module); a positional restore replaces the
        trim for those. All layers are validated before any mutation.
        """
        n = num_drafts - accepted
        if n <= 0:
            return True
        for c in cache:
            subs = getattr(c, "caches", None)
            candidates = subs if subs is not None else [c]
            for sub in candidates:
                if getattr(sub, "rollback_state", None) is not None:
                    if getattr(sub, "_mtp_draft_stash", None) is not None:
                        return False
                    continue
                if hasattr(sub, "is_trimmable") and sub.is_trimmable():
                    continue
                return False
        for c in cache:
            subs = getattr(c, "caches", None)
            candidates = subs if subs is not None else [c]
            for sub in candidates:
                rollback = getattr(sub, "rollback_state", None)
                if rollback is not None:
                    conv_snap, ssm_snap = rollback
                    sub[0] = conv_snap
                    sub[1] = ssm_snap
                    sub.rollback_state = None
                    continue
                trimmed = sub.trim(n)
                if trimmed != n:
                    logger.warning(
                        "glm5_next MTP rollback trim shortfall on %s",
                        type(sub).__name__,
                    )
                    return False
        return True

    if not init_wrapped:
        cls.__init__ = __init__
        cls._omlx_mtp_init_wrapped = True
    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls.get_mtp_module = get_mtp_module
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_forward = mtp_forward
    cls.mtp_partial_rollback = mtp_partial_rollback


def _make_mtp_block(glm: Any, layer_config: Any, args: Any):
    import mlx.nn as nn

    # The vendored package re-exports only the top-level classes; the
    # decoder layer lives on the language submodule.
    language_mod = getattr(glm, "language", None) or __import__(
        "mlx_vlm.models.glm5_next.language", fromlist=["Glm5NextDecoderLayer"]
    )
    Glm5NextDecoderLayer = language_mod.Glm5NextDecoderLayer

    class Glm5NextMTPBlock(nn.Module):
        """One MTP layer: enorm/hnorm/eh_proj fusion + a full glm5_next
        decoder layer.

        ``norm`` holds the checkpoint's ``shared_head.norm`` — applied to
        the block output before the shared lm_head (mtp_forward does the
        lm_head so ``logits_keep`` can shrink the vocab matmul).
        """

        def __init__(self):
            super().__init__()
            dim = args.hidden_size
            eps = args.rms_norm_eps
            self._hc_mult = int(getattr(args, "hc_mult", 4) or 4)
            self.enorm = nn.RMSNorm(dim, eps=eps)
            self.hnorm = nn.RMSNorm(dim, eps=eps)
            self.eh_proj = nn.Linear(2 * dim, dim, bias=False)
            self.norm = nn.RMSNorm(dim, eps=eps)
            self.block = Glm5NextDecoderLayer(
                layer_config, layer_config.num_hidden_layers
            )

        def __call__(self, h, embed_tokens, input_ids, cache):
            import mlx.core as mx

            e = self.enorm(embed_tokens(input_ids))
            x = self.eh_proj(mx.concatenate([e, self.hnorm(h)], axis=-1))
            # The decoder layer stack runs on the hyper-connection layout
            # (B, L, hc_mult, D): broadcast the fused input out and fold
            # the streams back with a mean afterwards, exactly like the
            # trunk's Glm5NextModel.__call__ does around its layer loop.
            x = mx.broadcast_to(
                x[:, :, None, :],
                (x.shape[0], x.shape[1], self._hc_mult, x.shape[2]),
            )
            x = mx.contiguous(x)
            out = self.block(x, mask=None, cache=cache)
            return out.mean(axis=2)

    return Glm5NextMTPBlock()
