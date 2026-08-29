# SPDX-License-Identifier: Apache-2.0
"""Expert streaming (SSD) patch for MoE models (glm_moe_dsa, deepseek_v4, ...)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False

SUPPORTED_TYPES = {
    "glm_moe_dsa",
    "deepseek_v32",
    "deepseek_v4",
    "deepseek_v4_mtp",
    "glm5_next",
    "glm5_next_text",
    "qwen4_exp",
    "qwen4_exp_text",
}


def is_supported_model_type(model_type: str | None) -> bool:
    if not model_type:
        return False
    return model_type.replace("-", "_").lower() in SUPPORTED_TYPES


def apply_expert_streaming_patch() -> bool:
    global _APPLIED
    if _APPLIED:
        return False
    _APPLIED = True
    logger.info("Expert streaming patch registered (post-load converter)")
    return True


def _get_budget_bytes(model_settings: Any | None, estimate: Any | None) -> int:
    if model_settings is not None:
        # Preferred name (model_settings.py:222) + legacy cache name.
        # Explicit 0 = page-cache only (no app-level LRU); None falls through
        # to the engine default below.
        for attr in ("expert_streaming_budget_gib", "expert_cache_budget_gib"):
            gib = getattr(model_settings, attr, None)
            if gib is not None:
                try:
                    return max(0, int(float(gib) * 1024**3))
                except (TypeError, ValueError):
                    continue
        # legacy mib
        for attr in ("expert_streaming_budget_mib", "expert_cache_budget_mib"):
            mib = getattr(model_settings, attr, None)
            if mib is not None and int(mib) > 0:
                return int(int(mib) * 1024 * 1024)
    # default: page-cache only. The OS file cache serves expert reuse from
    # clean evictable pages; measured A/B on Qwen3.8-Flash-Next and
    # GLM-5.3-Flash showed it beats a 1-8 GiB app-level LRU in both cold and
    # warm runs while using several GiB less RSS. Pass an explicit budget >0
    # to opt back into the LRU heap.
    return 0


# SwitchGLU bank key prefixes per main layer. GLM/Qwen nest the MoE under
# ``mlp``; DeepSeek V4 nests it under ``ffn`` (and MTP stages under
# ``mtp.<stage>`` — see _mtp_candidate_stacked_keys).
_MAIN_SWITCH_PREFIX_TEMPLATES = (
    "model.layers.{i}.mlp.switch_mlp",
    "model.layers.{i}.ffn.switch_mlp",
    "model.language_model.layers.{i}.mlp.switch_mlp",
    "model.language_model.layers.{i}.ffn.switch_mlp",
    "language_model.model.layers.{i}.mlp.switch_mlp",
    "language_model.model.layers.{i}.ffn.switch_mlp",
    "language_model.layers.{i}.mlp.switch_mlp",
    "language_model.layers.{i}.ffn.switch_mlp",
)


def _candidate_stacked_keys(layer_idx: int, proj: str, suffix: str) -> list[str]:
    return [
        f"{template.format(i=layer_idx)}.{proj}.{suffix}"
        for template in _MAIN_SWITCH_PREFIX_TEMPLATES
    ]


def _mtp_candidate_stacked_keys(stage_idx: int, proj: str, suffix: str) -> list[str]:
    """Bank key candidates for one DeepSeek V4 MTP/DSpark stage.

    DSpark checkpoints (0731) store ``mtp.<stage>.ffn.switch_mlp.*``; the
    legacy MTPBlock layout nests one level deeper under ``block``.
    """
    return [
        f"mtp.{stage_idx}.ffn.switch_mlp.{proj}.{suffix}",
        f"mtp.{stage_idx}.block.ffn.switch_mlp.{proj}.{suffix}",
    ]


def _resolve_stacked_key(
    candidates: list[str],
    proj: str,
    suffix: str,
    backing: Any | None,
    needle: str,
) -> str:
    """Pick the checkpoint key for one stacked bank.

    Prefers exact candidates present in the weight map, then any key
    containing *needle* (layer/scope disambiguation) plus the
    ``switch_mlp.<proj>.<suffix>`` middle. Falls back to the first
    candidate for RAM dicts / missing maps.
    """
    if backing is not None and hasattr(backing, "_weight_map"):
        wm = getattr(backing, "_weight_map", {}) or {}
        for cand in candidates:
            if cand in wm:
                return cand
        mid = f"switch_mlp.{proj}.{suffix}"
        for k in wm:
            if needle in k and mid in k:
                return k
    return candidates[0]


def _model_config_candidates(model: Any) -> list[Any]:
    """Collect potential config objects for dim resolution (LLM + VLM wrappers)."""
    candidates = []
    for obj in [
        getattr(model, "args", None),
        getattr(getattr(model, "model", None), "args", None),
        getattr(getattr(model, "language_model", None), "args", None),
        getattr(getattr(getattr(model, "language_model", None), "model", None), "args", None),
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "language_model", None), "config", None),
        getattr(getattr(getattr(model, "language_model", None), "model", None), "config", None),
    ]:
        if obj is not None:
            candidates.append(obj)
    return candidates


def _resolve_moe_dims(cfg_candidates: list[Any]) -> tuple[int, int]:
    """Resolve (hidden_size, moe_intermediate_size) from config candidates."""
    hidden: int | None = None
    moe_hidden: int | None = None
    for cand in cfg_candidates:
        try:
            h = getattr(cand, "hidden_size", None)
            if h is None and isinstance(cand, dict):
                h = cand.get("hidden_size")
            if h is not None:
                hidden = int(h)
            m = getattr(cand, "moe_intermediate_size", None)
            if m is None and isinstance(cand, dict):
                m = cand.get("moe_intermediate_size")
            if m is not None:
                moe_hidden = int(m)
            if hidden is not None and moe_hidden is not None:
                break
        except Exception:
            continue
    if hidden is None:
        hidden = 4096
    if moe_hidden is None:
        # qwen4_exp default is 640, glm5_next/deepseek_v4 are 2048; try to
        # infer from first expert — keep original default 1407 for glm_moe_dsa
        moe_hidden = 1407
    # Override for known types when fallback is still generic
    try:
        mt = None
        for c in cfg_candidates:
            mt = getattr(c, "model_type", None) or (c.get("model_type") if isinstance(c, dict) else None)
            if mt:
                break
        mt = str(mt).lower().replace("-", "_") if mt else ""
        if mt in ("qwen4_exp", "qwen4_exp_text") and moe_hidden == 1407:
            moe_hidden = 640
        elif mt in ("glm5_next", "glm5_next_text", "deepseek_v4", "deepseek_v4_mtp") and moe_hidden == 1407:
            moe_hidden = 2048
    except Exception:
        pass
    return hidden, moe_hidden


def _convert_switch_mlp_module(
    moe: Any,
    layer_idx: int,
    *,
    candidates_for: Any,
    needle: str,
    backing: Any,
    backing_kind: str,
    cache: Any,
    estimate: Any,
    hidden: int,
    moe_hidden: int,
    layer: Any | None = None,
) -> bool:
    """Replace *moe*.switch_mlp with a StreamingSwitchGLU. Returns True on success.

    ``candidates_for(proj, suffix)`` yields the checkpoint key candidates for
    this module's stacked banks; ``needle`` disambiguates weight-map fallback
    scans (e.g. ``layers.5.`` or ``mtp.2.``).
    """
    import mlx.core as mx

    from .streaming_switch import (
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    switch_mlp = getattr(moe, "switch_mlp", None)
    if switch_mlp is None:
        return False

    # Determine quantized vs bf16: QuantizedSwitchLinear has 'scales'
    is_quantized = False
    for attr in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
        proj = getattr(switch_mlp, attr, None)
        if proj is not None:
            if hasattr(proj, "scales") or "scales" in getattr(proj, "_data", {}):
                is_quantized = True
                break
            if proj.__class__.__name__ == "QuantizedSwitchLinear":
                is_quantized = True
                break

    n_experts = estimate.experts_per_layer

    fused = hasattr(switch_mlp, "gate_up_proj")
    inv_scatter = getattr(switch_mlp, "inverse_scatter", False)

    group_size = 64
    bits = 4
    mode = "affine"
    if is_quantized:
        for attr in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
            proj = getattr(switch_mlp, attr, None)
            if proj is not None:
                group_size = getattr(proj, "group_size", 64)
                bits = getattr(proj, "bits", 4)
                mode = getattr(proj, "mode", "affine")
                break

    streaming_glu = StreamingSwitchGLU(
        input_dims=hidden,
        hidden_dims=moe_hidden,
        num_experts=n_experts,
        layer_idx=layer_idx,
        backing=backing,
        cache=cache,
        fused_gate_up=fused,
        inverse_scatter=inv_scatter,
        quantized=is_quantized,
        group_size=group_size,
        bits=bits,
        mode=mode,
        # DeepSeek V4 uses LimitedSwiGLU (swiglu_limit / fp32 on MTP stages);
        # copying it keeps streaming bit-exact with the resident path.
        activation=getattr(switch_mlp, "activation", None),
    )

    # For RAM dict backing, populate dict from resident weights
    if backing_kind == "ram-dict":
        assert isinstance(backing, dict)
        # Map resident stacked banks to per-expert entries
        # need to know stacked keys for file backing naming, but for RAM we key by (layer, proj)
        for proj_name in (["gate_up_proj"] if fused else ["gate_proj", "up_proj", "down_proj"]):
            proj = getattr(switch_mlp, proj_name, None)
            if proj is None:
                continue
            # weight bank
            w = getattr(proj, "weight", None)
            if w is not None:
                mx.eval(w)
                # store stacked for slicing in streaming linear fallback
                backing[(layer_idx, proj_name)] = w  # type: ignore[index]
                # also for quantized scales/biases
                if is_quantized:
                    sc = getattr(proj, "scales", None)
                    if sc is not None:
                        mx.eval(sc)
                        backing[(layer_idx, proj_name, "weight")] = w  # type: ignore[index]
                        backing[(layer_idx, proj_name, "scales")] = sc  # type: ignore[index]
                        b = getattr(proj, "biases", None)
                        if b is not None:
                            mx.eval(b)
                            backing[(layer_idx, proj_name, "biases")] = b  # type: ignore[index]
                        else:
                            # ensure weight/scales keys exist for uniform fallback
                            pass
            # bias
            b = getattr(proj, "bias", None)
            if b is not None:
                mx.eval(b)

    # Now create streaming linears for the projections
    if fused:
        src = switch_mlp.gate_up_proj
        stacked_w_key = _resolve_stacked_key(
            candidates_for("gate_up_proj", "weight"), "gate_up_proj", "weight", backing, needle
        )
        if is_quantized:
            stacked_s_key = _resolve_stacked_key(
                candidates_for("gate_up_proj", "scales"), "gate_up_proj", "scales", backing, needle
            )
            stacked_b_key = _resolve_stacked_key(
                candidates_for("gate_up_proj", "biases"), "gate_up_proj", "biases", backing, needle
            )
            proj_stream = StreamingQuantizedSwitchLinear(
                layer_idx=layer_idx,
                proj_name="gate_up_proj",
                stacked_weight_key=stacked_w_key,
                stacked_scales_key=stacked_s_key,
                stacked_biases_key=stacked_b_key,
                num_experts=n_experts,
                input_dims=hidden,
                output_dims=moe_hidden * 2,
                backing=backing,
                cache=cache,
                group_size=group_size,
                bits=bits,
                mode=mode,
                has_bias=hasattr(src, "bias"),
            )
            if hasattr(src, "bias"):
                proj_stream.set_bias(src.bias)  # type: ignore[attr-defined]
        else:
            proj_stream = StreamingSwitchLinear(
                layer_idx=layer_idx,
                proj_name="gate_up_proj",
                stacked_key=stacked_w_key,
                num_experts=n_experts,
                input_dims=hidden,
                output_dims=moe_hidden * 2,
                backing=backing,
                cache=cache,
                bias=hasattr(src, "bias"),
            )
            if hasattr(src, "bias"):
                proj_stream.set_bias(src.bias)  # type: ignore[attr-defined]
        streaming_glu.gate_up_proj = proj_stream  # type: ignore[attr-defined]
        # down
        src_down = switch_mlp.down_proj
        stacked_w_key = _resolve_stacked_key(
            candidates_for("down_proj", "weight"), "down_proj", "weight", backing, needle
        )
        if is_quantized:
            stacked_s_key = _resolve_stacked_key(
                candidates_for("down_proj", "scales"), "down_proj", "scales", backing, needle
            )
            stacked_b_key = _resolve_stacked_key(
                candidates_for("down_proj", "biases"), "down_proj", "biases", backing, needle
            )
            down_stream = StreamingQuantizedSwitchLinear(
                layer_idx=layer_idx,
                proj_name="down_proj",
                stacked_weight_key=stacked_w_key,
                stacked_scales_key=stacked_s_key,
                stacked_biases_key=stacked_b_key,
                num_experts=n_experts,
                input_dims=moe_hidden,
                output_dims=hidden,
                backing=backing,
                cache=cache,
                group_size=group_size,
                bits=bits,
                mode=mode,
                has_bias=hasattr(src_down, "bias"),
            )
            if hasattr(src_down, "bias"):
                down_stream.set_bias(src_down.bias)  # type: ignore[attr-defined]
        else:
            down_stream = StreamingSwitchLinear(
                layer_idx=layer_idx,
                proj_name="down_proj",
                stacked_key=stacked_w_key,
                num_experts=n_experts,
                input_dims=moe_hidden,
                output_dims=hidden,
                backing=backing,
                cache=cache,
                bias=hasattr(src_down, "bias"),
            )
            if hasattr(src_down, "bias"):
                down_stream.set_bias(src_down.bias)  # type: ignore[attr-defined]
        streaming_glu.down_proj = down_stream  # type: ignore[attr-defined]
    else:
        for proj_name, out_dim, in_dim in [
            ("gate_proj", moe_hidden, hidden),
            ("up_proj", moe_hidden, hidden),
            ("down_proj", hidden, moe_hidden),
        ]:
            src = getattr(switch_mlp, proj_name, None)
            if src is None:
                continue
            stacked_w_key = _resolve_stacked_key(
                candidates_for(proj_name, "weight"), proj_name, "weight", backing, needle
            )
            if is_quantized:
                stacked_s_key = _resolve_stacked_key(
                    candidates_for(proj_name, "scales"), proj_name, "scales", backing, needle
                )
                stacked_b_key = _resolve_stacked_key(
                    candidates_for(proj_name, "biases"), proj_name, "biases", backing, needle
                )
                proj_stream = StreamingQuantizedSwitchLinear(
                    layer_idx=layer_idx,
                    proj_name=proj_name,
                    stacked_weight_key=stacked_w_key,
                    stacked_scales_key=stacked_s_key,
                    stacked_biases_key=stacked_b_key,
                    num_experts=n_experts,
                    input_dims=in_dim,
                    output_dims=out_dim,
                    backing=backing,
                    cache=cache,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    has_bias=hasattr(src, "bias"),
                )
                if hasattr(src, "bias"):
                    proj_stream.set_bias(src.bias)  # type: ignore[attr-defined]
            else:
                proj_stream = StreamingSwitchLinear(
                    layer_idx=layer_idx,
                    proj_name=proj_name,
                    stacked_key=stacked_w_key,
                    num_experts=n_experts,
                    input_dims=in_dim,
                    output_dims=out_dim,
                    backing=backing,
                    cache=cache,
                    bias=hasattr(src, "bias"),
                )
                if hasattr(src, "bias"):
                    proj_stream.set_bias(src.bias)  # type: ignore[attr-defined]
            setattr(streaming_glu, proj_name, proj_stream)

    # Replace
    moe.switch_mlp = streaming_glu  # type: ignore[attr-defined]
    # Disable decoder FFN compilation (GLM-5.3 Glm5NextDecoderLayer
    # compiles the FFN when compile_ffn is True): mx.eval(indices) inside
    # the streaming switch is illegal under mx.compile/vmap transforms.
    if layer is not None:
        try:
            layer.compile_ffn = False  # type: ignore[attr-defined]
            layer._ffn_c = None  # type: ignore[attr-defined]
        except Exception:
            pass
        # Evaluate the layer output so the lazy graph does not pin every
        # layer's mini-bank (42 layers x ~13 MB/expert) at once — without
        # this the accumulate graph swaps on GLM-class experts.
        try:
            layer._stream_eval = True  # type: ignore[attr-defined]
        except Exception:
            pass
    return True


def convert_model_to_streaming(
    model: Any,
    model_path: str | Path,
    model_settings: Any | None = None,
    *,
    budget_bytes: int | None = None,
    use_file_backing: bool = True,
) -> tuple[Any, Any]:
    """Convert MoE layers of *model* to streaming.

    Returns (model, backing_store) where backing_store must be kept alive
    for the model lifetime (holds mmap readers).  When no MoE layers are
    found, returns (model, None) unchanged.
    """
    from .residency import expert_streaming_estimate

    estimate = expert_streaming_estimate(str(model_path))
    if not estimate.supported:
        logger.info("Expert streaming: model %s not supported (%s)", model_path, estimate.reason)
        return model, None

    if budget_bytes is None:
        budget_bytes = _get_budget_bytes(model_settings, estimate)

    logger.info(
        "Expert streaming: converting %s: budget=%.2f GiB (%s), layers=%d, experts/layer=%d, per_expert=%.2f MB, slots/layer=%d",
        Path(model_path).name,
        budget_bytes / 1024**3,
        "page-cache only, no LRU" if budget_bytes <= 0 else "LRU heap",
        estimate.num_moe_layers,
        estimate.experts_per_layer,
        estimate.per_expert_bytes / 1024 / 1024,
        estimate.slots_for_budget(budget_bytes),
    )

    # Import here to avoid circular
    from .streaming_switch import ExpertLRUCache

    per_expert = estimate.per_expert_bytes or 0
    # One cache slot holds ONE projection's slice (gate/up/down are separate
    # keys), so slot sizing must divide by the projections per expert —
    # otherwise the LRU holds a third of the budget it was promised (F2).
    per_slot = max(1, per_expert // 3) if per_expert else 0
    cache = ExpertLRUCache(budget_bytes, per_slot, num_layers=estimate.num_moe_layers)

    # Backing store
    backing = None
    backing_kind = "ram"
    if use_file_backing:
        try:
            import os

            from .shard_bank import ExpertBackingStore

            extra_roots = [
                p
                for p in os.environ.get("OMLX_EXPERT_STREAMING_EXTRA_ROOTS", "").split(":")
                if p.strip()
            ]
            backing = ExpertBackingStore(model_path, extra_roots=extra_roots)
            # Guard metadata for the scheduler's prefill chunk sizing: the
            # lazy chunk forward holds every MoE layer's assembled mini-bank
            # until the chunk-end eval, so the peak carries ~one bank per
            # layer simultaneously. Without this term the guard under-predicts
            # and admits chunks whose real peak reaches ~26 GB on qwen4_exp
            # (48 layers x ~215 uniq experts x ~2.5 MB) and squeezes the
            # machine (docs F-series F1).
            backing.streaming_guard_info = {
                "num_moe_layers": estimate.num_moe_layers,
                "experts_per_layer": estimate.experts_per_layer,
                "per_expert_bytes": estimate.per_expert_bytes,
            }
            backing_kind = "mmap"
        except Exception as e:
            logger.warning("Expert streaming: file backing failed (%s), falling back to RAM dict", e)
            backing = None

    # RAM dict fallback: copy per-expert slices from resident model into dict
    ram_dict: dict[tuple, Any] | None = None
    if backing is None:
        ram_dict = {}
        backing = ram_dict  # type: ignore[assignment]
        backing_kind = "ram-dict"

    converted = 0
    # Walk model.layers — handle LLM (model.model.layers) and VLM wrappers
    # (language_model.model.layers via language_model indirection)
    layers = None
    layers_owner = None
    # candidate attribute paths to try
    candidate_paths = [
        ("model", "layers"),  # mlx_lm LanguageModel.model.layers
        ("layers",),  # VLM Model.layers property (glm5_next)
        ("language_model", "model", "layers"),  # VLM wrapper: Model.language_model.model.layers
        ("language_model", "layers"),  # alternative VLM wrapper
        ("model", "language_model", "model", "layers"),
    ]
    for path in candidate_paths:
        cur = model
        owner = None
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            owner = cur
            cur = getattr(cur, attr)
        if ok and cur is not None:
            # sanity: should be iterable with length ~ num_layers
            try:
                _ = len(cur)  # type: ignore[arg-type]
                layers = cur  # type: ignore[assignment]
                layers_owner = owner
                break
            except Exception:
                continue
    if layers is None:
        logger.warning("Expert streaming: could not find model.layers")
        return model, backing if isinstance(backing, dict) is False else None

    hidden, moe_hidden = _resolve_moe_dims(_model_config_candidates(model))

    # Main decoder layers. GLM/Qwen nest the MoE under ``mlp``; DeepSeek V4
    # nests it under ``ffn`` — prefer whichever holds a switch_mlp.
    for layer_idx, layer in enumerate(layers):
        if layer is None:
            continue
        moe = getattr(layer, "mlp", None)
        if moe is None or getattr(moe, "switch_mlp", None) is None:
            moe = getattr(layer, "ffn", None)
        if moe is None or getattr(moe, "switch_mlp", None) is None:
            continue
        if _convert_switch_mlp_module(
            moe,
            layer_idx,
            candidates_for=lambda proj, suffix, _i=layer_idx: _candidate_stacked_keys(_i, proj, suffix),
            needle=f"layers.{layer_idx}.",
            backing=backing,
            backing_kind=backing_kind,
            cache=cache,
            estimate=estimate,
            hidden=hidden,
            moe_hidden=moe_hidden,
            layer=layer,
        ):
            converted += 1

    # DeepSeek V4 MTP/DSpark stages carry their own SwitchGLU banks
    # (mtp.<stage>.ffn on DSpark checkpoints, mtp.<stage>.block.ffn on the
    # legacy MTPBlock layout). Streaming them keeps the ~3 GB/stage banks
    # out of RAM on low-memory hosts.
    mtp_stages = getattr(layers_owner, "mtp", None) if layers_owner is not None else None
    if not isinstance(mtp_stages, (list, tuple)) or not mtp_stages:
        mtp_stages = None
    mtp_converted = 0
    if mtp_stages:
        for stage_idx, stage in enumerate(mtp_stages):
            if stage is None:
                continue
            stage_moe = getattr(stage, "ffn", None)
            if stage_moe is None or getattr(stage_moe, "switch_mlp", None) is None:
                block = getattr(stage, "block", None)
                stage_moe = getattr(block, "ffn", None) if block is not None else None
            if stage_moe is None or getattr(stage_moe, "switch_mlp", None) is None:
                continue
            if _convert_switch_mlp_module(
                stage_moe,
                len(layers) + stage_idx,
                candidates_for=lambda proj, suffix, _s=stage_idx: _mtp_candidate_stacked_keys(_s, proj, suffix),
                needle=f"mtp.{stage_idx}.",
                backing=backing,
                backing_kind=backing_kind,
                cache=cache,
                estimate=estimate,
                hidden=hidden,
                moe_hidden=moe_hidden,
                layer=stage,
            ):
                mtp_converted += 1
                converted += 1
        if mtp_converted:
            logger.info(
                "Expert streaming: converted %d/%d MTP/DSpark stage MoE banks",
                mtp_converted,
                len(mtp_stages),
            )

    # The estimate counts MTP stages as MoE layers; when the runtime MTP is
    # inactive (no model.mtp) fewer layers were converted — rebalance the
    # per-layer LRU split so the converted layers keep a fair share.
    if converted and cache.num_layers != converted and cache.capacity > 0:
        cache.num_layers = converted
        cache._per_layer_cap = max(1, cache.capacity // converted)  # type: ignore[attr-defined]

    if converted:
        import mlx.core as mx

        mx.clear_cache()
        logger.info("Expert streaming: converted %d MoE layers (backing=%s, cache_capacity=%d experts)", converted, backing_kind, cache.capacity)
        # Opt-in adaptive top-k routing truncation (cumulative mass). Exact
        # (None/1.0) by default — no patch engagement, zero overhead.
        from .adaptive_topk import apply_qwen35_moe_topk_patch, configure_from_settings

        thr = configure_from_settings(model_settings)
        if thr is not None:
            apply_qwen35_moe_topk_patch()
        # PILOT: async router-lookahead prefetch into the LRU (glm5_next's
        # Glm5NextModel loop scores the next MoE layer's router against the
        # current layer output). mmap backing only; off when the RAM dict
        # fallback is in use or OMLX_EXPERT_STREAMING_PILOT=0.
        import os

        if os.environ.get("OMLX_EXPERT_STREAMING_PILOT", "0") == "1" and not isinstance(
            backing, dict
        ):
            try:
                from .prefetch import ExpertPrefetcher

                prefetcher = ExpertPrefetcher(cache)
                prefetcher.start()
                for obj_ in (
                    getattr(getattr(model, "language_model", None), "model", None),
                    getattr(model, "language_model", None),
                    model,
                ):
                    match_ = obj_ is not None and getattr(obj_, "layers", None) is layers
                    if match_:
                        obj_._expert_prefetcher = prefetcher  # type: ignore[attr-defined]
                        break
                else:
                    prefetcher.stop()
                    prefetcher = None
                    logger.warning(
                        "Expert streaming: PILOT prefetch attach point not found; disabled"
                    )
                if prefetcher is not None:
                    # Wire streaming linears to their prefetcher so the
                    # demand path can drain staged np bundles before a
                    # synchronous backing read.
                    wired = 0
                    for lyr_ in layers:
                        mlp_ = getattr(lyr_, "mlp", None)
                        sm_ = getattr(mlp_, "switch_mlp", None)
                        for proj_ in ("gate_proj", "up_proj", "down_proj"):
                            lin_ = getattr(sm_, proj_, None)
                            if lin_ is not None and hasattr(lin_, "_load_expert_np"):
                                lin_._prefetcher = prefetcher  # type: ignore[attr-defined]
                                wired += 1
                    logger.info(
                        "Expert streaming: PILOT async prefetch active (%d linears wired)",
                        wired,
                    )
            except Exception as e:
                logger.warning("Expert streaming: PILOT prefetch init failed: %s", e)

        # Opt-in warm-only page-cache prefetch + mlock pins (page-cache
        # complements; replaces the LRU's "keep hot experts in RAM" role).
        # F_RDADVISE readahead (RA) rides the same prediction flow with
        # kernel hints instead of reads and defaults ON, as does the
        # prefill-hotness cache seed (SEED).
        from . import warmer as _warmer_mod

        if (
            _warmer_mod.WARM_ENABLED
            or _warmer_mod.PIN_ENABLED
            or _warmer_mod.RA_ENABLED
            or _warmer_mod.SEED_ENABLED
        ):
            try:
                glus: dict[int, Any] = {}
                for layer_idx_, layer_ in enumerate(layers):
                    moe_ = getattr(layer_, "mlp", None) or getattr(layer_, "ffn", None)
                    sm_ = getattr(moe_, "switch_mlp", None)
                    if sm_ is not None and hasattr(sm_, "down_proj"):
                        glus[layer_idx_] = sm_
                linears_by_layer: dict[int, list] = {
                    i: [
                        getattr(g, p)
                        for p in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
                        if hasattr(g, p)
                    ]
                    for i, g in glus.items()
                }
                if _warmer_mod.WARM_ENABLED:
                    warmer = _warmer_mod.PageCacheWarmer(linears_by_layer)
                elif _warmer_mod.RA_ENABLED:
                    warmer = _warmer_mod.PageCacheWarmer(
                        linears_by_layer, advise_only=True
                    )
                else:
                    warmer = None
                pinner = None
                if _warmer_mod.PIN_ENABLED and backing is not None and not isinstance(backing, dict):
                    pinner = _warmer_mod.PinController(
                        linears_by_layer,
                        backing,
                        budget_bytes=_warmer_mod.PIN_BUDGET_BYTES,
                        observe_calls=_warmer_mod.PIN_OBSERVE_CALLS,
                        per_expert_bytes=estimate.per_expert_bytes,
                    )
                recorder = None
                if _warmer_mod.SEED_ENABLED and backing is not None and not isinstance(backing, dict):
                    recorder = _warmer_mod.PrefillHotnessRecorder(
                        linears_by_layer,
                        backing,
                        cache,
                        per_expert_bytes=estimate.per_expert_bytes,
                    )
                if warmer is not None or pinner is not None or recorder is not None:
                    hook = _warmer_mod.WarmPinHook(warmer, pinner, recorder)
                    for sm_ in glus.values():
                        sm_._warm_pins = hook  # type: ignore[attr-defined]
                    logger.info(
                        "Expert streaming: warm=%s pin=%s readahead=%s seed=%s attached (%d layers)",
                        bool(warmer and not warmer.advise_only),
                        bool(pinner),
                        bool(warmer and warmer.advise_only),
                        bool(recorder),
                        len(glus),
                    )
            except Exception as e:
                logger.warning("Expert streaming: warm/pin init failed: %s", e)
    else:
        logger.info("Expert streaming: no MoE layers converted")

    # ram-dict backing is internal only — never part of the public return
    # (existing contract; file-backed store is the only returned backing).
    return model, backing if not isinstance(backing, dict) else None


__all__ = [
    "apply_expert_streaming_patch",
    "convert_model_to_streaming",
    "is_supported_model_type",
    "SUPPORTED_TYPES",
]
