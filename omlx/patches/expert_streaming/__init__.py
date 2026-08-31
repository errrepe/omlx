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


def _io_overrides(model_settings: Any | None) -> dict[str, Any]:
    """Per-model streaming IO overrides with env-fallback semantics.

    Returns a dict whose values are None when the setting is unset (keep the
    env-var / built-in default) or the requested override otherwise.
    """
    raw = {
        "expert_streaming_io_depth": None,
        "expert_streaming_coalesce": None,
        "expert_streaming_readahead": None,
        "expert_streaming_seed": None,
        "expert_streaming_pilot": None,
        "expert_streaming_per_layer_eval": None,
        "expert_streaming_pins": None,
        "expert_streaming_pin_gib": None,
    }
    if model_settings is None:
        return raw
    for key in raw:
        raw[key] = getattr(model_settings, key, None)
    depth = raw["expert_streaming_io_depth"]
    if depth is not None:
        try:
            depth = int(depth)
        except (TypeError, ValueError):
            depth = None
        else:
            depth = max(1, min(64, depth)) if depth >= 1 else None
        raw["expert_streaming_io_depth"] = depth
    return raw


def _wire_streaming_io_overrides(
    layers: Any,
    mtp_stages: Any,
    io_depth: int | None,
    coalesce: bool | None,
) -> int:
    """Attach per-model IO pool / coalesce overrides to streaming linears.

    Returns the number of linears wired (0 when both overrides are unset —
    the module env defaults stay in effect).
    """
    if io_depth is None and coalesce is None:
        return 0
    from .streaming_switch import io_pool_for

    pool = io_pool_for(io_depth) if io_depth is not None else None
    wired = 0
    targets = list(layers or []) + list(mtp_stages or [])
    for lyr in targets:
        moe = getattr(lyr, "mlp", None) or getattr(lyr, "ffn", None)
        sm = getattr(moe, "switch_mlp", None)
        if sm is None:
            continue
        for proj in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
            lin = getattr(sm, proj, None)
            if lin is not None and hasattr(lin, "_io_pool_override"):
                if pool is not None:
                    lin._io_pool_override = pool  # type: ignore[attr-defined]
                if coalesce is not None:
                    lin._coalesce_override = bool(coalesce)  # type: ignore[attr-defined]
                wired += 1
    return wired


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

    # Cold precision tier (I5): when the backing serves this layer's banks
    # from expert_cold/, every projection of the layer computes at the
    # tier's packing — override the source bits/group size once, here, so
    # the fused and split branches both build with the tier parameters.
    if hasattr(backing, "cold_quant_params"):
        first_attr = next(
            (
                a
                for a in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
                if getattr(switch_mlp, a, None) is not None
            ),
            None,
        )
        if first_attr is not None:
            probe_key = _resolve_stacked_key(
                candidates_for(first_attr, "weight"),
                first_attr,
                "weight",
                backing,
                needle,
            )
            cold_params = backing.cold_quant_params(probe_key)
            if cold_params is not None:
                bits, group_size = cold_params

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


def _glu_projection_count(layers: Any) -> int:
    """Projections sharing one per-layer load context on a converted model.

    A converted ``StreamingSwitchGLU`` holds ``linears``: 2 when the
    checkpoint fuses ``gate_up_proj`` (plus ``down_proj``), 3 when gate/up
    are split. The scheduler's prefill guard charges
    ``min(2, projections)`` banks when a per-layer eval boundary is live, so
    any value >= 2 collapses to the same 2; only a 1 (no shared context)
    changes the charge. Defaults to 3 (the conservative split case) when no
    converted GLU is reachable.
    """
    try:
        for layer in layers or ():
            linears = getattr(getattr(layer, "switch_mlp", None), "linears", None)
            # An empty/unsized ``linears`` is not a converted GLU — keep
            # looking rather than reporting 0 projections.
            if linears and len(linears) > 0:
                return len(linears)
    except Exception:  # noqa: BLE001
        pass
    return 3


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

            from .shard_bank import ExpertBackingStore, cold_tier_status

            extra_roots = [
                p
                for p in os.environ.get("OMLX_EXPERT_STREAMING_EXTRA_ROOTS", "").split(":")
                if p.strip()
            ]
            # Cold precision tier (I5): expert_streaming_cold_tier ("2"/"3")
            # routes expert reads to <model>/expert_cold/ — a requantized
            # full expert set (tools/requant_cold_tier.py) that cuts the
            # bytes per token pinning decode to the NVMe I/O floor. Partial
            # tiers are rejected: the uniform-packing assumption the linears
            # build on would silently break.
            cold_root = None
            cold_setting = getattr(model_settings, "expert_streaming_cold_tier", None)
            if cold_setting and str(cold_setting) in ("2", "3"):
                ok, why = cold_tier_status(model_path)
                if ok:
                    cold_root = Path(model_path) / "expert_cold"
                    logger.info("Expert streaming: cold tier %s-bit active (%s)", cold_setting, why)
                else:
                    logger.warning(
                        "Expert streaming: cold tier %s requested but %s — disabled",
                        cold_setting,
                        why,
                    )
            backing = ExpertBackingStore(model_path, extra_roots=extra_roots, cold_root=cold_root)
            # Guard metadata for the scheduler's prefill chunk sizing: the
            # lazy chunk forward holds every MoE layer's assembled mini-bank
            # until the chunk-end eval, so the peak carries ~one bank per
            # layer simultaneously. Without this term the guard under-predicts
            # and admits chunks whose real peak reaches ~26 GB on qwen4_exp
            # (48 layers x ~215 uniq experts x ~2.5 MB) and squeezes the
            # machine (docs F-series F1).
            #
            # Fase J Etapa E: ``boundary_active`` starts False — the
            # per-layer bank charge is the safe default, and it is only
            # relaxed once a per-layer eval boundary has actually been
            # installed on a decoder class (set below, after conversion).
            # ``projections`` is the number of projections sharing one
            # per-layer load context (2 fused gate_up+down, or 3 split);
            # the guard charges min(2, projections) banks.
            # bf16/fp16 activation: one materialized layer output per token.
            # 0 when the config hid hidden_size — conservative.
            _hidden_size = int(getattr(estimate, "hidden_size", 0) or 0)
            backing.streaming_guard_info = {
                "num_moe_layers": estimate.num_moe_layers,
                "experts_per_layer": estimate.experts_per_layer,
                "per_expert_bytes": estimate.per_expert_bytes,
                "boundary_active": False,
                "projections": 3,
                "activation_bytes_per_token": 2 * _hidden_size,
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
        # Per-model IO overrides (autotune): pool depth + run coalescing ride
        # the streaming linears; unset values keep the env-var defaults.
        io_ov = _io_overrides(model_settings)
        io_wired = _wire_streaming_io_overrides(
            layers, mtp_stages, io_ov["expert_streaming_io_depth"], io_ov["expert_streaming_coalesce"]
        )
        if io_wired:
            logger.info(
                "Expert streaming: IO overrides wired (io_depth=%s coalesce=%s, %d linears)",
                io_ov["expert_streaming_io_depth"],
                io_ov["expert_streaming_coalesce"],
                io_wired,
            )
        # Opt-in adaptive top-k routing truncation (cumulative mass). Exact
        # (None/1.0) by default — no patch engagement, zero overhead.
        from .adaptive_topk import apply_qwen35_moe_topk_patch, configure_from_settings

        thr = configure_from_settings(model_settings)
        if thr is not None:
            apply_qwen35_moe_topk_patch()
        # Qwen3.5/3.8 prefill eval boundary (G4): the installed qwen decoder
        # ignores _stream_eval; wrap it so long prefill chunks evaluate per
        # layer instead of pinning every layer's mini-bank in the lazy graph
        # and retaining an allocator pool big enough to evict the page cache.
        # Bit-exact; prefill-shaped calls only (decode/MTP verify stay lazy).
        # Imported as a module (not by name) so it cannot shadow the
        # ``configure_from_settings`` bound above for adaptive_topk.
        from . import qwen35_stream_eval as qse

        eval_on = qse.configure_from_settings(
            io_ov["expert_streaming_per_layer_eval"]
        )
        if qse.apply_qwen35_moe_stream_eval():
            logger.info(
                "Expert streaming: qwen per-layer eval boundary %s",
                "on" if eval_on else "off",
            )
        # Etapa E: tell the scheduler's prefill guard whether a boundary is
        # really live. Only then may it stop charging one mini-bank per MoE
        # layer; the flag stays False when the knob is off or no decoder
        # class was wrapped, so glm5_next (which honors _stream_eval inline
        # and is not wrapped here) keeps the conservative charge.
        _guard_info = getattr(backing, "streaming_guard_info", None)
        if isinstance(_guard_info, dict):
            _guard_info["boundary_active"] = bool(
                eval_on and qse.boundary_active()
            )
            _guard_info["projections"] = _glu_projection_count(layers)
            if _guard_info["boundary_active"]:
                logger.info(
                    "Expert streaming: prefill guard boundary accounting on "
                    "(projections=%d, activation=%d B/token)",
                    _guard_info["projections"],
                    _guard_info.get("activation_bytes_per_token", 0),
                )
        # PILOT: async router-lookahead prefetch into the LRU (glm5_next's
        # Glm5NextModel loop scores the next MoE layer's router against the
        # current layer output). mmap backing only; off when the RAM dict
        # fallback is in use, the per-model setting disables it, or
        # OMLX_EXPERT_STREAMING_PILOT=0.
        import os

        pilot_requested = io_ov["expert_streaming_pilot"]
        if pilot_requested is None:
            pilot_requested = os.environ.get("OMLX_EXPERT_STREAMING_PILOT", "0") == "1"
        if pilot_requested and not isinstance(
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
        # prefill-hotness cache seed (SEED). The per-model readahead/seed
        # settings (autotune) override the env defaults when set.
        from . import warmer as _warmer_mod

        ra_setting = io_ov["expert_streaming_readahead"]
        ra_enabled = (
            _warmer_mod.RA_ENABLED if ra_setting is None else bool(ra_setting)
        )
        seed_setting = io_ov["expert_streaming_seed"]
        seed_enabled = (
            _warmer_mod.SEED_ENABLED if seed_setting is None else bool(seed_setting)
        )
        pins_setting = io_ov["expert_streaming_pins"]
        pins_enabled = (
            _warmer_mod.PIN_ENABLED if pins_setting is None else bool(pins_setting)
        )
        pin_gib = io_ov["expert_streaming_pin_gib"]
        pin_budget_bytes = (
            _warmer_mod.PIN_BUDGET_BYTES
            if pin_gib is None
            else max(0, min(64.0, float(pin_gib))) * 1024**3
        )

        if (
            _warmer_mod.WARM_ENABLED
            or pins_enabled
            or ra_enabled
            or seed_enabled
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
                elif ra_enabled:
                    warmer = _warmer_mod.PageCacheWarmer(
                        linears_by_layer, advise_only=True
                    )
                else:
                    warmer = None
                pinner = None
                if pins_enabled and backing is not None and not isinstance(backing, dict):
                    # Per-model learned-pin profile so the hot set is wired
                    # from token 1 on the next load (E3). The env path (bench
                    # override) wins when set; otherwise a .omlx sidecar in
                    # the model directory.
                    pin_profile_path = _warmer_mod.PIN_PROFILE_PATH or str(
                        Path(model_path) / ".omlx" / "expert_pin_profile.json"
                    )
                    pinner = _warmer_mod.PinController(
                        linears_by_layer,
                        backing,
                        budget_bytes=int(pin_budget_bytes),
                        observe_calls=_warmer_mod.PIN_OBSERVE_CALLS,
                        per_expert_bytes=estimate.per_expert_bytes,
                        profile_path=pin_profile_path,
                    )
                    # Save-on-unload hook: engines call save_expert_pin_profile()
                    # in stop() while the backing is still reachable.
                    backing._pin_controller = pinner  # type: ignore[attr-defined]
                recorder = None
                if seed_enabled and backing is not None and not isinstance(backing, dict):
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


def save_expert_pin_profile(engine: Any) -> None:
    """Persist the learned pin profile of a streaming engine, if any.

    Called from the engine ``stop()`` paths while the backing store (and the
    PinController attached to it) is still reachable — before teardown drops
    the references. Never raises: a failed save only costs the learned hot
    set, never correctness.
    """
    for holder in (
        engine,
        getattr(engine, "_model", None),
        getattr(engine, "_vlm_model", None),
    ):
        if holder is None:
            continue
        backing = getattr(holder, "_expert_streaming_backing", None)
        pinner = getattr(backing, "_pin_controller", None)
        if pinner is not None:
            try:
                pinner.save_profile()
            except Exception:
                logger.debug(
                    "Expert streaming: pin profile save failed", exc_info=True
                )
            return


__all__ = [
    "apply_expert_streaming_patch",
    "convert_model_to_streaming",
    "save_expert_pin_profile",
    "is_supported_model_type",
    "SUPPORTED_TYPES",
]
