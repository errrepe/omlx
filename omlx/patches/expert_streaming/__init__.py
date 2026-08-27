# SPDX-License-Identifier: Apache-2.0
"""Expert streaming (SSD) patch for MoE models (glm_moe_dsa)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False

SUPPORTED_TYPES = {"glm_moe_dsa", "deepseek_v32"}


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
        gib = getattr(model_settings, "expert_cache_budget_gib", None)
        if gib is not None and float(gib) > 0:
            return int(float(gib) * 1024**3)
        # legacy mib
        mib = getattr(model_settings, "expert_cache_budget_mib", None)
        if mib is not None and int(mib) > 0:
            return int(int(mib) * 1024 * 1024)
    # default 2 GiB
    return 2 * 1024 * 1024 * 1024


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
        "Expert streaming: converting %s: budget=%.2f GiB, layers=%d, experts/layer=%d, per_expert=%.2f MB, slots/layer=%d",
        Path(model_path).name,
        budget_bytes / 1024**3,
        estimate.num_moe_layers,
        estimate.experts_per_layer,
        estimate.per_expert_bytes / 1024 / 1024,
        estimate.slots_for_budget(budget_bytes),
    )

    # Import here to avoid circular
    from .streaming_switch import ExpertLRUCache, StreamingQuantizedSwitchLinear, StreamingSwitchGLU, StreamingSwitchLinear

    per_expert = estimate.per_expert_bytes or 0
    cache = ExpertLRUCache(budget_bytes, per_expert)

    # Backing store
    backing = None
    backing_kind = "ram"
    if use_file_backing:
        try:
            from .shard_bank import ExpertBackingStore

            backing = ExpertBackingStore(model_path)
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

    import mlx.core as mx

    converted = 0
    # Walk model.layers
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers  # type: ignore[assignment]
    else:
        logger.warning("Expert streaming: could not find model.layers")
        return model, backing if isinstance(backing, dict) is False else None

    for layer_idx, layer in enumerate(layers):
        if layer is None:
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        # Check if MoE
        switch_mlp = getattr(mlp, "switch_mlp", None)
        if switch_mlp is None:
            continue

        # Determine quantized vs bf16
        is_quantized = False
        # check any projection is QuantizedSwitchLinear
        for attr in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
            proj = getattr(switch_mlp, attr, None)
            if proj is not None:
                # QuantizedSwitchLinear has 'scales'
                if hasattr(proj, "scales") or "scales" in getattr(proj, "_data", {}):
                    is_quantized = True
                    break
                # also check class name
                if proj.__class__.__name__ == "QuantizedSwitchLinear":
                    is_quantized = True
                    break

        # Build streaming GLU
        # Need dims
        # Infer from existing proj weight shape or config
        # For simplicity, get from mlp.config or layer
        hidden = None
        moe_hidden = None
        n_experts = estimate.experts_per_layer
        try:
            # Try to infer from weight shape
            sample_proj = getattr(switch_mlp, "down_proj", None) or getattr(switch_mlp, "gate_up_proj", None) or getattr(switch_mlp, "gate_proj", None)
            if sample_proj is not None:
                w = getattr(sample_proj, "weight", None)
                if w is not None:
                    # weight shape (E, O, I) or (E, O, packed)
                    # input_dims derived from weight shape[2] * 32/bits for quant?
                    pass
        except Exception:
            pass

        # Use model config for dims
        cfg = getattr(model, "args", None) or getattr(getattr(model, "model", None), "args", None)
        if cfg is not None:
            hidden = getattr(cfg, "hidden_size", None)
            moe_hidden = getattr(cfg, "moe_intermediate_size", None)
        if hidden is None:
            hidden = 4096
        if moe_hidden is None:
            moe_hidden = 1407

        # Create streaming GLU shell
        fused = hasattr(switch_mlp, "gate_up_proj")
        inv_scatter = getattr(switch_mlp, "inverse_scatter", False)

        group_size = 64
        bits = 4
        mode = "affine"
        # Try to read from an existing quantized proj
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
            src = getattr(switch_mlp, "gate_up_proj")
            # Determine stacked keys for file backing (for mmap path)
            # Use checkpoint key pattern: model.layers.{l}.mlp.switch_mlp.gate_up_proj.weight
            stacked_w_key = f"model.layers.{layer_idx}.mlp.switch_mlp.gate_up_proj.weight"
            if is_quantized:
                stacked_s_key = f"model.layers.{layer_idx}.mlp.switch_mlp.gate_up_proj.scales"
                stacked_b_key = f"model.layers.{layer_idx}.mlp.switch_mlp.gate_up_proj.biases"
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
            src_down = getattr(switch_mlp, "down_proj")
            stacked_w_key = f"model.layers.{layer_idx}.mlp.switch_mlp.down_proj.weight"
            if is_quantized:
                stacked_s_key = f"model.layers.{layer_idx}.mlp.switch_mlp.down_proj.scales"
                stacked_b_key = f"model.layers.{layer_idx}.mlp.switch_mlp.down_proj.biases"
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
                stacked_w_key = f"model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}.weight"
                if is_quantized:
                    stacked_s_key = f"model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}.scales"
                    stacked_b_key = f"model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}.biases"
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
        mlp.switch_mlp = streaming_glu  # type: ignore[attr-defined]
        converted += 1
        # Free original tensors to save memory
        # Drop references; will be GC'd. For mmap backing, original stacked banks are no longer needed.
        # For RAM dict backing, we kept a copy; we can still free the original stacked mx arrays.
        # Delete original switch_mlp projections to release memory
        try:
            del switch_mlp  # type: ignore[assignment]
        except Exception:
            pass

    if converted:
        import mlx.core as mx

        mx.clear_cache()
        logger.info("Expert streaming: converted %d MoE layers (backing=%s, cache_capacity=%d experts)", converted, backing_kind, cache.capacity)
    else:
        logger.info("Expert streaming: no MoE layers converted")

    return model, backing


__all__ = [
    "apply_expert_streaming_patch",
    "convert_model_to_streaming",
    "is_supported_model_type",
    "SUPPORTED_TYPES",
]
