# SPDX-License-Identifier: Apache-2.0
"""Residency estimates for MoE expert streaming (SSD)."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_MODEL_OVERHEAD_FACTOR = 1.05


@dataclass(frozen=True)
class ExpertStreamingEstimate:
    """Estimated bytes with and without expert streaming."""

    supported: bool
    checkpoint_bytes: int
    expert_bytes: int
    dense_bytes: int
    resident_bytes: int
    streaming_bytes: int
    num_moe_layers: int
    experts_per_layer: int
    per_expert_bytes: int
    per_layer_expert_bytes: int
    reason: str | None = None

    def force_streaming(self, memory_ceiling: int) -> bool:
        """True when streaming turns an impossible load into a viable one."""
        return (
            self.supported
            and memory_ceiling > 0
            and self.resident_bytes > memory_ceiling
            and self.streaming_bytes <= memory_ceiling
        )

    def slots_for_budget(self, budget_bytes: int) -> int:
        """Slots per layer that fit in *budget_bytes*."""
        if not self.supported or self.per_expert_bytes <= 0 or self.num_moe_layers <= 0:
            return 0
        if budget_bytes <= 0:
            return 0
        per_layer = self.per_expert_bytes
        # budget is total across all MoE layers
        slots = budget_bytes // (self.num_moe_layers * per_layer)
        # clamp to experts_per_layer
        return int(max(0, min(slots, self.experts_per_layer)))

    def streaming_bytes_for_budget(self, budget_bytes: int) -> int:
        """Resident streaming bytes for a given cache budget."""
        if not self.supported:
            return self.resident_bytes
        # dense + cache
        cache = 0
        if budget_bytes > 0 and self.per_expert_bytes > 0:
            slots = self.slots_for_budget(budget_bytes)
            cache = slots * self.num_moe_layers * self.per_expert_bytes
        return int(self.dense_bytes * _MODEL_OVERHEAD_FACTOR + cache)


def _safetensors_header(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return {}
            hsize = struct.unpack("<Q", raw)[0]
            return json.loads(f.read(hsize))
    except Exception:
        return {}


def _load_config(model_path: Path) -> dict:
    try:
        return json.loads((model_path / "config.json").read_text())
    except Exception:
        return {}


def _detect_expert_keys(
    weight_map: dict[str, str],
    headers: dict[str, dict],
    config: dict,
) -> tuple[list[str], int, int]:
    """Return (expert_tensor_keys, num_moe_layers, experts_per_layer)."""
    # Prefer text_config for VLM wrappers (glm5_next, qwen4_exp)
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    n_routed = config.get("n_routed_experts")
    if n_routed is None:
        n_routed = text_cfg.get("n_routed_experts")
    if n_routed is None:
        # qwen4_exp uses num_experts
        n_routed = config.get("num_experts")
    if n_routed is None:
        n_routed = text_cfg.get("num_experts")
    try:
        n_routed = int(n_routed) if n_routed is not None else 0
    except Exception:
        n_routed = 0

    num_layers = config.get("num_hidden_layers") or config.get("num_layers") or 0
    if not num_layers and text_cfg:
        num_layers = text_cfg.get("num_hidden_layers") or text_cfg.get("num_layers") or 0
    try:
        num_layers = int(num_layers)
    except Exception:
        num_layers = 0

    model_type = str(config.get("model_type") or "")
    # also check text_config type for VLM wrappers
    if not model_type and text_cfg:
        model_type = str(text_cfg.get("model_type") or "")

    expert_keys: list[str] = []

    # Heuristics: stacked MoE banks contain switch_mlp and dimension 0 == n_routed
    for key in weight_map.keys():
        # Exclude PLE ngram tables and MTP heads — not MoE experts
        if ".ngram_embedding." in key or ".ple." in key:
            continue
        if ".mtp." in key or key.startswith("mtp.") or "nextn" in key.lower():
            continue
        is_expert = False
        if ".mlp.experts." in key:
            is_expert = True
        elif "switch_mlp" in key and ("gate_proj" in key or "up_proj" in key or "down_proj" in key or "gate_up_proj" in key):
            # check shape[0] == n_routed if we have headers
            # weight_map may point to sharded files, need headers per file
            # we will check later via header shape; for now consider candidate
            is_expert = True
        elif model_type.lower().replace("-", "_") in ("glm_moe_dsa", "glm5_next", "glm5_next_text", "qwen4_exp", "qwen4_exp_text") and "mlp.switch_mlp" in key:
            is_expert = True
        if is_expert:
            expert_keys.append(key)

    # Refine by checking header shape when possible
    if n_routed and expert_keys and headers:
        refined: list[str] = []
        for k in expert_keys:
            entry = headers.get(k)
            if entry is None:
                refined.append(k)
                continue
            shape = entry.get("shape") or []
            if shape and shape[0] == n_routed:
                refined.append(k)
            elif ".experts." in k:
                refined.append(k)
            # else: may be false positive, skip stacking check
        # if refined is non-empty, use it; otherwise keep original for per-expert files
        if refined:
            expert_keys = refined

    # Determine moe layers by distinct layer indices in expert keys
    import re

    layer_pat = re.compile(r"layers\.(\d+)\.")
    layers = set()
    for k in expert_keys:
        m = layer_pat.search(k)
        if m:
            try:
                idx = int(m.group(1))
                # Exclude extra MTP/nextn layers beyond num_hidden_layers (e.g. glm5_next layer 45)
                if num_layers and idx >= num_layers:
                    continue
                layers.add(idx)
            except Exception:
                pass
    num_moe_layers = len(layers) if layers else 0
    # fallback to config's derived count when headers incomplete
    if num_moe_layers == 0 and model_type.lower().replace("-", "_") in ("glm_moe_dsa", "glm5_next", "glm5_next_text", "qwen4_exp", "qwen4_exp_text"):
        # For glm5_next: use mlp_layer_types sparse count (most accurate)
        try:
            mlp_types = config.get("mlp_layer_types")
            if mlp_types is None and text_cfg:
                mlp_types = text_cfg.get("mlp_layer_types")
            if isinstance(mlp_types, list) and n_routed:
                cnt = sum(1 for t in mlp_types if str(t).lower() == "sparse")
                if cnt > 0:
                    num_moe_layers = cnt
            if num_moe_layers == 0 and n_routed:
                # generic first_k/moe_freq fallback
                first_k = int(config.get("first_k_dense_replace") or text_cfg.get("first_k_dense_replace") or 0)
                freq = int(config.get("moe_layer_freq") or text_cfg.get("moe_layer_freq") or 1)
                cnt = 0
                for i in range(num_layers):
                    if i >= first_k and i % freq == 0:
                        cnt += 1
                if cnt > 0:
                    num_moe_layers = cnt
                # qwen4_exp: every layer is MoE when num_experts present and no mlp types
                if num_moe_layers == 0 and model_type.lower().replace("-", "_") in ("qwen4_exp", "qwen4_exp_text") and n_routed:
                    num_moe_layers = int(num_layers) if num_layers else 0
        except Exception:
            pass

    # Filter MTP extra layers from expert_keys so expert_bytes excludes them
    if num_layers:
        try:
            filt_pat = re.compile(r"layers\.(\d+)\.")
            filtered = []
            for k in expert_keys:
                m = filt_pat.search(k)
                if m:
                    try:
                        if int(m.group(1)) >= num_layers:
                            continue
                    except Exception:
                        pass
                filtered.append(k)
            expert_keys = filtered
        except Exception:
            pass

    return expert_keys, num_moe_layers, n_routed


@lru_cache(maxsize=128)
def _cached_estimate(
    model_path_str: str,
    sig: tuple[tuple[str, int, int], ...],
    index_sig: tuple[int, int] | None,
) -> ExpertStreamingEstimate:
    model_path = Path(model_path_str)
    config = _load_config(model_path)

    checkpoint_files = [Path(p[0]) for p in sig]
    checkpoint_bytes = sum(p[1] for p in sig)

    # Load weight_map and headers
    weight_map: dict[str, str] = {}
    headers: dict[str, dict] = {}
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
        except Exception:
            weight_map = {}
        # build headers from per-file headers
        # only need expert keys' headers for refinement; load all file headers lazily via per-file
        needed_files = set(weight_map.values())
        for fname in needed_files:
            hdr = _safetensors_header(model_path / fname)
            for k, v in hdr.items():
                # headers are per-file, but weight_map keys are global; keep first occurrence
                if k not in headers:
                    headers[k] = v
    else:
        # single file or sharded without index: scan headers directly
        for fpath in checkpoint_files:
            hdr = _safetensors_header(fpath)
            for k, v in hdr.items():
                headers[k] = v
                # weight_map synthetic
                weight_map[k] = fpath.name

    expert_keys, num_moe_layers, experts_per_layer = _detect_expert_keys(
        weight_map, headers, config
    )

    # Sum expert_bytes via headers data_offsets size
    expert_bytes = 0
    for k in expert_keys:
        entry = headers.get(k)
        if entry and "data_offsets" in entry:
            try:
                s, e = entry["data_offsets"]
                expert_bytes += int(e) - int(s)
            except Exception:
                continue
        else:
            # fallback: estimate via shape*itemsize (rare)
            # we have checkpoint_bytes fallback later
            pass

    # If no expert_keys but model_type indicates MoE, try fallback scan of headers for switch_mlp
    if not expert_keys:
        model_type = str(config.get("model_type") or "")
        text_cfg_local = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
        if not model_type and text_cfg_local:
            model_type = str(text_cfg_local.get("model_type") or "")
        if model_type.lower().replace("-", "_") in ("glm_moe_dsa", "deepseek_v32", "glm5_next", "glm5_next_text", "qwen4_exp", "qwen4_exp_text") and headers:
            for k, entry in headers.items():
                if "switch_mlp" in k:
                    try:
                        s, e = entry["data_offsets"]
                        expert_bytes += int(e) - int(s)
                        expert_keys.append(k)
                    except Exception:
                        pass
            if expert_keys:
                import re

                layer_pat = re.compile(r"layers\.(\d+)\.")
                layers = {int(m.group(1)) for k in expert_keys if (m := layer_pat.search(k))}
                # Filter MTP extra layers
                cfg_layers = int(config.get("num_hidden_layers") or (config.get("text_config") or {}).get("num_hidden_layers") or 0) if config.get("num_hidden_layers") or (config.get("text_config") or {}).get("num_hidden_layers") else 0
                if cfg_layers:
                    layers = {l for l in layers if l < cfg_layers}
                num_moe_layers = len(layers)
                exp = config.get("n_routed_experts")
                if exp is None:
                    exp = (config.get("text_config") or {}).get("n_routed_experts")
                if exp is None:
                    exp = config.get("num_experts")
                if exp is None:
                    exp = (config.get("text_config") or {}).get("num_experts")
                experts_per_layer = int(exp or 0)

    supported = False
    reason: str | None = None
    per_expert = 0
    per_layer_expert = 0
    if expert_keys and num_moe_layers > 0 and experts_per_layer > 0 and expert_bytes > 0:
        # per_expert = expert_bytes / (layers*experts)
        try:
            per_expert = expert_bytes // (num_moe_layers * experts_per_layer)
            per_layer_expert = expert_bytes // num_moe_layers
            if per_expert > 0 and per_layer_expert > 0:
                supported = True
            else:
                reason = "per-expert bytes computed as 0"
        except Exception as e:
            reason = str(e)
    else:
        if not expert_keys:
            reason = "no expert tensors found"
        elif num_moe_layers <= 0:
            reason = "could not determine MoE layer count"
        elif experts_per_layer <= 0:
            reason = "n_routed_experts missing in config"
        elif expert_bytes <= 0:
            reason = "expert byte size 0"

    dense_bytes = max(0, checkpoint_bytes - expert_bytes)
    resident_bytes = int(checkpoint_bytes * _MODEL_OVERHEAD_FACTOR)
    # streaming with empty cache = dense only
    streaming_bytes_min = int(dense_bytes * _MODEL_OVERHEAD_FACTOR)
    # default cache 2 GiB
    default_budget = 2 * 1024 * 1024 * 1024
    streaming_bytes = streaming_bytes_min
    if supported and per_expert > 0:
        slots = min(experts_per_layer, default_budget // (num_moe_layers * per_expert) if num_moe_layers else 0)
        cache = int(slots * num_moe_layers * per_expert)
        streaming_bytes = int(dense_bytes * _MODEL_OVERHEAD_FACTOR + cache)

    return ExpertStreamingEstimate(
        supported=supported,
        checkpoint_bytes=checkpoint_bytes,
        expert_bytes=expert_bytes,
        dense_bytes=dense_bytes,
        resident_bytes=resident_bytes,
        streaming_bytes=streaming_bytes,
        num_moe_layers=num_moe_layers,
        experts_per_layer=experts_per_layer,
        per_expert_bytes=per_expert,
        per_layer_expert_bytes=per_layer_expert,
        reason=reason,
    )


def expert_streaming_estimate(model_path: str | Path) -> ExpertStreamingEstimate:
    """Inspect checkpoint headers without materializing tensor data."""

    p = Path(model_path).expanduser().resolve()
    files = {fp.resolve() for fp in p.glob("*.safetensors")}
    sig = tuple(
        sorted((str(fp), fp.stat().st_size, fp.stat().st_mtime_ns) for fp in files)
    )
    index_path = p / "model.safetensors.index.json"
    index_sig = None
    if index_path.is_file():
        st = index_path.stat()
        index_sig = (st.st_size, st.st_mtime_ns)
    return _cached_estimate(str(p), sig, index_sig)


def clear_estimate_cache() -> None:
    _cached_estimate.cache_clear()
