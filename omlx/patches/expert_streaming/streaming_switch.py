# SPDX-License-Identifier: Apache-2.0
"""Streaming MoE switch layers with per-expert LRU cache.

Implements a drop-in replacement for SwitchLinear / QuantizedSwitchLinear and
SwitchGLU that keeps a bounded number of experts resident as mx.arrays and
faults the rest from the SSD-backed ExpertBackingStore (or an in-RAM dict for
tests).  The budget is a total byte budget across all MoE layers; the cache
is global per model.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

try:
    from .shard_bank import ExpertBackingStore
except Exception:  # pragma: no cover
    ExpertBackingStore = Any  # type: ignore


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0


class ExpertLRUCache:
    """Global per-model LRU for expert slices.

    Each slot holds one expert's data for one layer and one projection.
    For fused gate+up, one slot is one expert's fused tensor.
    """

    def __init__(self, budget_bytes: int, per_expert_bytes: int):
        self.budget_bytes = int(budget_bytes)
        # number of experts (total across all layers) that fit
        if per_expert_bytes > 0:
            self.capacity = max(1, budget_bytes // per_expert_bytes) if budget_bytes > 0 else 0
        else:
            self.capacity = 0
        self._store: OrderedDict[tuple[int, int, str], mx.array] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: tuple[int, int, str]) -> mx.array | None:
        if key in self._store:
            self._store.move_to_end(key)
            self.stats.hits += 1
            return self._store[key]
        self.stats.misses += 1
        return None

    def put(self, key: tuple[int, int, str], value: mx.array) -> None:
        if self.capacity <= 0:
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        # evict
        while len(self._store) >= self.capacity:
            self._store.popitem(last=False)
            self.stats.evictions += 1
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()
        self.stats = CacheStats()

    @property
    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Helpers that mirror switch_layers.py
# ---------------------------------------------------------------------------

def _inverse_permutation(order, inverse_scatter=False):
    if inverse_scatter:
        return mx.put_along_axis(
            mx.zeros_like(order), order, mx.arange(order.size, dtype=order.dtype), axis=0
        )
    return mx.argsort(order)


def _gather_sort(x, indices, inverse_scatter=False):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = _inverse_permutation(order, inverse_scatter)
    lhs_indices = order // M
    x = x.flatten(0, -3)
    return x[lhs_indices], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


# ---------------------------------------------------------------------------
# Streaming SwitchLinear variants
# ---------------------------------------------------------------------------

class StreamingSwitchLinear(nn.Module):
    """BF16 SwitchLinear with streaming cache."""

    def __init__(
        self,
        layer_idx: int,
        proj_name: str,
        stacked_key: str,
        num_experts: int,
        input_dims: int,
        output_dims: int,
        backing: Any,
        cache: ExpertLRUCache,
        bias: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.proj_name = proj_name
        self.stacked_key = stacked_key
        self.num_experts = num_experts
        self._input_dims = input_dims
        self._output_dims = output_dims
        self.backing = backing
        self.cache = cache
        # Bias per expert (small, keep resident)
        self._bias: mx.array | None = None
        self._has_bias = bias

    @property
    def input_dims(self) -> int:
        return self._input_dims

    @property
    def output_dims(self) -> int:
        return self._output_dims

    def _load_expert_weight(self, expert_id: int) -> mx.array:
        key = (self.layer_idx, expert_id, self.stacked_key)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        # Load slice from backing
        if hasattr(self.backing, "load_expert"):
            w = self.backing.load_expert(self.stacked_key, expert_id)
        else:
            # dict-backed for tests: backing is dict[(layer, proj)] -> mx.array[E,O,I]
            bank = self.backing[(self.layer_idx, self.proj_name)]  # type: ignore[index]
            w = bank[expert_id]
            # ensure mx.array
            if not isinstance(w, mx.array):
                w = mx.array(w)
        self.cache.put(key, w)
        return w

    def set_bias(self, bias: mx.array | None) -> None:
        self._bias = bias

    def __call__(self, x, indices, sorted_indices=False):
        # Streaming path: we need to build a mini-bank for the unique experts in this batch
        # and remap indices.  To avoid host sync when indices is small, we still need to
        # materialize indices to find unique set. Decode typically has K=8 and B*L<=64 so
        # this is cheap; for large prefill (L>1k) we have many tokens but unique is bounded
        # by num_experts.
        # Evaluate indices to host for unique discovery
        # NOTE: this forces a sync per MoE layer — the fundamental cost of streaming.
        # Keep it on the same stream; caller already eval'd gate.
        # Use mx.eval + tolist pattern?
        # We avoid double eval by checking if indices is already evaluated (has no graph)
        # For simplicity, eval indices.
        mx.eval(indices)
        # Convert to numpy for unique
        try:
            flat_np = np.array(indices, copy=False).reshape(-1)  # device->host copy via mlx->numpy?
        except Exception:
            # fallback via tolist
            flat_list = indices.tolist()  # type: ignore[attr-defined]
            # flatten
            def _flatten(obj):
                if isinstance(obj, list):
                    for v in obj:
                        yield from _flatten(v)
                else:
                    yield obj
            flat_np = np.array(list(_flatten(flat_list)), dtype=np.int32)
        uniq = np.unique(flat_np)
        uniq_list = uniq.tolist()
        # Map global expert id -> compact id
        id_to_compact = {int(e): i for i, e in enumerate(uniq_list)}
        # Load each unique expert weight
        mini_weights = []
        for eid in uniq_list:
            w = self._load_expert_weight(int(eid))
            mini_weights.append(w)
        # Stack into mini-bank (U, O, I)
        if len(mini_weights) == 1:
            mini_bank = mx.expand_dims(mini_weights[0], 0)
        else:
            mini_bank = mx.stack(mini_weights, axis=0)
        # Remap indices
        # Build remapped array via numpy then mx.array
        remapped_np = np.vectorize(lambda x: id_to_compact[int(x)], otypes=[np.int32])(flat_np)
        remapped = mx.array(remapped_np.reshape(indices.shape))
        # Call gather_mm with mini-bank
        out = mx.gather_mm(x, mini_bank.swapaxes(-1, -2), rhs_indices=remapped, sorted_indices=sorted_indices)
        if self._bias is not None and self._has_bias:
            # bias slice per expert
            # gather bias rows
            # bias shape (E, O)
            # mini bias = stack of needed biases
            # we stored bias as mx.array[E,O] in _bias
            # slice similarly
            b_mini = mx.stack([self._bias[int(e)] for e in uniq_list], axis=0)  # (U,O)
            out = out + mx.expand_dims(b_mini[remapped], -2)
        return out


class StreamingQuantizedSwitchLinear(nn.Module):
    """INT4/INT8 quantized SwitchLinear with streaming cache."""

    def __init__(
        self,
        layer_idx: int,
        proj_name: str,
        stacked_weight_key: str,
        stacked_scales_key: str,
        stacked_biases_key: str | None,
        num_experts: int,
        input_dims: int,
        output_dims: int,
        backing: Any,
        cache: ExpertLRUCache,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        has_bias: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.proj_name = proj_name
        self.stacked_weight_key = stacked_weight_key
        self.stacked_scales_key = stacked_scales_key
        self.stacked_biases_key = stacked_biases_key
        self.num_experts = num_experts
        self._input_dims = input_dims
        self._output_dims = output_dims
        self.backing = backing
        self.cache = cache
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._has_bias = has_bias
        self._bias: mx.array | None = None

    @property
    def input_dims(self) -> int:
        return self._input_dims

    @property
    def output_dims(self) -> int:
        return self._output_dims

    def set_bias(self, bias: mx.array | None) -> None:
        self._bias = bias

    def _load_expert_bundle(self, expert_id: int) -> tuple[mx.array, mx.array, mx.array | None]:
        key = (self.layer_idx, expert_id, self.stacked_weight_key)
        # cache key includes weight; scales/biases are co-located, so we cache bundle
        cached_w = self.cache.get(key)
        if cached_w is not None:
            # scales/biases cached under companion keys
            s_key = (self.layer_idx, expert_id, self.stacked_scales_key)
            b_key = (self.layer_idx, expert_id, self.stacked_biases_key) if self.stacked_biases_key else None
            s = self.cache.get(s_key)
            b = self.cache.get(b_key) if b_key else None
            # if scales miss, reload bundle
            if s is None:
                # reload all
                pass
            else:
                return cached_w, s, b  # type: ignore[return-value]
        # load from backing
        if hasattr(self.backing, "load_expert"):
            w = self.backing.load_expert(self.stacked_weight_key, expert_id)
            s = self.backing.load_expert(self.stacked_scales_key, expert_id)
            b = None
            if self.stacked_biases_key:
                try:
                    b = self.backing.load_expert(self.stacked_biases_key, expert_id)
                except Exception:
                    b = None
        else:
            # dict backing for tests
            w_bank = self.backing[(self.layer_idx, self.proj_name, "weight")]
            s_bank = self.backing[(self.layer_idx, self.proj_name, "scales")]
            b_bank = self.backing.get((self.layer_idx, self.proj_name, "biases"))
            w = w_bank[expert_id] if isinstance(w_bank[expert_id], mx.array) else mx.array(w_bank[expert_id])
            s = s_bank[expert_id] if isinstance(s_bank[expert_id], mx.array) else mx.array(s_bank[expert_id])
            b = None
            if b_bank is not None:
                bb = b_bank[expert_id]
                b = bb if isinstance(bb, mx.array) else mx.array(bb)
        self.cache.put(key, w)
        self.cache.put((self.layer_idx, expert_id, self.stacked_scales_key), s)
        if b is not None and self.stacked_biases_key:
            self.cache.put((self.layer_idx, expert_id, self.stacked_biases_key), b)
        return w, s, b

    def __call__(self, x, indices, sorted_indices=False):
        mx.eval(indices)
        try:
            flat_np = np.array(indices, copy=False).reshape(-1)
        except Exception:
            flat_list = indices.tolist()  # type: ignore[attr-defined]
            def _flatten(obj):
                if isinstance(obj, list):
                    for v in obj:
                        yield from _flatten(v)
                else:
                    yield obj
            flat_np = np.array(list(_flatten(flat_list)), dtype=np.int32)
        uniq = np.unique(flat_np)
        uniq_list = uniq.tolist()
        id_to_compact = {int(e): i for i, e in enumerate(uniq_list)}
        # Load bundles
        mini_w, mini_s, mini_b = [], [], []
        has_b = False
        for eid in uniq_list:
            w, s, b = self._load_expert_bundle(int(eid))
            mini_w.append(w)
            mini_s.append(s)
            if b is not None:
                has_b = True
                mini_b.append(b)
        if len(mini_w) == 1:
            w_bank = mx.expand_dims(mini_w[0], 0)
            s_bank = mx.expand_dims(mini_s[0], 0)
            b_bank = mx.expand_dims(mini_b[0], 0) if has_b and mini_b else None
        else:
            w_bank = mx.stack(mini_w, axis=0)
            s_bank = mx.stack(mini_s, axis=0)
            b_bank = mx.stack(mini_b, axis=0) if has_b and mini_b else None
        remapped_np = np.vectorize(lambda x: id_to_compact[int(x)], otypes=[np.int32])(flat_np)
        remapped = mx.array(remapped_np.reshape(indices.shape))
        out = mx.gather_qmm(
            x,
            w_bank,
            s_bank,
            b_bank,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if self._bias is not None and self._has_bias:
            b_mini = mx.stack([self._bias[int(e)] for e in uniq_list], axis=0)
            out = out + mx.expand_dims(b_mini[remapped], -2)
        return out


class StreamingSwitchGLU(nn.Module):
    """Streaming SwitchGLU that delegates to streaming linears."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        layer_idx: int,
        backing: Any,
        cache: ExpertLRUCache,
        fused_gate_up: bool = False,
        inverse_scatter: bool = False,
        quantized: bool = False,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.fused_gate_up = fused_gate_up
        self.inverse_scatter = inverse_scatter
        self.quantized = quantized

        # We will be populated by the converter after construction
        # Placeholder attributes for introspection
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._num_experts = num_experts
        self._backing = backing
        self._cache = cache
        self._group_size = group_size
        self._bits = bits
        self._mode = mode

        # Create streaming linears lazily; actual keys set by converter
        self._initialized = False

    def _ensure_initialized(self, template_glu: Any) -> None:
        if self._initialized:
            return
        # Called once after we have a template to copy projection config
        self._initialized = True

    def __call__(self, x, indices, scores=None, weighted_sum: bool = False):
        # Mirror SwitchGLU.__call__ but route through streaming linears
        # Determine fused vs split by presence of gate_up_proj
        has_fused = hasattr(self, "gate_up_proj")
        x_exp = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x_exp, idx, inv_order = _gather_sort(x_exp, indices, inverse_scatter=self.inverse_scatter)

        if has_fused:
            x_gate_up = self.gate_up_proj(x_exp, idx, sorted_indices=do_sort)  # type: ignore[attr-defined]
            x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
            # swiglu
            from mlx_lm.models.activations import swiglu

            x_act = swiglu(x_gate, x_up)
            x_out = self.down_proj(x_act, idx, sorted_indices=do_sort)  # type: ignore[attr-defined]
        else:
            x_up = self.up_proj(x_exp, idx, sorted_indices=do_sort)  # type: ignore[attr-defined]
            x_gate = self.gate_proj(x_exp, idx, sorted_indices=do_sort)  # type: ignore[attr-defined]
            from mlx_lm.models.activations import swiglu

            x_act = swiglu(x_gate, x_up)
            x_out = self.down_proj(x_act, idx, sorted_indices=do_sort)  # type: ignore[attr-defined]

        # weighted sum fast path — keep compatible but may not use fast kernel when streaming
        if weighted_sum and scores is not None and do_sort:
            try:
                from .kernels import fast as glm_fast  # type: ignore

                if hasattr(glm_fast, "glm_moe_weighted_sum"):
                    return glm_fast.glm_moe_weighted_sum(x_out, inv_order, scores)
            except Exception:
                pass

        if do_sort:
            x_out = _scatter_unsort(x_out, inv_order, indices.shape)
        return x_out.squeeze(-2)
