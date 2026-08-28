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
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

_PROFILE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_PROFILE", "") == "1"

# Parallel os.pread pool for the demand-set of one MoE layer call. Workers
# return raw numpy slices only — MLX promotion happens on the inference
# thread. 8 workers lift a single 13 MB pread stream (QD1, ~1.4 GB/s) toward
# the NVMe's sequential ceiling.
_EXPERT_IO_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="omlx-expert-io")


@dataclass
class LayerProfile:
    calls: int = 0
    gate_eval_s: float = 0.0
    unique_s: float = 0.0
    load_s: float = 0.0
    stack_s: float = 0.0
    load_hits: int = 0
    load_misses: int = 0
    experts_requested: int = 0
    positions: int = 0
    # load-source split: staging (prefetch) vs synchronous backing read
    staged_hits: int = 0
    staged_s: float = 0.0  # take + promote (np -> mx on this thread)
    sync_loads: int = 0
    sync_s: float = 0.0  # backing read (np copy) + promote


class ProfileAccumulator:
    """Per-layer stage timing for the streaming switch (Fase 0 instrumentation).

    Buckets per layer, per token:
      gate_eval  – mx.eval(indices) + device->host copy
      unique     – np.unique + id remap
      load       – _load_expert_bundle total (split hits/misses)
      stack      – mx.stack of mini-bank + gather graph build (lazy; kernel cost
                   shows up in GLU wall time)
    Wall time (full GLU __call__) is tracked separately so kernel cost can be
    derived as wall − ∑(linears buckets).
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.layers: Dict[int, LayerProfile] = {}
        self.wall_s: Dict[int, float] = {}
        self.predicted: Dict[int, set] = {}
        self.observed: Dict[int, set] = {}

    def record_predicted(self, idx: int, ids: Any) -> None:
        if not self.enabled:
            return
        self.predicted.setdefault(idx, set()).update(int(v) for v in ids)

    def record_observed(self, idx: int, ids: Any) -> None:
        if not self.enabled:
            return
        self.observed.setdefault(idx, set()).update(int(v) for v in ids)

    def add(
        self,
        idx: int,
        *,
        gate: float,
        unique: float,
        load: float,
        stack: float,
        hits: int,
        misses: int,
        experts: int,
        positions: int,
    ) -> None:
        if not self.enabled:
            return
        lp = self.layers.setdefault(idx, LayerProfile())
        lp.calls += 1
        lp.gate_eval_s += gate
        lp.unique_s += unique
        lp.load_s += load
        lp.stack_s += stack
        lp.load_hits += hits
        lp.load_misses += misses
        lp.experts_requested += experts
        lp.positions += positions

    def add_wall(self, idx: int, dt: float) -> None:
        if not self.enabled:
            return
        self.wall_s[idx] = self.wall_s.get(idx, 0.0) + dt

    def add_load_source(self, idx: int, *, staged: bool, dt: float) -> None:
        if not self.enabled:
            return
        lp = self.layers.setdefault(idx, LayerProfile())
        if staged:
            lp.staged_hits += 1
            lp.staged_s += dt
        else:
            lp.sync_loads += 1
            lp.sync_s += dt

    def report(self) -> dict:
        per: Dict[str, dict] = {}
        totals = LayerProfile()
        for idx in sorted(self.layers):
            lp = self.layers[idx]
            c = max(lp.calls, 1)
            per[str(idx)] = {
                "calls": lp.calls,
                "gate_eval_ms": lp.gate_eval_s / c * 1e3,
                "unique_ms": lp.unique_s / c * 1e3,
                "load_ms": lp.load_s / c * 1e3,
                "stack_ms": lp.stack_s / c * 1e3,
                "wall_ms": self.wall_s.get(idx, 0.0) / c * 1e3,
                "load_hits": lp.load_hits,
                "load_misses": lp.load_misses,
                "hit_rate": lp.load_hits / max(lp.load_hits + lp.load_misses, 1),
                "staged_hits": lp.staged_hits,
                "staged_ms_per_hit": lp.staged_s / max(lp.staged_hits, 1) * 1e3,
                "sync_loads": lp.sync_loads,
                "sync_ms_per_load": lp.sync_s / max(lp.sync_loads, 1) * 1e3,
                "experts_req_per_call": lp.experts_requested / c,
                "positions_per_call": lp.positions / c,
            }
            totals.calls += lp.calls
            totals.gate_eval_s += lp.gate_eval_s
            totals.unique_s += lp.unique_s
            totals.load_s += lp.load_s
            totals.stack_s += lp.stack_s
            totals.load_hits += lp.load_hits
            totals.load_misses += lp.load_misses
            totals.experts_requested += lp.experts_requested
            totals.positions += lp.positions
            totals.staged_hits += lp.staged_hits
            totals.staged_s += lp.staged_s
            totals.sync_loads += lp.sync_loads
            totals.sync_s += lp.sync_s
        n = max(totals.calls, 1)
        tots = {
            "calls": totals.calls,
            "gate_eval_ms": totals.gate_eval_s / n * 1e3,
            "unique_ms": totals.unique_s / n * 1e3,
            "load_ms": totals.load_s / n * 1e3,
            "stack_ms": totals.stack_s / n * 1e3,
            "load_hits": totals.load_hits,
            "load_misses": totals.load_misses,
            "hit_rate_global": totals.load_hits / max(totals.load_hits + totals.load_misses, 1),
            "staged_hits": totals.staged_hits,
            "staged_ms_per_hit": totals.staged_s / max(totals.staged_hits, 1) * 1e3,
            "sync_loads": totals.sync_loads,
            "sync_ms_per_load": totals.sync_s / max(totals.sync_loads, 1) * 1e3,
            "wall_ms_per_call": sum(self.wall_s.values()) / n * 1e3,
            "layers": len(self.layers),
        }
        # Prediction accuracy: of the ids actually requested per layer, how
        # many had been predicted by the lookahead at least once
        pred_acc = {}
        pred_tot = obs_tot = hit_tot = 0
        for idx in sorted(set(self.predicted) | set(self.observed)):
            pr = self.predicted.get(idx, set())
            ob = self.observed.get(idx, set())
            hit = len(pr & ob)
            pred_tot += len(pr)
            obs_tot += len(ob)
            hit_tot += hit
            pred_acc[str(idx)] = {
                "predicted": len(pr),
                "observed": len(ob),
                "hit": hit,
                "recall": hit / max(len(ob), 1),
                "precision": hit / max(len(pr), 1),
            }
        return {
            "per_layer": per,
            "totals": tots,
            "prediction": pred_acc,
            "prediction_totals": {
                "predicted": pred_tot,
                "observed": obs_tot,
                "hit": hit_tot,
                "recall": hit_tot / max(obs_tot, 1),
            },
        }

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
    """Per-layer LRU for expert slices (global budget split evenly).

    Each slot holds one expert's bundle for one layer (weight+scales+biases).
    Budget is split across MoE layers → per-layer capacity = budget // (layers*per_expert)
    approximated via total capacity, but eviction is per-layer to avoid cross-layer thrashing.
    `size`/`capacity` remain global totals for logging.
    """

    def __init__(self, budget_bytes: int, per_expert_bytes: int, num_layers: int | None = None):
        self.budget_bytes = int(budget_bytes)
        self.per_expert_bytes = int(per_expert_bytes)
        self.num_layers = int(num_layers) if num_layers else 0
        if per_expert_bytes > 0:
            self.capacity = max(1, budget_bytes // per_expert_bytes) if budget_bytes > 0 else 0
        else:
            self.capacity = 0
        # per-layer stores to avoid global thrashing (layer 47 evicting layer 0)
        if self.num_layers > 0 and self.capacity > 0:
            per_layer = max(1, self.capacity // self.num_layers)
            # distribute remainder
            self._per_layer_cap = per_layer
            self._global_cap = self.capacity
        else:
            self._per_layer_cap = self.capacity
            self._global_cap = self.capacity
        self._store: OrderedDict[tuple[int, int, str], Any] = OrderedDict()
        # per-layer tracking for eviction
        self._layer_counts: Dict[int, int] = {}
        self.stats = CacheStats()
        self.profile = ProfileAccumulator(enabled=_PROFILE_ENV)

    def __contains__(self, key: tuple[int, int, str]) -> bool:
        return key in self._store

    def _layer_of(self, key: tuple[int, int, str]) -> int:
        try:
            return int(key[0])
        except Exception:
            return -1

    def get(self, key: tuple[int, int, str]) -> Any | None:
        if key in self._store:
            self._store.move_to_end(key)
            self.stats.hits += 1
            return self._store[key]
        self.stats.misses += 1
        return None

    def put(self, key: tuple[int, int, str], value: Any) -> None:
        if self.capacity <= 0:
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        # per-layer cap enforcement
        layer = self._layer_of(key)
        if self.num_layers > 0 and self._per_layer_cap:
            cnt = self._layer_counts.get(layer, 0)
            # evict oldest entry of same layer if per-layer full
            if cnt >= self._per_layer_cap:
                # find oldest entry of this layer
                for k in list(self._store.keys()):
                    if self._layer_of(k) == layer:
                        self._store.pop(k)
                        self.stats.evictions += 1
                        self._layer_counts[layer] = max(0, self._layer_counts.get(layer, 1) - 1)
                        break
                # if still over capacity due to rounding, fall through to global
        # global cap
        while len(self._store) >= self.capacity:
            old_k, _ = self._store.popitem(last=False)
            self.stats.evictions += 1
            old_layer = self._layer_of(old_k)
            self._layer_counts[old_layer] = max(0, self._layer_counts.get(old_layer, 1) - 1)
        self._store[key] = value
        self._layer_counts[layer] = self._layer_counts.get(layer, 0) + 1

    def clear(self) -> None:
        self._store.clear()
        self._layer_counts.clear()
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
# Shared per-layer routing plan (one host sync per MoE layer)
# ---------------------------------------------------------------------------

@dataclass
class _RemapPlan:
    """Routing plan shared by every streaming linear of one MoE layer call.

    The first linear invoked in a layer builds the plan (mx.eval + host copy
    + np.unique + compact remap); the other projections (up/gate/down) reuse
    it — one sync per MoE layer instead of three.
    """

    indices_shape: Tuple[int, ...] = ()
    flat_np: Any = None
    uniq_list: list = field(default_factory=list)
    remapped: Any = None  # mx.array of compact ids, original indices shape
    positions: int = 0
    gate_s: float = 0.0
    unique_s: float = 0.0


def _build_plan_into(plan: _RemapPlan, indices) -> None:
    """Populate a shared routing plan in place (called once per MoE layer)."""
    t0 = time.perf_counter()
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
    t1 = time.perf_counter()
    uniq_np = np.unique(flat_np)
    uniq_list = uniq_np.tolist()
    # compact remap via searchsorted (uniq is sorted ascending): vectorized C
    # lookup, replaces the per-element np.vectorize dict indirection
    remapped_np = np.searchsorted(uniq_np, flat_np).astype(np.int32)
    t2 = time.perf_counter()
    plan.indices_shape = tuple(indices.shape)
    plan.flat_np = flat_np
    plan.uniq_list = uniq_list
    plan.remapped = mx.array(remapped_np.reshape(indices.shape))
    plan.positions = int(flat_np.size)
    plan.gate_s = t1 - t0
    plan.unique_s = t2 - t1


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

    def __call__(self, x, indices, sorted_indices=False, plan: _RemapPlan | None = None):
        p = self.cache.profile
        if plan is None:
            plan = _RemapPlan()
        built = plan.flat_np is None
        if built:
            _build_plan_into(plan, indices)
        t2 = time.perf_counter()
        # Load each unique expert weight
        mini_weights = []
        t_load = 0.0
        hits = 0
        misses = 0
        for eid in plan.uniq_list:
            was_hit = (self.layer_idx, eid, self.stacked_key) in self.cache
            t_l = time.perf_counter()
            w = self._load_expert_weight(int(eid))
            t_load += time.perf_counter() - t_l
            if was_hit:
                hits += 1
            else:
                misses += 1
            mini_weights.append(w)
        # Stack into mini-bank (U, O, I)
        if len(mini_weights) == 1:
            mini_bank = mx.expand_dims(mini_weights[0], 0)
        else:
            mini_bank = mx.stack(mini_weights, axis=0)
        remapped = plan.remapped
        # Call gather_mm with mini-bank
        out = mx.gather_mm(x, mini_bank.swapaxes(-1, -2), rhs_indices=remapped, sorted_indices=sorted_indices)
        if self._bias is not None and self._has_bias:
            b_mini = mx.stack([self._bias[int(e)] for e in plan.uniq_list], axis=0)  # (U,O)
            out = out + mx.expand_dims(b_mini[remapped], -2)
        t4 = time.perf_counter()
        p.record_observed(self.layer_idx, plan.uniq_list)
        p.add(
            self.layer_idx,
            gate=plan.gate_s if built else 0.0,
            unique=plan.unique_s if built else 0.0,
            load=t_load,
            stack=t4 - t2 - t_load,
            hits=hits,
            misses=misses,
            experts=len(plan.uniq_list),
            positions=plan.positions,
        )
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

    def _slice_dtypes_lazy(self):
        if not hasattr(self, "_slice_dtypes"):
            td = getattr(self.backing, "tensor_dtype", None)
            self._slice_dtypes = (
                td(self.stacked_scales_key) if td else None,
                td(self.stacked_biases_key) if td and self.stacked_biases_key else None,
            )
        return self._slice_dtypes

    def _promote_np(self, v, dtype_str: str | None = None):
        """Promote a cached/staged np.ndarray to mx.array on this thread."""
        if v is None:
            return None
        if isinstance(v, mx.array):
            return v
        if dtype_str == "BF16" and v.dtype == np.uint16:
            # bf16 stored as raw uint16 bits — reinterpret directly (matches
            # mx.load; the old shift->f32->astype path flushed subnormals via
            # Metal FTZ and cost ~9x more on 4 MB slices).
            return mx.array(v).view(mx.bfloat16)
        return mx.array(v)  # np.ndarray -> mx.array copy on this thread

    def bundle_key(self, expert_id: int):
        return (self.layer_idx, expert_id, self.stacked_weight_key)

    def _load_expert_np(self, expert_id: int) -> tuple | None:
        """Numpy-only load for the prefetch worker.

        Never touches the LRU and never allocates MLX arrays (worker threads
        must not bind MLX ops to a non-existent default stream). Returns None
        when the backing has no slice-level API or the read fails.
        """
        if not hasattr(self.backing, "load_expert_slice"):
            return None
        try:
            w = self.backing.load_expert_slice(self.stacked_weight_key, expert_id)
            s = self.backing.load_expert_slice(self.stacked_scales_key, expert_id)
            b = None
            if self.stacked_biases_key:
                try:
                    b = self.backing.load_expert_slice(self.stacked_biases_key, expert_id)
                except Exception:
                    b = None
            return (w, s, b)
        except Exception:
            return None

    def _bundle_cached_or_staged(self, expert_id: int):
        """Resolve a bundle without touching the disk (inference thread only).

        Returns the cached bundle (mx or raw np tuple) or None when the expert
        must be fetched from the backing store.
        """
        key = (self.layer_idx, expert_id, self.stacked_weight_key)
        cached = self.cache.get(key)
        if cached is not None:
            # New format: bundle tuple stored under weight key
            if isinstance(cached, tuple) and len(cached) == 3:
                return cached  # type: ignore[return-value]
            # Legacy: companion keys (weight hit but scales separate) — upgrade to bundle
            if isinstance(cached, mx.array):
                s_key = (self.layer_idx, expert_id, self.stacked_scales_key)
                b_key = (self.layer_idx, expert_id, self.stacked_biases_key) if self.stacked_biases_key else None
                s = self.cache.get(s_key)
                b = self.cache.get(b_key) if b_key else None
                if s is not None:
                    bundle = (cached, s, b)
                    # Collapse 3 slots into 1 bundle slot (evict companions)
                    try:
                        self.cache._store.pop(s_key, None)  # type: ignore[attr-defined]
                        if b_key:
                            self.cache._store.pop(b_key, None)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    self.cache.put(key, bundle)  # type: ignore[arg-type]
                    return bundle  # type: ignore[return-value]
        # prefetch staging: worker already read the np slices — promote here
        # (inference thread) instead of re-reading from the backing.
        pf = getattr(self, "_prefetcher", None)
        if pf is not None:
            t_st = time.perf_counter()
            staged = pf.take(key)
            if staged is not None:
                dt = self._slice_dtypes_lazy()
                bundle = (
                    self._promote_np(staged[0]),
                    self._promote_np(staged[1], dt[0]),
                    self._promote_np(staged[2], dt[1]) if staged[2] is not None else None,
                )
                self.cache.put(key, bundle)  # type: ignore[arg-type]
                if getattr(self.cache, "profile", None) is not None:
                    self.cache.profile.add_load_source(
                        self.layer_idx, staged=True, dt=time.perf_counter() - t_st
                    )
                return bundle
        return None

    def _load_expert_bundle(self, expert_id: int) -> tuple[mx.array, mx.array, mx.array | None]:
        key = (self.layer_idx, expert_id, self.stacked_weight_key)
        # Cache / staging resolution (shared with the parallel demand-set path)
        resolved = self._bundle_cached_or_staged(expert_id)
        if resolved is not None:
            return resolved  # type: ignore[return-value]
        # 3) synchronous load from backing
        t_sy = time.perf_counter()
        if hasattr(self.backing, "load_expert_slice"):
            # Async-friendly: store plain np.ndarray slices in the cache and
            # promote them to mx.array on the inference thread at use time
            # (avoids cross-thread stream errors from MLX op allocation —
            # the prefetch worker must never allocate MLX arrays).
            w = self.backing.load_expert_slice(self.stacked_weight_key, expert_id)
            s = self.backing.load_expert_slice(self.stacked_scales_key, expert_id)
            b = None
            if self.stacked_biases_key:
                try:
                    b = self.backing.load_expert_slice(self.stacked_biases_key, expert_id)
                except Exception:
                    b = None
        elif hasattr(self.backing, "load_expert"):
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
        bundle = (w, s, b)
        self.cache.put(key, bundle)  # type: ignore[arg-type]
        if getattr(self.cache, "profile", None) is not None:
            self.cache.profile.add_load_source(
                self.layer_idx, staged=False, dt=time.perf_counter() - t_sy
            )
        return bundle

    def __call__(self, x, indices, sorted_indices=False, plan: _RemapPlan | None = None):
        p = self.cache.profile
        if plan is None:
            plan = _RemapPlan()
        built = plan.flat_np is None
        if built:
            _build_plan_into(plan, indices)
        t2 = time.perf_counter()
        # Load bundles: cache/staging resolution on this thread, then one
        # parallel os.pread fetch per missing bundle (QD1 -> QD8). Pool
        # workers return raw np slices only; promotion to mx.array happens
        # in the loop below on the inference thread.
        bundles: Dict[int, tuple] = {}
        mini_w, mini_s, mini_b = [], [], []
        has_b = False
        hits = 0
        misses = 0
        missing: list[int] = []
        t_res_start = time.perf_counter()
        for eid in plan.uniq_list:
            eid = int(eid)
            b = self._bundle_cached_or_staged(eid)
            if b is not None:
                bundles[eid] = b
                hits += 1
            else:
                misses += 1
                missing.append(eid)
        if missing:
            # ascending expert id = ascending file offset within the stacked
            # bank (row-major) — sorted reads keep the NVMe's locality
            missing.sort()
            if hasattr(self.backing, "load_expert_slice"):
                raws = list(_EXPERT_IO_POOL.map(self._load_expert_np, missing))
                dt_per = time.perf_counter() - t_res_start
                for eid, raw in zip(missing, raws):
                    if raw is None:
                        bundles[eid] = self._load_expert_bundle(eid)
                        continue
                    w, s, b = raw
                    bundle = (w, s, b)
                    self.cache.put((self.layer_idx, eid, self.stacked_weight_key), bundle)  # type: ignore[arg-type]
                    if p is not None:
                        p.add_load_source(self.layer_idx, staged=False, dt=dt_per / len(missing))
                    bundles[eid] = bundle
            else:
                # dict-backed test doubles: sequential fallback
                for eid in missing:
                    bundles[eid] = self._load_expert_bundle(eid)
        t_load = time.perf_counter() - t_res_start

        dt = self._slice_dtypes_lazy()
        for eid in plan.uniq_list:
            w, s, b = bundles[int(eid)]
            w = self._promote_np(w)
            s = self._promote_np(s, dt[0])
            if b is not None:
                has_b = True
                b = self._promote_np(b, dt[1])
            mini_w.append(w)
            mini_s.append(s)
            if b is not None:
                mini_b.append(b)
        if len(mini_w) == 1:
            w_bank = mx.expand_dims(mini_w[0], 0)
            s_bank = mx.expand_dims(mini_s[0], 0)
            b_bank = mx.expand_dims(mini_b[0], 0) if has_b and mini_b else None
        else:
            w_bank = mx.stack(mini_w, axis=0)
            s_bank = mx.stack(mini_s, axis=0)
            b_bank = mx.stack(mini_b, axis=0) if has_b and mini_b else None
        remapped = plan.remapped
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
            b_mini = mx.stack([self._bias[int(e)] for e in plan.uniq_list], axis=0)
            out = out + mx.expand_dims(b_mini[remapped], -2)
        t4 = time.perf_counter()
        p.record_observed(self.layer_idx, plan.uniq_list)
        p.add(
            self.layer_idx,
            gate=plan.gate_s if built else 0.0,
            unique=plan.unique_s if built else 0.0,
            load=t_load,
            stack=t4 - t2 - t_load,
            hits=hits,
            misses=misses,
            experts=len(plan.uniq_list),
            positions=plan.positions,
        )
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
        activation: Any | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.fused_gate_up = fused_gate_up
        self.inverse_scatter = inverse_scatter
        self.quantized = quantized
        # Original SwitchGLU activation (e.g. DeepSeek V4's LimitedSwiGLU with
        # swiglu_limit / fp32). None falls back to the stock mlx-lm swiglu.
        # Underscore attr keeps it out of the nn.Module parameter tree.
        self._activation = activation

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

    def _apply_activation(self, x_up: Any, x_gate: Any) -> Any:
        act = getattr(self, "_activation", None)
        if act is not None:
            # Same call order as the original SwitchGLU: activation(up, gate)
            return act(x_up, x_gate)
        from mlx_lm.models.activations import swiglu

        return swiglu(x_gate, x_up)

    def __call__(self, x, indices, scores=None, weighted_sum: bool = False):
        # Mirror SwitchGLU.__call__ but route through streaming linears
        p = getattr(self, "_cache", None).profile if hasattr(self, "_cache") else None
        t_wall0 = time.perf_counter() if (p is not None and p.enabled) else None
        # Determine fused vs split by presence of gate_up_proj
        has_fused = hasattr(self, "gate_up_proj")
        # Opt-in warm/pin hook (warmer.py): fires previous-token reads for
        # the next layer before this layer's demand loads; decode-only.
        hook = getattr(self, "_warm_pins", None)
        if hook is not None:
            hook.on_layer_start(self.layer_idx, int(indices.size))
        x_exp = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x_exp, idx, inv_order = _gather_sort(x_exp, indices, inverse_scatter=self.inverse_scatter)

        # One shared routing plan for the whole layer: the first linear
        # invoked builds it (single mx.eval + unique + remap), the rest reuse.
        plan = _RemapPlan()

        if has_fused:
            x_gate_up = self.gate_up_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
            x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
            x_act = self._apply_activation(x_up, x_gate)
            x_out = self.down_proj(x_act, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
        else:
            x_up = self.up_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
            x_gate = self.gate_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
            x_act = self._apply_activation(x_up, x_gate)
            x_out = self.down_proj(x_act, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]

        if hook is not None:
            hook.on_layer_plan(self.layer_idx, plan.uniq_list, plan.positions)

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
        out = x_out.squeeze(-2)
        if t_wall0 is not None and p is not None:
            p.add_wall(self.layer_idx, time.perf_counter() - t_wall0)
        return out
