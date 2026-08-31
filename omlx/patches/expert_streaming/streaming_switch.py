# SPDX-License-Identifier: Apache-2.0
"""Streaming MoE switch layers with per-expert LRU cache.

Implements a drop-in replacement for SwitchLinear / QuantizedSwitchLinear and
SwitchGLU that keeps a bounded number of experts resident as mx.arrays and
faults the rest from the SSD-backed ExpertBackingStore (or an in-RAM dict for
tests).  The budget is a total byte budget across all MoE layers; the cache
is global per model.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

_PROFILE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_PROFILE", "") == "1"
_COALESCE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_COALESCE", "") != "0"
# Prefill attribution diag: sync the GPU at every prefill-sized MoE GLU call
# and record the drain as a per-layer gpu bucket. Serializes CPU/GPU overlap
# (wall inflates), so use it for attribution only — never for latency claims.
_PREFILL_DIAG_ENV = os.environ.get("OMLX_EXPERT_STREAMING_PREFILL_DIAG", "") == "1"
# Routes above this count are treated as prefill-sized for the diag sync.
_PREFILL_DIAG_MIN_ROUTES = 512

# B5 admission filter (scan-resistant). When OMLX_EXPERT_STREAMING_ADMISSION=1,
# only experts seen >=2 times in the recent window enter the LRU. Disabled by
# default; operational default for this model/box is budget=0 (see docs).
_ADMISSION_ENV = os.environ.get("OMLX_EXPERT_STREAMING_ADMISSION", "") == "1"
_ADMISSION_WINDOW = 1024

# O2 cross-layer speculation (G2 F_RDADVISE + stash). RA is default-on (like G2)
# and can be disabled with OMLX_EXPERT_STREAMING_RA=0. When enabled, each
# MoE layer advises (or stashes, if OMLX_EXPERT_STREAMING_STASH=1) the next
# layer's previous-token experts via F_RDADVISE so the NVMe fetch overlaps
# compute. The stash is a small per-layer ring (<=256 experts total, ~650 MB
# worst) that bypasses the LRU and never blocks demand reads.
_RA_ENV = os.environ.get("OMLX_EXPERT_STREAMING_RA", "") != "0"
_STASH_ENV = os.environ.get("OMLX_EXPERT_STREAMING_STASH", "") == "1"
_STASH_MAX_ENTRIES = 256
_PREV_UNIQ_BY_LAYER: Dict[int, list[int]] = {}
_SPEC_STASH: Dict[Tuple[int, int, str], Any] = {}
_SPEC_STASH_ORDER: list[Tuple[int, int, str]] = []
_ADVISE_STATS = {"advised": 0, "advised_bytes": 0, "stash_hits": 0, "stash_misses": 0}

# Routing trace (Fase I3): when OMLX_EXPERT_STREAMING_TRACE is set, append one
# JSONL row per MoE layer call ({call, layer, positions, uniq}) so
# bench/lrc_analysis.py can compute routing-consistency (SRP/SCH) offline.
_TRACE_PATH = os.environ.get("OMLX_EXPERT_STREAMING_TRACE", "") or None
_TRACE_FILE = None
_TRACE_CALL = 0


def _trace_row(layer_idx: int, uniq_list: list, positions: int) -> None:
    global _TRACE_FILE, _TRACE_CALL
    if _TRACE_FILE is None:
        _TRACE_FILE = open(_TRACE_PATH, "a", buffering=1)  # noqa: SIM115
    _TRACE_CALL += 1
    _TRACE_FILE.write(
        json.dumps(
            {
                "call": _TRACE_CALL,
                "layer": layer_idx,
                "positions": positions,
                "uniq": [int(e) for e in uniq_list],
            }
        )
        + "\n"
    )

# Parallel os.pread pool for the demand-set of one MoE layer call. Workers
# return raw numpy slices only — MLX promotion happens on the inference
# thread. QD8 sustains ~1.5 GB/s on the reference NVMe; QD16 plateaus near
# ~2.5 GB/s (+34% decode) — see E1. OMLX_EXPERT_STREAMING_QD overrides.
#
# B1 correction (Fase J): _EXPERT_IO_POOL is a process-wide SINGLETON with
# 16 workers shared across all concurrent parents. With CTX_AHEAD=3, each
# parent sees ~5 workers on average, not 16. Device depth is 16 total,
# not N*16. Do not "fix" this to per-call pools — that oversubscribes and
# regressed at QD32 (commit 0a4d3c7). The sweep value 16 is process-wide.
_EXPERT_IO_POOL = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_QD", "") or 16)),
    thread_name_prefix="omlx-expert-io",
)

# Per-depth executors for models whose per-model settings override the pool
# depth (autotune). One shared executor per distinct depth value — repeated
# conversions of models tuned to the same depth must not multiply idle
# worker threads. depth None → the env-default module pool above.
_IO_POOLS: Dict[int, ThreadPoolExecutor] = {}
_IO_POOLS_LOCK = threading.Lock()


def io_pool_for(depth: int | None) -> ThreadPoolExecutor:
    """Return the expert IO pool for a per-model depth override."""
    if depth is None:
        return _EXPERT_IO_POOL
    try:
        d = int(depth)
    except (TypeError, ValueError):
        return _EXPERT_IO_POOL
    if d < 1:
        return _EXPERT_IO_POOL
    d = min(64, d)
    with _IO_POOLS_LOCK:
        pool = _IO_POOLS.get(d)
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=d, thread_name_prefix=f"omlx-expert-io-{d}"
            )
            _IO_POOLS[d] = pool
        return pool


@dataclass
class LayerProfile:
    calls: int = 0
    gate_eval_s: float = 0.0
    unique_s: float = 0.0
    load_s: float = 0.0
    stack_s: float = 0.0
    gpu_s: float = 0.0
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

    def add_gpu(self, idx: int, dt: float) -> None:
        if not self.enabled:
            return
        lp = self.layers.setdefault(idx, LayerProfile())
        lp.gpu_s += dt

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
                "gpu_ms": lp.gpu_s / c * 1e3,
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
            totals.gpu_s += lp.gpu_s
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
            "gpu_ms": totals.gpu_s / n * 1e3,
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
        # B5 admission filter: scan-resistant frequency window (only when env set)
        self._admission_enabled = bool(_ADMISSION_ENV and self.capacity > 0 and self.capacity < 4096)
        self._admission_counts: Dict[Tuple[int, int], int] = {}
        self._admission_order: deque[Tuple[int, int]] = deque()  # type: ignore[type-arg]
        self.admission_drops = 0

    def __contains__(self, key: tuple[int, int, str]) -> bool:
        return key in self._store

    def _layer_of(self, key: tuple[int, int, str]) -> int:
        try:
            return int(key[0])
        except Exception:
            return -1

    def get(self, key: tuple[int, int, str]) -> Any | None:
        # O2 stash check (decode speculation) - bypasses LRU, never counts as miss
        if _RA_ENV and _STASH_ENV:
            sv = _SPEC_STASH.get(key)  # type: ignore[arg-type]
            if sv is not None:
                _ADVISE_STATS["stash_hits"] += 1
                return sv
            # Count demand-path stash misses only when RA is active
            # (keeps stats meaningful without extra branching in hot path)
            # Miss counter incremented below after stash miss.
        if key in self._store:
            self._store.move_to_end(key)
            self.stats.hits += 1
            return self._store[key]
        if _RA_ENV and _STASH_ENV:
            _ADVISE_STATS["stash_misses"] += 1
        self.stats.misses += 1
        return None

    def _admission_should_insert(self, key: tuple[int, int, str]) -> bool:
        if not self._admission_enabled:
            return True
        # Frequency >=2 in recent window; count via hashed (layer, expert)
        lk = (int(key[0]), int(key[1]))
        c = self._admission_counts.get(lk, 0) + 1
        self._admission_counts[lk] = c
        self._admission_order.append(lk)
        if len(self._admission_order) > _ADMISSION_WINDOW:
            old = self._admission_order.popleft()
            oc = self._admission_counts.get(old, 0) - 1
            if oc <= 0:
                self._admission_counts.pop(old, None)
            else:
                self._admission_counts[old] = oc
        if c < 2:
            self.admission_drops += 1
            return False
        return True

    def put(self, key: tuple[int, int, str], value: Any) -> None:
        if self.capacity <= 0:
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        if not self._admission_should_insert(key):
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

    def retain_hot(self, hot_pairs: set) -> int:
        """Keep only entries whose (layer_idx, expert_id) is in hot_pairs.

        The prefill demand path fills the cache with the *last* chunks'
        experts; the hotness seeder replaces those contents with the
        prompt-wide hot set. Rebuilds per-layer counts; returns the number
        of evicted entries.
        """
        if self.capacity <= 0 or not self._store:
            return 0
        evicted = 0
        for key in list(self._store.keys()):
            if (key[0], key[1]) not in hot_pairs:
                del self._store[key]
                evicted += 1
        if evicted:
            counts: Dict[int, int] = {}
            for key in self._store:
                layer = self._layer_of(key)
                counts[layer] = counts.get(layer, 0) + 1
            self._layer_counts = counts
            self.stats.evictions += evicted
        return evicted

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
        # Per-model IO overrides (expert_streaming_io_depth/coalesce settings).
        # Consumed by the quantized demand path; inert here. None → module
        # env defaults (_EXPERT_IO_POOL / _COALESCE_ENV).
        self._io_pool_override: Any = None
        self._coalesce_override: bool | None = None

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
        # Per-model IO overrides (expert_streaming_io_depth/coalesce settings).
        # None → module env defaults (_EXPERT_IO_POOL / _COALESCE_ENV).
        self._io_pool_override: Any = None
        self._coalesce_override: bool | None = None
        # HOBBIT hot/cold split (Fase I6): hot experts keep the ORIGINAL
        # packing (source bits/gs below); the rest compute at the cold tier
        # (self._cold_bits/_cold_gs from expert_cold/ metadata). Empty set or
        # None bits = uniform tier (I5) — the single-bank path.
        self._hot_experts: set | None = None
        self._cold_bits: int | None = None
        self._cold_gs: int | None = None

    def set_hobbit_split(self, hot_experts, cold_bits: int, cold_gs: int) -> None:
        """Enable the dual-tier path for this linear (convert-time only)."""
        self._hot_experts = {int(e) for e in (hot_experts or [])}
        self._cold_bits = int(cold_bits)
        self._cold_gs = int(cold_gs)

    def _is_split_active(self) -> bool:
        return (
            self._hot_experts is not None
            and len(self._hot_experts) > 0
            and self._cold_bits is not None
            and self._cold_bits != self.bits
        )

    def _tier_of(self, expert_id: int) -> int:
        """0 = hot (source packing), 1 = cold (tier packing)."""
        return 0 if int(expert_id) in (self._hot_experts or ()) else 1

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
        # Tier-suffixed under the HOBBIT split so a hot (source-packing)
        # bundle and a cold (tier-packing) bundle of the same expert can
        # coexist in the LRU without aliasing.
        tier = self._tier_of(expert_id) if self._is_split_active() else 0
        base = self.stacked_weight_key if tier == 0 else self.stacked_weight_key + "#c"
        return (self.layer_idx, expert_id, base)

    def _load_expert_np(self, expert_id: int) -> tuple | None:
        """Numpy-only load for the prefetch worker.

        Never touches the LRU and never allocates MLX arrays (worker threads
        must not bind MLX ops to a non-existent default stream). Returns None
        when the backing has no slice-level API or the read fails.
        """
        if not hasattr(self.backing, "load_expert_slice"):
            return None
        # Tier contract: the backing's hot set (same ids as this linear's
        # _hot_experts) routes hot ids to the source shards; everyone else
        # reads expert_cold/. The LRU key (bundle_key) keeps the two apart.
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

    def _load_expert_run_np(self, first_id: int, count: int) -> list[tuple] | None:
        """Numpy-only load of *count* consecutive experts in one pread per key.

        Returns None when the run read is unsupported/fails (caller falls
        back to per-expert loads). Runs exploit row-major contiguity: one
        sequential transfer instead of *count* scattered ones.
        """
        if not hasattr(self.backing, "load_expert_run"):
            return None
        try:
            ws = self.backing.load_expert_run(self.stacked_weight_key, first_id, count)
            ss = self.backing.load_expert_run(self.stacked_scales_key, first_id, count)
            bs: list | None = None
            if self.stacked_biases_key:
                try:
                    bs = self.backing.load_expert_run(self.stacked_biases_key, first_id, count)
                except Exception:
                    bs = None
            return [
                (w, s, bs[i] if bs is not None and i < len(bs) else None)
                for i, (w, s) in enumerate(zip(ws, ss))
            ]
        except Exception:
            return None

    def _group_runs(self, sorted_ids: list[int], max_run: int = 16) -> list[tuple[int, int]]:
        """Split ascending expert ids into (first, count) contiguous runs.

        Under the HOBBIT split a run must NOT cross a tier boundary: the
        coalesced pread reads from ONE backing reader (resolved by the
        first id — source shard vs expert_cold/), so experts past the
        boundary would come back in the wrong packing. Runs therefore end
        at the first id whose tier differs from the run's first id."""
        tier_of = self._tier_of if self._is_split_active() else None
        runs: list[tuple[int, int]] = []
        i = 0
        n = len(sorted_ids)
        while i < n:
            first = sorted_ids[i]
            first_tier = tier_of(first) if tier_of else 0
            count = 1
            while (
                count < max_run
                and i + count < n
                and sorted_ids[i + count] == first + count
                and (tier_of is None or tier_of(sorted_ids[i + count]) == first_tier)
            ):
                count += 1
            runs.append((first, count))
            i += count
        return runs

    def _bundle_cached_or_staged(self, expert_id: int):
        """Resolve a bundle without touching the disk (inference thread only).

        Returns the cached bundle (mx or raw np tuple) or None when the expert
        must be fetched from the backing store.
        """
        key = self.bundle_key(expert_id)
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

    def _advise_next_layer_prev_token(self) -> None:
        if not _RA_ENV:
            return
        next_layer = self.layer_idx + 1
        prev = _PREV_UNIQ_BY_LAYER.get(next_layer)
        if not prev:
            return
        try:
            sorted_prev = sorted(int(e) for e in prev)
            if not sorted_prev:
                return
            # Coalesce into runs for single F_RDADVISE per run
            s = sorted_prev[0]
            cnt = 1
            runs: list[tuple[int, int]] = []
            for eid in sorted_prev[1:]:
                if eid == s + cnt:
                    cnt += 1
                else:
                    runs.append((s, cnt))
                    s, cnt = eid, 1
            runs.append((s, cnt))
            for first, count in runs:
                try:
                    ok = self.backing.advise_expert_run(self.stacked_weight_key, first, count)
                    if ok:
                        _ADVISE_STATS["advised"] += count
                except Exception:
                    pass
        except Exception:
            pass

    def _load_expert_bundle(self, expert_id: int) -> tuple[mx.array, mx.array, mx.array | None]:
        key = self.bundle_key(expert_id)
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
        if _RA_ENV and _PREV_UNIQ_BY_LAYER:
            try:
                self._advise_next_layer_prev_token()
            except Exception:
                pass
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
                io_pool = self._io_pool_override or _EXPERT_IO_POOL
                coalesce_on = (
                    _COALESCE_ENV
                    if self._coalesce_override is None
                    else bool(self._coalesce_override)
                )
                raws: list = [None] * len(missing)
                # Coalesce consecutive ids into single-pread runs (dense in
                # long-prompt prefill; rare in decode, where runs are size 1
                # and the path degenerates to the per-expert fetch).
                # B2/O6: map keeps a sliding window of 16 in flight (singleton
                # pool), so the device queue stays full; batch drain/sawtooth
                # is avoided without moving promotion off the inference thread.
                runs = self._group_runs(missing)
                if coalesce_on and len(runs) < len(missing):
                    results_by_run = list(
                        io_pool.map(lambda r: (r, self._load_expert_run_np(r[0], r[1])), runs)
                    )
                    idx_of = {eid: i for i, eid in enumerate(missing)}
                    leftover: list[int] = []
                    for (first, count), out in results_by_run:
                        if out is not None:
                            for j in range(count):
                                raws[idx_of[first + j]] = out[j]
                        else:
                            leftover.extend(range(first, first + count))
                    if leftover:
                        for eid, raw in zip(
                            leftover, io_pool.map(self._load_expert_np, leftover)
                        ):
                            raws[idx_of[eid]] = raw
                else:
                    raws = list(io_pool.map(self._load_expert_np, missing))
                dt_per = time.perf_counter() - t_res_start
                for eid, raw in zip(missing, raws):
                    if raw is None:
                        bundles[eid] = self._load_expert_bundle(eid)
                        continue
                    # Raw np bundles in the LRU by design: Metal only holds the
                    # per-call stack. Caching promoted mx copies here double-
                    # holds the same weights in wired memory when the budget
                    # is positive — measured: LRU(mx) + stacks summed 37GB
                    # Metal active on a 6GiB budget and the guard killed the
                    # second prefill outright (F2 post-mortem).
                    bundles[eid] = raw
                    # Tier-suffixed key (bundle_key): under the HOBBIT split a
                    # hot (source-packing) and cold (tier-packing) bundle of
                    # the same expert must never alias in the LRU — the raw
                    # pread path staged unsuffixed keys and served a cold-packing
                    # bundle to a hot slot (mixed widths, mx.stack crash).
                    self.cache.put(self.bundle_key(eid), raw)  # type: ignore[arg-type]
                    if p is not None:
                        p.add_load_source(self.layer_idx, staged=False, dt=dt_per / len(missing))
            else:
                # dict-backed test doubles: sequential fallback
                for eid in missing:
                    bundles[eid] = self._load_expert_bundle(eid)
        t_load = time.perf_counter() - t_res_start

        dt = self._slice_dtypes_lazy()
        split = self._is_split_active()
        # Per-tier bundle lists under the HOBBIT split: hot (source packing)
        # and cold (tier packing) widths differ (e.g. 8 vs 6 u32 cols per
        # row at gs 64), so a single stacked mini-bank is impossible — build
        # one per tier and combine the two gather_qmm outputs.
        tier_w = ([], [])  # hot, cold
        tier_s = ([], [])
        tier_b = ([], [])
        uniq: list[int] = []
        hot_idx: list[int] = []
        cold_idx: list[int] = []
        if split:
            uniq = [int(e) for e in plan.uniq_list]
            hot_idx = [i for i, e in enumerate(uniq) if self._tier_of(e) == 0]
            cold_idx = [i for i, e in enumerate(uniq) if self._tier_of(e) == 1]
            hot_rank = {i: r for r, i in enumerate(hot_idx)}
            cold_rank = {i: r for r, i in enumerate(cold_idx)}
            for i, eid in enumerate(uniq):
                w, s, b = bundles[eid]
                if i in hot_rank:
                    t = 0
                else:
                    t = 1
                tier_w[t].append(self._promote_np(w))
                tier_s[t].append(self._promote_np(s, dt[0]))
                if b is not None:
                    has_b = True
                    tier_b[t].append(self._promote_np(b, dt[1]))
        else:
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

        # HOBBIT dual-tier assembly (Fase I6): one mini-bank per tier and a
        # masked add — positions are mutually exclusive (each position
        # consumes exactly one expert), so the two gather_qmm outputs
        # partition the positions and zeros fill the rest.
        if split:
            flat_np = np.asarray(plan.flat_np).reshape(-1)
            out = None
            for t, idxs, bits_, gs_ in (
                (0, hot_idx, self.bits, self.group_size),
                (1, cold_idx, self._cold_bits, self._cold_gs),
            ):
                if not idxs:
                    continue
                ws, ss, bs_ = tier_w[t], tier_s[t], tier_b[t]
                if len(ws) == 1:
                    w_b = mx.expand_dims(ws[0], 0)
                    s_b = mx.expand_dims(ss[0], 0)
                    b_b = mx.expand_dims(bs_[0], 0) if bs_ else None
                else:
                    w_b = mx.stack(ws, axis=0)
                    s_b = mx.stack(ss, axis=0)
                    b_b = mx.stack(bs_, axis=0) if bs_ else None
                # expert-id -> rank within THIS tier's bank (flat ids here,
                # not compact uniq ranks); -1 where the other tier owns it.
                tier_map = np.full((self.num_experts,), -1, dtype=np.int32)
                for rank, i in enumerate(idxs):
                    tier_map[uniq[i]] = rank
                tier_remapped_np = tier_map[flat_np].reshape(plan.indices_shape)
                # gather_qmm takes UNSIGNED row indices — -1 wraps to a huge
                # OOB index (garbage/nan) that the keep mask cannot undo
                # (nan * 0 = nan). Clamp the gather indices to 0 (any valid
                # rank: the row is zeroed by the keep mask below); the -1
                # survives only in keep_np, which is what selects the tier.
                gather_np = np.maximum(tier_remapped_np, 0)
                tier_remapped = mx.array(gather_np)
                tier_out = mx.gather_qmm(
                    x,
                    w_b,
                    s_b,
                    b_b,
                    rhs_indices=tier_remapped,
                    transpose=True,
                    group_size=gs_,
                    bits=bits_,
                    mode=self.mode,
                    sorted_indices=sorted_indices,
                )
                # Mask: keep only the positions this tier owns (-1 elsewhere).
                # gather_qmm inserts the indices' shape at dims 2.. so the
                # keep mask is the (index-shaped) validity, expanded over the
                # trailing (x_exp singleton, output) dims: [.., topk, 1, 1].
                keep_np = (tier_remapped_np >= 0).astype(np.float32)
                keep_shape = tuple(plan.indices_shape) + (1,) * (tier_out.ndim - len(plan.indices_shape))
                keep = mx.array(keep_np).reshape(keep_shape)
                tier_out = tier_out * keep
                out = tier_out if out is None else out + tier_out
            if out is None:
                # Degenerate: every unique expert hot (hot bank == full uniq
                # order) — identical to the uniform path.
                ws, ss, bs_ = tier_w[0], tier_s[0], tier_b[0]
                w_b = mx.stack(ws, axis=0) if len(ws) > 1 else mx.expand_dims(ws[0], 0)
                s_b = mx.stack(ss, axis=0) if len(ss) > 1 else mx.expand_dims(ss[0], 0)
                b_b = (mx.stack(bs_, axis=0) if len(bs_) > 1 else mx.expand_dims(bs_[0], 0)) if (has_b and bs_) else None
                out = mx.gather_qmm(
                    x, w_b, s_b, b_b, rhs_indices=plan.remapped,
                    transpose=True, group_size=self.group_size, bits=self.bits,
                    mode=self.mode, sorted_indices=sorted_indices,
                )
            if self._bias is not None and self._has_bias:
                b_mini = mx.stack([self._bias[int(e)] for e in plan.uniq_list], axis=0)
                out = out + mx.expand_dims(b_mini[plan.remapped], -2)
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
            # O2: remember this layer's routing for next token's speculation
            _PREV_UNIQ_BY_LAYER[self.layer_idx] = [int(e) for e in plan.uniq_list]
            return out

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
        # O2: remember this layer's routing for next token's speculation
        _PREV_UNIQ_BY_LAYER[self.layer_idx] = [int(e) for e in plan.uniq_list]
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
            # Fase I6 hotness signal: per-TOKEN usage over the routing plan
            # (bincount of the flat ids), computed only when a consumer
            # wants it — the readahead warmer keeps the uniq-list contract
            # and pays nothing. flat_np is already on the host (built by
            # _build_plan_into), so the bincount is a cheap vectorized pass.
            counts = (
                np.bincount(
                    np.asarray(plan.flat_np).reshape(-1),
                    minlength=self._num_experts,
                )
                if getattr(hook, "wants_usage_counts", False)
                else None
            )
            hook.on_layer_plan(self.layer_idx, plan.uniq_list, plan.positions, counts)
        if _TRACE_PATH is not None:
            _trace_row(self.layer_idx, plan.uniq_list, plan.positions)

        if (
            _PREFILL_DIAG_ENV
            and p is not None
            and p.enabled
            and int(indices.size) >= _PREFILL_DIAG_MIN_ROUTES
        ):
            # Force-eval the layer's graph (everything upstream is a lazy
            # dependency of x_out, so with a sync at every MoE GLU each eval
            # covers exactly one layer's segment: attention + dense + GLU
            # QMMs). CPU buckets measured inside the linears remain valid;
            # absolute wall inflates because the CPU/GPU overlap is gone.
            t_gpu0 = time.perf_counter()
            mx.eval(x_out)
            p.add_gpu(self.layer_idx, time.perf_counter() - t_gpu0)

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
