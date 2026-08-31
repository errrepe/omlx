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
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# Opt-in per-layer / per-projection Metal memory trace (Fase J prefill-memory
# work). Null-tracer by default: call sites cost one attribute lookup.
from .memtrace import memtrace  # noqa: E402

_PROFILE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_PROFILE", "") == "1"
_COALESCE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_COALESCE", "") != "0"
_BANK_MAX_BYTES = max(
    1,
    int(os.environ.get("OMLX_EXPERT_STREAMING_BANK_MAX_BYTES", str(256 * 1024**2))),
)
_RUN_MAX = max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_RUN_MAX", "16")))
# Etapa A1: promote an all-miss demand bank with a single mx.array instead of
# U per-expert mx arrays followed by mx.stack. Bit-identical — gather_qmm
# receives the same bytes, dtype and shape — but it halves the Metal transient
# at the promotion point, where the U copies and the bank briefly coexist.
# 0 restores the per-expert promote + stack path.
_BANK_PROMOTE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_BANK_PROMOTE", "1") != "0"
# Etapa A1b: same single-promotion trick, but on the *layer-context* path,
# which is the one that actually runs when the Etapa B barrier is on (the
# default). The context reads the demand bank as NumPy on an IO pool worker
# and hands the raw buffers back; promoting them to MLX must happen here, on
# the inference thread, so no MLX op is ever bound off-stream. 0 restores the
# per-expert promote + stack path for A/B.
_BANK_PROMOTE_CTX_ENV = (
    os.environ.get("OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX", "1") != "0"
)
_LAYER_BARRIER_ENV = os.environ.get("OMLX_EXPERT_STREAMING_LAYER_BARRIER", "1") != "0"
# Etapa B: rolling per-projection bank load instead of the union load.
# 0 restores the legacy behaviour (every projection's NumPy bank resident at
# once) for A/B against the new pipelined path.
_CTX_ROLLING_ENV = os.environ.get("OMLX_EXPERT_STREAMING_CTX_ROLLING", "1") != "0"
# How many *following* projections to read in the background while the
# current one is promoted/computed. 0 disables prefetch entirely.
_CTX_PREFETCH_AHEAD = max(
    0, int(os.environ.get("OMLX_EXPERT_STREAMING_CTX_AHEAD", "1"))
)
# Banks larger than this are never held speculatively; they are read on demand.
_CTX_PREFETCH_MAX_BYTES = max(
    0,
    int(
        os.environ.get(
            "OMLX_EXPERT_STREAMING_CTX_AHEAD_BYTES", str(512 * 1024**2)
        )
    ),
)
# Prefill attribution diag: sync the GPU at every prefill-sized MoE GLU call
# and record the drain as a per-layer gpu bucket. Serializes CPU/GPU overlap
# (wall inflates), so use it for attribution only — never for latency claims.
_PREFILL_DIAG_ENV = os.environ.get("OMLX_EXPERT_STREAMING_PREFILL_DIAG", "") == "1"
# Routes above this count are treated as prefill-sized for the diag sync.
_PREFILL_DIAG_MIN_ROUTES = 512

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
_EXPERT_IO_POOL = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_QD", "") or 16)),
    thread_name_prefix="omlx-expert-io",
)

# Per-depth executors for models whose per-model settings override the pool
# depth (autotune). One shared executor per distinct depth value — repeated
# conversions of models tuned to the same depth must not multiply idle
# worker threads. depth None → the env-default module pool above.
_IO_POOLS: dict[int, ThreadPoolExecutor] = {}
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
        self.layers: dict[int, LayerProfile] = {}
        self.wall_s: dict[int, float] = {}
        self.predicted: dict[int, set] = {}
        self.observed: dict[int, set] = {}

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
        per: dict[str, dict] = {}
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
        else:
            self._per_layer_cap = self.capacity
        self._store: OrderedDict[tuple[int, int, str], Any] = OrderedDict()
        self._layer_store: dict[int, OrderedDict[tuple[int, int, str], None]] = {}
        self._lock = threading.RLock()
        # per-layer tracking for eviction
        self._layer_counts: dict[int, int] = {}
        self.stats = CacheStats()
        self.profile = ProfileAccumulator(enabled=_PROFILE_ENV)

    def __contains__(self, key: tuple[int, int, str]) -> bool:
        with self._lock:
            return key in self._store

    def _layer_of(self, key: tuple[int, int, str]) -> int:
        try:
            return int(key[0])
        except Exception:
            return -1

    def _remove_locked(self, key: tuple[int, int, str]) -> bool:
        if key not in self._store:
            return False
        del self._store[key]
        layer = self._layer_of(key)
        queue = self._layer_store.get(layer)
        if queue is not None:
            queue.pop(key, None)
            if not queue:
                self._layer_store.pop(layer, None)
        count = self._layer_counts.get(layer, 0) - 1
        if count > 0:
            self._layer_counts[layer] = count
        else:
            self._layer_counts.pop(layer, None)
        return True

    def get(self, key: tuple[int, int, str]) -> Any | None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                layer_queue = self._layer_store.get(self._layer_of(key))
                if layer_queue is not None and key in layer_queue:
                    layer_queue.move_to_end(key)
                self.stats.hits += 1
                return self._store[key]
            self.stats.misses += 1
            return None

    def put(self, key: tuple[int, int, str], value: Any) -> None:
        if self.capacity <= 0:
            return
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
                queue = self._layer_store.setdefault(self._layer_of(key), OrderedDict())
                queue.move_to_end(key)
                return
            layer = self._layer_of(key)
            queue = self._layer_store.setdefault(layer, OrderedDict())
            if self.num_layers > 0 and self._per_layer_cap and len(queue) >= self._per_layer_cap:
                old_key, _ = queue.popitem(last=False)
                self._remove_locked(old_key)
                self.stats.evictions += 1
            while len(self._store) >= self.capacity:
                old_key = next(iter(self._store))
                self._remove_locked(old_key)
                self.stats.evictions += 1
            self._store[key] = value
            self._layer_store.setdefault(layer, OrderedDict())[key] = None
            self._layer_counts[layer] = self._layer_counts.get(layer, 0) + 1

    def discard(self, key: tuple[int, int, str]) -> bool:
        """Remove a cache entry while keeping per-layer accounting correct."""
        with self._lock:
            return self._remove_locked(key)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._layer_store.clear()
            self._layer_counts.clear()
            self.stats = CacheStats()

    def retain_hot(self, hot_pairs: set) -> int:
        """Keep only entries whose (layer_idx, expert_id) is in hot_pairs.

        The prefill demand path fills the cache with the *last* chunks'
        experts; the hotness seeder replaces those contents with the
        prompt-wide hot set. Rebuilds per-layer counts; returns the number
        of evicted entries.
        """
        with self._lock:
            if self.capacity <= 0 or not self._store:
                return 0
            evicted = 0
            for key in list(self._store.keys()):
                if (key[0], key[1]) not in hot_pairs:
                    if self._remove_locked(key):
                        evicted += 1
            if evicted:
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
    *_, m_cols = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = _inverse_permutation(order, inverse_scatter)
    lhs_indices = order // m_cols
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

class _LayerLoadContext:
    """Shared quantized demand load for one MoE layer's projections.

    Scope: quantized only (Fase J G4). The context is driven through hooks
    that exist solely on ``StreamingQuantizedSwitchLinear`` —
    ``stacked_weight_key``, ``_bank_bytes_for`` and ``_load_expert_bank_np`` —
    and it is only constructed when the owning GLU is quantized, so
    ``StreamingSwitchLinear`` (bf16) never participates. That is intentional:
    the bf16 path resolves one projection at a time inside its own ``__call__``
    and therefore has no cross-projection union for the context to collapse.

    Two modes, selected by ``OMLX_EXPERT_STREAMING_CTX_ROLLING``:

    rolling (default — Etapa B)
        Each projection resolves its own bank on demand. At most
        ``_CTX_PREFETCH_AHEAD`` following projections are read on pool workers
        in the background, so the next bank is in flight while the current one
        is promoted and consumed on the GPU. Peak NumPy residency drops from
        the *union* of every projection (~3 banks) to ~1-2 banks.

    union (legacy — set the env var to 0)
        One ``pool.map`` across every projection; all banks are resident until
        the last projection is consumed. Maximum I/O parallelism, highest RSS.

    Both modes preserve the C6 contract: one shared routing plan, and reads
    performed on IO-pool workers that never allocate MLX arrays.
    """

    def __init__(self, linears: list[Any], cache: ExpertLRUCache):
        self.linears = linears
        self.cache = cache
        self.bundles: dict[int, dict[int, tuple]] = {}
        self.hits: dict[int, int] = {}
        self.misses: dict[int, int] = {}
        self.failed = False
        # Etapa A1b: the raw contiguous NumPy banks behind ``bundles``, kept so
        # the linear can promote a whole demand set with one mx.array per key
        # instead of U per-expert arrays plus a stack. Populated only when the
        # read covered the *entire* demand set (all-miss); ``bank_ids`` records
        # exactly which ids the bank holds, so a stale bank can never be
        # promoted against a demand set it does not describe.
        self.bank_raw: dict[int, tuple] = {}
        self.bank_ids: dict[int, list[int]] = {}
        # rolling state
        self._order: dict[int, int] = {id(lin): i for i, lin in enumerate(linears)}
        self._futures: dict[int, Any] = {}
        self._inflight: dict[int, int] = {}
        self._resolved: set[int] = set()
        self._expert_ids: list[int] = []
        # legacy union latch
        self._loaded = False

    # -- helpers ------------------------------------------------------------

    def _split(self, linear: Any, expert_ids: list[int]) -> tuple[dict, list[int]]:
        """Partition ``expert_ids`` into cached bundles and missing ids."""
        cached: dict[int, tuple] = {}
        missing: list[int] = []
        for eid in expert_ids:
            key = (linear.layer_idx, eid, linear.stacked_weight_key)
            value = self.cache.get(key)
            if value is None:
                missing.append(eid)
            else:
                cached[eid] = value
        return cached, missing

    @staticmethod
    def _pool_for(linear: Any):
        return getattr(linear, "_io_pool_override", None) or _EXPERT_IO_POOL

    @property
    def _inflight_bytes(self) -> int:
        return sum(self._inflight.values())

    # -- rolling path -------------------------------------------------------

    def _prefetch(self, linear: Any) -> None:
        """Start background reads for the following projections, bounded.

        Bounded two ways: at most ``_CTX_PREFETCH_AHEAD`` submissions per
        call, and no single bank larger than ``_CTX_PREFETCH_MAX_BYTES`` is
        held speculatively (it is read on demand instead).
        """
        if _CTX_PREFETCH_AHEAD <= 0:
            return
        start = self._order.get(id(linear), -1)
        if start < 0:
            return
        submitted = 0
        for nxt in self.linears[start + 1 :]:
            if submitted >= _CTX_PREFETCH_AHEAD:
                break
            nid = id(nxt)
            if nid in self._resolved or nid in self._futures:
                continue
            cached, missing = self._split(nxt, self._expert_ids)
            self.bundles[nid] = cached
            self.hits[nid] = len(cached)
            self.misses[nid] = len(missing)
            if not missing:
                # Fully cached: nothing to read, mark resolved so the linear
                # short-circuits when it asks.
                self._resolved.add(nid)
                continue
            bank_bytes = int(nxt._bank_bytes_for(len(missing)))
            if bank_bytes > _CTX_PREFETCH_MAX_BYTES:
                continue
            # Etapa A1b: ask for the raw contiguous banks as well when
            # single-promotion is on. The read is identical either way — only
            # what the worker hands back differs, and it stays NumPy, so no MLX
            # op is ever created on a pool thread.
            reader = (
                nxt._load_expert_bank_np_full
                if _BANK_PROMOTE_CTX_ENV
                else nxt._load_expert_bank_np
            )
            self._futures[nid] = self._pool_for(nxt).submit(reader, missing)
            self._inflight[nid] = bank_bytes
            submitted += 1

    def _ensure_rolling(self, linear: Any, expert_ids: list[int]) -> None:
        lid = id(linear)
        if lid in self._resolved:
            return
        self._resolved.add(lid)
        if not self._expert_ids:
            self._expert_ids = list(expert_ids)
        ids = self._expert_ids

        # A prefetch may already be in flight; the split is recomputed because
        # the cache can change between submit and await.
        fut = self._futures.pop(lid, None)
        self._inflight.pop(lid, None)
        cached, missing = self._split(linear, ids)
        self.bundles[lid] = cached
        self.hits[lid] = len(cached)
        self.misses[lid] = len(missing)

        if missing:
            if fut is not None:
                try:
                    got = fut.result()
                except Exception:
                    got = None
            else:
                got = (
                    linear._load_expert_bank_np_full(missing)
                    if _BANK_PROMOTE_CTX_ENV
                    else (None, None, linear._load_expert_bank_np(missing))
                )
            # A worker dispatched as bare rows (prefetch submitted before this
            # call, or with the knob off) yields a list; normalise to the
            # (keys, banks, rows) shape so the consumer has one contract.
            if got is not None and not isinstance(got, tuple):
                got = (None, None, got)
            rows = None if got is None else got[2]
            if rows is None or len(rows) != len(missing):
                self.failed = True
                if memtrace.enabled:
                    memtrace.record(
                        "ctx.ensure.fail",
                        layer=linear.layer_idx,
                        proj=getattr(linear, "proj_name", "?"),
                        uniq=len(ids),
                        miss=len(missing),
                    )
                return
            self.bundles[lid].update(zip(missing, rows))
            # Etapa A1b: single-promotion is only valid when the read covered
            # the *whole* demand set. A partial bank would have to be
            # concatenated with separately promoted cache hits, which changes
            # the layout contract, so it is left on the legacy path.
            if _BANK_PROMOTE_CTX_ENV and got[0] is not None and len(missing) == len(ids):
                self.bank_raw[lid] = (got[0], got[1])
                self.bank_ids[lid] = list(missing)

        self._prefetch(linear)
        if memtrace.enabled:
            memtrace.record(
                "ctx.ensure.exit",
                layer=linear.layer_idx,
                proj=getattr(linear, "proj_name", "?"),
                uniq=len(ids),
                miss=len(missing),
                bank_bytes=int(linear._bank_bytes_for(len(missing))),
                inflight=len(self._futures),
                inflight_bytes=self._inflight_bytes,
            )

    # -- legacy union path --------------------------------------------------

    def _ensure_union(self, linear: Any, expert_ids: list[int]) -> None:
        if self._loaded:
            return
        self._loaded = True
        tracing = memtrace.enabled
        layer = self.linears[0].layer_idx if self.linears else -1
        if tracing:
            memtrace.record(
                "ctx.ensure.enter",
                layer=layer,
                n_proj=len(self.linears),
                uniq=len(expert_ids),
            )
        jobs: list[tuple[Any, list[int]]] = []
        for proj in self.linears:
            cached, missing = self._split(proj, expert_ids)
            self.bundles[id(proj)] = cached
            self.hits[id(proj)] = len(cached)
            self.misses[id(proj)] = len(missing)
            if missing:
                jobs.append((proj, missing))
        if not jobs:
            return
        pool = self._pool_for(jobs[0][0])
        results = list(pool.map(lambda job: job[0]._load_expert_bank_np(job[1]), jobs))
        for (proj, ids), rows in zip(jobs, results):
            if rows is None or len(rows) != len(ids):
                self.failed = True
                return
            self.bundles[id(proj)].update(zip(ids, rows))
        if tracing:
            # Union retention: every projection's bank is resident at once
            # here, so the layer holds sum(miss_i * per_expert_bytes_i) bytes
            # of NumPy until the last projection is promoted. This is exactly
            # the term the rolling path eliminates.
            live = sum(proj._bank_bytes_for(len(ids)) for proj, ids in jobs)
            memtrace.record(
                "ctx.ensure.exit",
                layer=layer,
                n_proj=len(self.linears),
                uniq=len(expert_ids),
                n_loaded=len(jobs),
                miss_per_proj=[len(ids) for _, ids in jobs],
                bank_bytes=live,
            )

    # -- public API ---------------------------------------------------------

    def ensure(self, linear: Any, expert_ids: list[int]) -> None:
        """Resolve ``linear``'s demand set for this layer call."""
        if _CTX_ROLLING_ENV:
            self._ensure_rolling(linear, expert_ids)
        else:
            self._ensure_union(linear, expert_ids)


@dataclass
class _RemapPlan:
    """Routing plan shared by every streaming linear of one MoE layer call.

    The first linear invoked in a layer builds the plan (mx.eval + host copy
    + np.unique + compact remap); the other projections (up/gate/down) reuse
    it — one sync per MoE layer instead of three.
    """

    indices_shape: tuple[int, ...] = ()
    flat_np: Any = None
    uniq_list: list = field(default_factory=list)
    uniq_mx: Any = None  # MLX unique expert IDs reused by bias gather
    remapped: Any = None  # mx.array of compact ids, original indices shape
    positions: int = 0
    gate_s: float = 0.0
    unique_s: float = 0.0
    ctx: _LayerLoadContext | None = None


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
    plan.uniq_mx = mx.array(uniq_np)
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

    def _load_expert_weight(self, expert_id: int, *, cache_result: bool = True) -> mx.array:
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
        if cache_result:
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
        # NOTE (Fase J G4): ``plan.ctx`` is deliberately NOT consulted here.
        # ``_RemapPlan.ctx`` is populated only when the GLU is quantized
        # (StreamingSwitchGLU.__call__ gates on ``self.quantized``), and
        # ``_LayerLoadContext`` drives quantized-only hooks
        # (``stacked_weight_key``, ``_bank_bytes_for``, ``_load_expert_bank_np``).
        # The bf16 path never sees a context, and would not benefit from one
        # the way the quantized path does: each projection resolves its own
        # experts inside its own __call__, so there is no cross-projection
        # union to collapse — the context's win there is memory, here it would
        # only be I/O overlap. Left as-is; see docs/expert-streaming.md.
        #
        # Load each unique expert weight. C4 avoids retaining a large prefill
        # demand set while the hotness seeder is active.
        cache_result = not (
            getattr(self.cache, "prefill_bypass", False) and plan.positions > 64
        )
        mini_weights = []
        t_load = 0.0
        hits = 0
        misses = 0
        for eid in plan.uniq_list:
            was_hit = (self.layer_idx, eid, self.stacked_key) in self.cache
            t_l = time.perf_counter()
            w = self._load_expert_weight(int(eid), cache_result=cache_result)
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
            b_mini = mx.take(self._bias, plan.uniq_mx, axis=0)  # (U,O)
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
            object.__setattr__(
                self,
                "_slice_dtypes",
                (
                    td(self.stacked_scales_key) if td else None,
                    td(self.stacked_biases_key) if td and self.stacked_biases_key else None,
                ),
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

    def _slice_bytes(self, key: str) -> int:
        """Per-expert byte size of *key* (truthful: read from the backing reader)."""
        try:
            reader = self.backing._reader_for_key(key)
            return int(reader._rp_for(key).expert_bytes)
        except Exception:
            return 0

    def _per_expert_bytes(self) -> int:
        """Summed per-expert bytes across this projection's stacked tensors."""
        keys = [self.stacked_weight_key, self.stacked_scales_key]
        if self.stacked_biases_key:
            keys.append(self.stacked_biases_key)
        return sum(self._slice_bytes(k) for k in keys)

    def _bank_bytes_for(self, n_experts: int) -> int:
        """Bytes of raw NumPy bank needed to hold ``n_experts`` of this projection.

        Used by the memory trace to quantify per-projection retention and by
        the demand-set tiler (Etapa A) to size tiles under the bank cap.
        """
        if n_experts <= 0:
            return 0
        return n_experts * self._per_expert_bytes()

    def _slice_view(self, key: str, buf: np.ndarray) -> np.ndarray:
        """Reshape a raw uint8 expert buffer exactly as ``expert_slice`` would.

        Mirrors ``_ShardReader.expert_slice`` so the promoted mx.array is
        bit-identical to the legacy per-slice path (C2 correctness)."""
        reader = self.backing._reader_for_key(key)
        rp = reader._rp_for(key)
        return np.frombuffer(buf, dtype=rp.np_dtype).reshape(rp.per_shape)

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

    def _read_expert_banks(self, expert_ids: list[int]):
        """Read a contiguous demand bank per projection.

        Returns ``(keys, banks, rows)``: *banks* are raw ``(U, per_bytes)``
        uint8 buffers and *rows* are the per-expert typed views that the LRU
        caches. ``None`` when the backing cannot serve the demand set as banks
        (dict backing, unsupported layout, oversized demand set).
        """
        if not hasattr(self.backing, "read_expert_into") or not expert_ids:
            return None
        keys = [self.stacked_weight_key, self.stacked_scales_key]
        if self.stacked_biases_key:
            keys.append(self.stacked_biases_key)
        try:
            per_bytes = [self._slice_bytes(key) for key in keys]
            if any(size <= 0 for size in per_bytes):
                return None
            total = len(expert_ids) * sum(per_bytes)
            if total > _BANK_MAX_BYTES:
                return None
            banks = [
                np.empty((len(expert_ids), size), dtype=np.uint8) for size in per_bytes
            ]
            components = [(key, expert_ids) for key in keys]
            if not self.backing.read_expert_into(components, banks):
                return None
            rows = []
            for row in range(len(expert_ids)):
                w = self._slice_view(self.stacked_weight_key, banks[0][row])
                s = self._slice_view(self.stacked_scales_key, banks[1][row])
                b = (
                    self._slice_view(self.stacked_biases_key, banks[2][row])
                    if self.stacked_biases_key
                    else None
                )
                rows.append((w, s, b))
            return keys, banks, rows
        except Exception:
            return None

    def _load_expert_bank_np(self, expert_ids: list[int]) -> list[tuple] | None:
        """Read a demand set into one raw NumPy bank per projection.

        The backing performs coalesced contiguous reads into caller-owned banks;
        rows are then exposed as views for the existing LRU representation.
        Returning ``None`` preserves the legacy per-expert fallback for dict
        backings and unsupported/cold layouts.
        """
        got = self._read_expert_banks(expert_ids)
        return None if got is None else got[2]

    def _load_expert_bank_np_full(self, expert_ids: list[int]):
        """Like :meth:`_load_expert_bank_np`, but keeps the raw contiguous banks.

        Needed by the Etapa B layer context: the NumPy read may happen on an IO
        pool worker, yet promoting those buffers to MLX must happen later on the
        inference thread (MLX ops may not be bound off-stream). Same failure
        contract as :meth:`_load_expert_bank_np` — ``None`` whenever the backing
        cannot serve the demand set as banks.
        """
        return self._read_expert_banks(expert_ids)

    def _promote_banks(self, keys: list[str], banks: list, n: int):
        """Promote already-read contiguous banks into one mx.array per key.

        Shared by Etapa A1 (read + promote together) and Etapa A1b (read on a
        pool thread, promote here on the inference thread).

        Bit-identical to promoting U per-expert arrays and stacking them: each
        bank is reinterpreted with exactly the dtype and per-expert shape that
        :meth:`_slice_view` applies to a single row, so ``gather_qmm`` receives
        the same bytes, dtype and layout. Only the allocation count differs —
        one mx.array per key instead of U of them plus the stack copy.
        """
        try:
            dt = self._slice_dtypes_lazy()
            promoted = []
            for i, key in enumerate(keys):
                rp = self.backing._reader_for_key(key)._rp_for(key)
                # Same reinterpretation _slice_view applies to one row, applied
                # to the whole contiguous bank at once.
                typed = np.frombuffer(banks[i], dtype=rp.np_dtype).reshape(
                    n, *rp.per_shape
                )
                arr = mx.array(typed)
                # scales/biases can be bf16 stored as raw uint16 bits.
                dtype_str = dt[0] if i == 1 else (dt[1] if i == 2 else None)
                if dtype_str == "BF16" and arr.dtype == mx.uint16:
                    arr = arr.view(mx.bfloat16)
                promoted.append(arr)
            while len(promoted) < 3:
                promoted.append(None)
            return (promoted[0], promoted[1], promoted[2])
        except Exception:
            return None

    def _load_expert_bank_mx(self, expert_ids: list[int]):
        """Etapa A1: promote an all-miss demand bank in one shot.

        Returns ``(w_bank, s_bank, b_bank, rows)`` or ``None`` when the demand
        set cannot be served as a single bank. *rows* are the per-expert raw
        views the caller still has to seed into the LRU, so the hit-rate path
        is unaffected.

        This is **bit-identical** to promoting U per-expert arrays and stacking
        them: the bank is reinterpreted with exactly the dtype and per-expert
        shape that ``_slice_view`` uses per row, so ``gather_qmm`` receives the
        same bytes in the same layout. Only the allocation count differs — one
        mx.array instead of U of them plus the stack copy.
        """
        got = self._read_expert_banks(expert_ids)
        if got is None:
            return None
        keys, banks, rows = got
        promoted = self._promote_banks(keys, banks, len(expert_ids))
        if promoted is None:
            return None
        return (*promoted, rows)

    @staticmethod
    def _group_runs(
        sorted_ids: list[int], max_run: int | None = None
    ) -> list[tuple[int, int]]:
        """Split ascending expert ids into bounded contiguous runs."""
        max_run = _RUN_MAX if max_run is None else max(1, int(max_run))
        runs: list[tuple[int, int]] = []
        i = 0
        n = len(sorted_ids)
        while i < n:
            first = sorted_ids[i]
            count = 1
            while (
                count < max_run
                and i + count < n
                and sorted_ids[i + count] == first + count
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
                    self.cache.discard(s_key)
                    if b_key:
                        self.cache.discard(b_key)
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
        if hasattr(self.backing, "read_expert_into"):
            # Coalesced zero-copy read of all three components into caller-owned
            # uint8 buffers (one preadv per component across every missing
            # expert — avoids the per-expert reader resolution + slice
            # allocation the legacy path below does). Falls back to per-slice
            # reads if the backing cannot serve any component.
            try:
                per_w = self._slice_bytes(self.stacked_weight_key)
                per_s = self._slice_bytes(self.stacked_scales_key)
                w_buf = np.empty((1, per_w), dtype=np.uint8)
                s_buf = np.empty((1, per_s), dtype=np.uint8)
                comps = [(self.stacked_weight_key, [expert_id]), (self.stacked_scales_key, [expert_id])]
                outs = [w_buf, s_buf]
                if self.stacked_biases_key:
                    per_b = self._slice_bytes(self.stacked_biases_key)
                    b_buf = np.empty((1, per_b), dtype=np.uint8)
                    comps.append((self.stacked_biases_key, [expert_id]))
                    outs.append(b_buf)
                if per_w and per_s and (not self.stacked_biases_key or per_b):
                    if self.backing.read_expert_into(comps, outs):
                        w = self._promote_np(self._slice_view(self.stacked_weight_key, w_buf[0]))
                        s = self._promote_np(self._slice_view(self.stacked_scales_key, s_buf[0]))
                        b = None
                        if self.stacked_biases_key:
                            b = self._promote_np(self._slice_view(self.stacked_biases_key, b_buf[0]))
                    else:
                        raise ValueError("read_expert_into returned False")
                else:
                    raise ValueError("slice bytes unavailable")
            except Exception:
                w = self.backing.load_expert_slice(self.stacked_weight_key, expert_id)
                s = self.backing.load_expert_slice(self.stacked_scales_key, expert_id)
                b = None
                if self.stacked_biases_key:
                    try:
                        b = self.backing.load_expert_slice(self.stacked_biases_key, expert_id)
                    except Exception:
                        b = None
        elif hasattr(self.backing, "load_expert_slice"):
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
        cache_result = not (
            getattr(self.cache, "prefill_bypass", False) and plan.positions > 64
        )
        t2 = time.perf_counter()
        # Load bundles: cache/staging resolution on this thread, then one
        # parallel os.pread fetch per missing bundle (QD1 -> QD8). Pool
        # workers return raw np slices only; promotion to mx.array happens
        # in the loop below on the inference thread.
        bundles: dict[int, tuple] = {}
        mini_w, mini_s, mini_b = [], [], []
        has_b = False
        hits = 0
        misses = 0
        missing: list[int] = []
        t_res_start = time.perf_counter()
        context_bundles = None
        if plan.ctx is not None:
            # Etapa B: resolve *this* projection; the context prefetches the
            # next one in the background so banks are not all resident at once.
            plan.ctx.ensure(self, plan.uniq_list)
            if not plan.ctx.failed:
                context_bundles = plan.ctx.bundles.get(id(self))
                hits = plan.ctx.hits.get(id(self), 0)
                misses = plan.ctx.misses.get(id(self), 0)
                if context_bundles is not None and len(context_bundles) == len(plan.uniq_list):
                    bundles.update(context_bundles)
                else:
                    context_bundles = None
        if context_bundles is None:
            for eid in plan.uniq_list:
                eid = int(eid)
                b = self._bundle_cached_or_staged(eid)
                if b is not None:
                    bundles[eid] = b
                    hits += 1
                else:
                    misses += 1
                    missing.append(eid)
        banked = None
        if missing:
            # ascending expert id = ascending file offset within the stacked
            # bank (row-major) — sorted reads keep the NVMe's locality
            missing.sort()
            if (
                _BANK_PROMOTE_ENV
                and len(missing) == len(plan.uniq_list)
                and hasattr(self.backing, "read_expert_into")
            ):
                # Etapa A1: every demanded expert is a miss, so the demand set
                # is one contiguous bank — promote it once instead of building
                # U per-expert mx arrays and stacking them.
                #
                # SCOPE (measured, bench/prefill_mem_harness.py): this branch
                # is only reachable when the Etapa B layer context did NOT
                # pre-resolve the demand set — i.e. ``_LAYER_BARRIER`` off,
                # the context failed, or it was never built. With the barrier
                # on (the default) the context fills ``bundles`` above, so
                # ``missing`` is empty and this never runs: A1 measured 0
                # calls vs 24 for the legacy path. It is a fallback-path win,
                # not a main-path one — do not read its benchmark gain as a
                # default-configuration gain.
                banked = self._load_expert_bank_mx(missing)
            if banked is not None:
                rows = banked[3]
                dt_per = time.perf_counter() - t_res_start
                for eid, raw in zip(missing, rows):
                    bundles[eid] = raw
                    if cache_result:
                        self.cache.put(
                            (self.layer_idx, eid, self.stacked_weight_key), raw
                        )  # type: ignore[arg-type]
                    if p is not None:
                        p.add_load_source(
                            self.layer_idx, staged=False, dt=dt_per / len(missing)
                        )
            elif hasattr(self.backing, "load_expert_slice"):
                io_pool = self._io_pool_override or _EXPERT_IO_POOL
                # C2 bank-first path: read all missing experts into one raw
                # bank per projection, then expose rows as views. This avoids
                # one task/result allocation per expert on dense demand sets.
                raws = self._load_expert_bank_np(missing)
                if raws is None:
                    coalesce_on = (
                        _COALESCE_ENV
                        if self._coalesce_override is None
                        else bool(self._coalesce_override)
                    )
                    raws = [None] * len(missing)
                    # Legacy fallback: coalesce consecutive ids into runs.
                    runs = self._group_runs(missing)
                    if coalesce_on and len(runs) < len(missing):
                        results_by_run = list(
                            io_pool.map(
                                lambda r: (r, self._load_expert_run_np(r[0], r[1])),
                                runs,
                            )
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
                    if cache_result:
                        self.cache.put(
                            (self.layer_idx, eid, self.stacked_weight_key), raw
                        )  # type: ignore[arg-type]
                    if p is not None:
                        p.add_load_source(self.layer_idx, staged=False, dt=dt_per / len(missing))
            else:
                # dict-backed test doubles: sequential fallback
                for eid in missing:
                    bundles[eid] = self._load_expert_bundle(eid)
        t_load = time.perf_counter() - t_res_start

        if memtrace.enabled:
            memtrace.record(
                "linear.resolve",
                layer=self.layer_idx,
                proj=self.proj_name,
                uniq=len(plan.uniq_list),
                hits=hits,
                misses=misses,
                bank_bytes=self._bank_bytes_for(len(missing)),
                from_ctx=context_bundles is not None,
            )

        dt = self._slice_dtypes_lazy()
        ctx_bank = None
        if banked is None and plan.ctx is not None:
            # Etapa A1b: the layer context read this projection's demand set as
            # one contiguous NumPy bank (possibly on an IO pool worker). Promote
            # it *here*, on the inference thread, so MLX ops stay on-stream and
            # the U per-expert mx arrays plus the stack copy are both skipped.
            # Guarded by bank_ids: only a bank that describes exactly this
            # demand set may be promoted, so a stale or partial bank cannot
            # silently mis-pair experts.
            raw = plan.ctx.bank_raw.get(id(self))
            if (
                raw is not None
                and _BANK_PROMOTE_CTX_ENV
                and plan.ctx.bank_ids.get(id(self)) == plan.uniq_list
            ):
                ctx_bank = self._promote_banks(raw[0], raw[1], len(plan.uniq_list))
        if banked is not None:
            # Etapa A1: the bank is already promoted, so the U per-expert mx
            # copies and the stack (which briefly doubled the Metal footprint)
            # are both skipped.
            w_bank, s_bank, b_bank = banked[0], banked[1], banked[2]
            has_b = b_bank is not None
        elif ctx_bank is not None:
            w_bank, s_bank, b_bank = ctx_bank
            has_b = b_bank is not None
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
            if len(mini_w) == 1:
                w_bank = mx.expand_dims(mini_w[0], 0)
                s_bank = mx.expand_dims(mini_s[0], 0)
                b_bank = mx.expand_dims(mini_b[0], 0) if has_b and mini_b else None
            else:
                w_bank = mx.stack(mini_w, axis=0)
                s_bank = mx.stack(mini_s, axis=0)
                b_bank = mx.stack(mini_b, axis=0) if has_b and mini_b else None
        if memtrace.enabled:
            # Sampled *before* the QMM runs: at this instant the U promoted
            # per-expert mx copies and the freshly stacked bank coexist, which
            # is the transient double-buffer that demand-set tiling removes.
            memtrace.record(
                "linear.stack",
                layer=self.layer_idx,
                proj=self.proj_name,
                uniq=len(plan.uniq_list),
                bank_bytes=self._bank_bytes_for(len(plan.uniq_list)),
            )
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
            b_mini = mx.take(self._bias, plan.uniq_mx, axis=0)
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
        if memtrace.enabled:
            memtrace.record(
                "glu.enter", layer=self.layer_idx, positions=int(indices.size)
            )
        x_exp = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x_exp, idx, inv_order = _gather_sort(x_exp, indices, inverse_scatter=self.inverse_scatter)

        # One shared routing plan for the whole layer: the first linear
        # invoked builds it (single mx.eval + unique + remap), the rest reuse.
        plan = _RemapPlan()
        if self.quantized and _LAYER_BARRIER_ENV:
            projections = (
                [self.gate_up_proj, self.down_proj]
                if has_fused
                else [self.up_proj, self.gate_proj, self.down_proj]
            )
            if all(hasattr(proj, "_load_expert_bank_np") for proj in projections):
                plan.ctx = _LayerLoadContext(projections, self._cache)

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
                from omlx.patches.glm_moe_dsa.kernels import fast as glm_fast  # type: ignore

                if hasattr(glm_fast, "glm_moe_weighted_sum"):
                    return glm_fast.glm_moe_weighted_sum(x_out, inv_order, scores)
            except Exception:
                pass

        if do_sort:
            x_out = _scatter_unsort(x_out, inv_order, indices.shape)
        out = x_out.squeeze(-2)
        if t_wall0 is not None and p is not None:
            p.add_wall(self.layer_idx, time.perf_counter() - t_wall0)
        if memtrace.enabled:
            memtrace.record(
                "glu.exit",
                layer=self.layer_idx,
                uniq=len(plan.uniq_list),
                positions=plan.positions,
            )
        return out
