# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic dynamic budget over the V4.1 native expert adapter.

Split of responsibilities (clean-split design):

- Policy (model-agnostic): ``ExpertResidencyGovernor`` from
  ``expert_streaming.governor`` decides capacity from pressure (free RAM)
  and hunger (windowed decode stall). It only knows the ``cache``
  duck-type: ``capacity`` / ``resize`` / ``clear`` / ``set_layer_caps`` /
  ``stats``. No V4.1 knowledge, no kernels.
- Mechanics (model-specific): this backing translates those calls onto
  the partitioned per-layer ``_ExpertSlots`` storage (fixed mx arrays per
  layer, V4.1 packed layouts untouched).

Unit adaptation: V4.1 storage is partitioned per layer while the governor
reasons about one pool. The backing therefore presents capacity in
per-layer units — ``per_slot = bytes_per_expert * num_layers`` with
``num_layers=1`` at the governor — so every budget<->slots conversion is
exact and a global shrink can never starve one layer below its decode
working set. Targeting overrides (``set_layer_caps``) are per-layer
ceilings enforced lazily; ``capacity`` reports the uniform base.
"""

from __future__ import annotations

import logging
import os
import threading

from ..expert_streaming.memtrace import memtrace
from ..expert_streaming.slot_cache import DecodeVisitStats

logger = logging.getLogger(__name__)

# Per-layer staging recall gate (parity with the generic path's
# stage_gate in expert_streaming.speculation): the prev-token predictor
# must have covered at least this share of observed demand recently
# (EWMA, decay 0.9) or stage_predicted is skipped — a layer whose
# routing diverged pays one bounded burst of staged reads and then
# shuts itself off.
try:
    # `or "0.3"` — not `or 0`: an empty env must keep the default like
    # the sibling knobs; "" -> 0.0 would silently hold the recall gate
    # open and stage unconditionally.
    _STAGED_MIN_RECALL = float(
        os.environ.get("OMLX_V41_STAGE_MIN_RECALL", "0.3") or "0.3"
    )
except (TypeError, ValueError):
    # Same contract as the moe_offload env knobs (_fetch_threads,
    # _span_gap, _span_max_rows): a degenerate value keeps the default
    # instead of killing the import.
    logger.warning(
        "Invalid OMLX_V41_STAGE_MIN_RECALL value; using 0.3",
        exc_info=True,
    )
    _STAGED_MIN_RECALL = 0.3
_STAGED_RECALL_DECAY = 0.9


class _V41CacheStats(DecodeVisitStats):
    """Cumulative decode-visit counters for governor windows.

    The shared contract class (``expert_streaming.slot_cache``) is the
    shape ``ExpertResidencyGovernor._window`` duck-reads — same fields the
    legacy state and the unified CacheStats expose.

    Cadence note: V4.1 notes a visit per ``ensure()`` — one per decode
    CHUNK, and a B-row batched decode splits into ceil(B/step) chunks —
    while the generic cache notes one visit per layer-call. Governor
    windows therefore fill ~chunks-per-layer-call faster here; the stall
    signal stays a ratio so ``stall_target`` applies unchanged.
    """

    pass


class V41StreamingBacking:
    """Governor-facing cache over V4.1's per-layer expert slots."""

    def __init__(
        self,
        plan,
        layers: list,
        *,
        dynamic: bool = True,
        max_budget_bytes: int | None = None,
        min_budget_bytes: int | None = None,
        stall_target: float | None = None,
        min_cap: int | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.plan = plan
        self.slots_of = {}
        for layer_idx, slots in layers:
            self.slots_of[int(layer_idx)] = slots
            slots.backing = self
            slots.layer = int(layer_idx)
        self.num_layers = len(self.slots_of)
        # P8 staging predictor: last decode-token routing per layer, the
        # MoE layer order used to find the "next" layer to stage, and the
        # per-layer prev-token recall EWMA that gates speculative reads.
        self.prev_uniq: dict = {}
        self.recall_ewma: dict = {}
        self._order = sorted(self.slots_of)
        self.staged_skips = 0
        # plan.full_bytes spans ALL layers while count is per-layer: derive
        # per-expert size from resident_bytes (capacity x layers) so the
        # governor's GiB budgets stay truthful. Actions are scale-invariant
        # (halve/double/+25%), but byte floors and log labels are not.
        per_expert = max(
            1,
            int(plan.resident_bytes)
            // max(1, int(plan.capacity) * max(1, len(self.slots_of))),
        )
        # Per-layer units (see module docstring).
        self.per_slot = per_expert * max(1, self.num_layers)
        self.base_cap = int(plan.capacity)
        self.overrides: dict = {}
        self.stats = _V41CacheStats()
        self.governor = None
        if not dynamic or self.num_layers == 0:
            return
        try:
            from ..expert_streaming.governor import (
                ExpertResidencyGovernor,
                _max_dynamic_budget_bytes,
            )

            initial_total = self.base_cap * self.per_slot
            gov_max = (
                int(max_budget_bytes)
                if max_budget_bytes is not None
                else _max_dynamic_budget_bytes()
            )
            gov_min = (
                int(min_budget_bytes)
                if min_budget_bytes is not None
                else max(int(0.25 * 1024**3), initial_total // 4)
            )
            # Never shrink a layer below one token's decode working set:
            # below n_activated_experts `ensure` raises "working set
            # exceeds resident capacity" mid-generation. The old
            # min(8, cap) floor broke exactly that on top-k>8 models
            # (2.1). The plan already guarantees capacity >= n_activated.
            floor = (
                int(min_cap)
                if min_cap is not None
                else max(
                    1,
                    min(int(getattr(plan, "n_activated", 8) or 8), self.base_cap),
                )
            )
            kwargs = {}
            if stall_target is not None:
                kwargs["stall_target"] = stall_target
            self.governor = ExpertResidencyGovernor(
                self,
                self.per_slot,
                1,
                max(gov_max, initial_total),
                min_budget_bytes=gov_min,
                min_cap=max(1, floor),
                **kwargs,
            )
            logger.info(
                "V4.1 dynamic residency: governor armed "
                "(per-layer %d slots, min %d, max %.2f GiB)",
                self.base_cap,
                self.governor._min_cap_slots(),
                self.governor.max_budget_bytes / 1024**3,
            )
        except Exception:
            logger.debug("V4.1 governor arming failed", exc_info=True)
            self.governor = None

    # -- governor duck-type ------------------------------------------------
    def resize(self, cap: int, per_layer: int | None) -> None:
        """Retarget the uniform per-layer base (atomic under the lock).

        With ``num_layers=1`` the governor always passes ``per_layer ==
        cap``; growth materializes lazily inside ``ensure`` (realloc on
        demand), so resize only ever sets ceilings and compacts overs.
        Targeting overrides reset to uniform; the governor re-applies
        them via ``set_layer_caps`` right after (same contract as the
        generic cache).
        """
        want = max(1, int(per_layer if per_layer is not None else cap))
        # Never apply a per-layer ceiling below one token's working set:
        # a book.cap < n_activated makes ensure() raise "working set
        # exceeds resident capacity" mid-generation. base_cap keeps the
        # governor's raw request so consecutive shrinks do not flap.
        floor = max(1, int(getattr(self.plan, "n_activated", 1) or 1))
        eff = max(floor, want)
        with self._lock:
            self.base_cap = want
            self.overrides = {}
            for layer_idx, slots in self.slots_of.items():
                # Lock order backing -> slots (same as clear); the
                # cap write and the compact must be one atomic step
                # against a concurrent ensure on this layer.
                with slots._slots_lock:
                    slots.book.cap = eff
                    self._compact(slots, eff)

    def clear(self) -> None:
        with self._lock:
            for slots in self.slots_of.values():
                slots.clear_residency()

    def set_layer_caps(self, caps: dict) -> None:
        floor = max(1, int(getattr(self.plan, "n_activated", 1) or 1))
        with self._lock:
            self.overrides = {}
            for k, v in (caps or {}).items():
                eff = max(floor, int(v))
                # An override that lands on the applied base is not
                # targeting — dropping it keeps layer_cap_overrides()
                # truthful (the governor's num_layers==1 retarget sends
                # exactly this no-op).
                if eff != max(floor, self.base_cap):
                    self.overrides[int(k)] = eff
            for layer_idx, slots in self.slots_of.items():
                eff = max(floor, self.overrides.get(layer_idx, self.base_cap))
                with slots._slots_lock:
                    slots.book.cap = eff
                    self._compact(slots, eff)

    def layer_cap_overrides(self) -> dict:
        return dict(self.overrides)

    @property
    def capacity(self) -> int:
        # Generic-cache contract (governor line 309 gates on it): total
        # budget units, which for the per-layer-unit fiction equal the
        # uniform per-layer base.
        return self.base_cap

    @property
    def evictions(self) -> int:
        # Per-layer counters live on the slots now (the fetch path no
        # longer holds the global lock); the backing just aggregates.
        return sum(s.evictions for s in self.slots_of.values())

    @property
    def streaming_guard_info(self):
        """None on purpose.

        The scheduler's prefill-bank transient exists for the generic
        streaming path's lazy per-layer mini-banks. V4.1 expert slots are
        persistent pre-allocated buffers, so that term does not apply;
        the chunked gather outputs are bounded by the existing chunk
        machinery. Backing presence still matters: _streaming_backing_of
        finds it and serializes requests, which the per-layer LRU needs.
        """
        return None

    def close(self) -> None:
        """Release the plan's shard readers and fetch pool.

        ``shutdown_expert_streaming`` reaches this on engine stop; the
        model's own ``close()`` reaches the same plan via
        ``_moe_offload_plan`` — ``plan.close()`` is idempotent, so both
        paths are safe (2.8: without this method the shutdown hook did
        nothing for V4.1 and the shard fds could outlive the engine).
        """
        try:
            self.plan.close()
        except Exception:
            logger.debug("V4.1 plan close failed", exc_info=True)

    # -- stats -------------------------------------------------------------
    def note_visit(self, layer_idx: int, missed: bool) -> None:
        # Called by _ExpertSlots.ensure AFTER its per-layer lock is
        # released — taking the backing lock here keeps the order
        # slots._slots_lock then backing._lock impossible to invert.
        with self._lock:
            self.stats.note_visit(layer_idx, missed)
        # P8 parity with _LayerLoadContext.close(): mid-request governor
        # tick — one decode visit per layer-call is the same cadence the
        # generic path uses; tick() self-throttles on _GOV_TICK_S so the
        # cost is a monotonic compare until the interval elapses. Runs
        # outside the backing lock: observe() may resize() which retakes
        # it (RLock is reentrant, but keeping the call unscoped matches
        # the lock-ordering comment above).
        gov = self.governor
        if gov is not None:
            try:
                gov.tick()
            except Exception:
                logger.debug("governor tick failed", exc_info=True)

    def note_routing(self, layer_idx: int, experts) -> None:
        """Record a decode token's routed set as this layer's predictor
        for the next token (same temporal-locality assumption the generic
        path's SpeculationState uses; measured recall ~0.8 there).

        Deliberate simplification vs the generic path: ``prev_uniq``
        keeps only last token's set — no (layer, expert) -> next-expert
        transition table. Each staged payload here is a full projection
        row, so a looser predictor would spend real reads; instead the
        observed prev-token recall EWMA below gates ``stage_next``.
        """
        li = int(layer_idx)
        now = {int(e) for e in experts}
        prev = self.prev_uniq.get(li)
        if prev and now:
            obs = len(set(prev) & now) / len(now)
            self.recall_ewma[li] = (
                _STAGED_RECALL_DECAY * self.recall_ewma.get(li, 0.0)
                + (1.0 - _STAGED_RECALL_DECAY) * obs
            )
        self.prev_uniq[li] = now
        # V2-0 routing trace: decode-only by contract (ensure() calls this
        # solely on the decode branch — verify/prefill never pollute it).
        # A tracer write failure must never propagate into ensure() —
        # tracing is observability, not the demand path.
        try:
            memtrace.record(
                "routing",
                _light=True,
                src="v41",
                layer=int(layer_idx),
                experts=sorted(int(e) for e in experts),
                positions=1,
            )
        except Exception:
            logger.debug("v41 routing memtrace failed", exc_info=True)

    def stage_next(self, layer_idx: int) -> None:
        """Stage the NEXT MoE layer's predicted set (P8 prefetch).

        Called after a decode ensure — while layer ``layer_idx``'s MoE
        compute runs, the next layer's likely experts read on the plan's
        staging worker. Wraps to the first MoE layer so the last layer
        stages the next token's entry point. Failure is silent: staging
        is a hint, never on the demand path.
        """
        if not self._order:
            return
        try:
            pos = self._order.index(int(layer_idx))
        except (TypeError, ValueError):
            # int(None)/int(non-numeric) escapes ValueError alone; an
            # unconvertible or unknown layer just means "don't stage".
            return
        nxt = self._order[(pos + 1) % len(self._order)]
        pred = self.prev_uniq.get(nxt)
        if not pred:
            return
        # Recall gate (generic stage_gate parity): a layer whose
        # prev-token prediction stopped covering demand does not earn
        # speculative reads — the EWMA shuts it off after a bounded burst.
        if self.recall_ewma.get(nxt, 0.0) < _STAGED_MIN_RECALL:
            self.staged_skips += 1
            return
        if not self._stage_headroom():
            self.staged_skips += 1
            return
        try:
            self.slots_of[nxt].stage_predicted(pred)
        except Exception:
            pass

    def _stage_headroom(self) -> bool:
        """Speculative reads only pay when residency has slack.

        Measured on the starved 48GB/422GB regime (cap pinned at the
        working-set floor, governor clearing): staged_drops (270) >
        staged_hits (200) — mispredicted reads competed with demand
        fetches on a saturated disk for slots that cannot hold them.
        Suppress staging when the governor has shrunk to the floor or
        last saw free memory inside the desperate-clear band; keep it
        whenever capacity has room for a prediction to persist.
        """
        gov = self.governor
        if gov is None:
            return True
        try:
            # Public governor accessors when present (contract API);
            # fall back to the private reads while they are pending.
            at_floor = getattr(gov, "at_floor", None)
            floor_hit = (
                at_floor()
                if callable(at_floor)
                else self.base_cap <= gov._min_cap_slots()
            )
            if floor_hit:
                return False
            desperate = getattr(gov, "in_desperate_band", None)
            if callable(desperate):
                if desperate():
                    return False
            else:
                free = float(getattr(gov, "_last_free_gib", 0.0) or 0.0)
                if 0.0 < free < gov.low_free_bytes / 1024**3:
                    return False
        except Exception:
            return True
        return True

    # -- storage -----------------------------------------------------------
    # Realloc/compact live on _ExpertSlots (layer-3 mechanics: packed
    # layouts, QuantizedProjection rebind); the backing only orchestrates
    # ceilings and counts evictions.
    def _compact(self, slots, keep: int) -> None:
        # compact() takes the layer's own lock (it also runs standalone in
        # the ensure path); eviction counts live per-layer now.
        slots.compact(keep)

    def summary(self) -> dict:
        # The per-layer counters mutate under each slots._slots_lock, not
        # self._lock — summing under the backing lock can tear by a few
        # counts while an ensure is in flight. Accepted: this is a
        # reporting path, not accounting, and single-int reads are atomic.
        # hits/misses count demanded EXPERTS (SlotBookkeeping bumps once
        # per covered/missed id), not the generic cache's per-projection
        # puts — hit_rate denominators differ by ~3x vs the generic path.
        # The governor snapshot runs OUTSIDE the backing lock on purpose:
        # governor.summary() takes gov._lock while observe()/tick() hold
        # gov._lock across cache.resize() -> backing._lock. Taking the
        # backing lock first here would invert the mandated
        # governor->cache order and AB-BA deadlock against a mid-request
        # tick; the governor numbers don't need the backing lock anyway.
        gov = self.governor.summary() if self.governor else {}
        with self._lock:
            hits = sum(s.hits for s in self.slots_of.values())
            misses = sum(s.misses for s in self.slots_of.values())
            resident = sum(len(s.slot_of) for s in self.slots_of.values())
        total = hits + misses
        staged_hits = sum(getattr(s, "staged_hits", 0) for s in self.slots_of.values())
        staged_drops = sum(getattr(s, "staged_drops", 0) for s in self.slots_of.values())
        staged_failures = sum(
            getattr(s, "staged_failures", 0) for s in self.slots_of.values()
        )
        span_reads = sum(getattr(s, "span_reads", 0) for s in self.slots_of.values())
        span_demand = sum(getattr(s, "span_demand_rows", 0) for s in self.slots_of.values())
        span_phys = sum(getattr(s, "span_phys_rows", 0) for s in self.slots_of.values())
        span_fallbacks = sum(
            getattr(s, "span_fallbacks", 0) for s in self.slots_of.values()
        )
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": (hits / total) if total else 0.0,
            "evictions": self.evictions,
            "resident": resident,
            "capacity_per_layer": self.base_cap,
            "layers": self.num_layers,
            "staged_hits": staged_hits,
            "staged_drops": staged_drops,
            "staged_failures": staged_failures,
            "staged_skips": self.staged_skips,
            "span_reads": span_reads,
            "span_demand_rows": span_demand,
            "span_phys_rows": span_phys,
            "span_fallbacks": span_fallbacks,
            "governor": gov,
        }
