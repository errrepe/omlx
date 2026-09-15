# SPDX-License-Identifier: Apache-2.0
"""Shared expert-slot bookkeeping for the per-expert cache tracks.

The two non-unified caches — DeepSeek V4.1's ``_ExpertSlots`` and the
legacy ``ExpertCache`` — used to carry their own copies of the same
expert-id → row machinery: an LRU-ordered ``slot_of`` dict, a free-row
list, the rooms-vs-cap split (physical rows vs governor ceiling), and
the acquire/commit/rollback protocol that keeps a fetch failure from
orphaning a row or silently dropping the evicted victim's residency.
This module is that machinery, once.

The unified ``ExpertLRUCache`` keeps its own internals (per-projection
slots, s3fifo/route-frequency policies, cross-layer budget): only the
visit-stats contract below is shared, which is the shape the governor
duck-reads.

Locking stays with the cache: ``SlotBookkeeping`` itself is not
thread-safe; callers serialize under their own per-layer lock.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable

import mlx.core as mx

# OMLX_EXPERT_STREAMING_HOTPIN: pin the K hottest experts per book against
# eviction. Routing frequency is tracked on the demand set each ensure_set
# call; pins are recomputed per call and counters halve periodically so the
# protected set follows the current routing distribution instead of
# freezing history. 0 (default) keeps plain LRU victims.
_HOTPIN_K = int(os.environ.get("OMLX_EXPERT_STREAMING_HOTPIN", "0") or 0)
_HOTPIN_DECAY_CALLS = 512


@dataclass
class DecodeVisitStats:
    """Governor-facing visit counters: decode layer-calls and misses.

    ``decode_layers``/``decode_layers_missed`` count layer-CALLS (a visit
    that missed ≥1 expert stalls once regardless of miss count — the
    layer waits for its slowest read). ``decode_misses_by_layer`` is the
    governor's per-layer targeting signal: the per-expert tracks bump it
    once per stalled visit; the unified cache bumps it per missed slot —
    either way it ranks which layers hurt, which is all the governor reads.
    """

    decode_layers: int = 0
    decode_layers_missed: int = 0
    decode_misses_by_layer: dict = field(default_factory=dict)

    def note_visit(self, layer_idx: int, missed: bool) -> None:
        self.decode_layers += 1
        if missed:
            self.decode_layers_missed += 1
            self.decode_misses_by_layer[layer_idx] = (
                self.decode_misses_by_layer.get(layer_idx, 0) + 1
            )

    def reset_visits(self) -> None:
        self.decode_layers = 0
        self.decode_layers_missed = 0
        self.decode_misses_by_layer.clear()


def working_set_step(cap: int, top_k: int) -> int:
    """Tokens per forward chunk so a chunk's routed rows fit in *cap*.

    Every token routes to at most ``top_k`` experts, so ``cap // top_k``
    tokens can never touch more than ``cap`` distinct experts — the same
    bound the legacy halving scan approximated with repeated device→host
    syncs, in O(1). ``top_k <= cap`` is a caller precondition (enforced
    upstream: a cache smaller than the routing width cannot serve).
    """
    top_k = max(1, int(top_k))
    return max(1, int(cap) // top_k)


class SlotBookkeeping:
    """Expert-id → resident-row map with LRU eviction and fetch rollback.

    ``slot_of`` is insertion-ordered: first entry = eviction victim, last =
    most recently used. ``rooms`` counts the physical rows allocated in the
    slot tensors; ``cap`` is the governor-driven working ceiling — after a
    shrink, rooms can exceed cap (rows stay allocated but must not host
    residents). Growth past rooms is physical (the caller reallocs), then
    ``grew_to`` opens the new rows here.

    Acquire/commit/rollback: ``acquire`` reserves a row BEFORE the fetch
    payload exists, ``commit`` publishes residency AFTER it is in place,
    and ``rollback`` undoes a partially-run batch — restoring victims whose
    rows were never overwritten and freeing the row whose bytes are
    suspect.
    """

    def __init__(self, rooms: int, cap: int | None = None) -> None:
        self.slot_of: dict[int, int] = {}
        self.rooms = int(rooms)
        self.cap = self.rooms if cap is None else int(cap)
        self.free: list[int] = list(range(self.rooms))
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.freq: dict[int, int] = {}
        self.pinned: frozenset[int] = frozenset()
        self._demand_calls = 0

    def note_demand(self, needed: Iterable[int], pin_k: int) -> None:
        """Frequency-track the demanded set and refresh the pinned hot set.

        Each demanded expert gains one count per call (hit or miss — the
        routing signal is the demand, not residency). Counters halve every
        ``_HOTPIN_DECAY_CALLS`` calls so pins track the live distribution.
        ``pin_k`` is clamped to ``cap - len(needed)`` so a fully demanded
        book always leaves evictable rows for the working set itself.
        """
        needed_set = needed if isinstance(needed, (set, frozenset)) else set(needed)
        for expert in needed_set:
            self.freq[expert] = self.freq.get(expert, 0) + 1
        self._demand_calls += 1
        if self._demand_calls >= _HOTPIN_DECAY_CALLS:
            for expert in list(self.freq):
                halved = self.freq[expert] >> 1
                if halved:
                    self.freq[expert] = halved
                else:
                    del self.freq[expert]
            self._demand_calls = 0
        eff = min(int(pin_k), max(0, self.cap - len(needed_set)))
        if eff <= 0 or not self.freq:
            self.pinned = frozenset()
            return
        self.pinned = frozenset(
            sorted(self.freq, key=self.freq.get, reverse=True)[:eff]
        )

    def __contains__(self, expert: int) -> bool:
        return expert in self.slot_of

    def __len__(self) -> int:
        return len(self.slot_of)

    def touch(self, expert: int) -> bool:
        """Move *expert* to the MRU end when resident; returns residency."""
        slot = self.slot_of.pop(expert, None)
        if slot is None:
            return False
        self.slot_of[expert] = slot
        return True

    def evict_oldest_outside(
        self, needed: Iterable[int], on_evict: Callable[[int], None] | None = None
    ) -> tuple[int, int] | None:
        """Pop the LRU-oldest expert not in *needed* → ``(expert, row)``."""
        needed_set = needed if isinstance(needed, (set, frozenset)) else set(needed)
        victim = next(
            (
                e
                for e in self.slot_of
                if e not in needed_set and e not in self.pinned
            ),
            None,
        )
        if victim is None:
            return None
        row = self.slot_of.pop(victim)
        self.evictions += 1
        if on_evict is not None:
            on_evict(victim)
        return victim, row

    def trim_to_cap(
        self,
        needed: Iterable[int] = (),
        on_evict: Callable[[int], None] | None = None,
    ) -> int:
        """Evict LRU-oldest entries outside *needed* until within ``cap``."""
        trimmed = 0
        while len(self.slot_of) > self.cap:
            evicted = self.evict_oldest_outside(needed, on_evict)
            if evicted is None:
                break
            self.free.append(evicted[1])
            trimmed += 1
        return trimmed

    def acquire(self, needed: Iterable[int]) -> tuple[int, int | None, bool]:
        """Reserve a row for a missing expert.

        Returns ``(slot, victim, needs_grow)``: a free row, an evicted
        victim's row, or — when every resident is needed — ``(rooms,
        None, True)`` telling the caller to grow physical storage first,
        then take the new row from ``free``.
        """
        if self.free:
            return self.free.pop(), None, False
        needed_set = needed if isinstance(needed, (set, frozenset)) else set(needed)
        victim = next(
            (
                e
                for e in self.slot_of
                if e not in needed_set and e not in self.pinned
            ),
            None,
        )
        if victim is not None:
            self.evictions += 1
            return self.slot_of.pop(victim), victim, False
        return self.rooms, None, True

    def grew_to(self, new_rooms: int) -> None:
        """Open rows ``[rooms, new_rooms)`` after a physical grow."""
        new_rooms = int(new_rooms)
        if new_rooms > self.rooms:
            self.free.extend(range(self.rooms, new_rooms))
            self.rooms = new_rooms

    def commit(self, expert: int, slot: int, *, to_oldest: bool = False) -> None:
        """Publish ``expert -> slot`` and count the miss.

        ``to_oldest`` inserts at the eviction-candidate end — the DSv4.1
        verify path uses it so draft-only experts leave first without
        disturbing the decode-hot order.
        """
        if to_oldest:
            rest = dict(self.slot_of)
            self.slot_of.clear()
            self.slot_of[expert] = slot
            self.slot_of.update(rest)
        else:
            self.slot_of[expert] = slot
        self.misses += 1

    def rollback(
        self,
        fetch_list: list[tuple[int, int, int | None]],
        committed_through: int,
        pass3_started: bool,
        order_snapshot: list[int] | None = None,
    ) -> None:
        """Undo slot assignments for entries whose install did not finish.

        ``fetch_list`` entries are ``(expert, slot, victim)`` in commit
        order. Unstarted entries still hold the victim's bytes — restore
        it. The entry interrupted mid-commit has suspect bytes: its row
        goes back to ``free`` and the victim stays evicted.

        With ``order_snapshot`` — ``slot_of``'s key order captured just
        before the call's first victim pop — restored victims reinsert
        at their ORIGINAL recency positions: the map is rebuilt as the
        snapshot order minus the victims that stay evicted, with the
        committed entries appended in commit order (matching the end
        positions ``commit`` gave non-frozen entries; a frozen commit's
        to_oldest head placement is approximated — residency and rows
        stay exact either way, only that edge's recency differs).
        Without a snapshot the restore falls back to the MRU end —
        residency and rows stay consistent, only recency is approximate.
        """
        if order_snapshot is None:
            for j in range(committed_through + 1, len(fetch_list)):
                _expert, slot, victim = fetch_list[j]
                if pass3_started and j == committed_through + 1:
                    self.free.append(slot)
                elif victim is not None:
                    self.slot_of[victim] = slot
                    self.evictions -= 1
                else:
                    self.free.append(slot)
            return
        # Victims whose rows host committed experts stay evicted, as
        # does the mid-commit interruption's victim (suspect bytes).
        stay_evicted = {
            victim
            for _e, _s, victim in fetch_list[: committed_through + 1]
            if victim is not None
        }
        restored: dict[int, int] = {}
        for j in range(committed_through + 1, len(fetch_list)):
            _expert, slot, victim = fetch_list[j]
            if pass3_started and j == committed_through + 1:
                self.free.append(slot)
                if victim is not None:
                    stay_evicted.add(victim)
            elif victim is not None:
                restored[victim] = slot
                self.evictions -= 1
            else:
                self.free.append(slot)
        # Rebuild the order: snapshot keys keep their positions — the
        # still-resident ones at their current rows, restored victims at
        # the rows they were popped from. Entries committed before the
        # failure postdate the snapshot and append in commit order.
        rebuilt: dict[int, int] = {}
        for e in order_snapshot:
            if e in stay_evicted:
                continue
            if e in self.slot_of:
                rebuilt[e] = self.slot_of[e]
            elif e in restored:
                rebuilt[e] = restored[e]
        for e, s in self.slot_of.items():
            if e not in rebuilt:
                rebuilt[e] = s
        self.slot_of = rebuilt

    def release(self, expert: int) -> None:
        """Drop *expert*'s residency, returning its row to ``free``."""
        slot = self.slot_of.pop(expert, None)
        if slot is not None:
            self.free.append(slot)

    def rebuild(self, kept: list[int]) -> None:
        """Remap rows to 0..k-1 in *kept* order after a physical compact."""
        self.slot_of = {expert: row for row, expert in enumerate(kept)}
        self.rooms = len(kept)
        self.free = []

    def reset(self) -> list[int]:
        """Drop all residency; returns the evicted expert ids."""
        evicted = list(self.slot_of)
        self.slot_of.clear()
        self.free = list(range(self.rooms))
        return evicted


class SlotArena:
    """Fixed-row residency arena shared by the per-expert slot caches.

    The host module owns the stacked ``(rooms, *row_shape)`` arrays per
    projection field, bound into whatever module layout it uses —
    QuantizedProjection fields for DeepSeek V4.1, a plain bank for the
    unified streaming linears (V4-2b). The arena owns everything else
    about *where* residents live: the ``SlotBookkeeping`` map, the
    per-arena lock, the two-phase grow/compact of physical rows, the
    demand-set acquire/commit/rollback protocol, and the speculative
    prefetch side dict.

    Payload production — which bytes to read and how to decode them —
    stays with the caller through the ``produce`` callback to
    ``ensure_set``; the arena only knows rows. Because residents are
    bound into fixed rows once at admission, the consumer gathers by
    slot id (``rhs_indices``) and a cache hit costs zero assembly —
    the ~1 ms/projection ``mx.stack`` per call the bundle-dict
    representation pays is avoided entirely (bench_slot_arena.py).

    Host bindings supplied at construction:
      ``arrays(proj) -> {field: mx.array}``  currently bound rows
      ``bind(proj, {field: array})``       rebind grown/compacted storage
      ``eval_params()``                    mx.eval of host params post-rebind
    """

    def __init__(
        self,
        capacity: int,
        projections,
        arrays: Callable,
        bind: Callable,
        eval_params: Callable | None = None,
    ) -> None:
        self.book = SlotBookkeeping(capacity)
        self.projections = tuple(projections)
        self._arrays_of = arrays
        self._bind = bind
        self._eval_params = eval_params
        # Physical ceiling fixed at construction: the builder sized
        # ``capacity`` from a byte bound, so later cap retargets may only
        # move INSIDE the bound — never past it (that would commit
        # unbounded bank memory, the class of bug the ceiling exists to
        # prevent).
        self.rooms_max = self.book.rooms
        # Per-arena lock: residency mutation (and its fetches) serialize
        # per layer only — a global backing lock would hold ALL layers'
        # IO against the governor's resize.
        self.lock = threading.RLock()
        # Speculative prefetch side dict: expert -> future of the payload
        # a demand fetch returns. Entries are consumed by the producer's
        # join or dropped as mispredicts at the end of an ensure.
        self.staged = {}
        self.staged_hits = 0
        self.staged_drops = 0

    def write_row(self, slot: int, payload: dict) -> None:
        """Write one payload ``{proj: {field: array}}`` into row ``slot``."""
        for proj, fields in payload.items():
            arrays = self._arrays_of(proj)
            for field, array in fields.items():
                arrays[field][slot] = array

    def set_cap(self, cap: int) -> int:
        """Retarget the residency ceiling; returns the applied cap.

        Clamped to the construction-time physical bound ``rooms_max`` —
        the byte ceiling the builder sized is the hard limit, so a
        governor grow can only re-open rows the bound already paid for.
        Residents above the new cap are not evicted here; the next
        ``ensure_set`` trims them via ``trim_to_cap`` (eviction order
        stays demand-scoped).
        """
        with self.lock:
            self.book.cap = max(1, min(int(cap), self.rooms_max))
            return self.book.cap

    def grow(self, need: int) -> None:
        """Extend physical rooms to ``need`` (copy rows, extend free).

        Two-phase: every projection's grown arrays are built and evaluated
        BEFORE any rebind, so an OOM mid-way leaves the host on the old
        consistent storage instead of mixed-row projections.
        """
        if self.book.rooms >= need:
            return
        with self.lock:
            grown_all = {}
            for proj in self.projections:
                current = self._arrays_of(proj)
                grown = {
                    field: mx.zeros((need, *array.shape[1:]), dtype=array.dtype)
                    for field, array in current.items()
                }
                for field, array in current.items():
                    grown[field][: self.book.rooms] = array
                grown_all[proj] = grown
            mx.eval(*[a for g in grown_all.values() for a in g.values()])
            for proj in self.projections:
                self._bind(proj, grown_all[proj])
            if self._eval_params is not None:
                self._eval_params()
            self.book.grew_to(need)
            # Demand-driven growth re-bases the physical ceiling: rows
            # already committed are a paid cost, so a later set_cap must
            # clamp against what exists, not the construction-time size.
            self.rooms_max = max(self.rooms_max, self.book.rooms)

    def compact(self, keep: int) -> int:
        """Keep the most-recent ``keep`` entries, remap rows 0..k-1.

        Two-phase like ``grow``: stacks for ALL projections evaluate
        before the first rebind, so a mid-compact failure cannot serve a
        half-remapped layer.
        """
        with self.lock:
            order = list(self.book.slot_of)
            # No-op unless keep shrinks the live set — raising keep is a
            # ceiling change (lazy growth covers expansion); empty rooms
            # are the physical floor, not slack to release.
            if len(order) <= keep:
                return 0
            drop = len(order) - keep
            kept = order[drop:]
            stacked_all = {
                proj: {
                    field: (
                        mx.stack(
                            [array[self.book.slot_of[e]] for e in kept]
                        )
                        if kept
                        else mx.zeros(
                            (0, *array.shape[1:]), dtype=array.dtype
                        )
                    )
                    for field, array in self._arrays_of(proj).items()
                }
                for proj in self.projections
            }
            mx.eval(*[a for s in stacked_all.values() for a in s.values()])
            for proj in self.projections:
                self._bind(proj, stacked_all[proj])
            if self._eval_params is not None:
                self._eval_params()
            self.book.rebuild(kept)
            return drop

    def ensure_set(self, needed, frozen: bool, produce: Callable) -> bool:
        """Cover ``needed`` expert ids; returns True if any miss committed.

        Pass 1 reserves a row per missing expert (``acquire`` — free row,
        evicted victim's row, or physical grow). ``produce(fetch_list)``
        then runs the caller's IO+decode and returns payloads aligned
        with it. Pass 3 writes each payload into its row and publishes
        residency in fetch order — ``to_oldest`` under ``frozen`` so
        verify-path experts become first eviction candidates without
        disturbing the decode-hot order. A failure anywhere rolls back
        the uncommitted tail: untouched rows restore their victim, the
        interrupted row is freed.

        Physical growth runs at most once per ensure: a pre-pass predicts
        the row count the call can need (residents + misses − free rows −
        evictable victims) and grows straight to it, so an N-expert miss
        pays one realloc+rebind round instead of N.

        Leftover staged entries (predicted but not demanded) drop here —
        their futures are cancelled so an in-flight speculative read
        stops early instead of finishing into a dropped payload.
        """
        needed = set(needed)
        if len(needed) > self.book.cap:
            raise ValueError("Expert working set exceeds resident capacity")
        fetch_list = []
        committed_through = -1
        started = False
        order_snapshot = None
        try:
            if _HOTPIN_K:
                self.book.note_demand(needed, _HOTPIN_K)
            # Post-shrink trim: evict oldest non-needed down to the
            # ceiling. Safe: needed <= cap < len implies a non-needed
            # entry exists.
            self.book.trim_to_cap(needed)
            # Protect the entire working set, including hits after
            # misses.
            resident = 0
            for expert in needed:
                if expert in self.book:
                    self.book.hits += 1
                    resident += 1
                    if not frozen:
                        self.book.touch(expert)
            misses = len(needed) - resident
            # Grow-once pre-pass: the largest row count this call can
            # need is what the miss set requires after free rows and
            # evictable victims are spent. Pinned-but-not-needed rows are
            # NOT evictable, so demand may legitimately push rooms past
            # cap — occupancy over the ceiling is forced by the pins,
            # not by this bound.
            victims_available = sum(
                1
                for e in self.book.slot_of
                if e not in needed and e not in self.book.pinned
            )
            grow_by = max(
                0, misses - len(self.book.free) - victims_available
            )
            if grow_by:
                self.grow(self.book.rooms + grow_by)
            for expert in needed:
                if expert in self.book:
                    continue
                if order_snapshot is None and not self.book.free:
                    # acquire() evicts only once free rows are spent —
                    # capture the recency order just before this call's
                    # first victim pop so a rollback can reinsert victims
                    # at their ORIGINAL positions instead of the MRU end.
                    order_snapshot = list(self.book.slot_of)
                slot, victim, needs_grow = self.book.acquire(needed)
                if needs_grow:
                    # Safety net only: the pre-pass above opens every row
                    # the demand can need, so this cannot fire without
                    # bookkeeping drift. Keep the historical one-row grow
                    # rather than corrupting the book.
                    self.grow(self.book.rooms + 1)
                    slot = self.book.free.pop()
                fetch_list.append((expert, slot, victim))
            payloads = produce(fetch_list)
            if len(payloads) != len(fetch_list):
                raise RuntimeError(
                    "arena produce returned %d payloads for %d fetches"
                    % (len(payloads), len(fetch_list))
                )
            started = True
            for i, ((expert, slot, _victim), payload) in enumerate(
                zip(fetch_list, payloads)
            ):
                self.write_row(slot, payload)
                self.book.commit(expert, slot, to_oldest=frozen)
                committed_through = i
        except Exception:
            self.book.rollback(
                fetch_list, committed_through, started, order_snapshot
            )
            raise
        if self.staged:
            staged = list(self.staged.values())
            self.staged_drops += len(staged)
            self.staged.clear()
            for fut in staged:
                try:
                    fut.cancel()
                except Exception:
                    pass
        return committed_through >= 0

    def rows_for(self, expert_ids):
        """Slot row indices for ``expert_ids`` (all must be resident)."""
        return mx.array(
            [self.book.slot_of[int(e)] for e in expert_ids],
            dtype=mx.int32,
        )
