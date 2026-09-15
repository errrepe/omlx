# SPDX-License-Identifier: Apache-2.0
"""O2 cross-layer speculation state (per conversion).

Owns the routing history, converted-linears registry, advise stats and
the transition table that feed the next-layer F_RDADVISE advisor. Lives
in its own module so ``streaming_switch`` keeps the demand path; nothing
here imports the switch — this is a leaf like ``slot_cache``.

Fase K correction K1/K7: the O2 advisor's speculation state is PER
CONVERSION (one instance per backing/cache pair), never module-global.
Two engines (different checkpoints, same tensor keys) can never share
routing history or linears registries: the state dies with
the owning store, and close() drains the speculation workers the same
way ExpertBackingStore.close drains its readers.
FU1: transition-table overfetch. The (layer, expert) -> next-token expert
distribution (EWMA, temporal: same layer, token t-1 -> t) feeds the RA
advisor with one extra candidate per demanded expert (k+1 overfetch).
Hints only (F_RDADVISE), never changes output. 0 disables.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

from .memtrace import memtrace

_TRANSITION_ENV = os.environ.get("OMLX_EXPERT_STREAMING_TRANSITION", "1") != "0"
_TRANSITION_TOP = 8  # entries kept per (layer, expert) source
_TRANSITION_OVERFETCH = 1  # extra candidates per demanded expert

# Fase 1 (megaplan): staged decode prefetch — real reads into NumPy rows
# ahead of demand, replacing the dead stash and the hint-only F_RDADVISE.
# The predictor is prev_uniq_by_layer (previous token's routing for the
# next layer) plus transition-table candidates; a per-layer EWMA recall
# gate stops prefetching layers whose routing diverges, so a bad predictor
# costs a bounded burst and then shuts itself off.
_STAGED_ENV = os.environ.get("OMLX_EXPERT_STREAMING_STAGED", "1") != "0"
# Bound on staged rows + in-flight keys (rows are ~0.9 MB each here).
_STAGED_MAX = max(16, int(os.environ.get("OMLX_EXPERT_STREAMING_STAGED_MAX", "512")))
# Predicted set cap per (layer call, projection).
_STAGED_MAX_IDS = max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_STAGED_IDS", "48")))
# Per-layer recall gate: stage only when the prev-token prediction has
# covered at least this share of observed demand recently (EWMA).
_STAGED_MIN_RECALL = float(
    os.environ.get("OMLX_EXPERT_STREAMING_STAGED_MIN_RECALL", "0.25") or 0
)
_STAGED_RECALL_DECAY = 0.9

# P7 (megaplan): advise hygiene. The decode advisory re-issues F_RDADVISE
# for nearly the same prev-set every token — measured ~1.6x demand bytes
# in hints on a 256-token decode (325 GB advised / ~200 GB demand).
# _ADVISE_TTL dedupes per (layer, expert) across a window of layer-calls
# (~2 tokens at 48 layers). _ADVISE_TRANS_MIN_PRECISION gates the
# transition-table extras by their measured precision (the share of last
# token's extras that appeared in this token's demand). 0 disables.
_ADVISE_TTL = max(
    0, int(os.environ.get("OMLX_EXPERT_STREAMING_ADVISE_TTL", "96"))
)
_ADVISE_TRANS_MIN_PRECISION = float(
    os.environ.get("OMLX_EXPERT_STREAMING_TRANS_MIN_PRECISION", "0.05") or 0
)


class SpeculationState:
    """Per-conversion O2 speculation state (Fase K K1/K7).

    Owns the routing history used by the next-layer advisor, the
    converted-linears registry, the advise stats, and the transition
    table. Hangs off ``cache.spec_state``
    (and off ``backing.spec_state`` when the backing is an object) so the
    demand path, the advisor, and backing.close() all share one instance.
    All mutations are lock-guarded. A closed state stops advising so a
    drained engine never serves another conversion's speculation.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.closed = False
        self.prev_uniq_by_layer: Dict[int, list[int]] = {}
        self.linears_by_layer: Dict[int, list[Any]] = {}
        self.stats = {
            "advised": 0,
            "advised_runs": 0,
            "advised_experts": 0,
            "advised_bytes": 0,
            "advice_failures": 0,
            "advice_tier_segments": 0,
        }
        # FU1: transition table (layer, expert) -> {next_expert: weight}.
        # Temporal only (same layer, token t-1 -> t); cross-layer same-token
        # transitions are not observable without a model-loop hook (PILOT
        # covers glm5_next). Bounded: TOP entries per source, pruned on write.
        self.trans: Dict[Tuple[int, int], Dict[int, float]] = {}
        self.trans_updates = 0
        # Fase 1 staged prefetch state. `staged` holds completed NumPy rows
        # keyed by the TARGET linear's bundle_key; `staged_futs` maps the
        # same keys to (future, ids, linear) while the set read is in
        # flight — demand joins the in-flight read instead of re-issuing it.
        # recall_ewma[layer] is the prev-token prediction's observed recall.
        self.staged: Dict[Any, Any] = {}
        self.staged_futs: Dict[Any, Tuple[Any, list, Any]] = {}
        self.recall_ewma: Dict[int, float] = {}
        # P7: advise TTL dedup + transition-overfetch precision. The clock
        # advances one step per record_prev call (one MoE layer-call), so
        # TTL units are layer-calls (≈ num_layers per decode token).
        self._advise_seen: Dict[Tuple[int, int], int] = {}
        self._advise_clock = 0
        self._trans_pending: Dict[int, set] = {}
        self._trans_hits = 0
        self._trans_total = 0
        # Probe: same-token layer sets + cross-layer recall measurements.
        # Decode walks layers in order; a non-increasing layer index marks
        # a token boundary, so the just-finished token's sets roll into
        # `_prev_token_sets`.
        self._last_record_layer: Optional[int] = None
        self._cur_token_sets: Dict[int, list[int]] = {}
        self._prev_token_sets: Dict[int, list[int]] = {}
        self._xlayer_ewma: Dict[str, float] = {}

    # -- registry / history ----------------------------------------------

    def register_linears(self, layer_idx: int, linears: list[Any]) -> None:
        """Record one MoE layer's converted quantized linears (convert-time)."""
        kept = [lin for lin in linears if lin is not None]
        if not kept:
            return
        with self.lock:
            if self.closed:
                return
            self.linears_by_layer[int(layer_idx)] = kept

    def record_prev(
        self, layer_idx: int, ids: list[int], *, positions: int | None = None
    ) -> None:
        """Remember this layer's routing for the next token's speculation."""
        now = [int(e) for e in ids]
        li = int(layer_idx)
        # V2-0 routing trace: one event per MoE layer-call (emitted from
        # StreamingSwitchGLU.__call__ — the projections share the plan's
        # uniq_list, so a per-projection record triple-counted every
        # layer). _light skips the memory sampler: this fires on the
        # inference hot path.
        memtrace.record(
            "routing",
            _light=True,
            src="generic",
            layer=li,
            experts=now,
            positions=positions,
        )
        with self.lock:
            if self.closed:
                return
            self._advise_clock += 1
            last = self._last_record_layer
            if last is not None and li <= last:
                self._prev_token_sets = self._cur_token_sets
                self._cur_token_sets = {}
            self._last_record_layer = li
            self._cur_token_sets[li] = now
            now_set = set(now)
            if li > 0 and now_set:
                for key, src_map in (
                    ("xlayer_same", self._cur_token_sets),
                    ("xlayer_diag", self._prev_token_sets),
                ):
                    src = src_map.get(li - 1)
                    if src:
                        obs = len(set(src) & now_set) / len(now_set)
                        old = self._xlayer_ewma.get(key, 0.0)
                        new = (
                            _STAGED_RECALL_DECAY * old
                            + (1.0 - _STAGED_RECALL_DECAY) * obs
                        )
                        self._xlayer_ewma[key] = new
                        self.stats[key + "_recall"] = new
            # P7: transition-overfetch precision — score last call's extras
            # for this layer against what this call actually demanded.
            pend = self._trans_pending.pop(int(layer_idx), None)
            if pend:
                self._trans_total += len(pend)
                self._trans_hits += len(set(now) & pend)
            prev = self.prev_uniq_by_layer.get(int(layer_idx))
            if prev and now:
                # Fase 1: recall of the prev-token prediction for THIS call —
                # the gate that decides whether staging this layer pays.
                inter = len(set(prev) & set(now))
                # PAR-7 (measure-only): speculative-MoE feasibility. A
                # predicted-set execution recomputes every expert the
                # prediction missed (now \ prev) and wastes the extras it
                # loaded (prev \ now); full cover = zero-recompute calls.
                _ps, _ns = set(prev), set(now)
                self.stats["prev_cover_calls"] = (
                    self.stats.get("prev_cover_calls", 0) + 1
                )
                if _ps >= _ns:
                    self.stats["prev_cover_full"] = (
                        self.stats.get("prev_cover_full", 0) + 1
                    )
                self.stats["prev_cover_miss_sum"] = self.stats.get(
                    "prev_cover_miss_sum", 0
                ) + len(_ns - _ps)
                self.stats["prev_cover_extra_sum"] = self.stats.get(
                    "prev_cover_extra_sum", 0
                ) + len(_ps - _ns)
                obs = inter / max(1, len(now))
                old = self.recall_ewma.get(int(layer_idx), 0.0)
                self.recall_ewma[int(layer_idx)] = (
                    _STAGED_RECALL_DECAY * old
                    + (1.0 - _STAGED_RECALL_DECAY) * obs
                )
                told = self._xlayer_ewma.get("temporal", 0.0)
                tnew = (
                    _STAGED_RECALL_DECAY * told
                    + (1.0 - _STAGED_RECALL_DECAY) * obs
                )
                self._xlayer_ewma["temporal"] = tnew
                self.stats["temporal_recall"] = tnew
            if _TRANSITION_ENV and prev and now:
                # Credit temporal transitions prev -> now (EWMA without a
                # global decay pass: w = 1 + 0.9*w, normalized on read).
                for ep in prev:
                    row = self.trans.setdefault((int(layer_idx), int(ep)), {})
                    for en in now:
                        row[int(en)] = 1.0 + 0.9 * float(row.get(int(en), 0.0))
                    if len(row) > _TRANSITION_TOP:
                        for k in sorted(row, key=row.get)[: len(row) - _TRANSITION_TOP]:  # type: ignore[arg-type]
                            del row[k]
                self.trans_updates += 1
            self.prev_uniq_by_layer[int(layer_idx)] = now

    def to_payload(self) -> dict:
        """Serialize the transition table (fingerprint filled by caller)."""
        try:
            with self.lock:
                layers: dict[str, dict[str, dict[str, float]]] = {}
                for (layer, expert), row in self.trans.items():
                    layers.setdefault(str(int(layer)), {})[str(int(expert))] = {
                        str(int(k)): float(v) for k, v in row.items()
                    }
                return {"version": 1, "updates": int(self.trans_updates), "trans": layers}
        except Exception:
            return {"version": 1, "updates": 0, "trans": {}}

    def load_payload(self, payload: dict) -> int:
        """Load a transition payload; returns sources restored."""
        try:
            raw = (payload or {}).get("trans") or {}
            n = 0
            with self.lock:
                if self.closed:
                    return 0
                for layer_s, experts in raw.items():
                    for expert_s, row in (experts or {}).items():
                        clean = {
                            int(k): float(v)
                            for k, v in (row or {}).items()
                        }
                        if clean:
                            self.trans[(int(layer_s), int(expert_s))] = dict(
                                sorted(clean.items(), key=lambda kv: kv[1], reverse=True)[:_TRANSITION_TOP]
                            )
                            n += 1
                self.trans_updates += int((payload or {}).get("updates", 0) or 0)
            return n
        except Exception:
            return 0

    def predict_next(self, layer_idx: int, ids: list[int], k: int = _TRANSITION_OVERFETCH) -> list[int]:
        """FU1: top-k next-token candidates for this layer's demand set.

        Scores sum the transition rows of the demanded experts; the demanded
        ids themselves are excluded (the demand path already loads them).
        Returns at most k ids, possibly fewer (cold table). Lock-guarded.
        """
        if not _TRANSITION_ENV or k <= 0 or not ids:
            return []
        try:
            scores: Dict[int, float] = {}
            now = {int(e) for e in ids}
            with self.lock:
                if self.closed:
                    return []
                for e in now:
                    row = self.trans.get((int(layer_idx), int(e)))
                    if not row:
                        continue
                    for cand, w in row.items():
                        if cand not in now:
                            scores[int(cand)] = scores.get(int(cand), 0.0) + float(w)
            ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
            return [int(c) for c in ranked[: max(1, int(k))]]
        except Exception:
            return []

    def bump(self, key: str, amount: int = 1) -> None:
        with self.lock:
            if self.closed:
                return
            self.stats[key] = self.stats.get(key, 0) + amount

    # -- staged decode prefetch (Fase 1) ------------------------------------

    def stage_enabled(self) -> bool:
        return _STAGED_ENV

    def stage_recall(self, layer_idx: int) -> float:
        with self.lock:
            return float(self.recall_ewma.get(int(layer_idx), 0.0))

    def stage_gate(self, layer_idx: int) -> bool:
        """True when this layer's prediction earns its reads — the
        prev-token EWMA gates staging."""
        if not (_STAGED_ENV and not self.is_closed()):
            return False
        li = int(layer_idx)
        with self.lock:
            return self.recall_ewma.get(li, 0.0) >= _STAGED_MIN_RECALL

    def stage_room(self) -> bool:
        with self.lock:
            return (
                not self.closed
                and len(self.staged) + len(self.staged_futs) < _STAGED_MAX
            )

    def stage_pending(self, key: Any) -> bool:
        """True while a staged read covers *key* (done or in flight).

        Non-consuming probe for speculative prefetch membership: unlike
        ``stage_resolve`` it never pops a row or parks on a future, so a
        prefetch can skip experts whose read is already committed without
        racing the demand join.
        """
        if not _STAGED_ENV:
            return False
        with self.lock:
            return key in self.staged or key in self.staged_futs

    def stage_register(self, fut: Any, ids: list, linear: Any) -> int:
        """Map a submitted set-read future under each expert's bundle_key.

        Returns the number of NEW keys registered (ids already staged or in
        flight are skipped). On demand, ``stage_resolve`` awaits the future
        once and fans its rows out into ``staged`` for the siblings.
        """
        registered = 0
        with self.lock:
            if self.closed:
                return 0
            for eid in ids:
                key = linear.bundle_key(int(eid))
                if key in self.staged or key in self.staged_futs:
                    continue
                if len(self.staged) + len(self.staged_futs) >= _STAGED_MAX:
                    break
                self.staged_futs[key] = (fut, ids, linear)
                registered += 1
        return registered

    def stage_resolve(self, key: Any, wait: bool = True) -> Any:
        """Return a staged NumPy row for *key*, or None.

        Consumed rows are popped (the caller admits them to the LRU — the
        demand touch is the admission signal). An in-flight future is
        awaited once; its rows fan out to ``staged`` so sibling experts of
        the same set resolve without touching the pool again.
        ``wait=False`` only serves completed rows — used by the rolling
        prefetch's speculative split, which must never block the
        inference thread on a just-submitted read.
        """
        if not _STAGED_ENV:
            return None
        with self.lock:
            if self.closed:
                return None
            row = self.staged.pop(key, None)
            if row is not None:
                self.stats["staged_hits"] = self.stats.get("staged_hits", 0) + 1
                return row
            entry = self.staged_futs.get(key)
            if entry is not None and not wait and not entry[0].done():
                return None
        if entry is None:
            return None
        fut, ids, linear = entry
        # PAR-0: measure how long the demand path parks behind an in-flight
        # staged read — the signal that decides whether deeper staging
        # (PAR-4) or two-phase splits (PAR-2) are worth building.
        _waited = not fut.done()
        _wt0 = time.perf_counter() if _waited else 0.0
        try:
            got = fut.result()
        except Exception:
            got = None
        if _waited:
            _wdt = int((time.perf_counter() - _wt0) * 1e6)
            with self.lock:
                self.stats["stage_wait_us"] = self.stats.get("stage_wait_us", 0) + _wdt
                self.stats["stage_waits"] = self.stats.get("stage_waits", 0) + 1
        rows = got[1] if isinstance(got, tuple) else got
        with self.lock:
            if self.closed:
                # close() cleared staged/staged_futs mid-join: serve the
                # joined row for THIS key only — re-fanning it into the
                # cleared dicts would resurrect state on a closed store.
                if rows:
                    for eid, r in zip(ids, rows):
                        if (
                            r is not None
                            and linear.bundle_key(int(eid)) == key
                        ):
                            return r
                return None
            if rows:
                for eid, r in zip(ids, rows):
                    if r is None:
                        continue
                    k = linear.bundle_key(int(eid))
                    self.staged_futs.pop(k, None)
                    self.staged.setdefault(k, r)
            for eid in ids:
                self.staged_futs.pop(linear.bundle_key(int(eid)), None)
            row = self.staged.pop(key, None)
            if row is not None:
                self.stats["staged_hits"] = self.stats.get("staged_hits", 0) + 1
            else:
                self.stats["staged_misses"] = self.stats.get("staged_misses", 0) + 1
            return row

    def stage_drop(self, key: Any) -> None:
        """Remove *key* from staging; cancel its in-flight read when held.

        The set-read future is shared by every expert id it covers — it
        is cancelled only when no other staged_futs entry still references
        it, so dropping one key cannot kill a sibling's join.
        """
        with self.lock:
            self.staged.pop(key, None)
            entry = self.staged_futs.pop(key, None)
            fut = entry[0] if entry is not None else None
            if fut is not None and any(
                e[0] is fut for e in self.staged_futs.values()
            ):
                fut = None
        if fut is not None:
            try:
                fut.cancel()
            except Exception:
                pass

    # -- advise hygiene (P7) ----------------------------------------------

    def advise_fresh(self, layer_idx: int, eids: list) -> list:
        """Drop experts advised within the TTL window; mark the survivors."""
        if _ADVISE_TTL <= 0:
            return list(eids)
        with self.lock:
            if self.closed:
                return list(eids)
            now = self._advise_clock
            keep = []
            for e in eids:
                k = (int(layer_idx), int(e))
                if now - self._advise_seen.get(k, -10**9) >= _ADVISE_TTL:
                    keep.append(int(e))
                    self._advise_seen[k] = now
            if len(self._advise_seen) > 1 << 17:
                self._advise_seen = {
                    k: v
                    for k, v in self._advise_seen.items()
                    if now - v < _ADVISE_TTL
                }
            return keep

    def trans_overfetch_ok(self) -> bool:
        """True while the transition extras earn their hints (P7).

        Precision = share of last call's extras that this call actually
        demanded. Until enough evidence accumulates the extras stay on
        (optimistic); below the floor they stop being added.
        """
        with self.lock:
            if self._trans_total < 512:
                return True
            return (
                self._trans_hits / self._trans_total
                >= _ADVISE_TRANS_MIN_PRECISION
            )

    def note_trans_extras(self, layer_idx: int, extras: list) -> None:
        """Record the extras added to this call's advisory for scoring."""
        if not extras:
            return
        with self.lock:
            if self.closed:
                return
            self._trans_pending[int(layer_idx)] = set(int(e) for e in extras)

    def is_closed(self) -> bool:
        with self.lock:
            return self.closed

    def close(self) -> None:
        """Stop speculation: mark closed, cancel staged reads, clear state.

        Idempotent. In-flight staged futures are cancelled (a pending set
        read never starts; a running one finishes into the void) BEFORE the
        dicts drop, so a worker cannot fan rows out into a cleared
        ``staged`` after close — ``stage_resolve``'s join re-checks
        ``closed`` under the lock for the same reason. The routing history
        and registry are cleared so a closed state can never serve another
        conversion's speculation.
        """
        with self.lock:
            if self.closed:
                return
            self.closed = True
            futs = [entry[0] for entry in self.staged_futs.values()]
            self.prev_uniq_by_layer.clear()
            self.linears_by_layer.clear()
            self.staged.clear()
            self.staged_futs.clear()
            self._advise_seen.clear()
            self._trans_pending.clear()
            self.recall_ewma.clear()
            # Per-token routing scratch is per-conversion state — clear it
            # so a drained engine serves nothing stale. ``trans`` itself
            # survives: it is the learned profile that save-on-unload
            # persists after close.
            self._cur_token_sets.clear()
            self._prev_token_sets.clear()
            self._xlayer_ewma.clear()
        for fut in futs:
            try:
                fut.cancel()
            except Exception:
                pass
