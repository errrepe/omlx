# SPDX-License-Identifier: Apache-2.0
"""Warm-only page-cache prefetch and mlock pinning for streamed experts.

Both mechanisms exploit the page-cache-only streaming default (no LRU):
the OS file cache holds recently read expert pages, and reuse is served
from RAM at memory bandwidth instead of the NVMe.

PageCacheWarmer
    During decode, right before a MoE layer loads its experts, submit
    discarded reads for the PREVIOUS token's experts of the NEXT layer.
    Independent per-layer routing repeats ~35% of experts across adjacent
    tokens (measured on FlashNext-class checkpoints); those reads hit RAM
    when the next layer demands them. Results are discarded — nothing is
    stored, no heap, no LRU.

PinController
    Observe routed experts for the first N decode calls, then mlock the
    file-cache pages of the most frequent experts per layer within a byte
    budget. Locked pages are the file pages themselves (zero-copy) but
    become wired memory — they cannot be evicted. This substitutes a hot
    set for the LRU at a fraction of the accounting cost.

Both are opt-in via OMLX_EXPERT_STREAMING_WARM=1 / _PIN=1 and decode-only
(gated on routing row count).
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Decode rows are top_k * batch (8 * B); prefill chunks are much larger.
# A small prompt (<= this many rows) may warm needlessly once — bounded waste.
_MAX_WARM_ROWS = 64

WARM_ENABLED = os.environ.get("OMLX_EXPERT_STREAMING_WARM", "0") == "1"
PIN_ENABLED = os.environ.get("OMLX_EXPERT_STREAMING_PIN", "0") == "1"
# F_RDADVISE readahead (Fase G): same prediction flow as the read-warmer,
# but the submitted jobs are kernel readahead hints instead of discarded
# reads — no userspace copy, near-zero cost, so it defaults ON (disable
# with OMLX_EXPERT_STREAMING_RA=0).
RA_ENABLED = os.environ.get("OMLX_EXPERT_STREAMING_RA", "1") != "0"
# Prefill-hotness seeding (Fase G): after a streaming prefill, replace the
# expert cache contents with the prompt's hot experts (ds4's cache seeding).
SEED_ENABLED = os.environ.get("OMLX_EXPERT_STREAMING_SEED", "1") != "0"
SEED_BYTES = int(
    float(os.environ.get("OMLX_EXPERT_STREAMING_SEED_GIB", "2.0")) * 1024**3
)
PIN_BUDGET_BYTES = int(
    float(os.environ.get("OMLX_EXPERT_STREAMING_PIN_GIB", "1.25")) * 1024**3
)
PIN_OBSERVE_CALLS = max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_PIN_TOKENS", "8")))
# Learned pin store (colibri-style): persist observed per-layer frequencies
# to this JSON and reload them on the next load, skipping the observation
# window so the hot set is wired from token 1.
PIN_PROFILE_PATH = os.environ.get("OMLX_EXPERT_STREAMING_PIN_PROFILE", "") or None
_PIN_PROFILE_KEEP = 64

_WARM_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="omlx-expert-warm")


def _proj_keys(linear: Any) -> list[str]:
    keys = []
    w = getattr(linear, "stacked_weight_key", None) or getattr(linear, "stacked_key", None)
    if w:
        keys.append(w)
    s = getattr(linear, "stacked_scales_key", None)
    if s and hasattr(linear.backing, "load_expert_slice"):
        keys.append(s)
    b = getattr(linear, "stacked_biases_key", None)
    if b:
        keys.append(b)
    return keys


class PageCacheWarmer:
    """Fire-and-forget reads of the previous token's next-layer experts.

    With ``advise_only`` the jobs become F_RDADVISE kernel readahead hints
    (grouped per contiguous expert run) instead of discarded reads: same
    prediction, zero data copied into userspace.
    """

    def __init__(self, linears_by_layer: Dict[int, list], *, advise_only: bool = False):
        self.linears_by_layer = linears_by_layer
        self.last_uniq: Dict[int, list[int]] = {}
        self.warmed = 0
        self.warm_s = 0.0
        self.advised = 0
        self.advised_bytes = 0
        self.advise_failures = 0
        self.advise_only = advise_only
        self._inflight: set = set()

    def on_layer_start(self, layer_idx: int, positions: int) -> None:
        """Fire the next layer's previous-token warm reads before this
        layer's demand loads (maximum overlap with GPU compute)."""
        if positions > _MAX_WARM_ROWS:
            return
        nxt = layer_idx + 1
        prev = self.last_uniq.get(nxt)
        if prev:
            linears = self.linears_by_layer.get(nxt)
            if linears:
                self._submit(prev, linears)

    def on_layer_plan(self, layer_idx: int, uniq_list: list[int], positions: int) -> None:
        """Record this token's expert set for the next token's warm pass."""
        self.last_uniq[layer_idx] = [] if positions > _MAX_WARM_ROWS else list(uniq_list)

    def _submit(self, eids: list[int], linears: list) -> None:
        backing = getattr(linears[0], "backing", None)
        if backing is None:
            return
        if self.advise_only:
            if not hasattr(backing, "advise_expert_run"):
                return
            jobs = [(key, eids) for lin in linears for key in _proj_keys(lin)]
            if not jobs:
                return

            def _run():
                for key, ids in jobs:
                    # ids come from np.unique (ascending): group contiguous runs.
                    start = None
                    prev_id = None
                    for eid in ids:
                        if start is None:
                            start = prev_id = eid
                            continue
                        if eid == prev_id + 1:
                            prev_id = eid
                            continue
                        self._advise_one(backing, key, start, prev_id - start + 1)
                        start = prev_id = eid
                    if start is not None:
                        self._advise_one(backing, key, start, prev_id - start + 1)

            _WARM_POOL.submit(_run)
            return

        if not hasattr(backing, "load_expert_slice"):
            return
        jobs = []
        for lin in linears:
            for key in _proj_keys(lin):
                for eid in eids:
                    jobs.append((key, eid))
        if not jobs:
            return

        def _run():
            t0 = time.perf_counter()
            for key, eid in jobs:
                if (key, eid) in self._inflight:
                    continue
                self._inflight.add((key, eid))
                try:
                    backing.load_expert_slice(key, eid)
                except Exception:
                    pass
                finally:
                    self._inflight.discard((key, eid))
            self.warmed += len(jobs)
            self.warm_s += time.perf_counter() - t0

        _WARM_POOL.submit(_run)

    def _advise_one(self, backing: Any, key: str, first_id: int, count: int) -> None:
        try:
            if backing.advise_expert_run(key, first_id, count):
                self.advised += 1
            else:
                self.advise_failures += 1
        except Exception:
            self.advise_failures += 1


class PinController:
    """Observe routing for N calls, then mlock the hot experts per layer."""

    def __init__(
        self,
        linears_by_layer: Dict[int, list],
        backing: Any,
        *,
        budget_bytes: int = PIN_BUDGET_BYTES,
        observe_calls: int = PIN_OBSERVE_CALLS,
        per_expert_bytes: int = 0,
    ):
        self.linears_by_layer = linears_by_layer
        self.backing = backing
        self.budget_bytes = budget_bytes
        self.observe_calls = observe_calls
        self.per_expert_bytes = per_expert_bytes
        self.freq: Dict[int, Counter] = {}
        self.calls = 0
        self.pinned = False
        self.pin_jobs = 0
        self.profile_path = PIN_PROFILE_PATH
        if self.profile_path and self._load_profile():
            # Learned hot set available: pin immediately, no observation.
            self._pin_all()

    def _load_profile(self) -> bool:
        try:
            import json

            data = json.loads(open(self.profile_path).read())
            freq = data.get("freq") or {}
            if not freq:
                return False
            self.freq = {
                int(layer): Counter({int(e): int(c) for e, c in pairs})
                for layer, pairs in freq.items()
            }
            if data.get("per_expert_bytes"):
                self.per_expert_bytes = int(data["per_expert_bytes"])
            logger.info(
                "Expert streaming: loaded learned pin profile (%d layers) from %s",
                len(self.freq),
                self.profile_path,
            )
            return True
        except Exception as e:
            logger.debug("Failed to load pin profile %s: %s", self.profile_path, e)
            return False

    def save_profile(self) -> None:
        if not self.profile_path or not self.freq:
            return
        try:
            import json

            data = {
                "per_expert_bytes": self.per_expert_bytes,
                "freq": {
                    str(layer): counter.most_common(_PIN_PROFILE_KEEP)
                    for layer, counter in sorted(self.freq.items())
                },
            }
            tmp = self.profile_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.profile_path)
            logger.info("Expert streaming: saved pin profile to %s", self.profile_path)
        except Exception as e:
            logger.debug("Failed to save pin profile %s: %s", self.profile_path, e)

    def on_layer_plan(self, layer_idx: int, uniq_list: list[int], positions: int) -> None:
        if self.pinned or positions > _MAX_WARM_ROWS:
            return
        self.freq.setdefault(layer_idx, Counter()).update(int(e) for e in uniq_list)
        # One decode token = one plan per layer; pin after the window.
        self.calls += 1
        if self.calls >= self.observe_calls * max(len(self.linears_by_layer), 1):
            self._pin_all()

    def _pin_all(self) -> None:
        self.pinned = True
        num_layers = max(len(self.freq), 1)
        per_expert = self.per_expert_bytes
        if per_expert <= 0:
            per_expert = max(
                (getattr(l, "_per_expert_hint", 0) for ls in self.linears_by_layer.values() for l in ls),
                default=0,
            )
        slots_per_layer = 0
        if per_expert > 0:
            slots_per_layer = max(0, self.budget_bytes // (num_layers * per_expert))
        jobs = []
        for layer_idx, counter in self.freq.items():
            linears = self.linears_by_layer.get(layer_idx) or []
            keys = [k for lin in linears for k in _proj_keys(lin)]
            if not keys:
                continue
            top = [e for e, _ in counter.most_common(slots_per_layer)] if slots_per_layer else []
            for eid in top:
                for key in keys:
                    jobs.append((key, eid))

        def _run():
            t0 = time.perf_counter()
            for key, eid in jobs:
                try:
                    self.backing.pin_expert(key, eid)
                except Exception:
                    pass
            self.pin_jobs = len(jobs)
            self.save_profile()
            logger.info(
                "Expert streaming: pinned %d expert slices (%.2f GiB wired) in %.1fs",
                self.backing.pinned_count,
                self.backing.pinned_bytes / 1024**3,
                time.perf_counter() - t0,
            )

        _WARM_POOL.submit(_run)


def _infer_per_expert_bytes(linears_by_layer: Dict[int, list], backing: Any) -> int:
    """Per-expert byte size from the first stacked weight key's header."""
    for linears in linears_by_layer.values():
        for lin in linears:
            key = getattr(lin, "stacked_weight_key", None)
            if key and hasattr(backing, "expert_bytes"):
                try:
                    n = int(backing.expert_bytes(key))
                    if n > 0:
                        return n
                except Exception:
                    pass
    return 0


def deterministic_warmup(
    linears_by_layer: Dict[int, list],
    backing: Any,
    *,
    budget_bytes: int,
    num_experts: int,
    max_rows: int = 64,
) -> int:
    """Control-arm warmup: fire discarded reads of an evenly spread expert
    subset per layer until the byte budget is spent.

    The first request's routing is unpredictable, so a deterministic sweep
    has no correlation with what it needs — this measures exactly that
    (expected ~0 gain). Returns the number of read jobs submitted.
    """
    if num_experts <= 0 or budget_bytes <= 0:
        return 0
    per_expert = max(
        (getattr(l, "_per_expert_hint", 0) for ls in linears_by_layer.values() for l in ls),
        default=0,
    )
    if per_expert <= 0:
        per_expert = _infer_per_expert_bytes(linears_by_layer, backing)
    if per_expert <= 0:
        return 0
    per_layer = max(1, budget_bytes // max(len(linears_by_layer), 1) // per_expert)
    stride = max(1, num_experts // max(per_layer, 1))
    eids = list(range(0, num_experts, stride))[:per_layer]
    jobs = []
    for linears in linears_by_layer.values():
        for lin in linears:
            for key in _proj_keys(lin):
                for eid in eids:
                    jobs.append((key, eid))
    if not jobs:
        return 0

    def _run():
        for key, eid in jobs:
            try:
                backing.load_expert_slice(key, eid)
            except Exception:
                pass

    _WARM_POOL.submit(_run)
    return len(jobs)


class PrefillHotnessRecorder:
    """Seed the expert caches from prefill routing hotness (Fase G).

    The prefill demand path fills the LRU with whatever the *last* chunks
    read, so decode starts with a nearly useless cache (F2: hit_rate 0.002
    at budget 4 GiB). This recorder accumulates per-layer expert frequency
    over the prefill, then — on the first decode-sized call — swaps the
    cache to the prompt's hot set: LRU retain + missing-hot loads when a
    budget exists, a bounded page-cache seed burst otherwise (budget 0 =
    page-cache-only default). One-shot per prefill.
    """

    def __init__(
        self,
        linears_by_layer: Dict[int, list],
        backing: Any,
        cache: Any = None,
        *,
        per_expert_bytes: int = 0,
        seed_bytes: int = SEED_BYTES,
    ):
        self.linears_by_layer = linears_by_layer
        self.backing = backing
        self.cache = cache
        self.per_expert_bytes = per_expert_bytes
        self.seed_bytes = seed_bytes
        self.freq: Dict[int, Counter] = {}
        self.saw_prefill = False
        self.seeded = False
        self.seeded_experts = 0
        self.seeded_s = 0.0

    def on_layer_plan(self, layer_idx: int, uniq_list: list[int], positions: int) -> None:
        if self.seeded:
            return
        if positions > _MAX_WARM_ROWS:
            # Prefill-sized call: accumulate frequency (decode rows are
            # top_k * batch and would bias toward the first token).
            self.saw_prefill = True
            self.freq.setdefault(layer_idx, Counter()).update(
                int(e) for e in uniq_list
            )

    def maybe_seed(self, layer_idx: int, positions: int) -> None:
        """Fire once, at the first decode-sized call after a prefill."""
        if self.seeded or not self.saw_prefill or positions > _MAX_WARM_ROWS:
            return
        self.seeded = True
        if not self.freq:
            return
        t0 = time.perf_counter()
        try:
            if self.cache is not None and self.cache.capacity > 0:
                n = self._seed_lru()
            else:
                n = self._seed_page_cache()
        except Exception as e:
            logger.debug("Expert streaming: hotness seed failed: %s", e)
            return
        self.seeded_experts = n
        self.seeded_s = time.perf_counter() - t0
        logger.info(
            "Expert streaming: seeded %d hot expert slices from prefill "
            "routing in %.2fs",
            n,
            self.seeded_s,
        )

    def _hot_top(self, experts_per_layer: int) -> Dict[int, list[int]]:
        return {
            layer: [e for e, _ in counter.most_common(experts_per_layer)]
            for layer, counter in self.freq.items()
        }

    def _seed_lru(self) -> int:
        per_layer_cap = getattr(self.cache, "_per_layer_cap", 0) or 0
        if per_layer_cap <= 0:
            return 0
        # Slots hold one projection slice (~per_expert/3 bytes), bundles hold
        # all three: hot experts per layer = per-layer slots / 3.
        hot = self._hot_top(max(1, per_layer_cap // 3))
        hot_pairs = {(layer, eid) for layer, eids in hot.items() for eid in eids}
        retain = getattr(self.cache, "retain_hot", None)
        if callable(retain):
            retain(hot_pairs)
        n = 0
        for layer, eids in hot.items():
            for lin in self.linears_by_layer.get(layer) or []:
                loader = getattr(lin, "_load_expert_bundle", None)
                if loader is None:
                    continue
                for eid in eids:
                    key = (layer, eid, getattr(lin, "stacked_weight_key", None))
                    if self.cache.get(key) is not None:
                        n += 1
                        continue
                    try:
                        loader(eid)
                        n += 1
                    except Exception:
                        pass
        return n

    def _seed_page_cache(self) -> int:
        """Budget-0: discarded reads of the hot set into the page cache.

        Async on the warm pool (no LRU writes from worker threads), capped
        at seed_bytes across the whole model.
        """
        num_layers = max(len(self.freq), 1)
        per_expert = self.per_expert_bytes
        if per_expert <= 0:
            per_expert = _infer_per_expert_bytes(self.linears_by_layer, self.backing)
        if per_expert <= 0:
            return 0
        experts_per_layer = max(1, min(64, self.seed_bytes // (num_layers * per_expert)))
        hot = self._hot_top(experts_per_layer)

        def _run():
            t0 = time.perf_counter()
            n = 0
            for layer, eids in hot.items():
                for lin in self.linears_by_layer.get(layer) or []:
                    b = getattr(lin, "backing", None)
                    for key in _proj_keys(lin):
                        # ids from Counter.most_common — sort for run grouping.
                        for eid in sorted(eids):
                            try:
                                b.load_expert_slice(key, eid)
                                n += 1
                            except Exception:
                                pass
            self.seeded_s = time.perf_counter() - t0
            logger.info(
                "Expert streaming: page-cache seed burst done: %d slices in %.2fs",
                n,
                self.seeded_s,
            )

        _WARM_POOL.submit(_run)
        return sum(len(eids) for eids in hot.values())


class WarmPinHook:
    """Attachment point for StreamingSwitchGLU (attribute `_warm_pins`)."""

    def __init__(
        self,
        warmer: PageCacheWarmer | None,
        pinner: PinController | None,
        recorder: "PrefillHotnessRecorder | None" = None,
    ):
        self.warmer = warmer
        self.pinner = pinner
        self.recorder = recorder

    def on_layer_start(self, layer_idx: int, positions: int) -> None:
        if self.recorder is not None:
            self.recorder.maybe_seed(layer_idx, positions)
        if self.warmer is not None:
            self.warmer.on_layer_start(layer_idx, positions)

    def on_layer_plan(self, layer_idx: int, uniq_list: list[int], positions: int) -> None:
        if self.warmer is not None:
            self.warmer.on_layer_plan(layer_idx, uniq_list, positions)
        if self.pinner is not None:
            self.pinner.on_layer_plan(layer_idx, uniq_list, positions)
        if self.recorder is not None:
            self.recorder.on_layer_plan(layer_idx, uniq_list, positions)


