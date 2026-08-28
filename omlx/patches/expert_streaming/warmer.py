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
PIN_BUDGET_BYTES = int(
    float(os.environ.get("OMLX_EXPERT_STREAMING_PIN_GIB", "1.25")) * 1024**3
)
PIN_OBSERVE_CALLS = max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_PIN_TOKENS", "8")))

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
    """Fire-and-forget reads of the previous token's next-layer experts."""

    def __init__(self, linears_by_layer: Dict[int, list]):
        self.linears_by_layer = linears_by_layer
        self.last_uniq: Dict[int, list[int]] = {}
        self.warmed = 0
        self.warm_s = 0.0
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
        if backing is None or not hasattr(backing, "load_expert_slice"):
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
            logger.info(
                "Expert streaming: pinned %d expert slices (%.2f GiB wired) in %.1fs",
                self.backing.pinned_count,
                self.backing.pinned_bytes / 1024**3,
                time.perf_counter() - t0,
            )

        _WARM_POOL.submit(_run)


class WarmPinHook:
    """Attachment point for StreamingSwitchGLU (attribute `_warm_pins`)."""

    def __init__(self, warmer: PageCacheWarmer | None, pinner: PinController | None):
        self.warmer = warmer
        self.pinner = pinner

    def on_layer_start(self, layer_idx: int, positions: int) -> None:
        if self.warmer is not None:
            self.warmer.on_layer_start(layer_idx, positions)

    def on_layer_plan(self, layer_idx: int, uniq_list: list[int], positions: int) -> None:
        if self.warmer is not None:
            self.warmer.on_layer_plan(layer_idx, uniq_list, positions)
        if self.pinner is not None:
            self.pinner.on_layer_plan(layer_idx, uniq_list, positions)
