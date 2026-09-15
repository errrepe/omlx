# SPDX-License-Identifier: Apache-2.0
"""Read/pool telemetry for expert streaming.

Owns every observation counter the shard-bank read path reports:
the sampled page-cache probe (mincore), the per-backing phase-scoped
ReadTelemetry, and the process-wide RunPoolTelemetry for the shared
run-read executor. Also owns the C plumbing they share (libc handles,
the Py_buffer accessor, page size) — shard_bank's mlock/munlock pin
machinery imports those primitives from here.

Zero-cost contract: when profiling is disabled the read path pays one
attribute guard per call; the mincore probe is off unless
OMLX_EXPERT_STREAMING_MINCORE > 0.
"""

from __future__ import annotations

import ctypes
import itertools
import logging
import mmap
import os
import random
import threading
import time
import weakref
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PAGE_SIZE = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 4096

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mlock.restype = ctypes.c_int
_libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munlock.restype = ctypes.c_int
# mincore(2): one byte per page, low bit set == the page is in core. This is
# the direct way to ask "would this read hit the page cache?" instead of
# inferring it from a fitted model (see PageCacheProbe below).
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_libc.mincore.restype = ctypes.c_int


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_pyapi = ctypes.pythonapi
_pyapi.PyObject_GetBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_pyapi.PyObject_GetBuffer.restype = ctypes.c_int
_pyapi.PyBuffer_Release.argtypes = [ctypes.POINTER(_PyBuffer)]

# Page-cache residency probe. Off by default: mincore(2) costs one syscall
# per expert read and this exists to SETTLE A CONTRADICTION, not to run in
# production. The mihailescu2m/llama.cpp logs measured that "the page cache
# serves a few percent of expert reads" for multi-MiB expert slabs, and
# concluded all F_NOCACHE/L2 work there is dead -- while explicitly noting
# the opposite for 90-byte PLE rows, where the page cache "genuinely helps".
# Our own decode profile instead inferred "75% of expert bytes come from the
# page cache at 82 GiB/s" by FITTING 1/10.55 = 0.75/82 + 0.25/D, which is a
# model, not a measurement. Both cannot be true on the same hardware; this
# counter is what decides it.
# Sampling rate, not just an on/off flag: mincore(2) is one syscall per
# expert read, and measuring every read costs 3.6% decode throughput
# (bench/results/ab_mincore.json, qwen-jang, 4 reps) -- far too much for a
# diagnostic. 1-in-N sampling keeps the estimate: at N=32 the residency
# fraction is still +-0.2% absolute over ~150k reads per run, for ~0.1% of
# the cost. 0 disables; 1 samples every read (the original behaviour).
#
# MEASURED RESULT (warm page cache, qwen-jang, 4 reps, 38 GB sampled/run):
# hot_byte_frac = 0.563 +- 0.002. So on this hardware the page cache serves
# ~56% of expert read BYTES -- neither the fork's "a few percent" nor our
# fitted 75%. Their conclusion that page-cache work is dead does not
# transfer; ours was directionally right and numerically optimistic.
_PCACHE_PROBE_ENV = max(
    0, int(os.environ.get("OMLX_EXPERT_STREAMING_MINCORE", "") or 0)
)
# Atomic counter for 1-in-N sampling; shared across pool workers.
_PCACHE_SAMPLE_TICK = itertools.count()


class PageCacheProbe:
    """Process-wide page-cache hit/miss byte counters for expert reads.

    Diagnostic only. Sampling happens in ``_ShardReader._read_into``, the
    single funnel every expert read passes through, immediately before the
    preadv -- so "hot" means the pages were already resident and the read
    was a memcpy out of the page cache, and "cold" means the read had to go
    to the device.

    Lock-protected like RunPoolTelemetry: the read path runs on pool
    workers and never holds this lock across I/O.
    """

    __slots__ = (
        "_lock", "calls", "hot_bytes", "cold_bytes", "sampled_bytes",
        "failures", "reads_total", "bytes_total",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.hot_bytes = 0
        self.cold_bytes = 0
        self.sampled_bytes = 0
        self.failures = 0
        # Every read that passed the funnel, sampled or not. Without this,
        # `sampled_bytes` under 1-in-N sampling reads like total traffic.
        self.reads_total = 0
        self.bytes_total = 0

    def seen(self, n_bytes: int) -> None:
        """Count a read that passed the funnel, whether or not it was sampled."""
        with self._lock:
            self.reads_total += 1
            self.bytes_total += int(n_bytes)

    def record(self, hot_bytes: int, cold_bytes: int) -> None:
        with self._lock:
            self.calls += 1
            self.hot_bytes += int(hot_bytes)
            self.cold_bytes += int(cold_bytes)
            self.sampled_bytes += int(hot_bytes) + int(cold_bytes)

    def failed(self) -> None:
        with self._lock:
            self.failures += 1

    def summary(self) -> dict:
        with self._lock:
            total = self.hot_bytes + self.cold_bytes
            return {
                "calls": self.calls,
                "hot_bytes": self.hot_bytes,
                "cold_bytes": self.cold_bytes,
                "sampled_bytes": self.sampled_bytes,
                "failures": self.failures,
                "hot_byte_frac": (self.hot_bytes / total) if total else 0.0,
                # Denominators for the above: everything that went past,
                # so a sampled run can be read without knowing N.
                "reads_total": self.reads_total,
                "bytes_total": self.bytes_total,
                "sample_rate": (self.calls / self.reads_total) if self.reads_total else 0.0,
            }

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.hot_bytes = 0
            self.cold_bytes = 0
            self.sampled_bytes = 0
            self.failures = 0


_PCACHE_PROBE_SINGLETON: PageCacheProbe | None = None
_pcache_probe_lock = threading.Lock()


def page_cache_probe() -> PageCacheProbe:
    """Process-wide probe singleton (one set of counters, like the IO pool)."""
    global _PCACHE_PROBE_SINGLETON
    if _PCACHE_PROBE_SINGLETON is None:
        with _pcache_probe_lock:
            if _PCACHE_PROBE_SINGLETON is None:
                _PCACHE_PROBE_SINGLETON = PageCacheProbe()
    return _PCACHE_PROBE_SINGLETON


def _mincore_resident_bytes(mm: mmap.mmap, offset: int, length: int) -> int | None:
    """Bytes of [offset, offset+length) already resident in the page cache.

    ``None`` when the probe could not be taken (no mapping, mincore
    unavailable, range outside the mapping) -- callers must treat that as
    "unknown", never as "cold".

    Residency is reported per page, so the result is page-granular: a
    partially covered first/last page counts whole. Expert slices are
    MiB-scale, so the error is far below the effect being measured.
    """
    if mm is None or length <= 0:
        return None
    try:
        view = _PyBuffer()
        if _pyapi.PyObject_GetBuffer(mm, ctypes.byref(view), 0) != 0 or not view.buf:
            return None
        try:
            start = (offset // _PAGE_SIZE) * _PAGE_SIZE
            end = min(
                view.len,
                ((offset + length + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE,
            )
            if end <= start:
                return None
            span = end - start
            npages = (span + _PAGE_SIZE - 1) // _PAGE_SIZE
            vec = ctypes.create_string_buffer(npages)
            rc = _libc.mincore(
                ctypes.c_void_p(view.buf + start),
                ctypes.c_size_t(span),
                vec,
            )
            if rc != 0:
                return None
            # MINCORE_INCORE is bit 0 (bit 1 = REFERENCED, bit 2 = MODIFIED).
            # vec.raw is bytes, so indexing yields an int -- NOT vec[i], which
            # ctypes returns as a 1-byte `bytes` object in Python 3 and would
            # raise on `& 1` (that failure is swallowed below as "unknown",
            # which is silent and looks like a cold cache).
            # numpy rather than a Python loop: a coalesced prefill run spans
            # tens of thousands of 16 KB pages.
            flags = np.frombuffer(vec.raw, dtype=np.uint8)
            resident = int(np.count_nonzero(flags & 1))
            return min(resident * _PAGE_SIZE, length)
        finally:
            _pyapi.PyBuffer_Release(ctypes.byref(view))
    except Exception:
        logger.debug("mincore probe failed", exc_info=True)
        return None


# Fase 2/M3: demand-read telemetry. The env gates the DEFAULT enabled
# state of every backing's telemetry (the bench sets PROFILE=1); each
# ExpertBackingStore owns a ReadTelemetry instance, so engines never mix
# sessions. Zero instrumentation cost when a backing's telemetry is off:
# one attribute guard per read_expert_into call.
_PROFILE_READS = os.environ.get("OMLX_EXPERT_STREAMING_PROFILE", "") == "1"

# Runtime arm switch for demand-read telemetry. _PROFILE_READS freezes at
# import; long-lived processes (server engines) need to flip the default
# AFTER backings exist, without an env-var restart. arm_read_telemetry()
# flips both this global and every live backing instance (weak-refs, so a
# dead backing never blocks the loop).
_ARM_REGISTRY: weakref.WeakSet = weakref.WeakSet()


def register_backing(store: Any) -> None:
    """Track a live backing so arm_read_telemetry() can flip it in place."""
    _ARM_REGISTRY.add(store)


def profiling_enabled() -> bool:
    """Current default enabled state for new ReadTelemetry instances."""
    return _PROFILE_READS


def arm_read_telemetry(enabled: bool = True) -> bool:
    """Arm (or disarm) demand-read telemetry for live + future backings.

    Returns the previous default state. Existing ExpertBackingStore
    instances are flipped in place; backings created later inherit the
    new default because __init__ reads this module global.
    """
    global _PROFILE_READS
    prev = _PROFILE_READS
    _PROFILE_READS = bool(enabled)
    for store in list(_ARM_REGISTRY):
        tel = getattr(store, "read_telemetry", None)
        if tel is not None and tel.enabled != _PROFILE_READS:
            tel.enabled = _PROFILE_READS
    return prev

# Fase M2/A3 stage buckets (recorded per component except the per-run
# ones). A3 renamed the ambiguous window metrics and cut the old names
# directly (every pre-A artifact is archived as un-comparable):
#   queue_wait_us  -> worker_start_delay_us (submit -> worker start)
#   preadv_us      -> read_duration_us (inside _read_into: SSD/kernel)
#   future_tail_us -> last_future_wait_us (caller wait for the FINAL run)
# plus NEW window_wait_us (all caller blocks on window futures together).
# compare_results.py refuses comparisons whose stage-key vocabularies
# differ; tests/ snapshot this canonical set (Fase A6).
_READ_METRICS = (
    "component_e2e_us",
    "reader_resolve_us",
    "plan_us",
    "buffer_alloc_us",
    "worker_start_delay_us",
    "read_duration_us",
    "window_wait_us",
    "last_future_wait_us",
    "scatter_us",
    "fallback_us",
)
# Run sizes beyond this count collapse into one capped bucket.
_RUN_SIZE_CAP = 512


def read_stats(backing: Any = None) -> dict | None:
    """Fase M3: demand-read telemetry snapshot of one backing (None when
    unarmed or no backing given — telemetry is per-backing now)."""
    if backing is None:
        return None
    tel = getattr(backing, "read_telemetry", None)
    if tel is None or not tel.enabled:
        return None
    return tel.summary()


class _Percentile:
    """Bounded reservoir percentile with always-on count/sum/min/max.

    One metric of one accumulator. The reservoir keeps at most `capacity`
    samples (uniform reservoir sampling); dropped_samples = count - kept,
    reported at the summary level so an analysis knows the percentiles are
    approximate on long runs. No lock: insertion happens under the owning
    ReadTelemetry lock (one per read call, never per run).
    """

    __slots__ = ("capacity", "count", "sum_us", "min_us", "max_us", "_res")

    def __init__(self, capacity: int):
        self.capacity = max(8, int(capacity))
        self.count = 0
        self.sum_us = 0
        self.min_us: int | None = None
        self.max_us = 0
        self._res: list[int] = []

    def add(self, us: int) -> None:
        if us < 0:
            return
        self.count += 1
        self.sum_us += us
        if self.min_us is None or us < self.min_us:
            self.min_us = us
        if us > self.max_us:
            self.max_us = us
        if len(self._res) < self.capacity:
            self._res.append(us)
        elif random.random() < self.capacity / self.count:
            self._res[random.randrange(self.capacity)] = us

    def pct(self, p: float) -> int | None:
        if not self._res:
            return None
        vals = sorted(self._res)
        return int(vals[min(len(vals) - 1, int(len(vals) * p))])

    def report(self) -> dict:
        return {
            "count": self.count,
            "p50": self.pct(0.5),
            "p95": self.pct(0.95),
            "max": self.max_us,
            "sum": self.sum_us,
        }

    def merge(self, other: "_Percentile") -> None:
        """Merge for phase-summary aggregation (approximate percentiles:
        the reservoirs concatenate up to 2x capacity)."""
        self.count += other.count
        self.sum_us += other.sum_us
        if other.min_us is not None:
            if self.min_us is None or other.min_us < self.min_us:
                self.min_us = other.min_us
        if other.max_us > self.max_us:
            self.max_us = other.max_us
        self._res.extend(other._res[: max(0, self.capacity - len(self._res))])


class _ReadAccum:
    """One scope's counters + bounded histograms (lifetime or one phase)."""

    __slots__ = (
        "metrics",
        "calls",
        "runs",
        "bytes",
        "run_sizes",
        "requested_inflight_peak",
        "failed_calls",
    )

    def __init__(self, capacity: int):
        self.metrics: dict[str, _Percentile] = {
            m: _Percentile(capacity) for m in _READ_METRICS
        }
        self.calls = 0
        self.runs = 0
        self.bytes = 0
        self.run_sizes: dict[int, int] = {}
        self.requested_inflight_peak = 0
        self.failed_calls = 0

    def add_timings(self, timings: dict[str, list[int]]) -> None:
        for name, us_list in timings.items():
            p = self.metrics.get(name)
            if p is None:
                continue
            for us in us_list:
                p.add(us)

    def run_size(self, size: int) -> None:
        capped = min(int(size), _RUN_SIZE_CAP)
        self.run_sizes[capped] = self.run_sizes.get(capped, 0) + 1


class ReadTelemetry:
    """Fase M3: per-backing, phase-scoped, memory-bounded read telemetry.

    Ownership: ONE instance per ExpertBackingStore — two engines never
    share counters. Scopes: begin_phase(phase, request_id, engine_id,
    fingerprint) opens a per-request accumulator; end_phase() freezes it
    under "requests"; summary() merges per phase name (prefill/decode) and
    keeps the lifetime totals. The worker threads never take the lock: the
    read path collects local timings and inserts ONE aggregated record per
    call under the lock.
    """

    def __init__(self, enabled: bool = True, sample_capacity: int = 2048):
        self.enabled = bool(enabled)
        self.sample_capacity = max(64, int(sample_capacity))
        self._lock = threading.Lock()
        self._lifetime = _ReadAccum(self.sample_capacity)
        self._phase_acc: _ReadAccum | None = None
        self._phase_meta: dict = {}
        self._finished: dict[str, _ReadAccum] = {}

    def begin_phase(
        self,
        phase: str,
        request_id: str | None = None,
        engine_id: str | None = None,
        fingerprint: str | None = None,
    ) -> None:
        with self._lock:
            if self._phase_acc is not None:
                # Robustness: never lose a scope's counts to an unbalanced
                # begin; close it under its own key first.
                self._close_phase_locked()
            self._phase_meta = {
                "phase": phase,
                "request_id": request_id,
                "engine_id": engine_id,
                "fingerprint": fingerprint,
            }
            self._phase_acc = _ReadAccum(self.sample_capacity)

    def _close_phase_locked(self) -> None:
        if self._phase_acc is None:
            return
        meta = self._phase_meta
        key = (meta.get("request_id") or "anon") + "/" + (meta.get("phase") or "phase")
        self._finished[key] = self._phase_acc
        self._phase_acc = None
        self._phase_meta = {}

    def end_phase(self) -> dict | None:
        with self._lock:
            if self._phase_acc is None:
                return None
            out = self._report_readonly(self._phase_acc)
            self._close_phase_locked()
            return out

    def record_call(
        self,
        *,
        runs: int = 0,
        bytes_: int = 0,
        run_sizes: list[int] | None = None,
        requested_inflight: int = 0,
        failed: bool = False,
        timings: dict[str, list[int]] | None = None,
    ) -> None:
        """ONE lock per call: the caller aggregates run-level samples first."""
        if not self.enabled:
            return
        with self._lock:
            for acc in (self._lifetime, self._phase_acc):
                if acc is None:
                    continue
                acc.calls += 1
                acc.runs += runs
                acc.bytes += bytes_
                if acc.requested_inflight_peak < requested_inflight:
                    acc.requested_inflight_peak = requested_inflight
                if failed:
                    acc.failed_calls += 1
                for size in run_sizes or []:
                    acc.run_size(size)
                if timings:
                    acc.add_timings(timings)

    def reset(self) -> None:
        with self._lock:
            self._lifetime = _ReadAccum(self.sample_capacity)
            self._phase_acc = None
            self._phase_meta = {}
            self._finished = {}

    @staticmethod
    def _report_readonly(acc: _ReadAccum) -> dict:
        avg = (acc.bytes / acc.calls) if acc.calls else 0
        return {
            "calls": acc.calls,
            "runs": acc.runs,
            "bytes": acc.bytes,
            "bytes_per_call": int(avg),
            "run_sizes_buckets": dict(sorted(acc.run_sizes.items())),
            "run_size_max": max(acc.run_sizes) if acc.run_sizes else None,
            "requested_inflight_peak": acc.requested_inflight_peak,
            "failed_calls": acc.failed_calls,
            "stages_us": {m: acc.metrics[m].report() for m in _READ_METRICS},
        }

    def summary(self) -> dict:
        with self._lock:
            merged: dict[str, _ReadAccum] = {}

            def _merge_into(target: _ReadAccum, src: _ReadAccum) -> None:
                target.calls += src.calls
                target.runs += src.runs
                target.bytes += src.bytes
                target.failed_calls += src.failed_calls
                if target.requested_inflight_peak < src.requested_inflight_peak:
                    target.requested_inflight_peak = src.requested_inflight_peak
                for size, cnt in src.run_sizes.items():
                    target.run_sizes[size] = target.run_sizes.get(size, 0) + cnt
                for m in _READ_METRICS:
                    target.metrics[m].merge(src.metrics[m])

            for key, acc in self._finished.items():
                phase = key.split("/", 1)[-1]
                if phase not in merged:
                    merged[phase] = _ReadAccum(self.sample_capacity)
                _merge_into(merged[phase], acc)
            if self._phase_acc is not None:
                phase = (self._phase_meta.get("phase") or "phase")
                if phase not in merged:
                    merged[phase] = _ReadAccum(self.sample_capacity)
                _merge_into(merged[phase], self._phase_acc)
            out: dict = {
                "profiling_enabled": self.enabled,
                "dropped_samples": 0,
                "sample_capacity": self.sample_capacity,
                "lifetime": self._report_readonly(self._lifetime),
                "requests": {
                    key: self._report_readonly(acc)
                    for key, acc in sorted(self._finished.items())
                },
            }
            for phase, acc in merged.items():
                out[phase] = self._report_readonly(acc)
            dropped = 0
            for acc in [self._lifetime, self._phase_acc] + list(
                self._finished.values()
            ):
                if acc is None:
                    continue
                for m in acc.metrics.values():
                    dropped += max(0, m.count - len(m._res))
            out["dropped_samples"] = dropped
            return out


class RunPoolTelemetry:
    """Fase M4/A4: OBSERVED concurrency of the process-wide run pool.

    requested_inflight (the caller's window size) is not the effective
    depth: the pool is shared process-wide, so queued tasks may wait for
    workers busy elsewhere. Counters are cumulative; snapshot()/delta()
    attribute a phase from before/after samples. The worker wrapper takes
    two tiny locks (start/finish) and never covers the preadv itself.

    Fase A4 OWNERS: every task may carry an owner tag (e.g. id(backing)).
    The per-owner counters mirror the global ones and reconcile with them
    (submitted_A + submitted_B == submitted_total), so a multi-engine
    process can attribute pool activity to ONE engine. Callers pass an
    owner ONLY on the profiled path (PROFILE=1 gates wrap/submit_notice),
    so the off-profile cost stays zero.
    """

    _OWNER_KEYS = (
        "submitted",
        "queued",
        "started",
        "completed",
        "failed",
        "active",
        "active_peak",
        "queue_delay_us_max",
        "active_us_max",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.submitted = 0
        self.queued = 0
        self.started = 0
        self.completed = 0
        self.failed = 0
        self.active = 0
        self.active_peak = 0
        self.queue_delay_us_sum = 0
        self.queue_delay_us_max = 0
        self.active_us_sum = 0
        self.active_us_max = 0
        self._by_owner: dict[Any, dict] = {}

    def _owner_state(self, owner: Any) -> dict:
        st = self._by_owner.get(owner)
        if st is None:
            st = {k: 0 for k in self._OWNER_KEYS}
            self._by_owner[owner] = st
        return st

    def submit_notice(self, owner: Any = None) -> None:
        """Called on the submitting thread BEFORE executor.submit."""
        with self._lock:
            self.submitted += 1
            self.queued += 1
            if owner is not None:
                ost = self._owner_state(owner)
                ost["submitted"] += 1
                ost["queued"] += 1

    def wrap(self, submit_ts: int, fn, owner: Any = None) -> Any:
        """Wrap one run task. Runs INSIDE the worker: measures queue delay
        (submit -> start) and active duration (start -> finish). Never
        touches MLX. With an owner, the same counters accumulate per owner
        under the SAME lock — never an extra lock or an extra wall-clock
        read."""

        def _w(*a, **k):
            t0 = time.perf_counter_ns()
            # P1-8: bound `ost` up front. It used to be assigned inside the
            # `with self._lock` block, so if that block raised, the
            # `except BaseException` arm below hit an unbound `ost` and a
            # NameError replaced the real error.
            ost = None
            with self._lock:
                self.queued = max(0, self.queued - 1)
                self.started += 1
                self.active += 1
                if self.active > self.active_peak:
                    self.active_peak = self.active
                qd = max(0, t0 - submit_ts) // 1000
                self.queue_delay_us_sum += qd
                if qd > self.queue_delay_us_max:
                    self.queue_delay_us_max = qd
                ost = self._owner_state(owner) if owner is not None else None
                if ost is not None:
                    ost["queued"] = max(0, ost["queued"] - 1)
                    ost["started"] += 1
                    ost["active"] += 1
                    if ost["active"] > ost["active_peak"]:
                        ost["active_peak"] = ost["active"]
                    if qd > ost["queue_delay_us_max"]:
                        ost["queue_delay_us_max"] = qd
            try:
                result = fn(*a, **k)
            except BaseException:
                with self._lock:
                    self.active -= 1
                    self.failed += 1
                    if ost is not None:
                        ost["active"] -= 1
                        ost["failed"] += 1
                raise
            else:
                t1 = time.perf_counter_ns()
                with self._lock:
                    self.active -= 1
                    self.completed += 1
                    us = (t1 - t0) // 1000
                    self.active_us_sum += us
                    if us > self.active_us_max:
                        self.active_us_max = us
                    if ost is not None:
                        ost["active"] -= 1
                        ost["completed"] += 1
                        if us > ost["active_us_max"]:
                            ost["active_us_max"] = us
                return result

        return _w

    def snapshot(self, owner: Any = None) -> dict:
        """Cumulative counters of one owner, or of the whole process pool
        when owner is None (the pre-A4 behavior)."""
        with self._lock:
            if owner is not None:
                ost = self._owner_state(owner)
                return {k: ost[k] for k in self._OWNER_KEYS}
            return {
                "submitted": self.submitted,
                "queued": self.queued,
                "started": self.started,
                "completed": self.completed,
                "failed": self.failed,
                "active": self.active,
                "active_peak": self.active_peak,
                "queue_delay_us_max": self.queue_delay_us_max,
                "active_us_max": self.active_us_max,
            }

    def delta(self, before: dict, owner: Any = None) -> dict:
        """Phase attribution: cumulative counter differences plus the
        conservative observed-peak delta (the pool's all-time peak may
        predate the phase, so peak_delta only grows when the phase raised
        it). With an owner, only that owner's tasks are attributed —
        foreign engines' pool traffic never skews the owner's phase."""
        with self._lock:
            if owner is not None:
                ost = self._owner_state(owner)
                return {
                    "submitted": ost["submitted"] - before.get("submitted", 0),
                    "started": ost["started"] - before.get("started", 0),
                    "completed": ost["completed"] - before.get("completed", 0),
                    "failed": ost["failed"] - before.get("failed", 0),
                    "active": ost["active"],
                    "active_peak_delta": max(
                        0, ost["active_peak"] - before.get("active_peak", 0)
                    ),
                    "queue_delay_us_max": ost["queue_delay_us_max"],
                    "active_us_max": ost["active_us_max"],
                }
            return {
                "submitted": self.submitted - before.get("submitted", 0),
                "started": self.started - before.get("started", 0),
                "completed": self.completed - before.get("completed", 0),
                "failed": self.failed - before.get("failed", 0),
                "active": self.active,
                "active_peak_delta": max(
                    0, self.active_peak - before.get("active_peak", 0)
                ),
                "queue_delay_us_max": self.queue_delay_us_max,
                "active_us_max": self.active_us_max,
            }


# ---------------------------------------------------------------------------
# Per-layer stage profiler + routing trace (moved from streaming_switch).
# ---------------------------------------------------------------------------

_PROFILE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_PROFILE", "") == "1"

from dataclasses import dataclass  # noqa: E402
from typing import Dict  # noqa: E402
import json  # noqa: E402


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
