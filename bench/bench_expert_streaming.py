"""Fase 0 bench: expert streaming TTFT, steady-state tok/s, hit rate, memory, stage profile.

Usage:
    .venv/bin/python bench/bench_expert_streaming.py --model qwen --budget 1.0 --decode 96 --out bench/results/qwen_1g.json
    .venv/bin/python bench/bench_expert_streaming.py --model glm --budget 1.0 --decode 16

Protocol (B6): use --single-request and the same --decode for all A/B arms so
TTFT and tok/s are comparable. Every arm writes tokens + chunk_schedule + metal peaks.

Controls:
    OMLX_EXPERT_STREAMING_PROFILE=1  (integer per-stage profiling per layer)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_PATHS = {
    "qwen": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-oQ4e-mtp",
    "glm": "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e",
    "dsv4": "/Volumes/SSD 4TB/AI Models/DeepSeek-V4-Flash-0731-oQ4e-mtp",
}
DEFAULT_ENTRIES = {
    "qwen": "Qwen3.8-Flash-Next-oQ4e-mtp",
    "glm": "GLM-5.3-Flash-oQ4e",
    "dsv4": "DeepSeek-V4-Flash-0731-oQ4e-mtp",
}
PROMPTS = {
    "qwen": [{"role": "user", "content": "Hello, how are you?"}],
    "glm": [{"role": "user", "content": "Hello, how are you?"}],
    "dsv4": [{"role": "user", "content": "Hello, how are you?"}],
}

_FILLER = (
    "The scientist wrote a detailed report about the river ecosystem, "
    "describing how the water temperature changes with the seasons and "
    "which fish species migrate through the valley each year. "
)


def build_prompt(model_key: str, prompt_len: str) -> list[dict]:
    """Synthetic prompts: short (7 tok), 512, 2k, 8k (approximate word targets)."""
    if prompt_len == "short":
        return list(PROMPTS[model_key])
    words = {"512": 400, "2k": 1600, "8k": 6400}[prompt_len]
    content = (_FILLER * (words // 26 + 1))[: words * 7]
    return [{"role": "user", "content": content}]


class FakeEnforcer:
    memory_guard_tier = "balanced"

    def __init__(self, ceiling_gib=32.0):
        self._ceiling = int(ceiling_gib * 1024**3)

    def get_ceiling_breakdown(self):
        return {"static": self._ceiling, "dynamic": 64 * 1024**3, "metal_cap": 64 * 1024**3}

    def get_final_ceiling(self):
        return self._ceiling

    def get_admission_ceiling(self):
        return self._ceiling

    def get_admission_soft_target(self):
        return int(self._ceiling * 0.875)

    def wake(self, active=False):
        pass

    def _propagate_memory_limit(self):
        pass


def _bench_settings(
    pins: bool,
    pin_gib: float | None,
    pin_regime: str,
    budget: float,
    topk: float | None,
    cold_tier: str | None,
    hot_fraction: float | None,
    mtp: bool,
    mtp_block: int | None,
    ane: bool,
    specprefill_draft: str | None,
    specprefill_keep: float | None,
):
    """Fase M1: the bench's ModelSettings, wired EXPLICITLY.

    Pins arrive as model settings (pin regime + sync), never as a late
    os.environ mutation after engine load — the PinController is built
    inside get_engine, so env written later cannot be relied on.
    """
    from omlx.model_settings import ModelSettings

    return ModelSettings(
        expert_streaming_enabled=True,
        expert_streaming_budget_gib=budget,
        expert_streaming_topk_threshold=topk,
        expert_streaming_cold_tier=cold_tier,
        # Fase I6 HOBBIT split: top fraction of experts per layer (by
        # learned pin-profile frequency) keeps the ORIGINAL packing while
        # the rest read the cold tier. Requires --cold-tier + a profile.
        expert_streaming_hot_fraction=hot_fraction,
        # --pins (parity with the ppl harness): mlock the observed hot
        # experts and LEARN the pin profile this run persists on unload —
        # the decode-dominant hot set for the prefill x decode overlap study.
        expert_streaming_pins=pins or None,
        expert_streaming_pin_gib=(pin_gib if pin_gib is not None else 0.25)
        if pins
        else None,
        # Fase M1: explicit wiring — the controller receives these BEFORE
        # the first request; no reliance on late env mutation.
        expert_streaming_pin_regime=pin_regime if pins else None,
        expert_streaming_pin_sync=True if pins else None,
        qwen4_ple_ssd_offload=True,
        vlm_mtp_enabled=mtp,
        vlm_mtp_draft_block_size=mtp_block,
        qwen35_ane_prefill_enabled=ane,
        specprefill_enabled=bool(specprefill_draft),
        specprefill_draft_model=specprefill_draft,
        specprefill_keep_pct=specprefill_keep,
        # The bench prompt is 7440 tokens; the product default threshold
        # (8192) would never trigger. Score any long-prompt run.
        specprefill_threshold=2048,
    )


def find_streaming_cache(vlm_model):
    layers = None
    for path in [
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("layers",),
    ]:
        cur = vlm_model
        ok = True
        for a in path:
            if not hasattr(cur, a):
                ok = False
                break
            cur = getattr(cur, a)
        if ok and cur is not None and len(cur) > 0:
            layers = cur
            break
    if layers is None:
        return None
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        sm = getattr(mlp, "switch_mlp", None) if mlp else None
        if sm is None:
            continue
        cache = getattr(sm, "_cache", None) or getattr(sm, "cache", None)
        if cache is not None:
            return cache
        for attr in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
            proj = getattr(sm, attr, None)
            if proj is not None and hasattr(proj, "cache"):
                return proj.cache
    return None


async def run(
    model_key: str,
    budget: float,
    decode: int,
    mtp: bool,
    out: str | None,
    topk: float | None = None,
    cold_tier: str | None = None,
    prompt_len: str = "short",
    hot_fraction: float | None = None,
    pins: bool = False,
    mtp_block: int | None = None,
    ane: bool = False,
    warm_control: float = 0.0,
    mem_ceiling: float = 28.0,
    specprefill_draft: str | None = None,
    specprefill_keep: float | None = None,
    out_dir: str = "bench/results",
    single_request: bool = False,
    gate_tokens: bool = False,
    pin_gib: float | None = None,
    pin_regime: str = "decode",
):
    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettings
    from omlx.scheduler import SchedulerConfig
    from omlx.utils.proc_memory import get_phys_footprint
    import mlx.core as mx

    model_path = MODEL_PATHS[model_key]
    entry_name = DEFAULT_ENTRIES[model_key]

    print(f"=== {model_key} budget {budget}G decode {decode} mtp {mtp} block {mtp_block} ane {ane} prompt {prompt_len} ===")
    pool = EnginePool(scheduler_config=SchedulerConfig(hot_cache_max_size=0))
    pool._process_memory_enforcer = FakeEnforcer()
    pool.discover_models("/Volumes/SSD 4TB/AI Models")
    entry = pool.get_entry(entry_name)
    if not entry:
        print("entry not found")
        return
    pool._process_memory_enforcer = None  # keep _propagate no-op path quiet

    settings = _bench_settings(
        pins, pin_gib, pin_regime, budget, topk, cold_tier, hot_fraction,
        mtp, mtp_block, ane, specprefill_draft, specprefill_keep,
    )
    runtime = pool._entry_runtime_resident_size(entry, settings)
    print(f"runtime est {runtime / 1024**3:.2f}G")

    phys0 = get_phys_footprint() / 1024**3
    t0 = time.perf_counter()
    engine = await pool.get_engine(entry_name, runtime_settings=settings)
    t_load = time.perf_counter() - t0
    phys_loaded = get_phys_footprint() / 1024**3
    print(f"engine loaded {t_load:.1f}s phys {phys_loaded:.2f}G active {mx.get_active_memory() / 1024**3:.2f}G")

    # Honest memory limits: without an enforcer the scheduler's prefill
    # throttle/guard never engage (limits stay 0) and the lazy chunk forward's
    # measured ~17MB/token transient (streaming expert mini-banks) runs
    # unbounded — the Metal buffer pool reached ~30 GiB on 8k prompts and
    # squeezed the machine into swap (F-series F1). Set the same watermarks
    # the server's ProcessMemoryEnforcer would propagate.
    try:
        _eng = pool.get_entry(entry_name).engine
        _sched = getattr(getattr(getattr(_eng, "_engine", None), "engine", None), "scheduler", None)
        if _sched is not None:
            gib = 1024**3
            _sched._memory_hard_limit_bytes = int(mem_ceiling * gib)
            _sched._memory_limit_bytes = int(mem_ceiling * 0.9 * gib)
            _sched._memory_abort_limit_bytes = int(mem_ceiling * 0.95 * gib)
            print(
                f"scheduler memory limits: hard {mem_ceiling:.0f}G "
                f"soft {mem_ceiling * 0.9:.0f}G abort {mem_ceiling * 0.95:.0f}G"
            )
    except Exception as e:
        print(f"scheduler limit setup skipped: {e}")

    # Fase M1: pin sync/regime are wired through ModelSettings BEFORE
    # get_engine (see _bench_settings) — no late os.environ mutation here.

    vlm_model = getattr(engine, "_vlm_model", None)
    cache = find_streaming_cache(vlm_model)
    if warm_control:
        # E3 control arm: deterministic warmup (evenly spread experts) fired
        # post-load. Uncorrelated with the first request's routing by design.
        from omlx.patches.expert_streaming import warmer as _warmer

        linears_by_layer: dict[int, list] = {}
        layers = None
        for path in [
            ("language_model", "model", "layers"),
            ("language_model", "layers"),
            ("model", "layers"),
            ("layers",),
        ]:
            cur = vlm_model
            ok = True
            for a in path:
                if not hasattr(cur, a):
                    ok = False
                    break
                cur = getattr(cur, a)
            if ok and cur is not None and len(cur) > 0:
                layers = cur
                break
        if layers is not None:
            for i, layer in enumerate(layers):
                moe = getattr(layer, "mlp", None) or getattr(layer, "ffn", None)
                sm = getattr(moe, "switch_mlp", None)
                if sm is not None:
                    linears_by_layer[i] = [
                        getattr(sm, p)
                        for p in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
                        if hasattr(sm, p)
                    ]
        lin0 = next((l for ls in linears_by_layer.values() for l in ls), None)
        backing = getattr(lin0, "backing", None)
        if backing is not None:
            jobs = _warmer.deterministic_warmup(
                linears_by_layer,
                backing,
                budget_bytes=int(warm_control * 1024**3),
                num_experts=512,
            )
            time.sleep(3.0)  # let the 4-thread warm pool drain
            print(f"warm-control: {jobs} discard-reads fired")
    # Reference chunk schedule for bit-exactness (B4): fixed per prompt_len
    # so that divergence from different step sizes is explicit and comparable.
    _CHUNK_SCHEDULE_REF = {"short": 512, "512": 512, "2k": 1024, "8k": 4096}
    chunk_schedule = {
        "prompt_len": prompt_len,
        "reference_step": _CHUNK_SCHEDULE_REF.get(prompt_len, 512),
        "single_request": single_request,
    }
    results = {
        "model": model_key,
        "budget_gib": budget,
        "topk_threshold": topk,
        "cold_tier": cold_tier,
        "hot_fraction": hot_fraction,
        "mtp": mtp,
        "mtp_block": mtp_block,
        "ane": ane,
        "warm_control_gib": warm_control,
        "prompt_len": prompt_len,
        "single_request": single_request,
        "chunk_schedule": chunk_schedule,
        "runtime_est_gib": runtime / 1024**3,
        "load_s": t_load,
        "phys_before_gib": round(phys0, 2),
        "phys_after_load_gib": round(phys_loaded, 2),
    }
    if cache is not None:
        results["cache_per_expert_cap"] = getattr(cache, "capacity", None)
        results["cache_per_layer_cap"] = getattr(cache, "_per_layer_cap", None)

    messages = build_prompt(model_key, prompt_len)
    from resource_sampler import ResourceSampler

    sampler = ResourceSampler(
        interval=1.0,
        mlx_callbacks={
            "mlx_active_gib": mx.get_active_memory,
            "mlx_cache_gib": mx.get_cache_memory,
            # Fase J: high-water mark per phase to distinguish prefill transient
            # from decode residency (mlx_peak_gib is process-global, so reset per phase).
            "mlx_peak_gib": mx.get_peak_memory,
        },
    )
    _metal_peak: dict[str, float] = {}
    _reset_peak = getattr(mx, "reset_peak_memory", None)

    def _peak_phase(label: str) -> None:
        try:
            _metal_peak[label] = round(mx.get_peak_memory() / 1024**3, 3)
        except Exception:
            pass
        if _reset_peak is not None:
            try:
                _reset_peak()
            except Exception:
                pass

    if _reset_peak is not None:
        try:
            _reset_peak()
        except Exception:
            pass
    sampler.start()
    sampler.mark("prefill")
    # Fase M3: phase-scope the backing telemetry so read_stats splits
    # prefill vs decode without cross-contamination.
    _tel = getattr(_bk, "read_telemetry", None) if _bk is not None else None
    if _tel is not None and _tel.enabled:
        _tel.begin_phase("prefill", request_id="bench-1", engine_id=entry_name)
    if single_request:
        # Single-request avoids the second full prefill; TTFT is first streamed token.
        t_request = time.perf_counter()
        first_output_at = None
        out2 = None
        async for output in engine.stream_chat(
            messages, max_tokens=decode, temperature=0.0
        ):
            out2 = output
            if first_output_at is None and (
                output.completion_tokens > 0 or output.new_text or getattr(output, "tokens", None)
            ):
                first_output_at = time.perf_counter()
                _peak_phase("prefill")
                if _tel is not None and _tel.enabled:
                    _tel.end_phase()
                    _tel.begin_phase("decode", request_id="bench-1", engine_id=entry_name)
                sampler.mark("decode")
        if out2 is None:
            raise SystemExit("single-request benchmark produced no output")
        if _tel is not None and _tel.enabled and first_output_at is None:
            _tel.end_phase()
            _tel.begin_phase("decode", request_id="bench-1", engine_id=entry_name)
        if _tel is not None and _tel.enabled:
            _tel.end_phase()
        end_request = time.perf_counter()
        if first_output_at is None:
            first_output_at = end_request
            _peak_phase("prefill")
            sampler.mark("decode")
        ttft = first_output_at - t_request
        t_decode = end_request - first_output_at
        n = int(out2.completion_tokens)
        prompt_tokens = getattr(out2, "prompt_tokens", None)
        print(f"TTFT (stream first token) {ttft:.1f}s prompt {prompt_tokens}")
    else:
        if _tel is not None and _tel.enabled:
            _tel.end_phase()
            _tel.begin_phase("decode", request_id="bench-1", engine_id=entry_name)
        t1 = time.perf_counter()
        out1 = await engine.chat(messages, max_tokens=1, temperature=0.0)
        ttft = time.perf_counter() - t1
        _peak_phase("prefill")
        sampler.mark("decode")
        print(f"TTFT (1 tok) {ttft:.1f}s prompt {out1.prompt_tokens}")
        t2 = time.perf_counter()
        out2 = await engine.chat(messages, max_tokens=decode, temperature=0.0)
        t_decode = time.perf_counter() - t2
        n = int(out2.completion_tokens)
        if _tel is not None and _tel.enabled:
            _tel.end_phase()
    if n <= 0:
        raise SystemExit("benchmark produced zero completion tokens")
    tokps = n / max(t_decode, 1e-9)
    _peak_phase("decode")
    sampler.mark("teardown")
    sampler.stop()
    print(f"decode {n} tok in {t_decode:.1f}s -> {tokps:.3f} tok/s")
    res_summary = sampler.summary()
    print(f"resources {res_summary['phases']}")
    import json as _json

    # Side-effect artifacts land in out_dir so concurrent/sequential trials
    # (autotune) never overwrite each other's raw sampler series.
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    _json.dump(
        sampler.samples(),
        open(out_dir_p / f"{model_key}_{budget}g_samples.json", "w"),
    )
    # Generated output for bit-exactness comparison across runs. The VLM path
    # forwards RequestOutput.output_token_ids when available. Prefer token IDs
    # for the gate; keep textual fallback. Fail-high when neither exists.
    _text = getattr(out2, "text", None)
    _tokens = getattr(out2, "tokens", None)
    if _tokens is None:
        _tokens = getattr(out2, "token_ids", None)
    if isinstance(_tokens, list) and _tokens:
        _bit_exact = _tokens
        _bit_exact_kind = "tokens"
    elif isinstance(_text, str) and _text:
        _bit_exact = _text
        _bit_exact_kind = "text"
    else:
        raise SystemExit(
            f"bit-exactness gate FAILED: out2 has neither tokens nor text "
            f"(tokens={type(_tokens).__name__}, text={type(_text).__name__}); "
            "cannot compare runs. Aborting."
        )
    # Fase K K8: arms that REQUIRE the token-ID gate must fail high when
    # the engine produced no token list — a text-only gate cannot prove
    # identical token IDs, so it must never silently pass.
    if gate_tokens and _bit_exact_kind != "tokens":
        raise SystemExit(
            f"token-ID gate FAILED: bit_exact_kind={_bit_exact_kind} "
            f"(tokens={type(_tokens).__name__}); run with the engine fix that "
            "populates output_token_ids. Aborting."
        )
    _json.dump(
        {
            "bit_exact_kind": _bit_exact_kind,
            "text": _text if isinstance(_text, str) else None,
            "completion_tokens": n,
            "tokens": _tokens if isinstance(_tokens, list) else None,
        },
        open(out_dir_p / f"{model_key}_{budget}g_output.json", "w"),
    )

    stats = None
    profile = None
    pf_stats = None
    # Fase M3: the streaming backing is resolved ONCE (walked from the
    # engine holders) and feeds the read telemetry + ctx fallback + pin
    # exports below — one source of truth.
    _bk = None
    _pinner = None
    for holder in (
        engine,
        getattr(engine, "_model", None),
        getattr(engine, "_vlm_model", None),
    ):
        if holder is None:
            continue
        _cand = getattr(holder, "_expert_streaming_backing", None)
        if _cand is not None:
            _bk = _cand
            _pinner = getattr(_bk, "_pin_controller", None)
            break
    if cache is not None:
        stats = {
            "hits": cache.stats.hits,
            "misses": cache.stats.misses,
            "evictions": cache.stats.evictions,
            "hit_rate": cache.stats.hit_rate(),
            "size": cache.size,
            "capacity": cache.capacity,
        }
        print(f"cache {stats}")
        if cache.profile.enabled:
            profile = cache.profile.report()
            print(f"profile totals {profile['totals']}")
        # PILOT prefetcher stats (attached on language_model.model or wrapper)
        pf = None
        for holder in (
            getattr(vlm_model, "language_model", None),
            getattr(getattr(vlm_model, "language_model", None), "model", None),
            vlm_model,
        ):
            cand = getattr(holder, "_expert_prefetcher", None)
            if cand is not None:
                pf = cand
                break
        if pf is not None:
            pf_stats = dict(pf.stats)
            print(f"prefetcher {pf_stats}")
        # Fase K F3: export the O2 F_RDADVISE/stash speculation counters so
        # the readahead coverage is measurable (advised experts, stash hits).
        # K1: the counters live on the per-conversion SpeculationState.
        try:
            _cache_spec = getattr(cache, "spec_state", None)
            advise_stats = dict(_cache_spec.stats) if _cache_spec is not None else None
            print(f"advise {advise_stats}")
        except Exception:
            advise_stats = None
        # Fase 2: demand-read telemetry (armed only by PROFILE=1).
        try:
            from omlx.patches.expert_streaming.shard_bank import read_stats as _rs

            _read_stats_out = _rs(_bk)
        except Exception:
            _read_stats_out = None
        # Fase L1: per-frame ctx observability — memtrace aggregates
        # (ctx_mode/positions/bank/inflight/prefetch per ctx.ensure event)
        # and the fallback-to-legacy counter by reason.
        try:
            from omlx.patches.expert_streaming.memtrace import memtrace as _mt

            _memtrace_summary = _mt.summary() if _mt.enabled else None
        except Exception:
            _memtrace_summary = None
        try:
            _ctx_fb = cache.ctx_fallback_stats()
        except Exception:
            _ctx_fb = None

        # Fase L: pin accounting (only when --pins armed a PinController).
        _pin_out = {
            "requested": pins,
            "pin_budget_gib": round((pin_gib if pin_gib is not None else 0.25), 3)
            if pins
            else 0.0,
            "pinned_bytes": 0,
            "pinned_experts": 0,
            "pinned_pages_estimate": 0,
            "profile_regime": pin_regime if pins else None,
            "pin_sync_requested": pins,
            "pin_sync_effective": False,
            "pin_regime_requested": pin_regime if pins else None,
            "pin_regime_effective": None,
            "pin_profile_loaded_at_engine_load": False,
            "pin_applied_before_first_request": False,
            "profile_fingerprint_match": None,
            "pin_load_time_ms": 0.0,
        }
        if _pinner is not None:
            _pin_out.update(
                {
                    "pin_budget_gib": round(_pinner.budget_bytes / 1024**3, 3),
                    "pinned_bytes": getattr(_bk, "pinned_bytes", 0),
                    "pinned_experts": getattr(_bk, "pinned_count", 0),
                    "pinned_pages_estimate": _pinner.pinned_pages_estimate,
                    "profile_regime": _pinner.profile_regime,
                    "pin_sync_effective": getattr(_pinner, "pin_sync", False),
                    "pin_regime_effective": _pinner.pin_regime,
                    "pin_profile_loaded_at_engine_load": getattr(
                        _pinner, "pins_applied_at_load", False
                    ),
                    "pin_applied_before_first_request": (
                        getattr(_pinner, "pins_applied_at_load", False)
                        and bool(getattr(_pinner, "pin_sync", False))
                    ),
                    "profile_fingerprint_match": _pinner.fingerprint_match,
                    "pin_load_time_ms": round(_pinner.pin_load_time_ms, 1),
                }
            )

    phys_end = get_phys_footprint() / 1024**3
    try:
        from omlx.utils.proc_memory import get_lifetime_max_phys_footprint

        phys_lifetime_max = round(
            get_lifetime_max_phys_footprint() / 1024**3, 2
        )
    except Exception:
        phys_lifetime_max = None
    results.update(
        {
            "ttft_s": round(ttft, 2),
            "decode_tokens": n,
            "decode_s": round(t_decode, 2),
            "tok_s": round(tokps, 4),
            "phys_after_decode_gib": round(phys_end, 2),
            "phys_lifetime_max_gib": phys_lifetime_max,
            "metal_peak_prefill_gib": _metal_peak.get("prefill"),
            "metal_peak_decode_gib": _metal_peak.get("decode"),
            "active_after_decode_gib": round(mx.get_active_memory() / 1024**3, 2),
            "cache_stats": stats,
            "profile": profile,
            "prefetcher": pf_stats,
            "advise_stats": advise_stats,
            "read_stats": _read_stats_out,
            "memtrace_summary": _memtrace_summary,
            "ctx_fallback_to_legacy": _ctx_fb,
            "pin": _pin_out,
            "resources": res_summary,
            "tokens": _tokens if isinstance(_tokens, list) else None,
            "bit_exact_kind": _bit_exact_kind,
        }
    )

    # Persist the learned pin profile when pins are active (the server does
    # this in stop(); the harness tears down via release/unload, so save
    # explicitly — parity with the ppl harness, which needs the frequencies
    # for the next HOBBIT-split load).
    if pins:
        from omlx.patches.expert_streaming import save_expert_pin_profile

        try:
            save_expert_pin_profile(engine)
        except Exception as exc:  # never cost the run its numbers
            print(f"pin profile save failed: {exc}")

    await pool.release_engine(entry_name)
    await pool._unload_engine(entry_name)

    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out}")
    print("=== DONE ===")


def main():
    # INFO logs (streaming conversion, pool releases) are bench evidence —
    # without a handler Python drops them below WARNING.
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_PATHS))
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--decode", type=int, default=96)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--topk", type=float, default=None, help="adaptive top-k mass threshold (default exact)")
    ap.add_argument("--cold-tier", default=None, metavar="BITS",
                    help="route expert reads to the <model>/expert_cold/ 3/2-bit tier (I5)")
    ap.add_argument("--hot-fraction", type=float, default=None, metavar="FRAC",
                    help="HOBBIT split fraction (I6): with --cold-tier and a learned pin "
                         "profile, this fraction of each layer's most-used experts keeps the "
                         "original packing; the rest read the cold tier")
    ap.add_argument("--pins", action="store_true",
                    help="mlock-pin observed hot experts (default 0.25 GiB) and persist the "
                         "learned pin profile on unload (parity with the ppl harness)")
    ap.add_argument("--prompt-len", choices=["short", "512", "2k", "8k"], default="short")
    ap.add_argument("--mtp-block", type=int, default=None, help="vlm_mtp_draft_block_size (MTP tokens per round)")
    ap.add_argument("--ane", action="store_true", help="enable qwen35 ANE prefill")
    ap.add_argument("--specprefill", default=None, metavar="PATH",
                    help="draft model path for SpecPrefill (scores the prompt and prefills only the important tokens)")
    ap.add_argument("--specprefill-keep", type=float, default=None, metavar="PCT",
                    help="keep rate for SpecPrefill (default 0.2)")
    ap.add_argument("--warm-control", type=float, default=0.0, metavar="GIB", help="post-load deterministic warmup budget")
    ap.add_argument("--pin-gib", type=float, default=None, metavar="GIB",
                    help="pin budget for --pins arms (default 0.25) — L2 matrix: 0.25/0.5/1.25")
    ap.add_argument("--pin-regime", choices=["decode", "prefill"], default="decode",
                    help="regime whose learned profile drives the pin selection (arm E: prefill)")
    ap.add_argument("--min-free-gb", type=float, default=22.0, metavar="GB",
                    help="abort when available memory is below this (memory-starved runs fragment prefill "
                         "into many chunks, re-stream experts, and thrash the page cache)")
    ap.add_argument("--mem-ceiling-gib", type=float, default=28.0, metavar="GIB",
                    help="scheduler memory ceiling propagated as throttle/guard watermarks (the server "
                         "gets this from the ProcessMemoryEnforcer; the bench has no enforcer)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate-tokens", action="store_true",
                    help="require non-empty token-ID lists for the bit-exactness gate; fail high on empty")
    ap.add_argument("--out-dir", default="bench/results", metavar="DIR",
                    help="directory for the _samples/_output side-effect files (default bench/results)")
    ap.add_argument(
        "--single-request",
        action="store_true",
        help="measure TTFT and decode from one streaming request (avoids a second prefill; B6)",
    )
    args = ap.parse_args()
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1024**3
        if free_gb < args.min_free_gb:
            raise SystemExit(
                f"bench aborted: only {free_gb:.1f} GB available (need {args.min_free_gb:.0f}+). "
                "Memory-starved runs fragment prefill into many chunks and re-stream experts — "
                "close apps or lower --min-free-gb to override."
            )
        print(f"memory preflight ok: {free_gb:.1f} GB available", flush=True)
    except ImportError:
        pass
    asyncio.run(
        run(
            args.model,
            args.budget,
            args.decode,
            args.mtp,
            args.out,
            args.topk,
            args.cold_tier,
            prompt_len=args.prompt_len,
            hot_fraction=args.hot_fraction,
            pins=args.pins,
            mtp_block=args.mtp_block,
            ane=args.ane,
            specprefill_draft=args.specprefill,
            specprefill_keep=args.specprefill_keep,
            warm_control=args.warm_control,
            mem_ceiling=args.mem_ceiling_gib,
            out_dir=args.out_dir,
            single_request=args.single_request,
            gate_tokens=args.gate_tokens,
            pin_gib=args.pin_gib,
            pin_regime=args.pin_regime,
        )
    )


if __name__ == "__main__":
    main()