"""Fase 0 bench: expert streaming TTFT, steady-state tok/s, hit rate, memory, stage profile.

Usage:
    .venv/bin/python bench/bench_expert_streaming.py --model qwen --budget 1.0 --decode 96 --out bench/results/qwen_1g.json
    .venv/bin/python bench/bench_expert_streaming.py --model glm --budget 1.0 --decode 16

Controls:
    OMLX_EXPERT_STREAMING_PROFILE=1  (integer per-stage profiling per layer)
"""

import argparse
import asyncio
import json
import logging
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
    mtp_block: int | None = None,
    ane: bool = False,
    warm_control: float = 0.0,
    mem_ceiling: float = 28.0,
    specprefill_draft: str | None = None,
    specprefill_keep: float | None = None,
    out_dir: str = "bench/results",
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

    settings = ModelSettings(
        expert_streaming_enabled=True,
        expert_streaming_budget_gib=budget,
        expert_streaming_topk_threshold=topk,
        expert_streaming_cold_tier=cold_tier,
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
    results = {
        "model": model_key,
        "budget_gib": budget,
        "topk_threshold": topk,
        "cold_tier": cold_tier,
        "mtp": mtp,
        "mtp_block": mtp_block,
        "ane": ane,
        "warm_control_gib": warm_control,
        "prompt_len": prompt_len,
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
        },
    )
    sampler.start()
    t1 = time.perf_counter()
    sampler.mark("prefill")
    out1 = await engine.chat(messages, max_tokens=1, temperature=0.0)
    ttft = time.perf_counter() - t1
    sampler.mark("decode")
    print(f"TTFT (1 tok) {ttft:.1f}s prompt {out1.prompt_tokens}")

    t2 = time.perf_counter()
    out2 = await engine.chat(messages, max_tokens=decode, temperature=0.0)
    t_decode = time.perf_counter() - t2
    n = out2.completion_tokens or decode
    tokps = n / t_decode
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
    # Generated output for bit-exactness comparison across runs
    _text = getattr(out2, "text", None)
    _ids = getattr(out2, "completion_tokens", None) or getattr(out2, "token_ids", None)
    _json.dump(
        {
            "text": _text if isinstance(_text, str) else None,
            "completion_tokens": n,
            "token_ids": _ids if isinstance(_ids, list) else None,
        },
        open(out_dir_p / f"{model_key}_{budget}g_output.json", "w"),
    )

    stats = None
    profile = None
    pf_stats = None
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

    phys_end = get_phys_footprint() / 1024**3
    results.update(
        {
            "ttft_s": round(ttft, 2),
            "decode_tokens": n,
            "decode_s": round(t_decode, 2),
            "tok_s": round(tokps, 4),
            "phys_after_decode_gib": round(phys_end, 2),
            "active_after_decode_gib": round(mx.get_active_memory() / 1024**3, 2),
            "cache_stats": stats,
            "profile": profile,
            "prefetcher": pf_stats,
            "resources": res_summary,
        }
    )

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
    ap.add_argument("--prompt-len", choices=["short", "512", "2k", "8k"], default="short")
    ap.add_argument("--mtp-block", type=int, default=None, help="vlm_mtp_draft_block_size (MTP tokens per round)")
    ap.add_argument("--ane", action="store_true", help="enable qwen35 ANE prefill")
    ap.add_argument("--specprefill", default=None, metavar="PATH",
                    help="draft model path for SpecPrefill (scores the prompt and prefills only the important tokens)")
    ap.add_argument("--specprefill-keep", type=float, default=None, metavar="PCT",
                    help="keep rate for SpecPrefill (default 0.2)")
    ap.add_argument("--warm-control", type=float, default=0.0, metavar="GIB", help="post-load deterministic warmup budget")
    ap.add_argument("--min-free-gb", type=float, default=22.0, metavar="GB",
                    help="abort when available memory is below this (memory-starved runs fragment prefill "
                         "into many chunks, re-stream experts, and thrash the page cache)")
    ap.add_argument("--mem-ceiling-gib", type=float, default=28.0, metavar="GIB",
                    help="scheduler memory ceiling propagated as throttle/guard watermarks (the server "
                         "gets this from the ProcessMemoryEnforcer; the bench has no enforcer)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default="bench/results", metavar="DIR",
                    help="directory for the _samples/_output side-effect files (default bench/results)")
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
            mtp_block=args.mtp_block,
            ane=args.ane,
            specprefill_draft=args.specprefill,
            specprefill_keep=args.specprefill_keep,
            warm_control=args.warm_control,
            mem_ceiling=args.mem_ceiling_gib,
            out_dir=args.out_dir,
        )
    )


if __name__ == "__main__":
    main()