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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_PATHS = {
    "qwen": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-oQ4e-mtp",
    "glm": "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e",
}
DEFAULT_ENTRIES = {
    "qwen": "Qwen3.8-Flash-Next-oQ4e-mtp",
    "glm": "GLM-5.3-Flash-oQ4e",
}
PROMPTS = {
    "qwen": [{"role": "user", "content": "Hello, how are you?"}],
    "glm": [{"role": "user", "content": "Hello, how are you?"}],
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
    prompt_len: str = "short",
    mtp_block: int | None = None,
    ane: bool = False,
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
        qwen4_ple_ssd_offload=True,
        vlm_mtp_enabled=mtp,
        vlm_mtp_draft_block_size=mtp_block,
        qwen35_ane_prefill_enabled=ane,
    )
    runtime = pool._entry_runtime_resident_size(entry, settings)
    print(f"runtime est {runtime / 1024**3:.2f}G")

    phys0 = get_phys_footprint() / 1024**3
    t0 = time.perf_counter()
    engine = await pool.get_engine(entry_name, runtime_settings=settings)
    t_load = time.perf_counter() - t0
    phys_loaded = get_phys_footprint() / 1024**3
    print(f"engine loaded {t_load:.1f}s phys {phys_loaded:.2f}G active {mx.get_active_memory() / 1024**3:.2f}G")

    vlm_model = getattr(engine, "_vlm_model", None)
    cache = find_streaming_cache(vlm_model)
    results = {
        "model": model_key,
        "budget_gib": budget,
        "topk_threshold": topk,
        "mtp": mtp,
        "mtp_block": mtp_block,
        "ane": ane,
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

    _json.dump(
        sampler.samples(),
        open(f"bench/results/{model_key}_{budget}g_samples.json", "w"),
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
        open(f"bench/results/{model_key}_{budget}g_output.json", "w"),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_PATHS))
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--decode", type=int, default=96)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--topk", type=float, default=None, help="adaptive top-k mass threshold (default exact)")
    ap.add_argument("--prompt-len", choices=["short", "512", "2k", "8k"], default="short")
    ap.add_argument("--mtp-block", type=int, default=None, help="vlm_mtp_draft_block_size (MTP tokens per round)")
    ap.add_argument("--ane", action="store_true", help="enable qwen35 ANE prefill")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    asyncio.run(
        run(
            args.model,
            args.budget,
            args.decode,
            args.mtp,
            args.out,
            args.topk,
            prompt_len=args.prompt_len,
            mtp_block=args.mtp_block,
            ane=args.ane,
        )
    )


if __name__ == "__main__":
    main()