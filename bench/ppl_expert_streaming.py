"""Perplexity harness for expert-streaming checkpoints (Fase I4).

Computes token NLL / perplexity over a local corpus for any MLX (quantized)
checkpoint via mlx_lm's resident path. Streaming compute is bit-exact versus
the resident path (test-pinned in tests/test_expert_streaming.py), so the
resident measurement is representative — fast, and without touching the SSD
streaming machinery.

Primary purpose: the quality gate for precision changes (the Fase I5 cold
tier) — compare oQ4e vs oQ2.7-cold-tier checkpoints on the same corpus.

Usage:
    .venv/bin/python bench/ppl_expert_streaming.py \
        --model "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e" \
        --corpus corpus.txt --max-windows 64 --out bench/results/ppl_glm.json

    # streaming mode — for checkpoints whose expert banks far exceed RAM
    # (loads through the omlx expert-streaming engine, same as the server):
    .venv/bin/python bench/ppl_expert_streaming.py --streaming \
        --model glm --cold-tier none --budget 2.0 --ctx 1024 --max-windows 24 \
        --corpus bench/corpus/pg1342.txt --out bench/results/ppl_runs/glm_base.json
    .venv/bin/python bench/ppl_expert_streaming.py --streaming \
        --model glm --cold-tier 3 --budget 2.0 --ctx 1024 --max-windows 24 \
        --corpus bench/corpus/pg1342.txt --out bench/results/ppl_runs/glm_cold3.json

Corpus: a plain UTF-8 text file (one or more documents; whitespace-joined).
Windows are disjoint (no overlap) of --ctx tokens each; mean NLL over
predicted tokens only (the first token of each window is context).

The streaming cold-tier arm is the real shipped path: experts are read from
`<model>/expert_cold/` and dequantized with the tier bits recorded in the
shard metadata.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_PATHS = {
    "qwen": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-oQ4e-mtp",
    "glm": "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e",
    "dsv4": "/Volumes/SSD 4TB/AI Models/DeepSeek-V4-Flash-0731-oQ4e-mtp",
}


def iter_windows(token_ids: list[int], ctx: int, max_windows: int | None):
    """Disjoint [ctx]-token windows. Yields (start, window). The first token
    of each window is context only — it produces no NLL term."""
    step = ctx - 1
    yielded = 0
    start = 0
    while start + ctx <= len(token_ids):
        yield start, token_ids[start : start + ctx]
        yielded += 1
        if max_windows is not None and yielded >= max_windows:
            break
        start += step


def window_nll(logits: "np.ndarray", targets: "np.ndarray") -> tuple[float, int]:
    """Mean NLL of `targets` under `logits` ([ctx, vocab] float32), skipping
    the first position (context token). Returns (sum_nll, n_terms)."""
    # logits[:-1] predict targets[1:]
    lg = logits[:-1].astype(np.float32)
    tg = targets[1:].astype(np.int64)
    lg = lg - lg.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(lg).sum(axis=-1))
    picked = lg[np.arange(len(tg)), tg]
    nll = logsumexp - picked
    return float(nll.sum()), int(len(tg))


def run_streaming(model_path: str, text: str, args) -> dict:
    """Load through the omlx expert-streaming engine (models whose expert
    banks far exceed RAM) and run the same disjoint-window NLL over raw
    forwards. Streaming compute is bit-exact with the resident path
    (test-pinned in tests/test_expert_streaming.py). With --cold-tier the
    engine's backing store routes expert reads to `<model>/expert_cold/`
    exactly as the server would."""
    import asyncio

    import mlx.core as mx

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_expert_streaming import FakeEnforcer

    async def _run() -> dict:
        from mlx_lm.models.cache import make_prompt_cache

        from omlx.engine_pool import EnginePool
        from omlx.model_settings import ModelSettings
        from omlx.scheduler import SchedulerConfig
        from omlx.utils.proc_memory import get_phys_footprint

        model_dir = Path(model_path)
        entry_name = model_dir.name
        pool = EnginePool(scheduler_config=SchedulerConfig(hot_cache_max_size=0))
        pool._process_memory_enforcer = FakeEnforcer(args.mem_ceiling_gib)
        pool.discover_models(str(model_dir.parent))
        if pool.get_entry(entry_name) is None:
            raise SystemExit(f"engine pool has no entry for {entry_name}")

        settings = ModelSettings(
            expert_streaming_enabled=True,
            expert_streaming_budget_gib=args.budget,
            expert_streaming_cold_tier=(
                None if args.cold_tier == "none" else args.cold_tier
            ),
            qwen4_ple_ssd_offload=True,
        )
        t0 = time.perf_counter()
        engine = await pool.get_engine(entry_name, runtime_settings=settings)
        t_load = time.perf_counter() - t0
        print(
            f"engine loaded in {t_load:.1f}s "
            f"phys {get_phys_footprint() / 1024**3:.2f}G",
            flush=True,
        )

        # Same watermarks the server's ProcessMemoryEnforcer propagates;
        # without them the scheduler's prefill guard never engages.
        try:
            sched = getattr(
                getattr(getattr(pool.get_entry(entry_name).engine, "_engine", None), "engine", None),
                "scheduler",
                None,
            )
            if sched is not None:
                gib = 1024**3
                sched._memory_hard_limit_bytes = int(args.mem_ceiling_gib * gib)
                sched._memory_limit_bytes = int(args.mem_ceiling_gib * 0.9 * gib)
                sched._memory_abort_limit_bytes = int(args.mem_ceiling_gib * 0.95 * gib)
                print(
                    f"scheduler memory limits: hard {args.mem_ceiling_gib:.0f}G "
                    f"soft {args.mem_ceiling_gib * 0.9:.0f}G",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            print(f"scheduler limit setup skipped: {e}")

        tokenizer = engine.tokenizer
        vlm_model = getattr(engine, "_vlm_model", None)
        lm = getattr(vlm_model, "language_model", None) or vlm_model

        if os.environ.get("OMLX_PPL_DISABLE_STREAM_EVAL") == "1":
            # Diagnostic only: the per-layer eval/clear_cache boundary is
            # bit-exact (test-pinned), so NLL must not change. If it does,
            # the boundary itself is implicated at this sequence length.
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
            n_off = 0
            for layer in layers or []:
                if getattr(layer, "_stream_eval", False):
                    layer._stream_eval = False
                    n_off += 1
            print(f"diagnostic: disabled stream_eval on {n_off} layers", flush=True)

        ids = tokenizer.encode(text)
        windows = list(iter_windows(ids, args.ctx, args.max_windows))
        if not windows:
            raise SystemExit(
                f"corpus too short: {len(ids)} tokens < one {args.ctx}-token window"
            )
        print(
            f"{len(ids)} tokens -> {len(windows)} disjoint {args.ctx}-token windows",
            flush=True,
        )

        total_nll = 0.0
        total_terms = 0
        for i, (_, window) in enumerate(windows):
            t_w = time.perf_counter()
            arr = mx.array(np.array([window], dtype=np.int32))
            # Fresh cache per window (windows are disjoint). The glm5_next
            # decoder produces garbage without a cache — the scheduler always
            # passes one (self.model(chunk, cache=state.cache)).
            cache = make_prompt_cache(lm)
            logits = lm(arr, cache=cache)
            if hasattr(logits, "logits"):  # mlx_vlm LanguageModelOutput
                logits = logits.logits
            elif isinstance(logits, tuple):
                logits = logits[0]
            nll, n = window_nll(
                np.array(logits[0].astype(mx.float32)), np.array(window)
            )
            total_nll += nll
            total_terms += n
            mx.clear_cache()
            ppl_so_far = math.exp(total_nll / total_terms)
            print(
                f"  window {i + 1}/{len(windows)}  running ppl {ppl_so_far:.4f}"
                f"  ({time.perf_counter() - t_w:.1f}s)",
                flush=True,
            )

        await pool.release_engine(entry_name)
        await pool._unload_engine(entry_name)

        mean_nll = total_nll / total_terms
        return {
            "model": model_path,
            "corpus": str(args.corpus),
            "ctx": args.ctx,
            "windows": len(windows),
            "n_terms": total_terms,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "mode": "streaming",
            "cold_tier": args.cold_tier,
            "budget_gib": args.budget,
            "load_s": round(t_load, 1),
        }

    return asyncio.run(_run())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        required=True,
        help="checkpoint path or bench alias (qwen/glm/dsv4)",
    )
    ap.add_argument("--corpus", required=True, help="plain UTF-8 text file")
    ap.add_argument("--ctx", type=int, default=2048, help="window size in tokens")
    ap.add_argument(
        "--max-windows", type=int, default=64, help="stop after this many windows"
    )
    ap.add_argument("--out", default=None, help="write a JSON result here")
    ap.add_argument(
        "--streaming",
        action="store_true",
        help="load through the omlx expert-streaming engine (for checkpoints "
        "whose expert banks far exceed RAM; --cold-tier selects the arm)",
    )
    ap.add_argument(
        "--cold-tier",
        choices=["none", "2", "3"],
        default="none",
        help="expert_streaming_cold_tier for the streaming load (streaming mode only)",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=2.0,
        metavar="GIB",
        help="streaming cache budget (streaming mode only)",
    )
    ap.add_argument(
        "--mem-ceiling-gib",
        type=float,
        default=14.0,
        metavar="GIB",
        help="scheduler memory ceiling for the streaming load",
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=12.0,
        metavar="GB",
        help="abort when available memory is below this (ppl is latency-"
        "insensitive so this can be lower than the tok/s bench floor)",
    )
    args = ap.parse_args()

    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1024**3
        if free_gb < args.min_free_gb:
            raise SystemExit(
                f"aborted: only {free_gb:.1f} GB available (need {args.min_free_gb:.0f}+)"
            )
        print(f"memory preflight ok: {free_gb:.1f} GB available", flush=True)
    except ImportError:
        pass

    model_path = MODEL_PATHS.get(args.model, args.model)
    text = Path(args.corpus).read_text(encoding="utf-8")
    if not text.strip():
        print("corpus is empty", file=sys.stderr)
        sys.exit(2)

    if args.streaming:
        result = run_streaming(model_path, text, args)
        print(json.dumps(result, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
            print(f"wrote {args.out}")
        return

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    import mlx.core as mx

    print(f"loading {model_path} ...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    ids = tokenizer.encode(text)
    windows = list(iter_windows(ids, args.ctx, args.max_windows))
    if not windows:
        print(
            f"corpus too short: {len(ids)} tokens < one {args.ctx}-token window",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        f"{len(ids)} tokens -> {len(windows)} disjoint {args.ctx}-token windows",
        flush=True,
    )

    total_nll = 0.0
    total_terms = 0
    for i, (_, window) in enumerate(windows):
        arr = mx.array(np.array([window], dtype=np.int32))
        cache = make_prompt_cache(model)
        logits = model(arr, cache=cache)
        # np has no bfloat16 — promote before the numpy NLL.
        nll, n = window_nll(
            np.array(logits[0].astype(mx.float32)), np.array(window)
        )
        total_nll += nll
        total_terms += n
        ppl_so_far = math.exp(total_nll / total_terms)
        print(
            f"  window {i + 1}/{len(windows)}  running ppl {ppl_so_far:.4f}",
            flush=True,
        )

    mean_nll = total_nll / total_terms
    result = {
        "model": model_path,
        "corpus": str(args.corpus),
        "ctx": args.ctx,
        "windows": len(windows),
        "n_terms": total_terms,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
