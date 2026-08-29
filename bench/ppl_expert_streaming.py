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

Corpus: a plain UTF-8 text file (one or more documents; whitespace-joined).
Windows are disjoint (no overlap) of --ctx tokens each; mean NLL over
predicted tokens only (the first token of each window is context).
"""

import argparse
import json
import math
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
    args = ap.parse_args()

    model_path = MODEL_PATHS.get(args.model, args.model)
    text = Path(args.corpus).read_text(encoding="utf-8")
    if not text.strip():
        print("corpus is empty", file=sys.stderr)
        sys.exit(2)

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
