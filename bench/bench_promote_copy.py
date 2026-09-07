"""What does the numpy -> MLX promote actually cost at decode geometry?

The union path reads one contiguous NumPy bank per (projection, key) and
promotes it with ``mx.array`` before feeding ``gather_qmm``. Per layer that
is ~28.2 MB: 3 projections x 10 experts x 940,800 B. The profile attributes
2.872 ms/call to ``load``, i.e. ~9.8 GB/s — far below the memory ceiling, so
either the copy is slow or the read dominates.

This bench isolates the copy so the two can be told apart. If ``mx.array``
for 28 MB costs ~2.5 ms, the promote is the wall and the read is nearly free.
If it costs ~0.4 ms, the read dominates and the copy is not worth attacking.

Usage:
    .venv/bin/python bench/bench_promote_copy.py
"""

import json
import time
from pathlib import Path

import numpy as np

GIB = 1024**3
MB = 1024**2

# Real geometry, Qwen3.8-Flash-Next-JANG_4M (48 layers x 512 experts, split).
PER_SLOT = 940_800          # bytes, one projection of one expert
EXPERTS_PER_CALL = 10       # decode demand set (num_experts_per_tok)
N_PROJ = 3                  # gate_proj / up_proj / down_proj
PER_PROJ_BANK = PER_SLOT * EXPERTS_PER_CALL   # 9.41 MB
LAYER_BYTES = PER_PROJ_BANK * N_PROJ          # 28.2 MB per layer call
ITERS = 300
WARMUP = 20


def _time(fn, iters=ITERS):
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return {
        "mean_ms": sum(samples) / len(samples) * 1e3,
        "p50_ms": samples[len(samples) // 2] * 1e3,
        "p90_ms": samples[int(len(samples) * 0.9)] * 1e3,
    }


def _rate(stats, nbytes):
    gb_s = nbytes / (stats["p50_ms"] * 1e-3) / GIB
    return gb_s


def main() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        print("mlx not importable — run inside the project venv")
        raise SystemExit(1) from None

    results = {}

    # -- floor 1: pure numpy memcpy of the same bytes ----------------------
    src = np.ones(PER_PROJ_BANK, dtype=np.uint8)
    dst = np.empty(PER_PROJ_BANK, dtype=np.uint8)

    def np_copy():
        np.copyto(dst, src)

    results["numpy_memcpy_28MB"] = _time(
        lambda: [np_copy() for _ in range(N_PROJ)]
    )

    # -- floor 2: fresh allocation + fill (page-fault cost) ---------------
    def np_alloc_copy():
        tmp = np.empty(PER_PROJ_BANK, dtype=np.uint8)
        np.copyto(tmp, src)

    results["numpy_alloc_copy_28MB"] = _time(
        lambda: [np_alloc_copy() for _ in range(N_PROJ)]
    )

    # -- the real thing: 3 mx.array promotes (one per projection) ---------
    def promote_3():
        out = [mx.array(src) for _ in range(N_PROJ)]
        mx.eval(out)

    results["mx_array_3x9.4MB_eval"] = _time(promote_3)

    def promote_3_noeval():
        return [mx.array(src) for _ in range(N_PROJ)]

    results["mx_array_3x9.4MB_noeval"] = _time(promote_3_noeval)

    # -- 9 smaller calls (w, s, b per projection) -------------------------
    small = np.ones(PER_PROJ_BANK // 3, dtype=np.uint8)

    def promote_9():
        out = [mx.array(small) for _ in range(9)]
        mx.eval(out)

    results["mx_array_9x3.1MB_eval"] = _time(promote_9)

    # -- zero-copy probes: anything that avoids the copy? -----------------
    probes = {}
    mv = memoryview(src)
    try:
        a = mx.array(mv)
        mx.eval(a)
        probes["mx.array(memoryview)"] = f"accepted, dtype={a.dtype}, size={a.size}"
    except Exception as exc:
        probes["mx.array(memoryview)"] = f"rejected: {type(exc).__name__}: {exc}"
    try:
        a = mx.array(src, copy=False)  # type: ignore[call-arg]
        mx.eval(a)
        probes["mx.array(copy=False)"] = f"accepted, dtype={a.dtype}"
    except Exception as exc:
        probes["mx.array(copy=False)"] = f"rejected: {type(exc).__name__}: {exc}"

    lines = []
    lines.append(f"layer bytes promoted per call: {LAYER_BYTES / MB:.2f} MB")
    lines.append(f"iters: {ITERS} (after {WARMUP} warmup)\n")
    header = f"{'arm':<28}{'p50 ms':>10}{'GB/s':>10}{'ms/token':>11}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, stats in results.items():
        gb_s = _rate(stats, LAYER_BYTES)
        # 48 layers per token
        lines.append(
            f"{name:<28}{stats['p50_ms']:>10.3f}{gb_s:>10.2f}"
            f"{stats['p50_ms'] * 48:>11.1f}"
        )
    lines.append("")
    lines.append("zero-copy probes:")
    for k, v in probes.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("profile says load_ms/call = 2.872 (read + promote).")
    promote = results["mx_array_3x9.4MB_eval"]["p50_ms"]
    lines.append(
        f"promote alone = {promote:.3f} ms -> "
        f"{promote / 2.872 * 100:.0f}% of load_ms; "
        f"read is the remaining ~{max(0.0, 2.872 - promote):.3f} ms"
        if promote < 2.872
        else f"promote alone = {promote:.3f} ms -> EXCEEDS load_ms (2.872); "
        f"the profile number must include overlap or smaller demand sets"
    )

    text = "\n".join(lines)
    print(text)

    out = Path("bench/results/decode_profile/promote_copy.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "layer_bytes": LAYER_BYTES,
                "iters": ITERS,
                "arms": results,
                "zero_copy_probes": probes,
                "profile_load_ms": 2.872,
            },
            indent=2,
        )
    )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
