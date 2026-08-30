#!/usr/bin/env python3
"""Validate the SwiGLU-in-ANE procedure bank on real hardware.

Feature A moves the SwiGLU activation out of the Metal merge kernel and into
the ANE program: each procedure runs two convs (gate and up) followed by
``silu`` and ``mul``, so it returns fused activation rows instead of doubled
gate/up rows.

This probe answers the two questions that decide whether the feature can
ship, using synthetic weights so it needs no checkpoint:

1. Does the private ANE runtime accept the two-conv + silu + mul program?
   Rejection is the main risk of the whole slice.
2. Does the merged activation match a plain GPU SwiGLU within the INT8
   approximation tolerance of the existing path?

Usage:
    python benchmarks/qwen35_ane_silu_in_ane_poc.py
    python benchmarks/qwen35_ane_silu_in_ane_poc.py --hidden 1024 \
        --intermediate 3072 --tokens 2048 --fraction 0.5 --repeats 20
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm.models.activations import swiglu

from omlx.custom_kernels.qwen35_prefill import fast


def _cosine(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    value = mx.sum(af * bf) / (
        mx.sqrt(mx.sum(mx.square(af))) * mx.sqrt(mx.sum(mx.square(bf)))
    )
    mx.eval(value)
    return float(value.item())


def _quantize_affine(
    weight: mx.array, group_size: int, bits: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array, mx.array]:
    """Match an mlx-lm affine QuantizedLinear: uint32 packed rows with
    scales/biases in the activation dtype."""
    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    return (
        mx.contiguous(packed),
        mx.contiguous(scales.astype(dtype)),
        mx.contiguous(biases.astype(dtype)),
    )


def _measure(call, repeats: int) -> tuple[float, list[float]]:
    value = call()
    mx.eval(value)
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = call()
        mx.eval(value)
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=64, choices=(64, 128))
    parser.add_argument("--bits", type=int, default=4, choices=(4, 5, 6, 8))
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Hidden-channel fraction assigned across both ANE instances",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--no-compare",
        dest="compare",
        action="store_false",
        help="Skip the baseline gate+up bank + Metal SwiGLU merge comparison",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.99,
        help="Absolute cosine gate against exact GPU SwiGLU",
    )
    args = parser.parse_args()

    if not fast.qwen35_ane_available():
        raise SystemExit("Private ANE runtime is unavailable on this machine.")
    for symbol in ("AneLinearBankBuilder", "qwen35_ane_dual_q4_act_t",
                   "qwen35_ane_dual_affine_act_t"):
        if not fast.has_symbol(symbol):
            raise SystemExit(
                f"Native symbol {symbol} is missing; rebuild the extension with "
                "OMLX_WITH_CUSTOM_KERNEL=1."
            )

    hidden = args.hidden
    intermediate = args.intermediate
    tokens = args.tokens
    bits = args.bits
    group_size = args.group_size

    # Split the hidden channels across the two ANE instances, mirroring the
    # production alignment: 128-wide ANE coverage, 64 per instance.
    alignment = 128
    ane_outputs = (int(intermediate * args.fraction) // alignment) * alignment
    if ane_outputs <= 0:
        raise SystemExit("--fraction is too small to cover one ANE tile")
    split = ane_outputs // 2
    gpu_outputs = intermediate - ane_outputs
    if gpu_outputs <= 0 or gpu_outputs % 64:
        raise SystemExit("--fraction must leave a positive multiple-of-64 GPU suffix")

    print(
        f"geometry: hidden={hidden} intermediate={intermediate} tokens={tokens} "
        f"bits={bits} group_size={group_size} ane={ane_outputs} "
        f"({split}/instance) gpu={gpu_outputs}"
    )

    mx.random.seed(0)
    gate = mx.random.normal((intermediate, hidden)) * 0.02
    up = mx.random.normal((intermediate, hidden)) * 0.02
    x = mx.random.normal((1, tokens, hidden)).astype(mx.float16)
    mx.eval(gate, up, x)

    def dense_rows(start: int, end: int, source: mx.array) -> mx.array:
        row = mx.contiguous(source[start:end].astype(mx.float32))
        # The builder reads the raw fp32 buffer from C++ under a released GIL,
        # so the GPU writes must be complete and visible first.
        mx.eval(row)
        mx.synchronize()
        return row

    def dense_rows_concat(start: int, end: int) -> mx.array:
        """Baseline layout: gate and up rows concatenated into one procedure."""
        row = mx.contiguous(
            mx.concatenate(
                (gate[start:end], up[start:end]), axis=0
            ).astype(mx.float32)
        )
        mx.eval(row)
        mx.synchronize()
        return row

    builder0 = fast.qwen35_ane_linear_bank_builder(tokens)
    builder1 = fast.qwen35_ane_linear_bank_builder(tokens)
    builder0.add_swiglu(dense_rows(0, split, gate), dense_rows(0, split, up))
    builder1.add_swiglu(
        dense_rows(split, ane_outputs, gate), dense_rows(split, ane_outputs, up)
    )

    print("compiling the SwiGLU-in-ANE procedure bank ...")
    started = time.perf_counter()
    models0 = builder0.compile(1, 0, 1)
    models1 = builder1.compile(2, 0, 1)
    compile_seconds = time.perf_counter() - started
    print(f"  compiled in {compile_seconds:.2f}s")

    model0 = models0[0]
    model1 = models1[0]
    print(
        f"  ANE output rows: instance0={model0.output_dim} "
        f"instance1={model1.output_dim} (expect {split} each = "
        f"half of the {2 * split} rows of a plain gate+up program)"
    )

    # Quantized GPU suffix: gate and up rows past the ANE prefix, concatenated
    # exactly like the production split.
    suffix = mx.concatenate((gate[ane_outputs:], up[ane_outputs:]), axis=0)
    gpu_weight, gpu_scales, gpu_biases = _quantize_affine(
        suffix, group_size, bits, x.dtype
    )
    mx.eval(gpu_weight, gpu_scales, gpu_biases)

    def act_call():
        if bits == 4:
            return fast.qwen35_ane_dual_q4_act_t(
                x, gpu_weight, gpu_scales, gpu_biases, model0, model1, 8, group_size
            )
        return fast.qwen35_ane_dual_affine_act_t(
            x,
            gpu_weight,
            gpu_scales,
            gpu_biases,
            model0,
            model1,
            bits,
            8,
            group_size,
        )

    try:
        median, samples = _measure(act_call, args.repeats)
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise SystemExit(
            f"SwiGLU-in-ANE evaluation failed: {exc}\n"
            "The private runtime rejected the program; feature A cannot be "
            "enabled on this machine."
        ) from exc

    actual = act_call()
    mx.eval(actual)
    print(f"  activation shape: {actual.shape} (expect (1, {tokens}, {intermediate}))")

    reference = swiglu(x @ gate.T, x @ up.T)
    mx.eval(reference)
    cosine = _cosine(actual, reference)
    max_abs = float(
        mx.max(mx.abs(actual.astype(mx.float32) - reference.astype(mx.float32))).item()
    )

    print(f"\nSwiGLU-in-ANE merge: {median * 1e3:.2f} ms median over {args.repeats} runs")
    print(f"  best {min(samples) * 1e3:.2f} ms / worst {max(samples) * 1e3:.2f} ms")
    print(f"  cosine vs exact GPU SwiGLU : {cosine:.6f}")
    print(f"  max abs error              : {max_abs:.5f}")

    if actual.shape[-1] != intermediate:
        raise SystemExit("FAIL: activation width does not match intermediate_size")

    # The decision-relevant comparison is not against exact arithmetic (both
    # ANE paths approximate) but against the path this feature replaces. Run
    # the same split through the plain gate+up bank + Metal SwiGLU merge.
    baseline_cosine = None
    if args.compare:
        plain0 = fast.qwen35_ane_linear_bank_builder(tokens)
        plain1 = fast.qwen35_ane_linear_bank_builder(tokens)
        plain0.add(
            dense_rows_concat(0, split)
        )
        plain1.add(
            dense_rows_concat(split, ane_outputs)
        )
        base_models0 = plain0.compile(1, 0, 1)
        base_models1 = plain1.compile(2, 0, 1)
        baseline = fast.qwen35_ane_dual_q4_swiglu_t(
            x,
            gpu_weight,
            gpu_scales,
            gpu_biases,
            base_models0[0],
            base_models1[0],
            8,
            group_size,
        )
        mx.eval(baseline)
        baseline_cosine = _cosine(baseline, reference)
        baseline_median, _ = _measure(
            lambda: fast.qwen35_ane_dual_q4_swiglu_t(
                x,
                gpu_weight,
                gpu_scales,
                gpu_biases,
                base_models0[0],
                base_models1[0],
                8,
                group_size,
            ),
            args.repeats,
        )
        cross = _cosine(actual, baseline)
        print(
            f"\nbaseline (gate+up program + Metal SwiGLU merge): "
            f"{baseline_median * 1e3:.2f} ms median"
        )
        print(f"  cosine vs exact GPU SwiGLU : {baseline_cosine:.6f}")
        print(f"  cosine vs SwiGLU-in-ANE    : {cross:.6f}")
        speedup = baseline_median / median if median else 0.0
        print(f"  speedup of SwiGLU-in-ANE   : {speedup:.2f}x")

    if cosine < args.min_cosine:
        raise SystemExit(
            f"FAIL: cosine {cosine:.6f} below the {args.min_cosine} gate; the "
            "ANE program is producing incorrect activations."
        )
    if baseline_cosine is not None and cosine < baseline_cosine - 0.01:
        raise SystemExit(
            f"FAIL: SwiGLU-in-ANE cosine {cosine:.6f} is materially worse than "
            f"the baseline {baseline_cosine:.6f}"
        )
    print("\nPASS: the ANE accepted the program and the activation matches.")


if __name__ == "__main__":
    main()
