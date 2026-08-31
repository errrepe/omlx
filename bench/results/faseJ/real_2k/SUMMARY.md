# Fase J — real-model A/B (Qwen3.8-Flash-Next-oQ4e-mtp, qwen4_exp)

Protocol: `bench/bench_expert_streaming.py --model qwen --budget 0
--prompt-len 2k --decode 48 --single-request`. Prompt 1898 tokens, 48 decode
tokens, greedy (temperature 0). Baseline = `c1be4b98`; Fase J = `80bf9b9f`.
Both arms run the **same** instrumented bench script (copied into a scratch
worktree), so the comparison is apples-to-apples.

Machine: 48 GiB, ~27 GiB free at start.

## Headline: Etapa C fixes the memory; Etapa E spends it on chunk size

| arm | TTFT | decode tok/s | Metal peak | phys_footprint max | MLX pool max | output |
|---|---|---|---|---|---|---|
| baseline (c1be4b98) | 85.7 s | **3.059** | 7.01 GiB | 37.03 GiB | 30.46 GiB | **A** |
| Fase J (C+E on) | **32.9 s** | 1.991 | 8.33 GiB | 11.25 GiB | 2.51 GiB | B |
| Fase J, Etapa E off | 92.3 s | 2.143 | **6.68 GiB** | **11.09 GiB** | **2.37 GiB** | **A** |
| Fase J, per-layer eval off | 116.2 s | 1.825 | 7.01 GiB | 37.73 GiB | 30.35 GiB | A |

`mlx_peak` = `mx.get_peak_memory()` (active buffers only).
`phys_footprint max` = whole-process lifetime high-water mark (includes the
IOAccelerator-backed allocator pool). **This is the unit the Fase J problem was
reported in (~35.7 GiB), and it is the one that matters.**

## What each etapa actually does

**Etapa C (per-layer eval boundary) delivers the entire memory win, and is
bit-exact.** Turning it off returns phys_footprint to 37.73 GiB and the pool to
30.35 GiB — i.e. baseline. With it on and Etapa E off, the output is identical
to baseline (hash `1765b80737`) in 2/2 runs.

**Etapa E (guard accounting) converts the freed memory into larger prefill
chunks.** That is where the 62% TTFT win comes from — and it is exactly what
breaks bit-exactness: a different chunk size means different GEMM shapes, hence
a different reduction order, hence different logits, hence different tokens.
With `OMLX_STREAMING_BANK_BOUNDARY_ACCOUNT=0` the output returns to baseline
while the memory win is fully retained; the TTFT win is lost.

So the trade is: **bit-exact and 70% less memory (C only), or 62% faster TTFT
and a different greedy decode (C+E).** You cannot have both.

## The output difference is deterministic, not noise

Every arm is perfectly reproducible within itself. Across all 11 real runs there
are exactly two distinct outputs, and they partition perfectly on whether the
per-layer eval boundary + Etapa E accounting are active:

- output **A** (`1765b80737`) — baseline r1, baseline r2, Fase J per-layer-eval
  off, Fase J Etapa E off r1, Fase J Etapa E off r2, e43dcfff r1, e43dcfff r2
- output **B** (`40e84090dd`) — Fase J r1..r4, Fase J barrier=0, Fase J promote=0,0

They diverge at token 3 of 48:

- A: "We need **respond** to user. User **pasted** repeated sentence many times, ending truncated: ..."
- B: "We need **answer** to user. User **repeated** sentence many times, ending truncated: ..."

Both are coherent continuations of the same thought. This is a tie-break between
near-equal logits, not obviously a degradation — but it does fail the project's
`bit_exact_kind=tokens` gate (acceptance criterion 1), and the handoff never
flagged that Etapa E costs bit-exactness.

## Open: decode throughput regresses ~30%, and it predates the memory work

    baseline (c1be4b98)   3.059 tok/s   (3.047, 3.071)
    e43dcfff              2.349 tok/s   (2.526, 2.171)
    Fase J (80bf9b9f)     1.991 tok/s   (1.802, 2.103, 2.064)

The bisect says most of it came from the *earlier* commits
(`5ef31dd6..e43dcfff`: async seed, shared layer I/O, routing-plan reuse), not
from the prefill-memory etapas. Kill-switch arms on Fase J HEAD:

    barrier=0              2.283 tok/s  (2.180, 2.387)   partial recovery
    promote=0,0            2.100 tok/s  (2.005, 2.196)   no real effect (A1/A1b exonerated)
    Etapa E off            2.143 tok/s  (2.106, 2.181)   no recovery

So: A1b is **not** the cause. Etapa B accounts for ~11% of the ~35%. The rest is
in the async-seed / shared-layer-IO commits and is still unlocalised.

End-to-end for this 48-token workload Fase J still wins (58 s vs 101 s) because
TTFT dominates, but for generation-heavy loads the decode term dominates and
Fase J loses.

## Reproducing

```sh
PYTHONPATH=<worktree> <python> bench/bench_expert_streaming.py \
  --model qwen --budget 0 --prompt-len 2k --decode 48 --single-request \
  --min-free-gb 10 --out-dir <dir> --out <dir>/result.json
```

Kill switches used above:
`OMLX_EXPERT_STREAMING_PER_LAYER_EVAL=0` (Etapa C),
`OMLX_STREAMING_BANK_BOUNDARY_ACCOUNT=0` (Etapa E),
`OMLX_EXPERT_STREAMING_LAYER_BARRIER=0` (Etapa B),
`OMLX_EXPERT_STREAMING_BANK_PROMOTE=0` + `OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX=0` (A1/A1b).
