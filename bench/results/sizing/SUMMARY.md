# Expert streaming sizing — 48 GB unified memory

Machine: M4 Pro, 48 GB unified. Usable bands: conservative 34 GiB, nominal 38 GiB, aggressive 42 GiB.

Arithmetic: `per_slot = per_expert // n_proj`, `capacity = budget // per_slot`, `per_layer_cap = capacity // num_moe_layers`. The `broken_*` columns are what the `n_proj` short-circuit bug (`__init__.py:744`) delivered for the same budget.


## 1. Geometry

| model | type | L | E/layer | n_proj | per_expert | per_slot | expert GiB | dense GiB | PLE GiB | floor GiB (PLE mmap, budget 0) |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash-0731-JANG | deepseek_v4 | 43 | 256 | 3 | 8.14 MB | 2.71 MB | 87.50 | 7.48 | 0.00 | 7.85 |
| GLM-5.3-Flash-JANG-MTP | glm5_next | 42 | 288 | 3 | 7.10 MB | 2.37 MB | 83.88 | 11.58 | 0.00 | 12.16 |
| Qwen3.8-Flash-Next-JANG_4M | qwen4_exp | 48 | 512 | 3 | 2.69 MB | 0.90 MB | 64.60 | 31.43 | 25.33 | 6.40 |
| Qwen3.8-Flash-Next-JANG_4S | qwen4_exp | 48 | 512 | 3 | 1.99 MB | 0.66 MB | 47.85 | 23.98 | 17.88 | 6.40 |
| mlx-community--Florence-2-base-ft-4bit | florence2 | — | — | — | — | — | — | — | — | unsupported: model type 'florence2' is not in the expert-streaming allowlist |
| mlx-community--Qwen2.5-1.5B-Instruct-4bit | qwen2 | — | — | — | — | — | — | — | — | unsupported: model type 'qwen2' is not in the expert-streaming allowlist |
| mlx-community--Qwen2.5-VL-3B-Instruct-4bit | qwen2_5_vl | — | — | — | — | — | — | — | — | unsupported: model type 'qwen2_5_vl' is not in the expert-streaming allowlist |
| mlx-community--Qwen3-4B-Instruct-2507-4bit | qwen3 | — | — | — | — | — | — | — | — | unsupported: model type 'qwen3' is not in the expert-streaming allowlist |

## 2. Budget sweep — DeepSeek-V4-Flash-0731-JANG

| budget GiB | slots total | slots/layer | experts/layer | coverage | cache GiB (fixed) | resident GiB | broken experts/layer | broken cache GiB | underfill |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.0% | 0.00 | 7.85 | 0 | 0.00 | 0.00x |
| 0.5 | 172 | 4 | 1 | 0.4% | 0.46 | 8.31 | 0 | 0.11 | 4.00x |
| 1 | 344 | 8 | 2 | 0.8% | 0.91 | 8.77 | 0 | 0.23 | 4.00x |
| 2 | 731 | 17 | 5 | 2.0% | 1.94 | 9.79 | 1 | 0.57 | 3.40x |
| 4 | 1505 | 35 | 11 | 4.3% | 3.99 | 11.84 | 3 | 1.25 | 3.18x |
| 8 | 3010 | 70 | 23 | 9.0% | 7.98 | 15.83 | 7 | 2.62 | 3.04x |
| 12 | 4515 | 105 | 35 | 13.7% | 11.96 | 19.82 | 11 | 3.99 | 3.00x |
| 16 | 6020 | 140 | 46 | 18.0% | 15.95 | 23.81 | 15 | 5.24 | 3.04x |
| 24 | 9030 | 210 | 70 | 27.3% | 23.93 | 31.78 | 23 | 7.98 | 3.00x |
| 32 | 12040 | 280 | 93 | 36.3% | 31.90 | 39.76 | 31 | 10.60 | 3.01x |

Full residency (streaming becomes pointless) needs 87.50 GiB of cache — NOT reachable on this box.

Fit against the 48 GB bands (floor + budget <= usable):

| band | usable GiB | headroom GiB | max budget GiB | slots/layer | experts/layer | coverage |
|---|---|---|---|---|---|---|
| conservative | 34 | 26.1 | 26.1 | 229 | 76 | 29.7% |
| nominal | 38 | 30.1 | 30.1 | 264 | 88 | 34.4% |
| aggressive | 42 | 34.1 | 34.1 | 299 | 99 | 38.7% |

## 2. Budget sweep — GLM-5.3-Flash-JANG-MTP

| budget GiB | slots total | slots/layer | experts/layer | coverage | cache GiB (fixed) | resident GiB | broken experts/layer | broken cache GiB | underfill |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.0% | 0.00 | 12.16 | 0 | 0.00 | 0.00x |
| 0.5 | 210 | 5 | 1 | 0.3% | 0.49 | 12.65 | 0 | 0.10 | 5.00x |
| 1 | 420 | 10 | 3 | 1.0% | 0.97 | 13.13 | 1 | 0.29 | 3.33x |
| 2 | 840 | 20 | 6 | 2.1% | 1.94 | 14.11 | 2 | 0.58 | 3.33x |
| 4 | 1722 | 41 | 13 | 4.5% | 3.98 | 16.14 | 4 | 1.26 | 3.15x |
| 8 | 3444 | 82 | 27 | 9.4% | 7.96 | 20.12 | 9 | 2.62 | 3.04x |
| 12 | 5166 | 123 | 41 | 14.2% | 11.94 | 24.11 | 13 | 3.98 | 3.00x |
| 16 | 6888 | 164 | 54 | 18.8% | 15.92 | 28.09 | 18 | 5.24 | 3.04x |
| 24 | 10374 | 247 | 82 | 28.5% | 23.98 | 36.14 | 27 | 7.96 | 3.01x |
| 32 | 13818 | 329 | 109 | 37.8% | 31.94 | 44.11 | 36 | 10.58 | 3.02x |

Full residency (streaming becomes pointless) needs 83.88 GiB of cache — NOT reachable on this box.

Fit against the 48 GB bands (floor + budget <= usable):

| band | usable GiB | headroom GiB | max budget GiB | slots/layer | experts/layer | coverage |
|---|---|---|---|---|---|---|
| conservative | 34 | 21.8 | 21.8 | 224 | 74 | 25.7% |
| nominal | 38 | 25.8 | 25.8 | 266 | 88 | 30.6% |
| aggressive | 42 | 29.8 | 29.8 | 307 | 102 | 35.4% |

## 2. Budget sweep — Qwen3.8-Flash-Next-JANG_4M

| budget GiB | slots total | slots/layer | experts/layer | coverage | cache GiB (fixed) | resident GiB | broken experts/layer | broken cache GiB | underfill | Belady | seed+LRU | cold LRU | % of cold ceil |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.0% | 0.00 | 6.40 | 0 | 0.00 | 0.00x | 0.000 | 0.000 | 0.000 | 0% |
| 0.5 | 528 | 11 | 3 | 0.6% | 0.46 | 6.86 | 1 | 0.13 | 3.67x | 0.253 | 0.001 | 0.000 | 0% |
| 1 | 1104 | 23 | 7 | 1.4% | 0.97 | 7.37 | 2 | 0.29 | 3.29x | 0.469 | 0.013 | 0.012 | 2% |
| 2 | 2256 | 47 | 15 | 2.9% | 1.98 | 8.38 | 5 | 0.63 | 3.13x | 0.643 | 0.429 | 0.426 | 55% |
| 4 | 4560 | 95 | 31 | 6.1% | 4.00 | 10.40 | 10 | 1.30 | 3.06x | 0.740 | 0.637 | 0.629 | 82% |
| 8 | 9120 | 190 | 63 | 12.3% | 7.99 | 14.39 | 21 | 2.65 | 3.02x | 0.774 | 0.753 | 0.723 | 97% |
| 12 | 13680 | 285 | 95 | 18.6% | 11.99 | 18.39 | 31 | 4.00 | 3.00x | 0.775 | 0.825 | 0.762 | 106% |
| 16 | 18240 | 380 | 126 | 24.6% | 15.98 | 22.38 | 42 | 5.30 | 3.02x | 0.775 | 0.872 | 0.773 | 112% |
| 24 | 27360 | 570 | 190 | 37.1% | 23.97 | 30.37 | 63 | 7.99 | 3.00x | 0.775 | 0.922 | 0.775 | 119% |
| 32 | 36480 | 760 | 253 | 49.4% | 31.96 | 38.36 | 84 | 10.64 | 3.00x | 0.775 | 0.929 | 0.775 | 120% |

Cold-start ceiling: **0.775** — the hit rate when every expert a layer ever touches is resident, so only first touches miss. This is the max for any policy that starts empty.

Belady = oracle that starts empty. seed+LRU = today's policy: prefill top-k seed, then LRU. cold LRU = no seed. Mean over layers of the captured decode trace.

**seed+LRU can exceed both** (values >100% in the last column): the seed is built from *prefill* frequencies, which are future information relative to decode. That is real and achievable in production — the seeder runs before the first decode token — not a simulator artifact.

**Knee: 8 GiB** already reaches 95% of the ceiling. More budget buys no hit rate.


Full residency (streaming becomes pointless) needs 64.60 GiB of cache — NOT reachable on this box.

Fit against the 48 GB bands (floor + budget <= usable):

| band | usable GiB | headroom GiB | max budget GiB | slots/layer | experts/layer | coverage |
|---|---|---|---|---|---|---|
| conservative | 34 | 27.6 | 27.6 | 656 | 218 | 42.6% |
| nominal | 38 | 31.6 | 31.6 | 751 | 250 | 48.8% |
| aggressive | 42 | 35.6 | 35.6 | 846 | 282 | 55.1% |

## 2. Budget sweep — Qwen3.8-Flash-Next-JANG_4S

| budget GiB | slots total | slots/layer | experts/layer | coverage | cache GiB (fixed) | resident GiB | broken experts/layer | broken cache GiB | underfill | Belady | seed+LRU | cold LRU | % of cold ceil |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.0% | 0.00 | 6.40 | 0 | 0.00 | 0.00x | 0.000 | 0.000 | 0.000 | 0% |
| 0.5 | 768 | 16 | 5 | 1.0% | 0.50 | 6.90 | 1 | 0.16 | 3.20x | 0.375 | 0.001 | 0.000 | 0% |
| 1 | 1536 | 32 | 10 | 2.0% | 1.00 | 7.40 | 3 | 0.31 | 3.20x | 0.551 | 0.265 | 0.263 | 34% |
| 2 | 3072 | 64 | 21 | 4.1% | 1.99 | 8.40 | 7 | 0.65 | 3.05x | 0.693 | 0.540 | 0.536 | 69% |
| 4 | 6144 | 128 | 42 | 8.2% | 3.99 | 10.39 | 14 | 1.31 | 3.05x | 0.762 | 0.690 | 0.674 | 89% |
| 8 | 12288 | 256 | 85 | 16.6% | 7.98 | 14.38 | 28 | 2.65 | 3.01x | 0.778 | 0.806 | 0.755 | 104% |
| 12 | 18480 | 385 | 128 | 25.0% | 11.99 | 18.40 | 42 | 3.99 | 3.01x | 0.778 | 0.878 | 0.776 | 113% |
| 16 | 24624 | 513 | 171 | 33.4% | 15.98 | 22.38 | 57 | 5.33 | 3.00x | 0.778 | 0.915 | 0.778 | 118% |
| 24 | 36960 | 770 | 256 | 50.0% | 23.99 | 30.39 | 85 | 7.98 | 3.01x | 0.778 | 0.930 | 0.778 | 120% |
| 32 | 49296 | 1027 | 342 | 66.8% | 31.99 | 38.40 | 114 | 10.65 | 3.00x | 0.778 | 0.930 | 0.778 | 120% |

Cold-start ceiling: **0.778** — the hit rate when every expert a layer ever touches is resident, so only first touches miss. This is the max for any policy that starts empty.

Belady = oracle that starts empty. seed+LRU = today's policy: prefill top-k seed, then LRU. cold LRU = no seed. Mean over layers of the captured decode trace.

**seed+LRU can exceed both** (values >100% in the last column): the seed is built from *prefill* frequencies, which are future information relative to decode. That is real and achievable in production — the seeder runs before the first decode token — not a simulator artifact.

**Knee: 8 GiB** already reaches 95% of the ceiling. More budget buys no hit rate.


Full residency (streaming becomes pointless) needs 47.85 GiB of cache — NOT reachable on this box.

Fit against the 48 GB bands (floor + budget <= usable):

| band | usable GiB | headroom GiB | max budget GiB | slots/layer | experts/layer | coverage |
|---|---|---|---|---|---|---|
| conservative | 34 | 27.6 | 27.6 | 885 | 295 | 57.6% |
| nominal | 38 | 31.6 | 31.6 | 1014 | 338 | 66.0% |
| aggressive | 42 | 35.6 | 35.6 | 1142 | 380 | 74.2% |

## 3. Measured points (real runs, 1 rep each — +/-4% noise)

| artifact | model | budget GiB | tok/s | hit rate | per-layer cap | phys GiB |
|---|---|---|---|---|---|---|
| qwen_0p5g.json | qwen | 0.5 | 1.029 | — | 4 | 6.03 |
| qwen_1g.json | qwen | 1.0 | 0.825 | — | 8 | 6.2 |
| qwen_2g.json | qwen | 2.0 | 0.855 | — | 16 | 6.53 |
| final_default_4g.json | qwen-jang4m | 4.0 | 2.647 | 0.0624 | 95 | 12.8 |
| nproj_fix_lru_4g.json | qwen-jang4m | 4.0 | 2.866 | 0.0619 | 95 | 12.99 |
| qwen_4g.json | qwen | 4.0 | 0.907 | 0.2335 | 32 | 7.21 |
| qwen_8g.json | qwen | 8.0 | 0.343 | 0.3201 | 64 | 8.57 |

## 4. Recommended defaults (48 GB)

Rule used: take the knee (smallest budget at >=95% of the cold-start ceiling) when a trace exists; otherwise 8 GiB; then clamp to the conservative band's headroom so the box never swaps.

| model | floor GiB | knee GiB | recommended budget GiB | resident GiB | experts/layer |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash-0731-JANG | 7.85 | no trace | 8 | 15.85 | 23 |
| GLM-5.3-Flash-JANG-MTP | 12.16 | no trace | 8 | 20.16 | 27 |
| Qwen3.8-Flash-Next-JANG_4M | 6.40 | 8 | 8 | 14.40 | 63 |
| Qwen3.8-Flash-Next-JANG_4S | 6.40 | 8 | 8 | 14.40 | 85 |

## 5. What this table does *not* say

- **Budget is not the binding constraint on throughput.** On JANG_4M at 4 GiB the offline policy ceiling is 0.637 hit rate but the measured e2e hit rate was 0.062 — 10% of attainable. The decode path resolves experts through the layer-context union and never writes back to the LRU (`puts == size`, `evict == 0` in the run logs), so the cache is frozen after seeding. See bench/results/residency_fix/SUMMARY.md.
  - **Correction (2026-09-07, instrumented run):** the "10% of attainable" comparison above is apples-to-oranges — the e2e `hit_rate` mixes prefill and speculation lookups into the denominator (one `get` per projection: decode 139,680 + prefill 89,853 + transitions 87,122 = 316,655). The decode-only static-seed hit rate is **~0.149** (vs a 0.063 random baseline — the seeder works and delivers 2.35x chance), and the remaining gap to 0.637 is the **frozen cache** (no write-back), not a broken seed. Full accounting: bench/results/ubc_budget/SUMMARY.md §4.1 and §7.
- **More budget can be slower.** The legacy qwen sweep measured 1.029 tok/s at 0.5 GiB and 0.343 tok/s at 8 GiB on the same prompt. A user-space cache competes with the kernel page cache for the same unified memory; the near-empty-cache arm (admission filter rejecting the seeder) had the lowest disk read of every run at 681,773 KB/s.
- **Single rep.** Every measured point is one run. Three runs of identical code spanned 2.647-2.867 tok/s (+/-4%), so no effect below ~8% is resolvable here.
- **Coverage is not hit rate.** 49% of experts resident (32 GiB on 4M) buys exactly 0% more hit rate than 12% (8 GiB), because the trace's per-layer working set is ~103 experts median / 193 max, not 512.
- **Models without a trace** (DeepSeek-V4-Flash-0731-JANG, GLM-5.3-Flash-JANG-MTP) have no hit-rate curve. Their 8 GiB default is a placeholder copied from the JANG geometry, not a measurement.

