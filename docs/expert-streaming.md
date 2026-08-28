# Expert Streaming (SSD)

Run Mixture-of-Experts (MoE) models that are larger than your Mac's RAM by keeping only the hot experts resident and streaming the rest from SSD.

Inspired by [slipstream](https://github.com/dwijenpatel/slipstream) (Swift/Metal expert LRU from SSD) and [colibri](https://github.com/JustVugg/colibri) (learned pin store + multi-tier memory), ported to oMLX's Python/MLX stack.

## When to use

- **You have a large MoE** (e.g. `glm_moe_dsa` / Qwen3.6-35B-A3B) and a **16–24 GB Mac**. Without streaming the model needs `resident_bytes` (checkpoint × 1.05) — often above the wired limit. With streaming it needs `dense_bytes × 1.05 + cache_budget` (default `2 GiB`), so a 35B MoE that needs ~21 GB resident fits in ~5–8 GB.
- You care about **fitting, not single-stream speed**. Streaming is slower than fully resident and disables continuous batching for that model (one request at a time). Use the default resident mode when the model already fits.

## How it works

- **Dense stays resident**: attention, shared experts, embeddings, LM head — always in unified memory.
- **Experts live on SSD**: the stacked `switch_mlp.(gate|up|down)_proj` banks (`(E, O, I)` tensors) stay memory-mapped from the original safetensors files (MADV_RANDOM, like the Qwen4 PLE `DiskBackedShardedEmbedding`). No duplicate copy is made.
- **Per-expert LRU**: one slot = one expert's weight for one layer. The cache is global per model and bounded by a single budget knob (GiB) — `slots_per_layer = budget / (num_moe_layers × per_expert_bytes)`. On a batch, the union of routed experts is looked up; hits run immediately, misses fault the expert slice via a `memcpy` from the mmap and evict the LRU entry. Quantized scales/biases are co-cached with the weight.
- **One budget, auto-forced**: set `Expert Streaming` on in the per-model settings (or leave the budget empty for `~2 GiB`). If `resident_bytes > ceiling ≥ streaming_bytes`, oMLX auto-enables it and shows an amber "auto-enabled" hint — the same pattern as `qwen4_ple_ssd_offload`.

## Supported models

| model_type | example | MoE | per-expert (oQ4e) |
|---|---|---|---|
| `glm5_next` | GLM-5.3-Flash-oQ4e (190G) | 42 layers × 288 routed | ~13 MB weight + scales |
| `qwen4_exp` | Qwen3.8-Flash-Next-oQ4e (99G) | 48 layers × 512 routed + PLE | ~2.7 MB weight + scales |
| `glm_moe_dsa` | GLM-5.2 MoE DSA | — | ~1.7 MB (BF16) / quantized packed |
| `deepseek_v4` | DeepSeek-V4-Flash-0731-oQ4e-mtp (166G) | 43 layers + 3 MTP stages × 256 routed | ~12.6 MB (mxfp4 gs32) |

Loading a glm5_next / qwen4_exp checkpoint with `expert_streaming_enabled` uses the lazy loader (`lazy=True`) and converts to streaming **before** `materialize_lazy_state` — the multi-hundred-GB MoE banks are dropped as lazy arrays instead of ever being materialized. GLM decoders additionally get `compile_ffn` disabled and a per-layer `mx.eval(out)` + `mx.clear_cache()` so the per-layer expert mini-banks (~3.4 GB at prefill) do not accumulate in the lazy graph / allocator and swap the machine. Text-engine loads (BatchedEngine) apply the same lazy + convert-before-materialize order for streaming-supported model types — this is what makes `deepseek_v4` viable on 16 GB Macs.

### DeepSeek V4 Flash (oQ4e-mtp)

`deepseek_v4` nests the MoE under `layer.ffn` (not `mlp`) and keeps one routed bank per **MTP/DSpark stage** under `mtp.<stage>[.block].ffn.switch_mlp`. The converter walks both: 43 main layers + 3 draft stages (layer ids `43..45` share the same LRU). Notes:

- **Residency**: the `mtp.<stage>` banks count as expert bytes in `expert_streaming_estimate`, so `resident_bytes`/`streaming_bytes` and the admin capability flags stay accurate for the `-mtp` checkpoints (~9.7 GB of draft-stage experts at oQ4e would otherwise sit resident). When the runtime MTP is inactive the converter simply converts fewer layers and the per-layer LRU split is rebalanced.
- **Activation**: DeepSeek V4's `SwitchGLU` uses `LimitedSwiGLU(swiglu_limit)` (fp32 on draft stages); the streaming GLU copies the original activation, keeping output bit-exact with the resident path.
- **Gate**: layers `0..2` are hash-routed (`tid2eid[input_ids]`); routing is untouched by streaming — only the expert banks are swapped.
- **Budget**: dense ≈ 5 GB, ~12.6 MB/expert × 256 experts × 46 layers. The default 1 GiB gives ~1 slot/layer (GLM-class numbers, ~0.07 tok/s on the measured baseline); 4–8 GiB is the sensible range if you have the RAM headroom.

## Settings

| field | type | notes |
|---|---|---|
| `expert_streaming_enabled` | bool | hardware-specific; excluded from profiles/templates |
| `expert_streaming_budget_gib` | float? | `null` / `0` or empty → auto `1 GiB`, clamp `0–64` |

Both are load-time settings: toggling unloads (and re-loads if pinned) the model. The runtime signature includes the *effective* (forced-or-requested) value, so the `GET /admin/api/models` capability flags `expert_streaming_supported / forced / reason / *_bytes / moe_layers / per_expert_bytes` always reflect what would actually run.

## UI

- **WebUI**: card `Advanced → Expert Streaming (SSD)` → toggle + `Cache budget (GiB)` input. Visible only when `supported`; disabled amber hint when `forced`. Lives under the `Experimental Features` header, before the Qwen ANE block.
- **macOS app**: `Model Settings → Advanced` → same toggle + conditional `Cache budget` row (auto-save, like `qwen4_ple_ssd_offload`). The `GiB` field clears to `null` when empty; values outside `0–64` are rejected.

## Measured baseline (48 GiB Mac, external SSD, warm page cache)

Decode 48 tokens after a short prefill, `OMLX_EXPERT_STREAMING_PROFILE=1`:

| model | budget | hit rate | tok/s | notes |
|---|---|---|---|---|
| Qwen 99G | 0.5 GiB (8/layer) | 0 % | ~1.0 | flat — bottleneck is per-call serial load, not hits |
| Qwen 99G | 1–2 GiB (8–16/layer) | 0 % | ~0.85 | same |
| Qwen 99G | 4 GiB (32/layer) | 23 % | ~0.9 | knee — more memory buys nothing |
| Qwen 99G | 8 GiB (64/layer) | 32 % | 0.34 | **negative**: big cache evicts OS page cache → misses re-read SSD |
| GLM 190G | 1/2/4 GiB | 0 % | 0.065–0.072 | 13 MB/expert; ~120 ms/call serialized copies dominate |

Per-stage profile (per call): Qwen — `gate_eval 1.5 ms` + `load 16 ms`; GLM — `gate_eval 1.4 ms` + `load ~120 ms`. Physical footprint is bounded: Qwen 5–7 GiB, GLM ~14 GiB (dense materialized 10.5 GiB + cache). Conclusions:

- **The bottleneck is the synchronous per-layer path** (router eval → host round-trip → serial expert loads → stack → gather), not hit rate. Cache size saturates quickly (slipstream's 6.25 % finding holds at 48 GiB).
- **Bigger caches go negative** past the knee (8 GiB on Qwen): expert cache competes with the OS page cache that makes misses cheap.
- Default `1 GiB` is right; invest in overlap/prefetch (next phase), not more memory.

## Cold-cache baseline + PILOT A/B (phase 1 result)

Same bench with a cold page cache (SSD delivering ~320–390 MB/s sustained, `bench/resource_sampler.py` measuring GPU/CPU/disk/RSS per phase), decode 8–16, GLM 190G:

| config | tok/s | disk read (decode) | GPU util | sync load |
|---|---|---|---|---|
| GLM 4 GiB, PILOT off | 0.037 | 386 MB/s | 13 % | 10.9 ms/miss |
| GLM 4 GiB, PILOT on (staging) | 0.011 | 326 MB/s | 8 % | 13.1 ms/miss |
| GLM 8 GiB, PILOT off | 0.063 | 316 MB/s | 11 % | 9.2 ms/miss |

Key findings:

- **Decode is I/O-bound at the SSD, not latency-bound**: GPU idles at 8–13 %, disk read saturates at ~330 MB/s. The working set per token (42 layers × 10 experts × 13 MB ≈ 5.5 GB) ÷ 330 MB/s ≈ 16.6 s/token ≈ **0.06 tok/s — the measured 8 GiB result sits on the physical I/O floor**. No software overlap can beat this without reducing bytes per token.
- **PILOT (router-lookahead prefetch) is strictly negative in this regime.** Workers read numpy slices into a staging buffer (0.49 ms/staged hit vs 10.9 ms sync — the mechanics work), but the router predicts one layer ahead and the disk is already saturated: 94 % of staged bundles were dropped unconsumed (~380 GB of wasted reads) and the demand path slowed by contention. A/B: 0.037 → 0.011 tok/s.
- **No inter-token expert reuse on GLM**: hit rate stays 0 % even with 606 slots (14/layer, 63 % of one token's per-layer demand) at temperature 0. GLM routers change completely between tokens (unlike Qwen's 23–32 % inter-token hits at 4 GiB).
- PILOT is therefore **default OFF** (`OMLX_EXPERT_STREAMING_PILOT=1` to opt in). It only pays when I/O is *not* saturated (warm cache, budget ≥ working set) — the exact regime where slipstream found prefetch didn't pay either.

Strategic conclusion: for models ≫ RAM the levers that matter are (a) **batch decode** (amortizes the per-token working set across requests), (b) budgets ≥ the batch working set, and (c) reducing bytes/token (lower-bit requant of cold experts). Prefetch/overlap alone cannot. Fase A below tested (a) and (b) directly — both failed on a 48 GiB box; the levers that remain are bytes/token and bandwidth.

## Fase A — capacity, MTP, batch (decision experiments)

Protocol: cold page cache between runs (`bench/cache_cool.py` touches ~72 % of available anon memory), PILOT off, `bench/resource_sampler.py` on, GLM 190 G unless noted.

### A1 — capacity sweep (diagonal-reuse hypothesis, Patterns Ob2)

| budget | hit | tok/s | phys at end | note |
|---|---|---|---|---|
| 8 GiB (ref) | 0 % | 0.063 | 14.2 G | the I/O-floor result |
| 16 GiB | 15.9 % | 0.040 | 19.4 G | disk **writes** 340–580 MB/s during decode (swap) |
| 24 GiB | 19.5 % | 0.053 | 22.2 G | still swap-bound, below 8 GiB |

- Inter-token reuse on GLM **exists** (0 % → 16 % → 19.5 % as capacity grows — the diagonal from Patterns Ob2 is real), but capturing ≥ 2 tokens of working set needs ≥ 16 GiB of heap, which pushes the 48 GiB box into swap: the *capacity point sits below the reuse point* on this hardware. The budget sweet spot stays ≤ 8 GiB.

### A2 — MTP (SP-MoE amortization)

- GLM-oQ4e 8 GiB + `--mtp`: 0.073 tok/s (+16 %) — within noise; the checkpoint has no `-mtp` suffix (draft weights likely stripped by the publisher).
- Qwen 4 GiB + `--mtp`: 0.326 tok/s ≈ the no-MTP cold single run. Draft steps fault *their own* experts → bytes/step rise faster than accepted tokens pay back, in this I/O-bound regime.
- Verdict: **MTP stays opt-in** (not default) for streaming models.

### A3 — concurrent (batched) decode, GLM 8 GiB, distinct prompts

| N | aggregate tok/s | per request | hit |
|---|---|---|---|
| 1 (ref) | 0.063 | 0.063 | 0 % |
| 2 | 0.053 | 0.026 | — |
| 4 | 0.043 | 0.011 | 0 % |

- Distinct prompts route to distinct experts: the per-step working set scales ~linearly with N while the SSD stays saturated — batch **amortizes nothing** here (the ≥ 1.8× criterion failed; aggregate declines monotonically). Batch would only pay when requests share routing (same domain/language) — worth re-testing with homogeneous traffic, not the default serving assumption.

### Fase A conclusion

On a 48 GiB Mac with one ~330 MB/s SSD, GLM-190G decode is pinned to the I/O floor and **every byte-adding strategy makes it worse** (MTP drafts, batch). The remaining levers are both physical: **bytes/token** (oQ2e requant of cold experts ≈ 2×) and **bandwidth** (multi-SSD striping, now supported — see below).

## Trade-offs

- **Prefill is burst-heavy**: a long prompt touches almost every expert once → many faults on first prefill. Keep the existing **paged SSD KV cache** on — a repeated prefix is restored from disk in milliseconds instead of re-faulting experts. The first prefill still warms the LRU, so the following decode hits more.
- **Decode sync per layer**: the router's `top-k` indices are read back to the CPU to decide which experts to fault. Like slipstream's ~200 µs/layer wake cost, this is the floor; batch decode multiplies the per-step working set and makes it *worse* (Fase A3: aggregate falls with N), so streaming models effectively stay single-request.
- **Quantized path**: expert weight + scales + biases are restored as a co-located bundle. Fused `gate_up_proj` is handled as a single streaming bank when present; otherwise `gate/up/down` are three independent banks. The output is bit-exact versus the resident path (gather indices are remapped to a compact mini-bank).
- **GLM memory bombs are fixed but still bounded**: without the per-layer eval+`clear_cache`, decode/prefill pins 42 layers × ~3.4 GB of mini-banks and swaps the machine (40–50 GB observed).

## Measuring

- `GET /admin/api/models` returns `expert_streaming_bytes` vs `expert_resident_bytes` so you can size the budget before enabling.
- `OMLX_EXPERT_STREAMING_PROFILE=1` prints a per-layer, per-stage profile (gate_eval / unique / load hit+miss / staged vs sync split / stack / wall ms) with hit rate per layer.
- `python bench/bench_expert_streaming.py --model qwen|glm --budget N --decode M [--out file.json]` reproduces the tables above (TTFT, steady-state tok/s, cache stats, profile, prefetcher stats, phys) and samples GPU/CPU/disk/RSS per phase via `bench/resource_sampler.py` (per-phase means/max saved into the result JSON).
- `python bench/bench_expert_batch.py --model qwen|glm --budget N --concurrency K --decode M` measures batched decode (Fase A3).
- `python bench/cache_cool.py --gb N` evicts the page cache without root between runs.

## Limitations

- No learned pin store (colibri's `.coli_usage` heat) yet — the LRU is cold after a restart. A sidecar that persists routing heat per model is queued. Note: inter-token reuse is model-dependent (Qwen hits 23–32 %, GLM 0 %) — pins pay on Qwen-like routing, not GLM.
- Async prefetch (PILOT) exists behind `OMLX_EXPERT_STREAMING_PILOT=1` but is **negative by default**: on a saturated SSD it wastes bandwidth (see phase-1 A/B above).
- Dual-SSD striping is supported: copy alternating shards to a second disk (`python bench/stripe_model.py --model <dir> --target <dir2>`) and run with `OMLX_EXPERT_STREAMING_EXTRA_ROOTS=<dir2>`. Mirrored shards are served from the stripe root, the rest fall back to the primary — no RAID, original dir untouched.
- TTFT on GLM-class models is high (prefill faults ~all experts once); KV snapshot helps repeated prompts only.

## References

- slipstream thesis + measurements: per-layer cache slots, 6.25 % hot-expert locality, decode attention near roofline, per-layer CPU wake floor.
- colibri expert atlas: routing heat is measurably structured and therefore cacheable.
