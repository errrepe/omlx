# Expert Streaming (SSD)

Run Mixture-of-Experts (MoE) models that are larger than your Mac's RAM by keeping only the hot experts resident and streaming the rest from SSD.

Inspired by [slipstream](https://github.com/dwijenpatel/slipstream) (Swift/Metal expert LRU from SSD) and [colibri](https://github.com/JustVugg/colibri) (learned pin store + multi-tier memory), ported to oMLX's Python/MLX stack.

## When to use

- **You have a large MoE** (e.g. `glm_moe_dsa` / Qwen3.6-35B-A3B) and a **16–24 GB Mac**. Without streaming the model needs `resident_bytes` (checkpoint × 1.05) — often above the wired limit. With streaming it needs `dense_bytes × 1.05` (page-cache-only default), so a 35B MoE that needs ~21 GB resident fits in ~5–8 GB.
- You care about **fitting, not single-stream speed**. Streaming is slower than fully resident and disables continuous batching for that model (one request at a time). Use the default resident mode when the model already fits.

## How it works

- **Dense stays resident**: attention, shared experts, embeddings, LM head — always in unified memory.
- **Experts live on SSD**: the stacked `switch_mlp.(gate|up|down)_proj` banks (`(E, O, I)` tensors) stay memory-mapped from the original safetensors files (MADV_RANDOM, like the Qwen4 PLE `DiskBackedShardedEmbedding`). No duplicate copy is made.
- **Per-layer demand loads**: on a batch, the union of routed experts is resolved per layer; hits in the (optional) LRU run immediately, misses fault the expert slice with one `os.pread` each on an 8-thread pool (QD8) and the mini-bank is assembled on the inference thread. Quantized scales/biases ride along. The default cache policy is **page-cache only** (no LRU) — see Fase B below; `expert_streaming_budget_gib > 0` re-enables a bounded LRU.
- **One budget, auto-forced**: set `Expert Streaming` on in the per-model settings (or leave the budget empty for the page-cache-only default). If `resident_bytes > ceiling ≥ streaming_bytes`, oMLX auto-enables it and shows an amber "auto-enabled" hint — the same pattern as `qwen4_ple_ssd_offload`.

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
| `expert_streaming_budget_gib` | float? | `null` / `0` = **page-cache only** (default; no app-level LRU), `>0` = fixed LRU heap (opt-in), clamp `0–64` |
| `expert_streaming_topk_threshold` | float? | `null` / `>= 1.0` = exact routing (default, bit-exact); `0.05–0.95` = adaptive top-k mass truncation (approximate, changes outputs) |

All are load-time settings: toggling unloads (and re-loads if pinned) the model. The runtime signature includes the *effective* (forced-or-requested) value and the threshold when `< 1.0` (it changes outputs), so the `GET /admin/api/models` capability flags `expert_streaming_supported / forced / reason / *_bytes / moe_layers / per_expert_bytes` always reflect what would actually run.

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

### A2 — MTP (SP-MoE amortization) — SUPERSEDED, see Fase B retest

- GLM-oQ4e 8 GiB + `--mtp`: 0.073 tok/s (+16 %) — within noise; the checkpoint has no `-mtp` suffix (draft weights likely stripped by the publisher).
- Qwen 4 GiB + `--mtp`: 0.326 tok/s ≈ the no-MTP cold single run. Draft steps fault *their own* experts → bytes/step rise faster than accepted tokens pay back, in this I/O-bound regime.
- Fase A verdict "MTP stays opt-in" was an **access-method artifact**: the mmap+LRU stack put wall/call at 30 ms, so the cycle's 3.4 forwards/token (164 calls / 48 layers) dominated. Retested after pread + page-cache-only (B2) — see Fase B below: MTP is now a **win (+27–37 %)**.

### A3 — concurrent (batched) decode, GLM 8 GiB, distinct prompts

| N | aggregate tok/s | per request | hit |
|---|---|---|---|
| 1 (ref) | 0.063 | 0.063 | 0 % |
| 2 | 0.053 | 0.026 | — |
| 4 | 0.043 | 0.011 | 0 % |

- Distinct prompts route to distinct experts: the per-step working set scales ~linearly with N while the SSD stays saturated — batch **amortizes nothing** here (the ≥ 1.8× criterion failed; aggregate declines monotonically). Batch would only pay when requests share routing (same domain/language) — worth re-testing with homogeneous traffic, not the default serving assumption.

### Fase A conclusion

On a 48 GiB Mac with one ~330 MB/s SSD, GLM-190G decode is pinned to the I/O floor and **every byte-adding strategy makes it worse** (MTP drafts, batch). The remaining levers are both physical: **bytes/token** (oQ2e requant of cold experts ≈ 2×) and **bandwidth** (multi-SSD striping, now supported — see below).

> **Correction (post-Fase A):** the "I/O floor" above was an artifact of the
> access method, not the hardware. The 4 TB external SSD is a ~3.7 GB/s
> NVMe; `mmap` + `MADV_RANDOM` cold page faults deliver only 0.2–0.4 GB/s.
> Switching to one `os.pread` per contiguous expert slice lifted decode from
> 0.063 to 0.336 tok/s (5.3×) — see the next section. Byte-adding strategies
> (MTP drafts) also flipped sign once the per-call cost dropped — see B5.

## pread + parallel demand loads (the real fix)

Micro-benchmark on a GLM oQ4e shard (40 × 4 MB expert slices, cold page cache):

| method | cold GB/s |
|---|---|
| mmap + MADV_RANDOM (page faults) | **0.35** |
| `os.pread` (single contiguous read) | **10.6** |

`_ShardReader.expert_slice` now uses one `os.pread` per expert slice, and the
quantized streaming linear resolves the whole per-layer demand set with a
thread pool (`_EXPERT_IO_POOL`, 8 workers, QD8) — workers return raw numpy
slices, MLX promotion stays on the inference thread.

GLM 190G oQ4e, 8 GiB budget, decode 16, cold cache, PILOT off:

| load path | tok/s | TTFT | disk (decode) | load ms/call | GPU |
|---|---|---|---|---|---|
| mmap faults (Fase 0/1) | 0.063 | ~200 s | 330 MB/s | ~190 ms | 11 % |
| pread, serial | 0.249 | ~40 s | 1.4 GB/s | 22.5 ms | 29 % |
| pread, 4 workers | 0.313 | 22.9 s | 1.85 GB/s | 15.4 ms | 35 % |
| pread, 8 workers | **0.336** | 22.8 s | 1.85–2.3 GB/s | 14.7 ms | 35 % |

Qwen 4 GiB cold single generation: 0.853 tok/s (was 0.30). PILOT remains
negative even on the fast path (38 ms vs 30 ms wall/call — 89 % of staged
bundles dropped), so it stays default-off. Batch decode still amortizes
nothing (see Fase A3). Remaining bottlenecks: disk random-read ceiling
(~2.3 GB/s of the 3.7 sequential spec) and the mini-bank assemble/promote
(~5 ms/call); bytes/token (oQ2e) is the next real lever.

## Fase B — FlashNext comparison (page-cache era)

Ported the transferable techniques from [macqwen-releases](https://github.com/1architect/macqwen-releases) ("FlashNext", MIT). Their README claims 1.9 tok/s on Qwen3.8-Flash-Next; their own code comments pin down the components: threshold-1.0 exact = 0.53, 0.85 mass truncation = 0.94 ("the cliff"), 1.9 = 0.85 truncation + mlock pins + warm file cache + steady state. Our bit-exact pread path already measured 0.853 cold where they measure 0.53 exact.

### B1 — fixed-cost cuts (bit-exact) — commit 7e958a6

- BF16 promoted via `mx.array(v).view(mx.bfloat16)` bit-reinterpret: ~9× faster on 4 MB slices, and it matches `mx.load` exactly — the old `shift → f32 → astype` roundtrip flushed bf16 subnormals to zero (Metal FTZ), i.e. the *old* path was the inexact one.
- One host sync per MoE layer: the first streaming linear builds a shared `_RemapPlan` (eval + unique + compact remap via `np.searchsorted`), gate/up/down reuse it. (MLX 0.32 refuses zero-copy CPU dlpack — `mx.from_dlpack(np, copy=False)` — so the view-reinterpret is the practical promote.)
- Demand reads sorted by ascending expert id (= ascending file offset, row-major banks).
- Profile (Qwen 4G cold): wall/call 21.8 → 9.2 ms; stack bucket 5.4 → 0.5 ms. Decode is now **disk-bound** — remaining levers are fewer bytes (truncation) and fewer disk reads (page cache / pins).

### B2 — page-cache-only is the new default — commit cd10501

Budget semantics: `null`/`0` = no app-level LRU — expert reuse rides the OS file cache (clean, evictable pages, never swapped). `>0` = fixed LRU heap (opt-in). Admission charges dense bytes only for the default (file cache is not committed memory → more load headroom).

Cold A/B on the 48 GiB Mac (16-token decode):

| config | tok/s | RSS (decode avg) |
|---|---|---|
| Qwen LRU 4G | 0.834 | 7.3 GB |
| Qwen **page-cache only** | **1.007** (warm 1.133) | 5.6–5.8 GB |
| GLM LRU 8G | 0.363 — 0% hit rate, 8 GB of bundles pinned in RSS | 11.7 GB |
| GLM **page-cache only** | **0.381** | 4.8 GB |

Even at a 14.5% LRU hit rate (Qwen) the heap loses: it pins RSS that the OS could have used for page cache. Matches FlashNext's finding that their app-level LRU always lost.

### B4 — adaptive top-k truncation (opt-in, approximate) — commit 15071667

`expert_streaming_topk_threshold`: after top-k selection, keep the smallest score-descending prefix whose relative mass reaches the threshold; dropped slots reuse the top expert (duplicates collapse in the streaming plan — no extra I/O); kept scores renormalize to the original total top-k mass. `null`/`>= 1.0` bypasses everything — bit-exact by construction (verified: identical generated text on the real checkpoint with `--topk 1.0`).

Cold sweep (16-token decode, page-cache only):

| threshold | Qwen | GLM |
|---|---|---|
| exact | ~1.0 | 0.381 |
| 0.85 | 1.091 | **0.485 (+27%)** |
| 0.70 | 1.197 | — |

Outputs diverge by design below 1.0 (FlashNext measured 7/10 identical tokens at 0.70); the WebUI/Swift hint carries that warning.

### B3 — mlock pins (opt-in) and warm-only prefetch (experimental)

- **Pins** (`OMLX_EXPERT_STREAMING_PIN=1`): observe routing for the first 8 decode calls, then `mlock` the page-aligned file ranges of the most frequent experts per layer within `OMLX_EXPERT_STREAMING_PIN_GIB` (default 1.25 GiB, `OMLX_EXPERT_STREAMING_PIN_TOKENS` for the window). Zero-copy — the locked pages ARE the file cache pages — but they become wired memory. The mapping address is obtained via `PyObject_GetBuffer` (read-only mmap cannot expose a writable buffer). Qwen 48-token decode: **1.538 → 1.764 tok/s (+15%)**; GLM short decode: neutral (13 MB experts → the budget covers ~2 experts/layer, and the 8-token observation eats half a 16-token run).
- **Warm-only prefetch** (`OMLX_EXPERT_STREAMING_WARM=1`): before a layer's demand loads, fire discarded reads for the previous token's next-layer experts (independent routing repeats ~35% across adjacent tokens). On the 48 GiB Mac: **neutral to negative** (1.531 vs 1.538 alone; drags pins 1.764 → 1.624) — the QD8 demand path already saturates the NVMe, and warming adds read traffic. It exists for the 16 GB-class case FlashNext targets (small page cache, less demand pressure); leave it off otherwise. The old PILOT staging prefetcher (default OFF) is superseded by this.

### B5 — MTP retest after the pread/page-cache stack: now a win

Upstream v0.6.3 fixed Qwen4 Lightning MTP weight detection (#3200: qwen4_exp binds only the embedded `mtp.*` head; the draft layer carries its own `switch_mlp` bank, so the draft pass streams its experts like any other layer). Retested on the Fase B stack — Qwen, 0G page-cache-only, 48-token decode, cold:

| config | tok/s | vs no-MTP (1.538) |
|---|---|---|
| MTP | 1.958 / 2.110 (with profile) | **+27–37 %** |
| MTP + pins | 2.058 | **+34 %** |

Why it flipped: the draft/verify cycle runs **3.4 target forwards per generated token** (164 MoE-layer calls / 48 layers — acceptance is modest), but the B1/B2 stack cut wall/call from 30 ms to **3.97 ms**. Per-token cost went from ~4.9 s (unpayable) to ~0.65 s of MoE I/O, and the wider verify reads reuse page-cache-resident experts from the draft rounds (`sync_ms_per_load` 0.16 ms). TTFT unchanged (~10 s cold). Conclusion: **enable Lightning MTP by default for streaming qwen4_exp checkpoints that ship `mtp.*` weights**; keep it opt-in elsewhere (GLM has no draft weights, DeepSeek untested on the new stack).

### Recommendation matrix

| hardware | budget | pins | threshold | MTP |
|---|---|---|---|---|
| 48 GB+, model >> RAM | default (0) | optional (+15% Qwen) | 1.0 exact | on (qwen4_exp with `mtp.*`) |
| 48 GB+, quality flexible | default | optional | 0.85 | on (qwen4_exp with `mtp.*`) |
| 16 GB-class | default | `PIN=1` | 0.85 (+ `WARM=1` to test) | on (qwen4_exp with `mtp.*`) |

## Fase E — bottleneck experiments (QD, coalescing, learned pins, MTP tuning, ANE)

All runs cold (28 GB `cache_cool`), Qwen 0G page-cache-only unless noted; run-to-run noise ~±5%.

### E1 — QD16 is the new pool default (+34%)

| QD | tok/s (48-tok decode) |
|---|---|
| 8 (old default) | 1.538 |
| **16** | **2.061** |
| 32 | 2.097 (plateau) |

Disk read max ~2.5 GB/s of the 3.7 GB/s sequential spec; short-prompt TTFT 10.4 → 8.9 s. `OMLX_EXPERT_STREAMING_QD` overrides.

**Run coalescing** (consecutive expert ids → one `pread` per bank key, `_ShardReader.expert_run`, verified bit-exact): +4% decode (adjacency ~2%), ~2% at 8k prefill (the union already covers ~the full bank). Kept — free and no-regression; `OMLX_EXPERT_STREAMING_COALESCE=0` disables.

**Long-prompt prefill finding**: 8k prompt = ~90 s TTFT streaming ~66 GB (full expert bank per layer) — but disk only sustains 1.85-1.89 GB/s of it and the GPU sits at 81-85%: **long-prompt prefill is ~60% GPU/CPU-bound** (expert QMM over ~7.4k positions + mini-bank assemble/promote). Decode immediately after a long prefill runs ~4.2 tok/s (page cache hot). The real long-prompt levers are chunked prefill / assemble cost — not bandwidth.

### E2 — MTP tuning on the QD16 stack

MTP's edge shrank from +27-37% (QD8) to +4-5% (QD16): the pool resize cut the per-forward I/O cost MTP was amortizing. Sweep (`--mtp-block`, no-MTP baseline 2.135):

| block | 2 | 3 | 4 | 8 |
|---|---|---|---|---|
| tok/s | 2.097 | 2.165 | 2.213 | 2.248 |

Pins no longer stack with deep drafting (block 8 + pins: 2.012/2.069, reproduced twice). Guidance: MTP stays default-off; when enabled use `draft_block_size` 4-8 with pins off — or QD16 alone, which beats MTP alone.

### E3 — learned pin store (persisted routing frequencies)

`OMLX_EXPERT_STREAMING_PIN_PROFILE=<path>`: PinController saves observed per-layer frequencies after pinning; the next load reloads them and wires the hot set from token 1 (no 8-call window). The mlock also populates the pages, so the reload doubles as a *targeted* warmup.

| arm | TTFT | decode |
|---|---|---|
| baseline | 8.3-8.9 s | 2.06-2.14 |
| (b) deterministic warmup 1.25 G post-load | 7.1-8.0 s | 1.71-1.93 — ~0/negative |
| (c) learned pins loaded at start | **6.9 s (−22%)** | **2.273 (+6-10%)** |

(b) confirms the prediction: warmup uncorrelated with the first request's routing is wasted I/O. (c) is the optimistic bound (profile from the same prompt); arbitrary first requests gain proportionally to overlap. Server follow-up: save on unload, reload on load, per model.

### E4 — ANE prefill: attaches to qwen4_exp, not the streaming lever

Required building the native extension (nanobind 2.13.0 ABI-matched to MLX 0.32.0; upstream `setup.py` cannot pass `-DPython_EXECUTABLE` through `CMAKE_ARGS` for paths with spaces — space-free venv symlink workaround). With it present, ANE attaches to the vendored qwen4_exp **without any port** ("48 MLP + 36 GDN procedures into 2 instance-pinned ANE programs").

Cold TTFT: 2k 25.7 → 26.3 s (no gain); 8k 89.7 → 87.1 s (~3%). Prefill is GPU-bound (81-85%) in both arms — time goes to expert QMMs and assembly, which ANE does not touch (dense MLPs + GDN projections only). Keep ANE for resident models; streaming TTFT needs chunked prefill / assemble work instead.

### E5 — GLM on the QD16 stack

GLM 0G decode 64: **0.697 tok/s vs 0.381 measured pre-E1 (+83%)** — large experts benefit more from deeper queues. TTFT 23 → 18.5 s. Pins 2.5 GiB: neutral (2.5 G = ~4-5 experts/layer of 288 — too sparse). GLM's levers remain QD16 (now default) + topk 0.85 (+27%).

## Trade-offs

- **Prefill is burst-heavy**: a long prompt touches almost every expert once → many faults on first prefill. Keep the existing **paged SSD KV cache** on — a repeated prefix is restored from disk in milliseconds instead of re-faulting experts. The first prefill still warms the LRU, so the following decode hits more.
- **Decode sync per layer**: the router's `top-k` indices are read back to the CPU to decide which experts to fault. With pread + the parallel demand loader this sync overlaps useful I/O (QD8); batch decode still multiplies the per-step working set and makes it *worse* (Fase A3), so streaming models effectively stay single-request.
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
- Dual-SSD striping is supported: copy alternating shards to a second disk (`python bench/stripe_model.py --model <dir> --target <dir2>`) and run with `OMLX_EXPERT_STREAMING_EXTRA_ROOTS=<dir2>`. Mirrored shards are served from the stripe root, the rest fall back to the primary — no RAID, original dir untouched. Note: only pays if the second disk is *fast* — in our setup the 4 TB NVMe (~3.7 GB/s) is the fast one and the 2 TB (~875 MB/s) would slow reads down.
- TTFT on GLM-class models is high (prefill faults ~all experts once); KV snapshot helps repeated prompts only.

## References

- slipstream thesis + measurements: per-layer cache slots, 6.25 % hot-expert locality, decode attention near roofline, per-layer CPU wake floor.
- colibri expert atlas: routing heat is measurably structured and therefore cacheable.
