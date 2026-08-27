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

Loading a glm5_next / qwen4_exp checkpoint with `expert_streaming_enabled` uses the lazy loader (`lazy=True`) and converts to streaming **before** `materialize_lazy_state` — the multi-hundred-GB MoE banks are dropped as lazy arrays instead of ever being materialized. GLM decoders additionally get `compile_ffn` disabled and a per-layer `mx.eval(out)` + `mx.clear_cache()` so the per-layer expert mini-banks (~3.4 GB at prefill) do not accumulate in the lazy graph / allocator and swap the machine.

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

## Trade-offs

- **Prefill is burst-heavy**: a long prompt touches almost every expert once → many faults on first prefill. Keep the existing **paged SSD KV cache** on — a repeated prefix is restored from disk in milliseconds instead of re-faulting experts. The first prefill still warms the LRU, so the following decode hits more.
- **Decode sync per layer**: the router's `top-k` indices are read back to the CPU to decide which experts to fault. Like slipstream's ~200 µs/layer wake cost, this is the floor; batching would multiply the working set and defeat the cache, so the scheduler caps streaming models to **one concurrent request**.
- **Quantized path**: expert weight + scales + biases are restored as a co-located bundle. Fused `gate_up_proj` is handled as a single streaming bank when present; otherwise `gate/up/down` are three independent banks. The output is bit-exact versus the resident path (gather indices are remapped to a compact mini-bank).
- **GLM memory bombs are fixed but still bounded**: without the per-layer eval+`clear_cache`, decode/prefill pins 42 layers × ~3.4 GB of mini-banks and swaps the machine (40–50 GB observed).

## Measuring

- `GET /admin/api/models` returns `expert_streaming_bytes` vs `expert_resident_bytes` so you can size the budget before enabling.
- `OMLX_EXPERT_STREAMING_PROFILE=1` prints a per-layer, per-stage profile (gate_eval / unique / load hit+miss / stack / wall ms) with hit rate per layer.
- `python bench/bench_expert_streaming.py --model qwen|glm --budget N --decode M [--out file.json]` reproduces the table above (TTFT, steady-state tok/s, cache stats, profile, phys).
- Future: per-model hit-rate / slots-in-use counters in the status payload (v1 currently counts via `ExpertLRUCache.stats`; expose is a follow-up).

## Limitations

- No learned pin store (colibri's `.coli_usage` heat) yet — the LRU is cold after a restart. A sidecar that persists routing heat per model is queued.
- No async overlap: expert loads wait on the GPU pipeline instead of prefetching the next layer's union (slipstream notes this pays exactly when the model doesn't fit — our case).
- No dual-SSD striping.
- TTFT on GLM-class models is high (prefill faults ~all experts once); KV snapshot helps repeated prompts only.

## References

- slipstream thesis + measurements: per-layer cache slots, 6.25 % hot-expert locality, decode attention near roofline, per-layer CPU wake floor.
- colibri expert atlas: routing heat is measurably structured and therefore cacheable.
