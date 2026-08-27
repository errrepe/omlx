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

## Supported models (v1)

| model_type | example | per-expert |
|---|---|---|
| `glm_moe_dsa` | GLM-5.2 MoE DSA | ~1.7 MB (BF16) / quantized packed |

Qwen MoE families reuse the same `SwitchGLU` seam and will follow the same path once their per-shard layout is validated.

## Settings

| field | type | notes |
|---|---|---|
| `expert_streaming_enabled` | bool | hardware-specific; excluded from profiles/templates |
| `expert_streaming_budget_gib` | float? | `null` / `0` or empty → auto `2 GiB`, clamp `0–64` |

Both are load-time settings: toggling unloads (and re-loads if pinned) the model. The runtime signature includes the *effective* (forced-or-requested) value, so the `GET /admin/api/models` capability flags `expert_streaming_supported / forced / reason / *_bytes / moe_layers / per_expert_bytes` always reflect what would actually run.

## UI

- **WebUI**: card `Advanced → Expert Streaming (SSD)` → toggle + `Cache budget (GiB)` input. Visible only when `supported`; disabled amber hint when `forced`. Lives under the `Experimental Features` header, before the Qwen ANE block.
- **macOS app**: `Model Settings → Advanced` → same toggle + conditional `Cache budget` row (auto-save, like `qwen4_ple_ssd_offload`). The `GiB` field clears to `null` when empty; values outside `0–64` are rejected.

## Trade-offs

- **Prefill is burst-heavy**: a long prompt touches almost every expert once → many faults on first prefill. Keep the existing **paged SSD KV cache** on — a repeated prefix is restored from disk in milliseconds instead of re-faulting experts. The first prefill still warms the LRU, so the following decode hits more.
- **Decode sync per layer**: the router's `top-k` indices are read back to the CPU to decide which experts to fault. Like slipstream's ~200 µs/layer wake cost, this is the floor; batching would multiply the working set and defeat the cache, so the scheduler caps streaming models to **one concurrent request**.
- **Quantized path**: expert weight + scales + biases are restored as a co-located bundle. Fused `gate_up_proj` is handled as a single streaming bank when present; otherwise `gate/up/down` are three independent banks. The output is bit-exact versus the resident path (gather indices are remapped to a compact mini-bank).

## Measuring

- `GET /admin/api/models` returns `expert_streaming_bytes` vs `expert_resident_bytes` so you can size the budget before enabling.
- Future: per-model hit-rate / slots-in-use counters in the status payload (v1 currently counts via `ExpertLRUCache.stats`; expose is a follow-up).

## Limitations (v1)

- Peak load still materializes the full checkpoint once to convert to streaming — a machine that cannot even hold the peak will OOM before conversion. A true `lazy=True` mmap-only load that skips the resident copy is the planned v1.1.
- No learned pin store (colibri's `.coli_usage` heat) yet — the LRU is cold after a restart. A sidecar that persists routing heat per model is queued for v2.
- No dual-SSD striping.

## References

- slipstream thesis + measurements: per-layer cache slots, 6.25 % hot-expert locality, decode attention near roofline, per-layer CPU wake floor.
- colibri expert atlas: routing heat is measurably structured and therefore cacheable.
