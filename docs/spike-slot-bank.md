# Spike: Slot-bank + grouped-GEMM (Fase 3 backlog)

**Status**: spike/scoping — wiring plan + supporting numbers, sem código. Fundações já existentes:
`deepseek_mxfp4_gather_qmm_blocks` + variantes pair/concat e `glm_moe_weighted_sum` em `omlx/custom_kernels/glm_moe_dsa/csrc/fused_moe.{h,cpp,metal}`, expostos em `fast.py` e consumidos por `switch_layers.py`/`deepseek_v32.py`.

## Numbers from the GLM-JANG decode (budget 1 GiB, prior 2.0, PROFILE=1, 42 layers × 12,726 calls, wall 62.6 s)

| stage | total s | share of wall | ms/call |
|---|---|---|---|
| load (LRU take → promote np→mx) | 26.97 | **43.1%** | 2.12 |
| stack (mini-bank assembly + graph build) | 17.83 | **28.5%** | 1.40 |
| gate_eval (route eval + d2h copy) | 14.10 | 22.5% | 1.11 |
| unique (np.unique remap) | 0.11 | 0.2% | 0.01 |

Hit 28% (32,261 hits / 82,630 misses global). O caminho custa ~3.5 ms/call em load+stack sozinhos — é daí que vem o teto de 1.1–1.3 tok/s.

## Diagnosis

Todo token paga `load+stack` **por chamada** mesmo em hit: o hit path toma U slices numpy do LRU e os re-promove a mx + re-stacka o mini-bank a cada chamada. O load é dominado pelo take do LRU + np→mx copy; o stack, pela montagem (U,O,I) + graph build. Ou seja: o custo por uso é proporcional ao número de chamadas × experts, não à novidade (misses). Com hit 28%, ~72% do tráfego de montagem é redundante — o mesmo expert é re-promovido e re-stackado token após token.

## Proposal: slot-bank

1. **Layout**: alocar S slots por (layer, proj) — buffer persistente mx (S, O, I) + escala/bias. Slot = residência de um expert. Mapa expert_id→slot_idx em numpy no host; MRU evict quando cheio.
2. **Insert**: no miss, escrever o slice direto no slot (uma cópia np→device, substituindo take+stack).
3. **Gather**: usar `mx.take`/indices do slot no `gather_qmm` existente (remap muda de rank-of-uniq para slot_idx — mesmo shape, zero mudança nos kernels do caminho atual). O `deepseek_mxfp4_gather_qmm_blocks` com block_meta/block_count (o path Metal direto) fica como fase 2 do spike, pois o JANG é affine-int8 (group 64) e o kernel affine (`deepseek_affine_gather_qmm_blocks`) já existe com a assinatura certa.
4. **Evict**: invalidar slot (mapa host) + overwrite no próximo insert; sem free/alloc de mx arrays.

## Esperado (bound calculado)

- Elimina o stack per-call nos hits: −17.8s de 62.6s wall (−28.5%) → 44.8s se load e gate não mudarem.
- Load em hit degrada para custo O(1) de map-lookup (o slice já está no device): praticamente zera o load_ms nos hits → load 26.97s × 72% hit-path share... na verdade com hit 28%: load já é 43% do wall **já** majoritariamente em hits — o take + promote acontece também nos hits. Se o slot-bank cobre 100% dos hits, remove ~28% das chamadas de load mais barato... conservador: −10 a −15s adicionais.
- Estimativa total: 62.6s → ~35–40s decode wall ⇒ 1.3 → **~1.9–2.2 tok/s** (mesmo sem mexer em gate_eval).

## Kill-gate

+10% tok/s no protocolo congelado (3 reps prior 2.0 short vs 3 reps exact). Se o slot-bank não entregar, reverter sem drama — nenhum path compartilhado é alterado (o cache LRU numpy permanece para o cold path / page-cache-only).

## Wiring points (já mapeados)

- `streaming_switch.py:2011` `StreamingQuantizedSwitchLinear.__call__` — resolve demand set, chama `_promote_banks` (2011–2493):
  - `_split` (917) → LRU hit/miss
  - ctx fast path (2041–2054) + Etapa A1b bank promote (2196–2214)
  - tier_single promote direto (2216–2229, 2450–2473) — **ponto de inserção do slot-bank**: substituir `_stack_tier`/`mx.stack` por take no slot-bank
  - `gather_qmm` final (2462–2473) — remap troca para slot_idx
- `shard_bank.py:1273+` `load_expert*`/`read_expert_into` — leitura do disco; inalterado (miss ainda lê, mas escreve no slot)
- `ExpertLRUCache` (`streaming_switch.py:660+`) — budget/capacity; o slot-bank usa budget próprio em device (S × (O,I) bytes × dtypes) e contabiliza no mesmo `budget_gib`
- Kernels: `deepseek_affine_gather_qmm_blocks(x, weight, scales, biases, block_meta, block_count, group_size, bits, variant)` em `fused_moe.h:55` — assinatura casa perfeitamente com o slot-bank (weight/scales/biases = slots; block_meta/block_count = mapa de posição)

## Riscos

- **Fidelidade**: take/gather no slot é bit-identical por construção (mesmos bytes, mesma ordem) — mas a invalidação de slot tem que ser à prova de staleness (mapa expert→slot no host, uma entrada por expert; sem duplicatas).
- **Memória**: slots residem em wired/device memory — com 45 layers × 2 projs × S... para GLM: 204 experts/layer? Não — checar `n_routed_experts` do config JANG: com budget 1 GiB e (O,I) do MoE GLM, S ≈ 40–80 experts globais. `mx.metal.set_cache_limit` / budget accounting já existe (`ExpertLRUCache.capacity`).
- **Arena de slots vs promove a cada call**: o ganho real é eliminar re-promoção de hits. Se o hit rate for baixo (28%), o ganho limita-se a ~1/3 do tráfego — por isso o gate de +10% é o critério de vida.
- **Desafio prévio documentado**: single-promotion parcial (scoreboard) mostrou concat ≈ stack em wall — **o custo está no tráfego de montagem, não nas alocações**. O slot-bank ataca exatamente o tráfego (sem tráfego nos hits, só o gather final).

## Estimativa de esforço

2–3 dias: (1) slot map + insert/evict no linear, atrás de env `OMLX_EXPERT_STREAMING_SLOT_BANK=1`; (2) remap slot_idx + take no caminho dual-tier e uniform; (3) bench A/B congelado + kill-gate; (4) se passar: testes de fidelidade (bateria 6 prompts greedy + ppl canário) + auditor + settings/ui plumbing se virar default.
