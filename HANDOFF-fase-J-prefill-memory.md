# Handoff — Fase J: Otimização de Memória do Prefill (MoE Expert Streaming)

> **Data:** 2026-08-30
> **Sessão:** Continuação da execução do Plano Fase J (otimizações de streaming de experts do SSD)
> **Estado:** Código implementado até C13; nova frente de otimização de memória do prefill identificada e planejada

---

## 1. Contexto da Sessão

### 1.1 O que foi feito nesta sessão

Esta sessão continuou a execução do **Plano Fase J** — um plano de 14 commits (C0→C13) para otimizar a latência de streaming de experts MoE do SSD no servidor de inferência `omlx` (baseado em MLX). O trabalho acumulado até o início desta sessão:

- **C0–C13:** Todos os commits foram implementados, testados e commitados. A documentação foi atualizada (`docs/expert-streaming.md`).
- **Benchmarks:** Executados com sucesso em Qwen (modelo canônico). GLM está bloqueado por hardware (prefill requer ~46 GiB, máquina tem 48 GiB total).
- **Diagnóstico de memória:** Confirmado que o N-gram está em `mmap` (streaming do SSD) e **não é o culpado** pelo alto uso de memória. O problema é o **Metal/MLX buffers** durante o prefill (~34,5 GiB de `IOAccelerator`).

### 1.2 Evidências coletadas

| Métrica | Valor |
|---------|-------|
| Physical footprint (pico) | 35,7 GiB |
| IOAccelerator (graphics) | 34,5 GiB |
| Mapped file virtual | 143,7 GiB |
| Mapped file resident | 60,5 MiB |
| MLX cache durante prefill | 30,36 GiB |
| MLX cache durante decode | 0,06 GiB |
| Swap usado (global) | 5,80 GiB |

**Conclusão do diagnóstico:** O pico de memória ocorre durante o **prefill**, não durante o decode. O N-gram (tabela de embeddings) está corretamente em streaming via `mmap` e não está carregado integralmente na RAM.

### 1.3 Arquivos-chave modificados

- `omlx/patches/expert_streaming/streaming_switch.py` — C2, C3, C4, C6, C10, C12
- `omlx/patches/expert_streaming/shard_bank.py` — C1, C2 (read_expert_into)
- `omlx/patches/expert_streaming/warmer.py` — C5, C7, C8
- `omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py` — C9
- `omlx/patches/expert_streaming/qwen35_stream_eval.py` — C9
- `omlx/scheduler.py` — C11
- `bench/bench_expert_streaming.py` — C0, single-request protocol, token-ID gate
- `omlx/engine/vlm.py` — Propagação de token IDs
- `docs/expert-streaming.md` — C13, documentação de resultados
- `tests/test_expert_streaming.py` — Testes para C1, C2, C3, C12
- `tests/test_vlm_engine.py` — Testes de propagação de tokens

### 1.4 Commits criados

```
70742ea7 bench: record optimized streaming measurements
7b787641 perf: add shared layer I/O kill switch
9cfb40a7 perf: reuse routing plan indices for bias gathers
e43dcfff perf: complete async seed and shared layer I/O
5ef31dd6 perf: execute Fase J streaming optimizations
26e124c7 bench: add single-request memory-aware protocol
7b10a94c docs: correct benchmark TTFT units
```

### 1.5 Resultados de benchmark (Qwen, estado otimizado combinado)

| Braço | TTFT | tok/s | hit_rate |
|-------|------|-------|----------|
| M0 A0 (baseline) | 198,83 s | 0,3004 | 0 |
| Otimizado A0 | 84,27 s | 0,4514 | 0 |
| M0 B3a (baseline) | 106,42 s | 0,3451 | 0,0877 |
| Otimizado B3 | 85,20 s | 0,4141 | 0,0323 |

**Nota:** O `hit_rate` do B3 mudou de 0,0877 para 0,0323 — precisa ser investigado antes de claims definitivos de performance.

### 1.6 Novo protocolo de benchmark

Foi adicionado `--single-request` ao `bench/bench_expert_streaming.py`:
- Usa uma única requisição `stream_chat` (não faz segundo prefill)
- Mede TTFT até o primeiro token
- Mede decode após o primeiro token
- Propaga token IDs reais (`bit_exact_kind=tokens`)

---

## 2. Problema Identificado: Pico de Memória no Prefill

### 2.1 Onde a memória fica retida (evidência no código)

**A. `_LayerLoadContext` mantém todos os bancos NumPy vivos simultaneamente**
- `streaming_switch.py:472-512` — `ensure()` preenche `bundles` para **todos** os lineares (gate + up + down) de uma vez
- Cada linha é uma *view* sobre o banco NumPy inteiro, mantendo-o vivo
- Bancos de gate + up + down ficam residentes do início da camada até o retorno: **~3× U × per_expert_bytes** de RAM por camada

**B. Grafo lazy do chunk inteiro (dominante em qwen4_exp)**
- `Qwen4ExpDecoderLayer.__call__` (`qwen4_exp/language.py:1496-1544`) **não tem** verificação de `_stream_eval`
- `qwen35_stream_eval.py` só envolve `Qwen3_5MoeDecoderLayer` — **inerte em qwen4_exp**
- O scheduler só faz `mx.eval` no final do chunk
- Cópias promovidas por linha e outputs de stack de **todas** as camadas MoE permanecem vivos como inputs não avaliados do grafo
- Pico F1 ~26 GB documentado: 48 layers × ~215 uniq × ~2.5 MB

**C. `mx.stack` gera double-buffer**
- Linhas (U cópias MLX) + saída do stack coexistem durante o kernel → **transient 2× bank** por projeção

**D. Linhas NumPy no LRU fixam bancos inteiros**
- `streaming_switch.py:1126-1130` armazena views `raw`; evictar uma entrada não libera até a última linha do banco ser evictada
- Intencional (F2) e limitado pelo orçamento LRU — **não mexer**

### 2.2 Por que TurboQuant 8-bit não resolve

- O patch foi ativado corretamente (`TurboQuant KV cache enabled: 8.0 bits`)
- O teste terminou com `rc=143` sem JSON válido
- KV estimado: 27,28 KB/token × 1.898 tokens ≈ **52 MB** (insignificante vs 34,5 GiB de Metal)
- O problema dominante é o **transient do prefill** (~62,49 GiB predito), não o KV

---

## 3. Plano de Otimização Proposto

### 3.1 Visão geral

Transformar o prefill em um **pipeline de memória limitada**, mantendo o plano de roteamento compartilhado mas eliminando a retenção simultânea de bancos de múltiplas projeções.

```
┌─────────────────────────────────────────────────────────────┐
│  Prompt completo → não materializar tudo de uma vez         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Chunk adaptativo: 512 → 256 → 128 tokens                   │
│  (reduz tamanho do temporário Metal)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Gate   │ →  │   Up    │ →  │  Down   │
│ carrega │    │ libera  │    │ carrega │
│ e avalia│    │ o banco │    │ só então│
└─────────┘    └─────────┘    └─────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Limite de working set: Metal + ativações + experts ≤       │
│  orçamento                                                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Etapas de implementação

#### Etapa A: Tilear o demand set em `StreamingQuantizedSwitchLinear.__call__`

**Objetivo:** Dividir `plan.uniq_list` em tiles de tamanho fixo para limitar o banco de experts por projeção.

**Implementação:**
- Dividir `plan.uniq_list` em tiles (padrão 32-64 experts)
- Limitar bytes do tile ≤ `_BANK_MAX_BYTES` (256 MiB)
- Por tile:
  - Fatia contígua de `x`/`remapped` (offset remapped pelo início do tile)
  - Carregamento via ladder existente
  - Montagem com **promoção única** quando all-miss: `mx.array(bank).view(...).reshape(U, *per_shape)` em vez de U cópias + `mx.stack`
  - `gather_qmm` na fatia
  - **`mx.eval(tile_out)`** (ou `async_eval` por flag) e drop refs
- Concatenar saídas em ordem ascendente
- Bias por tile via `self._bias[uniq_mx[tile]]`

**Ganho:** Elimina o double-buffer do `mx.stack` e reduz o footprint de montagem pela metade.

**Código atual relevante:** `streaming_switch.py:835-873` (`_load_expert_bank_np`), `streaming_switch.py:1140-1158` (promoção e stack)

#### Etapa B: Remover retenção union de `_LayerLoadContext`

**Objetivo:** Manter sobreposição de I/O sem reter bancos de todas as projeções simultaneamente.

**Implementação:**
- Substituir `_LayerLoadContext.ensure()` por prefetch assíncrono NumPy byte-capped (ex: 2 bancos)
- Disparar leitura de `down` enquanto `up`/`gate` promove/computa
- Linhas NumPy são promovidas e descartadas **por projeção**
- Kill-switch env espelha `_LAYER_BARRIER_ENV`
- Manter `prefill_bypass` exatamente onde está (nunca dentro de `_load_expert_bundle`)

**Ganho:** Elimina a retenção de ~2 bancos de projeção em RSS por camada.

**Código atual relevante:** `streaming_switch.py:472-512` (`_LayerLoadContext`), `streaming_switch.py:1269-1277` (anexação do ctx)

#### Etapa C: Definir pontos de boundary de eval

**Objetivo:** Forçar avaliação de resultados intermediários para liberar buffers do grafo lazy.

**Implementação:**
- Começar com **2 syncs por GLU** (após up+gate, após down) em vez de evals bloqueantes por tile
- Ou usar `mx.async_eval` + drop refs, que libera buffers conforme a GPU drena
- Qualquer `mx.clear_cache` deve passar por `_sync_and_clear_cache` (metal_sync.py:36-63)
- Aplicar o mesmo a `StreamingSwitchLinear` ou documentar escopo quantizado-only
- Opcionalmente patchar `Qwen4ExpDecoderLayer.__call__` para adicionar boundary `_stream_eval`

**Ganho:** Pool MLX estagna em ~1 camada de working set em vez de crescer 48×.

**Código atual relevante:** `qwen4_exp/language.py:1496-1544` (sem `_stream_eval`), `qwen35_stream_eval.py:88-96` (só Qwen3_5)

#### Etapa D: Threshold-gate limpezas de pool

**Objetivo:** Limpar o pool MLX apenas quando necessário, não incondicionalmente.

**Implementação:**
- Dentro do linear: `get_cache_memory() >= OMLX_EXPERT_STREAMING_CACHE_THRESH` (default 2 GiB) antes de `mx.clear_cache()`
- Limpezas de limite de chunk do scheduler tornam-se threshold-gated
- **Manter uma limpeza incondicional no final do prefill** para decode começar com pool limpo
- Todas as limpezas continuam em `_sync_and_clear_cache`
- Preservar fallback `get_cache_memory is None → clear`

**Ganho:** Evita overhead de limpezas desnecessárias mantendo controle de memória.

**Código atual relevante:** `scheduler.py:3544, 3750, 3797, 5242, 5665` (limpezas incondicionais atuais)

#### Etapa E: Atualizar contabilidade do guard (OFF por padrão — ver nota)

**Objetivo:** Fazer o scheduler refletir o novo pico de memória limitado por tile.

**Implementação:**
- Adicionar flag `boundary_active` a `streaming_guard_info` (setada em `__init__.py:642-646`)
- Fazer `_streaming_bank_bytes` cobrar `min(2, projections) × tile_bytes + ativação de uma camada` quando ativo
- Deixar o caminho medido por EWMA intocado

**Ganho:** O throttle/admission para de cobrar o pior caso de 26 GB e permite que o ganho apareça no comportamento.

> **Decisão de execução (2026-08-30): Etapa E fica OFF por padrão.** A medição
> prévia com E ativo divergiu do baseline em *token-ID bit-exactness* (primeira
> divergência no token 3 de 48), violando o critério de aceitação #1 (bit-exactness
> em primeiro lugar). Como a contabilidade do guard estruturalmente **não** deveria
> alterar saídas, se E for necessário ligar no futuro ele precisa ser re-verificado
> como *output-neutral*; se alterar os numerics, é um bug latente a corrigir à
> parte — não se liga E para contorná-lo. O ganho de comportamento (admission
> deixar de cobrar o pior caso de 26 GiB) fica condicionado a essa re-verificação.

**Código atual relevante:** `scheduler.py:3863-3882` (`_streaming_bank_bytes`), `scheduler.py:3952-3955` (`max()` no per_token)

---

## 4. Ganhos Adicionais Identificados

### G1: Montagem de banco com promoção única
No hot path de prefill (prefill_bypass on → all-miss), as linhas do tile vêm de um único banco NumPy fresco, então `w_bank` pode ser construído com **uma** chamada `mx.array(bank).view(...).reshape(U, *per_shape)` em vez de U cópias `mx.array` + `mx.stack`. Isso **divide pela metade** o footprint de montagem e remove U−1 alocações Metal por projeção.

### G2: Boundary de 6 linhas para qwen4_exp
Adicionar o boundary `_stream_eval` em `Qwen4ExpDecoderLayer.__call__` (antes do return em `language.py:1544`) obtém a maior parte do ganho cross-layer sem tocar nos lineares. Fazer ambos (boundary na camada + nos lineares) é redundância benéfica.

### G3: Atualizar termo estático de banco no guard (OFF por padrão — ver nota na Etapa E)
Sem isso, o scheduler continua rejeitando e sub-dimensionando como antes, e o ganho nunca aparece no comportamento.

### G4: Paridade no variant BF16
`StreamingSwitchLinear.__call__` (linhas 636-687) nunca usa `plan.ctx` e retém U arrays mx por projeção da mesma forma — aplicar o mesmo tratamento ou documentar escopo quantizado-only.

---

## 5. Riscos e Mitigações

| Risco | Descrição | Mitigação |
|-------|-----------|-----------|
| **R1** | Bit-exactness de `gather_qmm` em tiles | Gate de token-ids idênticos; concat em ordem ascendente é provadamente correta |
| **R2** | Stub de tracing OQ (`mx.eval` é noop) | Eval dentro do linear não é load-bearing para valores, apenas para lifetime |
| **R3** | Ladder de carregamento por tile | Manter cache hit → prefetch → bank read → fallback legacy; kill-switch |
| **R4** | Hooks e contratos alterados | `_warm_pins.on_layer_start`/`on_layer_plan`, `_trace_row`, `weighted_sum` — todos preservados |
| **R5** | Double boundary GLM | GLM já avalia por camada; evals extras são baratos; medir TTFT nos braços GLM |
| **R6** | Ruído no ledger de reclaim | Verificar que predictor não cobra em dobro após chunks com pool retido |
| **R7** | Gap DFlash (pré-existente) | DFlashEngine não tem termo de banco; verificar se serve modelo streaming |

---

## 6. Critérios de Aceitação

A mudança só deve ser considerada segura se:

1. ✅ Gate por token IDs continuar passando (`bit_exact_kind=tokens`)
2. ✅ Texto e 48 token IDs forem iguais ao baseline
3. ✅ Pico de `IOAccelerator` cair substancialmente (esperado: ~29 GiB → low single-digit GiB)
4. ✅ Swap delta por execução diminuir
5. ✅ Não houver crescimento de `load_ms` desproporcional
6. ✅ Throughput não cair mais do que o custo aceitável dos tiles
7. ✅ Caminho decode continuar usando tiles maiores/LRU, sem contenção agressiva do prefill
8. ✅ Hit-rate inalterado a budget fixo
9. ✅ RSS não aumentar mais de 5%

---

## 7. Testes Necessários

- `tests/test_expert_streaming.py` existentes devem passar
- Novos testes:
  - Tile concat vs single-shot `mx.array_equal`
  - Single-promotion all-miss vs stack por-linha
  - Fallback ladder por tile
  - Contagens LRU após puts tileados
- Benchmark de bit-exactness por token-ids
- Braços Qwen e GLM (se possível)

---

## 8. Ordem de Execução Recomendada

```
1. Instrumentar por camada e projeção (active, cache, footprint, IOAccelerator)
2. A/B do C6 (ligado vs desligado) para confirmar retenção do ctx
3. Implementar Etapa B (remover retenção union do ctx)
4. Implementar Etapa A (tiles de experts)
5. Implementar Etapa C (boundary de eval)
6. Implementar Etapa D (threshold-gate limpezas)
7. Implementar Etapa E (atualizar guard) — **OFF por padrão** (ver nota na Etapa E: conflita com bit-exactness)
8. Rodar testes e benchmarks de validação
9. Documentar resultados em docs/expert-streaming.md
```

---

## 9. Arquivos para Modificar

| Arquivo | Mudança |
|---------|---------|
| `omlx/patches/expert_streaming/streaming_switch.py` | Tiles, promoção única, boundary de eval, ctx sem retenção union |
| `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py` | Adicionar boundary `_stream_eval` |
| `omlx/scheduler.py` | Threshold-gate limpezas, atualizar `_streaming_bank_bytes` |
| `omlx/patches/expert_streaming/__init__.py` | Adicionar `boundary_active` flag |
| `tests/test_expert_streaming.py` | Novos testes de tile e promoção única |
| `bench/bench_expert_streaming.py` | Instrumentação de memória por fase |
| `docs/expert-streaming.md` | Documentar resultados |

---

## 10. Referências

- **Plano original:** `PLAN-fase-J-streaming-otimizacoes.md`
- **Documentação de resultados:** `docs/expert-streaming.md`
- **Log de memória:** `.workbuddy-ai/memory/2026-08-30.md`
- **Benchmarks:** `bench/results/faseJ/`
- **Código principal:** `omlx/patches/expert_streaming/streaming_switch.py`

---

## 11. Notas Finais

- O N-gram **já está em streaming via mmap** e não é o problema
- TurboQuant 8-bit **não resolve** o problema dominante de memória
- O problema é o **working set do prefill no Metal** (bancos de experts + grafo lazy)
- A solução é **limitar o working set por projeção/tile**, não reduzir o N-gram ou o KV
- Manter o protocolo `--single-request` para medir sem segundo prefill escondido
- GLM permanece **não mensurável** nesta máquina (prefill requer ~46 GiB, máquina tem 48 GiB total)

---

**Fim do handoff.**
