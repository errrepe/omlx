# Cache residency: diagnóstico e correção (Fase M4)

Pergunta de partida: *"o uso de memória não subiu muito — talvez ele não tenha
mantido experts suficientes em cache para bater o hit rate?"*

**Resposta: sim, e era bug de sizing, não de política nem de orçamento.**

Ambiente: M4 Pro 48 GB, `Qwen3.8-Flash-Next-JANG_4M` (48 camadas × 512 experts,
`per_expert = 2.822.400 B`, 3 projeções split), budget 4 GiB.

---

## 1. Causa raiz — `n_proj` com curto-circuito

`omlx/patches/expert_streaming/__init__.py` (~linha 744):

```python
n_proj = len(
    [a for a in ("gate_up_proj", "down_proj") if hasattr(g, a)]
    or [a for a in ("gate_proj", "up_proj", "down_proj") if hasattr(g, a)]
)
```

`down_proj` existe nos **dois** layouts, então a primeira lista nunca é vazia e o
`or` faz curto-circuito → **todo GLU split recebe `n_proj = 1` em vez de 3**.
Fused (`gate_up_proj` + `down_proj`) dava 2 → não era afetado.

Cascata: `_majority = 1` → `_want_slot = per_expert` (em vez de `per_expert // 3`)
→ `capacity = budget // per_expert` → **o `/3` aplicado duas vezes**.

| | capacity | slots/layer | experts/layer | bytes reais |
|---|---|---|---|---|
| quebrado | 1.521 | 31 | 10 | **1,33 GiB** de 4,00 |
| corrigido | 4.565 | 95 | 31 | **4,00 GiB** |

Corroborado por artefatos: `jang_4M_b4.json` (16:07) reporta 4565/95; os e2e da
Trilha A (19:50+, após `6d3bc913`) passaram a reportar 1521/31.

### Correções

1. `__init__.py` — discriminar por `hasattr(gate_up_proj)`, fallback `or 1`.
2. `warmer.py::_seed_lru` — `per_layer_cap // 3` hardcoded → `_projections_per_expert()`
   derivado de `linears_by_layer`. `per_layer_cap` é contado em **slots** (uma
   projeção cada), então o 3 fixo sub-semeava modelos fused em 1,5×.

---

## 2. Evicção per-layer era O(store) — agora O(1)

`ExpertLRUCache.put` varria **todo** o store para achar a vítima da layer assim
que cada layer sentava no seu cap. Micro-bench (capacity 4565, 48 layers, 23
puts/layer):

```
antes: 122,0 us/put  ->  134,66 ms por token de decode
depois:  2,48 us/put ->    2,74 ms por token   (49x)
```

Nunca apareceu antes porque o LRU só era escrito pelo seeder (praticamente zero
puts em produção). Correção: índice `self._layer_orders: dict[int, OrderedDict]`
(ordem de recência por layer) na classe base, mantido em `get` / `put` / `clear` /
`retain_hot` e no `governor._apply`; `_evict_layer()` O(1) com fallback para o scan
linear (subclasses com store próprio não mantêm o índice → ele pode ficar stale).

---

## 3. O LRU de experts **não** está no caminho quente do decode

Instrumentação nova (`CacheStats.puts`, `.retain_evicted`) deu a prova:

```
puts == size == 4464,  evictions == 0
```

Toda admissão veio do seeder. Os 297k misses são prefill com `cache_result=False`
(`prefill_bypass`). Decode resolve por `plan.ctx` → `_LayerLoadContext`:

- `_layer_ctx_mode()`: decode (`positions <= 64`) → **union**; prefill → rolling.
- `_ensure_union` lê o cache via `_split`, lê os misses do disco… **e nunca
  escreve de volta**. O `put` do `__call__` só existe no ramo `context_bundles is None`.

Ou seja: no decode o LRU era **somente-leitura**.

---

## 4. Write-back no union: melhora hit rate, piora vazão — fica desligado

Admitir o que o union lê (`put(proj.bundle_key(eid), raw)` após o `pool.map`):

| métrica | baseline | n_proj fix | **+ write-back** | + 2-strike adm |
|---|---|---|---|---|
| hit rate | 0,0193 | 0,0619 | **0,2261** | 0,0099 |
| size | 1.440 | 4.464 | 4.560 | 120 |
| evictions | 0 | 0 | 67.793 | 0 |
| puts | — | 4.464 | 72.353 | 120 |
| adm_drops | — | 0 | 0 | 140.898 |
| **tok_s** | 2,7093 | **2,8665** | **2,0682** | 2,8512 |
| decode_s | 35,43 | 33,49 | 46,42 | 33,67 |
| disk KiB/s | 843.702 | 949.696 | 798.718 (−16%) | 681.773 |
| **disco total (GiB)** | **29,9** | **31,8** | **37,1 (+17%)** | **22,9 (−28%)** |
| sys_cpu % | 21,17 | 16,19 | **33,63** | 24,23 |
| RSS max (GiB) | 7,24 | 4,89 | **9,30** | 8,84 |
| phys after decode (GiB) | 10,09 | 12,99 | 16,06 | 9,30 |

### Correção: o "−16% de disco" do write-back era uma TAXA, não bytes

A linha `disk KB/s` é `resources.phases.decode.disk_read_kib_s_avg` — uma
**média por segundo**. O write-back reduziu a taxa em 16% mas alongou o decode
em 39% (33,49 → 46,42 s). Em bytes absolutos (taxa × decode_s):

| braço | KiB/s | decode_s | total lido |
|---|---|---|---|
| baseline | 843.702 | 35,43 | 29,9 GiB |
| n_proj fix | 949.696 | 33,49 | 31,8 GiB |
| write-back | 798.718 | 46,42 | **37,1 GiB** |
| 2-strike | 681.773 | 33,67 | **22,9 GiB** |

Ou seja: **o write-back leu 17% MAIS bytes no total**, não 16% menos. O hit rate
mais alto (0,226 vs 0,062) não se traduziu em menos I/O absoluto — cada token
ficou 39% mais caro e a taxa de leitura caiu junto. Só o braço 2-strike (cache
praticamente vazio) reduziu bytes de verdade, e é ele que sustenta a conclusão
sobre page cache.

**Conclusão corrigida:** o write-back piora tudo que importa — vazão (−28%),
bytes lidos (+17%) e `sys_cpu` (16% → 34%). Não é "o objetivo foi atingido mas
custou caro": o objetivo **não** foi atingido. O mecanismo exato do custo
permanece em aberto; o que é medido é o resultado.

Fica **opt-in** via `OMLX_EXPERT_STREAMING_CTX_WRITEBACK=1`
(`_CTX_WRITEBACK_ENV`, default off).

### Correção 2: o "−28%" era ~2/3 bug de evicção — re-medido como −3,4%

Esta tabela foi medida **antes** do fix O(1) da evicção per-layer (§2). O braço
write-back teve 67.793 evictions a ~120 µs/put do scan O(store) ≈ **8,1 s dos
12,9 s** de penalidade. Re-medido depois do fix, 2 reps por braço
(`bench/results/wb_remeasure/`):

| métrica | off | on | delta |
|---|---|---|---|
| tok_s | 2,8570 | 2,7593 | **−3,4%** (dentro do ruído ±4%) |
| hit rate | 0,0618 | 0,2262 | 3,7× |
| **disco total** | **27,4 GiB** | **31,9 GiB** | **+16%** |
| rss_max | 13,75 GiB | 17,70 GiB | +3,95 GiB |

O custo em vazão praticamente desapareceu. O que **não** mudou — e é o número
que desqualifica o write-back — é o I/O: com 3,7× mais hits ele lê **16% mais
disco**, porque os 4 GiB do LRU saem do mesmo orçamento de memória unificada
do page cache do kernel, encolhendo o UBC. Ver
`bench/results/wb_remeasure/SUMMARY.md` §4.

### Efeito colateral: o cache de usuário compete com o page cache
O run com 2-strike (cache praticamente vazio, 120 entradas) teve a **menor**
leitura de disco de todos (681.773 KB/s, −28% vs. o n_proj fix). Sem cache de
usuário sobra memória para o UBC, que serve repetições ~13× mais barato que o
disco (medido na Trilha D: 0,42 ms morno vs 5,65 ms frio) e sem churn de
alocação.

---

## 5. Bug encontrado de quebra: filtro de admissão rejeita o seeder

`OMLX_EXPERT_STREAMING_ADMISSION=1` ativa `_admission_should_insert` (`c < 2` →
drop). Como o seeder escreve via `cache.put`, **todo put da semente é rejeitado**
(primeira referência): o cache fica em 120 entradas, `adm_drops = 140.898`,
hit rate 0,0099. O filtro precisa isentar puts explícitos de warm/seed.

---

## 6. Ruído de medição: o ganho de decode NÃO é resolvível nesta máquina

Três runs do **mesmo** caminho de código (write-back off; a evicção O(1) é
irrelevante porque `evict=0`):

| run | memória livre no preflight | tok_s | rss_max |
|---|---|---|---|
| n_proj fix | 23,4 GB | 2,8665 | 4,89 |
| 2-strike (cache vazio) | 27,8 GB | 2,8512 | 8,84 |
| FINAL (default) | **28,9 GB** | **2,6473** | 12,50 |

Spread ±4% em torno de 2,79. O baseline (capacity 1521, seed 1440) deu 2,7093 —
**dentro do ruído**. Mais memória livre não implicou mais vazão; `rss_max` variou
4,89 → 12,50 GiB entre runs idênticos.

**Conclusão honesta**: triplicar o cache residente (1,33 → 4,00 GiB) e o hit rate
(0,019 → 0,062) **não move o decode de forma mensurável**. O que é determinístico
e verificado:

- capacity 1521 → 4565 e residente 1,33 → 4,00 GiB (aritmética, verificada isolada).
- hit rate 0,0193 → 0,062/0,062 (contadores, reproduzido em dois runs).
- evicção per-layer 134,66 → 2,74 ms/token (micro-bench isolado, 49×).
- write-back: 2,068 vs 2,65–2,87 → perda clara, fora do ruído.

O que **não** é mensurável: qualquer efeito de decode ≤ ~8%. Portanto o critério
global "decode ≥ 2× baseline" **não** será atingido por residência de cache.

## 7. Onde está a alavanca restante

Oráculo de Belady na trace real (`bench/bench_residency_diagnosis.py`), média das
48 camadas:

| cap (experts/layer) | SCH (oráculo) | seed+LRU | LRU frio |
|---|---|---|---|
| 10 (bug) | 0,556 | 0,278 | 0,270 |
| 31 (corrigido) | **0,740** | 0,683 | 0,629 |
| 64 | 0,774 | 0,813 | 0,725 |

4 GiB suportam ~74% de hit rate. O medido é 6,2% — o gap não é tamanho de cache,
é o caminho `ctx` (seção 3) somado ao fato de que admitir custa mais do que
economiza (seção 4).

Próximos candidatos, em ordem de custo/benefício:
1. **Reuso de buffers** no caminho union (ler para um slab por slot em vez de
   alocar ~1 MB por admissão) — ataca diretamente o churn da seção 4.
2. Isentar o seeder do filtro de admissão (seção 5) e reavaliar 2-strike.
3. Deixar o UBC trabalhar: reduzir o budget do LRU e medir (o run de 2-strike
   sugere que menos cache de usuário pode significar menos disco).

---

## Arquivos

- `omlx/patches/expert_streaming/__init__.py` — fix `n_proj`.
- `omlx/patches/expert_streaming/warmer.py` — `_projections_per_expert()`.
- `omlx/patches/expert_streaming/streaming_switch.py` — índice per-layer O(1),
  `CacheStats.puts` / `.retain_evicted`, `_CTX_WRITEBACK_ENV`.
- `omlx/patches/expert_streaming/governor.py` — `_apply` usa `_evict_layer` O(1).
- `tests/test_expert_streaming.py` — 2 testes novos (write-back opt-in, consistência
  do índice per-layer).
- `bench/bench_residency_diagnosis.py` — simulador offline SCH/LRU.
- Rodadas: `nproj_fix_lru_4g.json` / `.log` (n_proj fix), `/tmp/m4_wb.json`
  (write-back), `/tmp/m4_adm.json` (2-strike), `/tmp/m4_final.json` (default).
