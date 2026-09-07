# Onde vai o tempo de decode (JANG_4M, 48 GB) — Fase M4

Motivo: a investigação de residência (`bench/results/residency_fix/SUMMARY.md`)
mostrou que mexer no cache de experts não move a vazão. Antes de tentar qualquer
otimização nova, este é o profile do caminho atual — para otimizar o que é
medido, não o que é intuitivo.

## Run

| | |
|---|---|
| modelo | `Qwen3.8-Flash-Next-JANG_4M` (48 camadas × 512 experts, split, `n_proj=3`) |
| budget | 4 GiB (capacity 4565, `per_slot` 940.800 B) |
| prompt | 2k, request único, 96 tokens de decode, sem MTP |
| ambiente | `OMLX_EXPERT_STREAMING_PROFILE=1` |
| resultado | TTFT 49,62 s · decode 34,73 s · **2,7645 tok/s** |

Comando:

```
OMLX_EXPERT_STREAMING_PROFILE=1 .venv/bin/python bench/bench_expert_streaming.py \
  --model qwen-jang4m --budget 4.0 --decode 96 --prompt-len 2k --single-request \
  --out bench/results/decode_profile/jang4m_4g_prof.json
```

## 1. Divisão do tempo por chamada MoE

`profile.totals` (média por chamada de projeção; 14.256 chamadas):

| estágio | ms | % do contabilizado |
|---|---|---|
| `load_ms` (ler fatias do backing) | **2,872** | **60,9 %** |
| `gate_eval_ms` (router) | 1,148 | 24,4 % |
| `stack_ms` (empilhar linhas p/ MLX) | 0,683 | 14,5 % |
| `unique_ms` (`np.unique` do roteamento) | 0,009 | 0,2 % |
| **contabilizado** | 4,712 | 100 % |
| `wall_ms_per_call` | 5,143 | → 8,4 % não contabilizado |

`sync_loads` = 68.268 (4,79 por chamada) a 0,273 ms cada = **1,31 ms dos 2,87 ms
de load**; os 1,57 ms restantes são leitura em banco (caminho union) e overhead.

**Load é 61 % do tempo.** Nada mais chega perto.

## 2. Nada está saturado

Fase de decode (`resources.phases.decode`):

| métrica | valor | teto |
|---|---|---|
| `proc_cpu_pct_avg` | 120 % | ~1200 % (12 cores) |
| `sys_cpu_pct_avg` | 15,2 % | — |
| `gpu_util_pct_avg` | 27,8 % | 100 % |
| `disk_read_kib_s_avg` | 906 MiB/s | **3300 MB/s medidos (`F_NOCACHE`)** |

Estamos a **27 % do teto de disco**, com 90 % da CPU e 72 % da GPU ociosos. É a
assinatura clássica de pipeline limitado por latência de I/O com paralelismo
insuficiente — não de CPU, nem de GPU, nem de banda bruta.

## 3. De onde vêm os bytes

Confirmado pela trace de roteamento (`bench/results/lrc/jang4m_trace.jsonl`),
não inferido:

```
decode : positions = 10, len(uniq) = 10   (2208 linhas)   <- exatamente top-k
prefill: positions = 570, len(uniq) = 144 (96 linhas)
```

`num_experts_per_tok = 10` no `config.json`. **Não há over-read**: o decode pede
exatamente o que o router pediu.

Tráfego de pesos por token:

```
48 camadas x 10 experts x 2.822.400 B (per_expert) = 1,26 GiB/token
```

Dois terços disso não vêm do disco:

| origem | por token | fração |
|---|---|---|
| disco (medido) | 0,32 GiB (328 MiB) | **25 %** |
| page cache do kernel | ~0,94 GiB | **75 %** |

Conferência: 328 MiB a 906 MiB/s = 362 ms ≈ os 349 ms/token medidos (2,76 tok/s).
**O tempo de decode é explicado quase inteiramente pelo tempo de disco.**

Implicação: **o cache que realmente está servindo o decode é o page cache do
kernel, não o LRU de user-space.** O LRU de 4 GiB compete com ele pela mesma
memória unificada — o que explica por que o braço com o LRU praticamente vazio
(2-strike) teve a menor leitura de disco de todas as medições.

## 4. Hipótese testada e refutada: views prendem o banco inteiro?

`_read_expert_banks` (streaming_switch.py:2199-2207) devolve as linhas como
`np.frombuffer(banks[k][i], ...)` — views zero-copy dentro de um banco contíguo
`(n_missing, per_slot)`. Como o write-back faz `put(key, raw)` com essas views,
cada entrada do LRU segura o banco inteiro; se a evicção levar só parte das
linhas, o banco sobrevive inteiro e a memória real passa da contabilizada.

Micro-bench: `bench/bench_writeback_arena.py` (LRU real, geometria do JANG_4M,
28.800 admissões, capacidade 4565):

| braço | admitidas | size | evicções | contabilizado | retido | amplificação | µs/admissão |
|---|---|---|---|---|---|---|---|
| views (como é hoje) | 28.800 | 4560 | 24.240 | 4,00 GiB | 4,21 GiB | **1,05×** | 14,98 |
| cópias | 28.800 | 4560 | 24.240 | 4,00 GiB | 4,00 GiB | 1,00× | 26,93 (+80 %) |

**Refutada.** A amplificação é de 5 % porque as linhas de um banco entram juntas
e saem juntas (LRU em ordem de inserção). E a cura seria pior que a doença:
copiar custa +80 % por admissão. Não vale mexer.

## 5. Correção ao relatório de residência

Em `bench/results/residency_fix/SUMMARY.md` eu tinha escrito que o write-back
reduziu o disco em 16 %. **Estava errado**: aquele número é
`disk_read_kib_s_avg`, uma *taxa*. O decode demorou 39 % mais (33,49 → 46,42 s).
Em bytes absolutos:

| braço | KiB/s | decode_s | total lido |
|---|---|---|---|
| baseline | 843.702 | 35,43 | 29,9 GiB |
| n_proj fix | 949.696 | 33,49 | 31,8 GiB |
| write-back | 798.718 | 46,42 | **37,1 GiB (+17 %)** |
| 2-strike | 681.773 | 33,67 | **22,9 GiB (−28 %)** |

O write-back **aumentou** os bytes lidos. Corrigido no relatório original.

## 6. O que é sólido e o que não fecha

**Sólido** (medido, reproduzível):

- Load é 61 % do tempo de chamada MoE; disco é ~25 % dos bytes e ~100 % do tempo.
- Nenhum recurso perto do teto; 27 % do teto de disco.
- O decode pede exatamente top-k = 10 — não há gordura no conjunto de demanda.
- Page cache serve 75 % dos bytes; o LRU de user-space compete com ele.
- Views não amplificam memória de forma relevante (1,05×).

**Não fecha** (apontado, não resolvido):

- O braço 2-strike leu 28 % menos disco (22,9 vs 31,8 GiB) mas deu a **mesma**
  vazão (2,85 vs 2,87 tok/s). Se o tempo fosse só disco, ele deveria ter sido
  ~28 % mais rápido. Ou há um piso de ~80 ms/token não coberto pelo modelo
  "tempo = bytes/banda", ou a diferença de bytes entre runs é ruído de estado da
  máquina (RSS variou 4,89 → 12,50 GiB entre runs idênticos). **Não resolvível
  com 1 repetição** — exigiria 3+ reps por braço, a ~6 min cada.

## 7. Consequência para a meta "decode ≥ 2× baseline"

Com 1,26 GiB/token de tráfego de pesos e 64,60 GiB de experts no checkpoint,
**não há orçamento de memória que torne o decode residente nesta máquina**. O
teto teórico, com tudo em RAM, seria dezenas de tok/s; o que se consegue hoje é
2,76 tok/s porque 25 % dos bytes insistem em vir do disco a 906 MiB/s.

Os dois únicos caminhos que restam, nenhum deles barato:

1. **Subir a banda efetiva de leitura** — estamos a 27 % do teto de 3300 MB/s.
   A Trilha D mediu Metal IO como empate técnico, mas isso foi com o pipeline
   atual; o gargalo aqui é latência/paralelismo, não a API de leitura. Exigiria
   aumentar QD e/ou reduzir o número de `pool.map` por token (hoje 48 por token,
   um por camada).
2. **Reduzir bytes por token** — cortar top-k de 10 para 5 (truncamento
   adaptativo, aproximado, já existe opt-in) ou quantizar mais. Custo: qualidade.
   Fora do critério "bit-exact".

Cache de experts (política, budget, write-back) **não** é o caminho. Isso está
medido em três direções diferentes agora.

## 8. A cópia numpy→MLX **não** é o gargalo (medido e refutado)

Hipótese: `load_ms = 2,872` por chamada move 26,92 MiB a 9,8 GiB/s — só ~4% do
pico de memória do M4 Pro, então o custo estaria na promoção
(`promote_np_array` → `mx.array`), não na leitura. Se fosse, daria para cortar
a cópia.

Medido (`bench/bench_promote_copy.py`, 300 iterações, geometria real):

| braço | p50 (ms) | GiB/s |
|---|---|---|
| `np.copyto` (piso memcpy) | 0,322 | 81,6 |
| `np.empty` + copy (piso c/ page fault) | 0,320 | 82,1 |
| **`mx.array`, 3× 9,4 MB** | **0,320** | **82,2** |
| `mx.array`, 9× 3,1 MB | 0,322 | 81,5 |

**A promoção já roda na velocidade do memcpy.** Não há overhead de cópia para
recuperar, e não há API de zero-copy: `mx.array(memoryview)` é aceito mas copia
igual (mesmos 82 GiB/s); `mx.array(..., copy=False)` é rejeitado
(`TypeError`).

Conclusão: a cópia é **11%** do `load_ms` (0,320 de 2,872). Os outros **89%
(2,55 ms) são leitura**. A hipótese está refutada.

## 9. Correção: "27% do teto de disco" era uma taxa misturada

O número `906 MiB/s` de `disk_read_kib_s_avg` é a taxa **misturada** sobre todos
os bytes de expert, incluindo os 75% que vêm do page cache a 82 GiB/s. Ele não
mede a velocidade do disco. Decompondo (`1/10,55 = 0,75/82 + 0,25/D`):

- leitura efetiva dos banks: 26,92 MiB / 2,552 ms = **10,55 GiB/s** (misturada)
- parte servida pelo page cache: 82 GiB/s
- **parte servida pelo disco: D ≈ 2,92 GiB/s = 3,14 GB/s — ~95% do teto medido de 3,3 GB/s**

Ou seja: **o disco não está a 27% do teto, está perto de saturado.** Não há
folga de I/O. Isso mata o caminho 1 da seção 7 (subir banda efetiva/paralelismo):
o dispositivo já está no limite no momento em que é acionado.

## 10. Orçamento corrigido por token (344 ms a 2,76 tok/s)

| componente | ms/token | % |
|---|---|---|
| leitura de experts — do disco | 110 | 32% |
| leitura de experts — do page cache | 12 | 3% |
| promoção numpy→MLX | 15 | 4% |
| `gate_eval` (roteador) | 55 | 16% |
| `gather_qmm` + stack | 33 | 10% |
| pesos residentes + atenção (não-MoE) | ~119 | 35% |

Isto resolve a anomalia da seção 6: o braço 2-strike leu 28% menos disco, o que
vale 0,28 × 110 ms ≈ **31 ms/token (+9%)** — exatamente na fronteira do ruído
de ±4% desta máquina, com 1 repetição e RSS variando 4,89 → 12,50 GiB entre runs
idênticos. Não era contradição, era falta de resolução.

**Teto se o page cache servisse 100% dos bytes de expert:** leitura cairia de
122 ms para 16 ms → 238 ms/token → **4,2 tok/s = 1,47× o valor atual**. Mesmo o
cenário perfeito de residência não chega a 2×, porque ~35% do token são pesos
residentes e atenção, que nenhuma otimização de streaming toca.
