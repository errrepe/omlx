# Re-medição do write-back após o fix O(1) de evicção (Fase M4)

Pergunta: o veredito "write-back custa 28% de decode" ainda vale agora que a
evicção per-layer é O(1)?

**Resposta: não. O custo real é −3,4% — dentro do ruído. O que desqualifica o
write-back é outro número: ele lê 16% MAIS disco.**

Ambiente: M4 Pro 48 GB, `Qwen3.8-Flash-Next-JANG_4M`, budget 4,0 GiB,
prompt 2k, 96 tokens, 1 request. 2 reps por braço, alternadas (off/on/off/on).

---

## 1. Por que re-medir

O veredito original (`residency_fix/SUMMARY.md` §4) foi obtido num run com
**67.793 evictions**. Na época a evicção per-layer ainda era O(store): cada
`put` que estourava o cap da layer varria as 4.560 entradas para achar a
vítima — 122 µs/put (micro-bench), contra 2,48 µs depois do índice
`_layer_orders` (`streaming_switch.py:889`).

`67.793 × ~120 µs ≈ 8,1 s` de puro scan linear. A penalidade total medida foi
`46,42 − 33,49 = 12,9 s`. Ou seja, **~63% do "custo do write-back" era o bug
de evicção, não o write-back.**

Não foi possível datar o run antigo pelos artefatos (o json não foi
preservado), então a hipótese foi testada re-medindo, não arqueologicamente.

## 2. Resultado

Comando:

```
.venv/bin/python bench/bench_expert_streaming.py --model qwen-jang4m \
  --budget 4.0 --decode 96 --prompt-len 2k --single-request \
  --out bench/results/wb_remeasure/{off,on}_rN.json
# braço ON: OMLX_EXPERT_STREAMING_CTX_WRITEBACK=1
```

| métrica | off (write-back desligado) | on (`_CTX_WRITEBACK=1`) |
|---|---|---|
| **tok_s** | **2,8570** | **2,7593** (**−3,4%**) |
| decode_s | 33,62 | 34,80 |
| hit rate (todas os acessos) | 0,0618 | **0,2262** (3,7×) |
| hits / misses | 19,6k / 297,1k | 71,6k / 245,0k |
| puts | 4.464 | 72.312 |
| evictions | 0 | 67.752 |
| **disco total (GiB)** | **27,4** | **31,9** (**+16%**) |
| sys_cpu % | 14,39 | 16,23 |
| RSS máx (GiB) | 13,75 | **17,70** |

Por rep:

| | r1 | r2 |
|---|---|---|
| off | 2,9133 | 2,8007 |
| on | 2,7471 | 2,7715 |

O spread do braço **off** sozinho é ±2% (2,91 vs 2,80); o ruído histórico da
máquina é ±4%. Um delta de −3,4% **não é resolvível** aqui.

## 3. O hit rate reportado subestima o decode

`hit_rate` = `hits / (hits + misses)` soma prefill e decode. O prefill usa
`prefill_bypass` e **nunca acerta** — contribui 297k misses garantidos nos dois
braços (denominador idêntico: 316.655 acessos).

O write-back converteu exatamente `297.073 − 245.038 = 52.035` misses em hits.
Todos em decode. Como decode faz `96 × 48 × 10 × 3 = 138.240` acessos de slot:

- decode-only (off) ≈ 0,14
- decode-only (on) ≈ 0,52

(estimativa: assume que todo hit do braço off foi em decode — plausível, já que
o prefill não consulta o cache.)

Ou seja: o write-back **funciona** como cache. Ele só não serve para nada.

## 4. O mecanismo: o cache de usuário canibaliza o page cache

Triplicar o hit rate de decode deveria cortar ~0,5 GiB/token de tráfego e
derrubar o decode de ~33 s para ~19 s. Medido: **34,8 s**.

E o disco vai para **cima**, não para baixo:

- RSS máx 13,75 → 17,70 GiB (**+3,95 GiB**, exatamente o cache enchendo).
- Disco 27,4 → 31,9 GiB (**+16%**).

Num Mac de memória unificada, os 4 GiB do LRU saem do mesmo orçamento do
buffer cache do kernel (UBC). O LRU guarda uma **segunda cópia** de bytes que o
UBC já tem; o UBC encolhe; leituras que antes eram servidas da RAM passam a ir
ao disco. Os hits que o LRU fabrica são hits sobre bytes que o UBC teria
servido quase de graça.

Isto é a mesma conclusão do braço 2-strike em `residency_fix` (cache
praticamente vazio → **menor** leitura de disco de todos os runs), agora
medida na direção oposta e com o confound removido.

## 5. Consequência: residência de cache está fechada como alavanca

Três medições independentes convergem:

1. **n_proj fix** — cache 1,33 → 4,00 GiB, hit 0,019 → 0,062: decode dentro do ruído.
2. **write-back re-medido** — hit 0,062 → 0,226 (decode ~0,14 → ~0,52): −3,4%, ruído; disco +16%.
3. **2-strike** — 28% MENOS disco, mesma vazão.

Bytes movidos e vazão de decode **não são proporcionais** neste sistema. Existe
um piso por token que o cache não atinge. O critério global "decode ≥ 2×
baseline" não será alcançado por residência, política ou orçamento de cache —
que era exatamente a hipótese de partida do usuário ("talvez ele não tenha
mantido experts suficientes em cache").

## 6. Onde a alavanca pode estar (não medida, explícito)

O profile (`bench/results/decode_profile/SUMMARY.md`) mostra nada saturado:
cpu 120% de ~1200, gpu 27,8%, disco 27% do teto. Assinatura de cadeia serial
por camada, não de recurso esgotado. Por token:

| fase | ms/token (48 camadas) |
|---|---|
| load (incl. promote numpy→MLX) | 138 |
| gate_eval | 55 |
| stack | 33 |

O `load` move 1,26 GiB/token a ~9,8 GB/s — ele já roda perto da banda de
memória. O que sobra é **a cópia**: leitura vai para buffer numpy e depois é
promovida a `mx.array`. Ler direto para memória do MLX eliminaria metade
desse tráfego (~64 ms/token → ~3,6 tok/s, +29%). É especulação: a Trilha D
mediu Metal IO como empate técnico contra `preadv`, mas mediu isso com o
pipeline atual, que copia de qualquer forma.

Não é 2×, e não foi medido.
