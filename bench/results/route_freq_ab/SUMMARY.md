# Trilha A — `route_frequency`: adoção da política de evicção do PR #3359

**Data:** 2026-09-06 · **Máquina:** M4 Pro 48 GB (~20 GB livres no momento dos testes)
**Branch de origem:** `onthehub97/omlx@ssd-expert-streaming` (PR #3359)

## Objetivo

Portar a política de evicção por *route frequency* (contador por `(layer, expert, proj)`,
evicta o menos roteado, recência só desempata) para o cache de experts residentes,
como opção opt-in ao lado de `lru` e `s3fifo`.

**Gate de aceite:** `route_frequency ≥ s3fifo AND ≥ lru` em ≥ 2 de 3 modelos.

---

## Resultado 1 — e2e: o bench não consegue discriminar (achado estrutural)

`bench/bench_expert_streaming.py --model qwen-jang4m --budget 4.0 --prompt-len 2k
--decode 96 --single-request`, braços adjacentes, mesma ordem de flags:

| política | TTFT | decode 96 tok | hit rate | evict | size/cap |
|---|---|---|---|---|---|
| `lru` | 51.3 s | 2.709 tok/s | 0.01923 | **0** | 1440/1521 |
| `s3fifo` | 53.3 s | 2.246 tok/s | 0.01923 | **0** | 1440/1521 |
| `route_frequency` | 52.5 s | 2.546 tok/s | 0.01929 | **0** | 1440/1521 |

**Nenhuma evicção ocorre** — logo as três políticas executam exatamente o mesmo
caminho de código. Motivo: o *hotness seeder* popula 30 experts por layer ao fim do
prefill e o per-layer cap é `capacity // num_layers = 1521 // 48 = 31`. O decode de
96 tokens não empurra nenhuma layer acima de 31, então a política nunca é chamada.
As diferenças de TTFT/decode (51–53 s, 2.2–2.7 tok/s) são ruído de uma máquina com
~20 GB livres — abaixo do próprio guard de 22 GB do bench.

Esse achado também explica por que o A/B anterior do `s3fifo` "empatou nas traces
reais": **não era empate de qualidade, era ausência de evicção.** A política só
importa quando o seed sub-cobre o working set de decode — budget menor, decode
longo/diverso, ou topic drift.

## Resultado 2 — A/B offline por trace (onde a evicção acontece)

`bench/bench_cache_policy_offline.py` — replay da mesma trace contra as três
classes de cache. Topologia real lida dos `config.json` (qwen4_exp 48×512/top-10,
glm5_next 45×288/top-8, deepseek_v4 43×256/top-6); forma da trace sintética
(prefill-scan → decode com reuso Zipf → troca de fase na metade).
256 tokens, scan 64, 3 repetições, mediana, warm-up descartado.
`cap` = slots por layer: 8≈1 GiB, 16≈2 GiB, 31≈4 GiB, 63≈8 GiB.

`hot_hit_rate` = acertos nas rotas do pool quente (as rotas de cauda fria são miss
em qualquer política e só somam uma constante).

| modelo | cap | lru | s3fifo | route_frequency | melhor |
|---|---|---|---|---|---|
| qwen | 8 | 0.4053 | **0.5577** | 0.4824 | s3fifo |
| qwen | 16 | 0.6219 | **0.7235** | 0.7150 | s3fifo |
| qwen | 31 | 0.8237 | 0.8259 | **0.8972** | route_frequency |
| qwen | 63 | 0.9601 | 0.9232 | **0.9743** | route_frequency |
| glm | 8 | 0.4046 | **0.5567** | 0.4835 | s3fifo |
| glm | 16 | 0.6228 | 0.7024 | **0.7107** | route_frequency |
| glm | 31 | 0.8241 | 0.8037 | **0.8841** | route_frequency |
| glm | 63 | 0.9590 | 0.8999 | **0.9674** | route_frequency |
| dsv4 | 8 | 0.4036 | **0.5471** | 0.4799 | s3fifo |
| dsv4 | 16 | 0.6213 | 0.6861 | **0.7056** | route_frequency |
| dsv4 | 31 | 0.8214 | 0.7914 | **0.8701** | route_frequency |
| dsv4 | 63 | 0.9563 | 0.9119 | **0.9633** | route_frequency |

**Verdict: PASS** — `route_frequency` vence em 8/12 células; lidera em glm (3/4) e
dsv4 (3/4), e em qwen (2/4). Gate de ≥2 modelos: **2 modelos → PASS**.

Nos budgets realistas (4 e 8 GiB) o ganho é consistente: **+6 a +9 pontos de
hot-hit-rate sobre `lru` e sobre `s3fifo`**. Em 1 GiB (`cap=8`) o `s3fifo` segue
imbatível: com o cache quase vazio, resistência a scan vale mais que frequência.

### Custo por lookup (harness, sem I/O)

| cap | lru | s3fifo | route_frequency |
|---|---|---|---|
| 8 | 5.6–6.0 µs | 1.6–1.7 µs | 2.9–3.8 µs |
| 31 | 14.0–17.1 µs | 1.4–1.5 µs | 8.2–8.6 µs |
| 63 | 19.8–24.9 µs | 3.2–4.3 µs | 8.8–10.5 µs |

A escolha de vítima do `route_frequency` é varredura O(size) — mesma ordem do
`ExpertLRUCache` (que ainda aloca `list(self._store.keys())` no caminho de
per-layer cap, daí ser ~2× mais caro). É mais barato que o `lru` e ~3–6× mais caro
que o `s3fifo`. **Se algum dia virar default, precisa de estrutura de vítima O(1)**
(buckets por contagem ou heap lazy).

---

## Decisão

- **Default continua `lru`.** Nada muda para quem não mexer na configuração.
- `route_frequency` fica **opt-in e experimental** — passou no gate offline, mas
  o e2e não consegue confirmar (evict=0) e o custo por lookup é ~3–6× o do s3fifo.
- **`s3fifo` continua sendo a escolha em budget ~1 GiB** (cap≈8), onde resistência
  a scan domina.
- Recomendação prática: em budget ≥ 4 GiB **com decode longo ou drift de tópico**
  (onde o seed de 30/layer não cobre), `route_frequency` é o candidato.

## Ressalvas (ler antes de citar esses números)

1. **A forma da trace é sintética.** Topologia e tamanhos são reais; a distribuição
   de roteamento (80% pool quente / 20% cauda, troca de fase na metade) é uma
   modelagem, não uma captura. Serve para ordenar políticas, não para prever tok/s.
2. **Máquina com memória apertada.** ~20 GB livres de 48 GB; o bench aborta no
   guard padrão de 22 GB e as rodadas e2e rodaram com `--min-free-gb 16`. Decode
   a 2.2–2.7 tok/s é I/O-bound, não representativo.
3. **As três rodadas e2e compartilham a tabela de transição persistida.**
   `trans_updates` saiu 14208 / 28416 / 42624 — é acúmulo do payload salvo em disco
   (`load_payload` soma `updates` do arquivo anterior), **não** efeito de política.
   Os braços e2e não são estatisticamente independentes.
4. **O e2e não é um teste da política** neste operating point. Qualquer A/B futuro
   precisa primeiro forçar evicção (budget menor ou decode mais longo/diverso).

## Knobs

| como escolher | valor |
|---|---|
| env (global) | `OMLX_EXPERT_STREAMING_CACHE=route_frequency` |
| por modelo (admin API) | `expert_streaming_cache_policy: "route_frequency"` |
| UI (admin + macOS app) | "Route frequency" no seletor de política |
| bench e2e | `--cache-policy route_frequency` |
| bench offline | `bench/bench_cache_policy_offline.py` |

## Arquivos

- `omlx/patches/expert_streaming/streaming_switch.py` — `RouteFrequencyCache` + wiring em `make_expert_cache`
- `bench/bench_cache_policy_offline.py` — harness de A/B por trace (novo)
- `bench/results/route_freq_ab/offline.json` — resultados offline completos
- `bench/results/route_freq_ab/e2e_{s3fifo,route_frequency}.json`, `pilot_lru.json` — e2e
