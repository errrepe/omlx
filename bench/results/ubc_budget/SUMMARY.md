# Cache de experts vazio vs 4 GiB (Fase M4)

Pergunta: se o cache de experts de user-space compete com o page cache do
kernel, um budget menor deveria **aumentar** a vazão devolvendo memória ao UBC?

**Resposta: sim, mas pouco — e o efeito grande e reprodutível é no disco, não
na vazão.**

Ambiente: M4 Pro 48 GB, `Qwen3.8-Flash-Next-JANG_4M`, prompt 2k, 96 tokens,
1 request. Budget 0,0 (n=3) vs 4,0 GiB (n=2), alternados.

---

## 1. Resultado

| métrica | budget 0,0 (LRU vazio) | budget 4,0 | delta |
|---|---|---|---|
| **tok_s** | **3,0745** | 2,9604 | **+3,9%** |
| decode_s | 31,24 | 32,44 | −3,7% |
| TTFT | 47,3 | 47,4 | igual |
| hit rate | 0,0000 | 0,0619 | — |
| **disco total (GiB)** | **18,8** | **25,8** | **−27%** |
| sys_cpu % | 17,76 | 13,84 | +3,9 pp |
| RSS máx (GiB) | 13,75 | 13,76 | **igual** |

Por rep:

| | r1 | r2 | r3 |
|---|---|---|---|
| budget 0,0 | 3,1076 | 2,9851 | 3,1308 |
| budget 4,0 | 3,0092 | 2,9115 | — |

Significância: somando os dois runs de budget 4,0 da re-medição do write-back
(2,9133 e 2,8007, mesma configuração) para um n=4 → média 2,9087. Diferença
0,166 tok/s com SEM combinado 0,062 → **t ≈ 2,7, p ≈ 0,045**. Marginal: está
na fronteira do que esta máquina resolve, mas a direção é consistente com as
outras duas medições independentes (§2).

## 2. O efeito no disco é reprodutível, o da vazão não

Três experimentos independentes em que o cache de user-space ficou
praticamente vazio:

| experimento | disco vs. braço com cache | vazão |
|---|---|---|
| 2-strike admission (`residency_fix` §4) | −28% | igual (2,85 vs 2,87) |
| este A/B, rep 1 | −29% | +3,3% |
| este A/B, média | **−27%** | **+3,9%** |

O corte de ~28% de disco aparece nas três vezes. O ganho de vazão previsto
pelo orçamento da seção 3 é de ~9%; o medido é ~4%. A diferença está dentro do
espalhamento (budget 0,0 variou 2,985 → 3,131, ±2,4%).

## 3. Onde o ganho deveria vir — e por que é pequeno

Do orçamento por token em `decode_profile/SUMMARY.md` §10, a leitura de experts
pelo disco é 110 ms de 344 ms (32%). Cortar 27% do disco vale:

```
0,27 x 110 ms = 30 ms/token  ->  314 ms  ->  3,18 tok/s  (+9%)
```

Medido +3,9%. A folga some porque **os 6,2% de hit rate que o cache produz
também têm valor** — e porque ~35% do token é não-MoE (pesos residentes +
atenção), intocável por qualquer otimização de streaming.

## 4. Mecanismo: **não** confirmado, e a história do UBC não fecha

A explicação que eu vinha sustentando — o LRU rouba memória unificada do page
cache — **não é sustentada pelos números de memória**:

- `phys_lifetime_max_gib`: 13,75 (b0) vs 13,76 (b4). Idêntico.
- `phys_after_load_gib`: 6,94 vs 6,95. Idêntico.
- `metal_peak_decode_gib`: 6,754 vs 6,757. Idêntico.

Motivo provável: o seeder popula o cache com **views do mmap** dos
safetensors (`np.frombuffer`), não com memória anônima. As 4.464 entradas são
páginas de arquivo — as mesmas que o UBC já contaria. Não há duplicação de
memória física, logo não há competição mensurável.

Resta sem explicação o **+7 GiB de leitura** no braço com cache. Hipóteses não
testadas: (a) as leituras do próprio seeder sendo contabilizadas na fase de
decode; (b) `trans_updates` maiores com cache (213.120 vs 198.912) puxando
mais leitura de transição. **Não resolvido — não afirmar mecanismo.**

## 4.1 O hit rate do LRU é estatisticamente igual à cobertura

Achado posterior, e a razão pela qual os 6,2% de hit não compram nada:

```
cobertura em espaço de slot = 95 / (512 x 3) = 0,061849
hit rate medido (2 runs)    = 0,062399 e 0,061900
```

Quatro casas decimais. O conteúdo do cache é indistinguível de uma amostra
uniforme de 6,2% dos `(camada, slot)` — **zero correlação com roteamento**.

O teto offline para a mesma geometria (`sizing/SUMMARY.md` §2, budget 4 GiB) é
Belady 0,740 e seed+LRU 0,637. O sistema entrega **10% do atingível**.

Corolário: um hit no LRU não é mais barato que um hit no page cache, porque as
entradas do LRU *são* views do mmap — páginas de arquivo. O LRU é um segundo
índice sobre o mesmo page cache, com overhead de dict Python por entrada e
4 GiB de working set a mais para o kernel considerar. Ele não adiciona
residência; ele a duplica.

Isso também explica parte do +7 GiB: os dois braços usam seeders diferentes.
Com budget > 0 o `_seed_lru` semeia `per_layer_cap // n_proj` = 31
especialistas/camada (~3,9 GiB lidos do disco); com budget 0 o
`_seed_page_cache` semeia `min(64, SEED_BYTES // (layers * per_expert))` = 15
especialistas/camada (~1,9 GiB). São ~2 GiB da diferença. O restante segue sem
explicação.

## 5. O que fazer com isso

O cache de 4 GiB custa ~27% a mais de disco e rende ~4% a menos de vazão, para
um hit rate de 6,2% que não se traduz em nada (um hit no caminho union ainda
paga a promoção numpy→MLX inteira — ver `decode_profile/SUMMARY.md` §8).

Há um argumento razoável para **reduzir o default de budget**, mas isso mexe
num default de produção e não foi pedido — fica como recomendação, não mudança:

1. Revisar o default de budget para algo bem menor (ou 0) em modelos split com
   `n_proj=3`, onde o cache rende 6% de hit rate.
2. Se o objetivo for só reduzir disco, o caminho é **não admitir**, não admitir
   melhor.

**Não** é uma alavanca para a meta "decode ≥ 2×": o teto com page cache
servindo 100% dos bytes é 1,47×.

## 6. Tentativa seguinte (seed de page cache 2 vs 6 GiB) — INCONCLUSIVA

Com budget 0 o único mecanismo ativo é `_seed_page_cache()`, que aquece o page
cache do kernel com leituras descartadas — o único caminho que dá residência
aos experts quentes **sem** duplicá-los em user space. O tamanho do conjunto
quente é `min(64, SEED_BYTES // (layers * per_expert_bytes))`, confirmado nos
logs: 720 slices semeados = 15 experts/camada com o default de 2 GiB.

A pergunta era se subir para 6 GiB (47 experts/camada) rende vazão. A/B
alternado, 2 reps, `--min-free-gb 18`:

| rep | seed 2 GiB | seed 6 GiB |
|---|---|---|
| 1 | 2,340 tok/s | 1,744 tok/s |
| 2 | 2,557 tok/s | 2,573 tok/s |

**Sem efeito resolvível, e o experimento não vale nada nesta máquina agora.**
Dois fatos:

- O spread entre reps do *mesmo* braço (2,34→2,56; 1,74→2,57) chega a 47%,
  contra ±2,4% medido na seção 1. Efeito de ordem forte: r1 sempre pior.
- Todos os números estão muito abaixo dos 3,07–3,13 tok/s da seção 1. A caixa
  está em estado degradado — `vm_stat` mostrava 231 MiB livres com 18,85 GiB
  inativos, e o bench recusa rodar com o default (`only 20.9 GB available,
  need 22+`).

Ruído de ±25% não resolve nem o efeito da seção 1 (+3,9%). **Qualquer A/B de
page cache fica suspenso até a máquina ter folga.** Não afirmar resultado
nenhum deste experimento — ele está registrado só para que ninguém o repita às
cegas.

Nota lateral: `trans_updates` variou 326.784 → 340.992 → 355.200 → 369.408,
exatamente +14.208 por run, em processos distintos. Não é uma métrica por run;
é estado acumulado. Não usá-la como sinal.

## 7. O seed estático deveria dar 0,1935 — e dá 0,0618

`bench/bench_seed_static.py` roda offline sobre `bench/results/lrc/jang4m_trace.jsonl`
e **reproduz os quatro números publicados** em `sizing/SUMMARY.md` §2, o que
valida o simulador:

| métrica | este script | sizing §2 |
|---|---|---|
| Belady | 0,7396 | 0,740 |
| seed + LRU | 0,6375 | 0,637 |
| cold LRU | 0,6287 | 0,629 |
| working set/camada (mediana, máx) | 103, 193 | ~103, 193 |

A quinta linha é nova. O sistema real **congela** o cache depois de semear
(sem write-back: `evictions=0`, `puts == size`), então a política dele não é
`seed+LRU` — é **seed estático**:

```
seed ESTATICO  = 0,1935     <- o que a producao deveria dar
medido e2e     = 0,0618     <- o que da
                 32% do potencial, gap de 3,13x
```

E não é o algoritmo de escolha que é ruim: 48,3% do top-31 por frequência de
prefill está dentro do working set de decode. O overlap **implicado** pelo hit
rate medido é 0,154 — *abaixo* do acaso (0,201 = 103/512). Ou seja: o que
chega ao cache não está correlacionado com o top-31, apesar de o seeder
escolher por `Counter.most_common`.

**Isso localiza o bug.** Não é "frequência de prefill não prevê decode" (ela
prevê: 48% de overlap). É que o conjunto que efetivamente fica residente não é
o conjunto escolhido. Candidatos, em ordem do que verificaria primeiro:

1. Chave: o seeder põe `(layer, eid, stacked_weight_key)`; o caminho union
   busca por chave de projeção. Se divergirem, o hit é espúrio.
2. `retain_hot(hot_pairs)` roda antes dos puts assíncronos no `_WARM_POOL` —
   se o pool perder jobs (exceções são silenciadas com `continue`), o cache
   fica com o que sobrou. `size=4464` diz que chegou, mas não diz *quais*.
3. Ordem de chegada: os puts assíncronos não são ordenados por hotness, e o
   cap por camada (95) pode truncar justamente os mais quentes.

Próximo passo concreto: dump dos `(layer, eid)` realmente residentes no fim do
decode e comparar com o top-31 do seeder. É uma instrumentação pequena e
responde em uma run — e não depende da máquina estar folgada, ao contrário dos
A/Bs de vazão.

**Ressalva, para não supervalorizar o 0,637:** mesmo o seed+LRU cheio não
converte linearmente em tok/s. O experimento de write-back subiu o hit rate
para 0,226 e a vazão caiu 3,4% com disco +16%: um hit no LRU de user-space
ainda paga a promoção numpy→MLX e a entrada é uma view de mmap que pode não
estar residente. O 0,637 é teto de *bytes evitados*, não de tempo.
