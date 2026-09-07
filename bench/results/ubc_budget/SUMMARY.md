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

## 4.1 ~~O hit rate do LRU é igual à cobertura~~ — RETIFICADO: o hit rate e2e é uma métrica diluída

> Esta seção afirmava, com base na coincidência numérica abaixo, que o cache
> tinha "zero correlação com roteamento". **Estava errado.** A coincidência era
> exatamente isso, coincidência. A medição instrumentada da seção 7 derrubou a
> hipótese; o texto correto segue.

A evidência que me enganou:

```
cobertura em espaço de slot = 95 / (512 x 3) = 0,061849
hit rate e2e medido (2 runs) = 0,062399 e 0,061900
```

Quatro casas decimais — e era acaso. A conta certa fecha o denominador da
métrica e2e (`hits + misses` = 316.655 numa run de 2k/96):

```
demanda de decode  : 46.560 experts x 3 projecoes = 139.680 lookups
demanda de prefill : 29.951 experts x 3 projecoes =  89.853 lookups
lookups de transicao/especulacao                  =  87.122
                                                   --------
                                                   316.655
```

O caminho union faz **um get por projeção** (`for proj in self.linears`), e o
contador engole prefill e especulação junto. O lado dos *hits* confirma: a
previsão estática do decode é 20.783 e o medido é 19.878 (−4,4%). Ou seja, o
cache **está** servindo o que deve servir.

**O número certo** (run instrumentada, budget 4 GiB, trace da própria run):

```
hit de decode, seed estatico da producao : 0,1488
baseline aleatorio (31/512, mesmo trace) : 0,0634
                                          2,35x sobre o aleatorio
```

O seeder funciona: escolhe por uso real (`counts`), 100% do escolhido fica
residente (§7), e entrega 2,35x o acaso. No trace antigo o replay dava 0,1935.

O que **segue valendo** desta seção:

- Um hit no LRU não é mais barato que um hit no page cache — as entradas são
  views do mmap, páginas de arquivo. O LRU é um segundo índice sobre o mesmo
  page cache, com overhead de dict por entrada e 4 GiB de working set a mais
  para o kernel considerar. Não adiciona residência; duplica.
- Os dois braços usam seeders diferentes (budget>0: `_seed_lru`, 31
  experts/camada ≈ 3,9 GiB; budget 0: `_seed_page_cache`, 15/camada ≈ 1,9 GiB)
  — ~2 GiB do +7 GiB de disco da seção 1. O restante segue sem explicação.

O que **cai**: "zero correlação", "10% do atingível", e qualquer conclusão
derivada delas. A comparação correta com o offline é seed estático ~0,15–0,19
vs seed+LRU 0,50–0,64 — o resto do gap é o cache **congelado** (sem write-back),
não o seed.

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

## 7. Instrumentação seed-vs-residente: as três hipóteses de bug morreram, e a história certa é outra

`bench/bench_seed_static.py` roda offline sobre `bench/results/lrc/jang4m_trace.jsonl`
e **reproduz os quatro números publicados** em `sizing/SUMMARY.md` §2, o que
valida o simulador:

| métrica | este script | sizing §2 |
|---|---|---|
| Belady | 0,7396 | 0,740 |
| seed + LRU | 0,6375 | 0,637 |
| cold LRU | 0,6287 | 0,629 |
| working set/camada (mediana, máx) | 103, 193 | ~103, 193 |

A primeira versão desta seção lia a linha nova (seed estático 0,1935) contra o
hit rate e2e (0,0618) como um "gap de 3,13x" e apontava três candidatos de bug.
A instrumentação derrubou os três e a própria leitura:

### O que foi medido (budget 4 GiB, 2k/96, `OMLX_BENCH_SEED_DUMP=1` + `OMLX_EXPERT_STREAMING_TRACE`)

```
seed_vs_resident : seed_total=1488, resident_total=1488,
                   intersection=1488, seed_kept_frac=1.0
```

**O conjunto residente é exatamente o escolhido.** Sem chave divergente, sem
jobs perdidos no `_WARM_POOL`, sem truncamento do cap. As hipóteses 1–3 estão
mortas.

### O replay no trace da própria run (o passo que faltava)

Replayando o seed **da produção** como estático sobre o routing **desta mesma
run** (4.752 linhas, decode = 46.560 requests):

```
seed ESTATICO (producao)  0,1488   <- melhor que o top-k do simulador (0,0875)
baseline aleatorio        0,0634   <- o seed vale 2,35x o acaso
seed+LRU (sim)            0,4988   <- o que o LRU atualizando daria
medido e2e                0,0628   <- metrica diluida (ver 4.1)
```

### A história correta, em três frases

1. **O seeder funciona de ponta a ponta**: escolhe por uso real (`counts`,
   não presença — e é por isso que bate o top-k por presença do simulador),
   100% do escolhido fica residente, e vale 2,35x o acaso.
2. **O "0,0618 == cobertura" era coincidência**: a métrica e2e mistura no
   denominador prefill e especulação (§4.1). O hit de decode real é ~0,15.
3. **O gap que sobra é o cache congelado**: sem write-back o teto é o seed
   estático (~0,15–0,19); com write-back o simulador dá 0,50–0,64.

### Consequência prática

A alavanca não é consertar o seed — não há o que consertar. É decidir se vale
destravar o LRU. O comentário em `streaming_switch.py` (~1745) já registra que
o custo de CPU do write-back caiu para ~3% (dentro do ruído) depois do fix
O(1) de `b73a0270` — o "-28%" histórico era o scan O(store). O que
desqualifica é **I/O**: +16% de bytes de disco (27,4 → 31,9 GiB), porque o
write-back preenche 4 GiB com uma cópia do que o page cache já serve.

**Ressalva, para não supervalorizar o 0,637:** mesmo o seed+LRU cheio não
converte linearmente em tok/s. Um hit no LRU de user-space ainda paga a
promoção numpy→MLX e a entrada é uma view de mmap que pode não estar residente.
O 0,637 é teto de *bytes evitados*, não de tempo.
