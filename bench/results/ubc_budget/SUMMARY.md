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
