# Trilha C — auditoria de layout "direct-compatible"

**Data:** 2026-09-06 · **Ferramenta:** `scripts/audit_expert_streaming_layout.py`
**Relatório:** `bench/results/layout_audit/audit.json`

## O que é "direct-compatible"

Trilha D (Fast Resource Loading nativo) quer publicar os bytes do expert na GPU
direto do arquivo mapeado, sem cópia de staging e sem conversão de dtype no
caminho quente. Isso só é possível se os bytes em disco já forem os bytes que o
kernel espera:

| papel | dtype alvo |
|---|---|
| `weight` | **U32** (palavras quantizadas packed em uint32) |
| `scales` | **F16** |
| `biases` | **F16** |

O auditor é **header-only**: lê o prefixo de 8 bytes e o JSON de cabeçalho de
cada shard. Nunca toca payload — é seguro contra volume montado read-only e
custa milissegundos mesmo num checkpoint de 102 shards.

## Resultado — corpus local

| checkpoint | veredito | tensores expert | routed | BF16 → converter |
|---|---|---|---|---|
| Qwen3.8-Flash-Next-JANG_4M | `direct_compatible` | 438 | 64.60 GiB | — |
| Qwen3.8-Flash-Next-JANG_4S | `direct_compatible` | 438 | 47.85 GiB | — |
| GLM-5.3-Flash-JANG-MTP | `direct_compatible` | 774 | 87.06 GiB | — |
| DeepSeek-V4-Flash-0731-JANG | `direct_compatible` | 99 459 | 88.57 GiB | — |
| Qwen3.6-35B-A3B-oQ4e-mtp | `needs_conversion` | 369 | 17.30 GiB | **1.92 GiB** |
| Tiel-Coder-35B-A3B-MLX-oQ4e | `needs_conversion` | 360 | 16.88 GiB | **1.88 GiB** |
| Qwen3.8-27B-oQ4e-mtp | `no_expert_tensors` | 0 | — | denso, sem MoE |

**Nenhum tensor é `incompatible`.** Os 4 modelos JANG — que são os que o
streaming realmente serve — já estão 100% no layout alvo. O BF16 aparece
**apenas** nos side tensors (`scales`/`biases`) de dois checkpoints
Qwen3-35B-A3B; o `weight` é U32 em todos os casos.

## Consequência para a Trilha D

- **Nenhuma conversão é pré-requisito para Trilha D.** Os 4 modelos que
  exercitam o expert streaming já podem publicar direto. Isso remove o item de
  risco "preciso converter antes de medir".
- O item 6 (conversor BF16→F16) continua **justificado** para
  Qwen3.6-35B-A3B e Tiel-Coder (~3.8 GiB de side tensors no total), mas é
  **fora do caminho crítico** — pode vir depois de D1/D2.

## Decisões de classificação (importantes para não reler errado)

1. **Shared experts ficam fora do veredito.** `mlp.shared_experts.*` /
   `ffn.shared_experts.*` são densos e residentes (rodam em todo token), não
   são streamados nem evictados. São reportados com `[shared]` mas não gatilham
   o veredito.
2. **O router não é expert.** `layers.N.ffn.gate.weight` é F32 por design
   (`moe_router_dtype: float32`). O fallback genérico **não** casa `ffn` sozinho
   — se casasse, todo modelo seria falsamente `incompatible`.
3. **MTP conta como routed.** `mtp.layers.N.mlp.experts.*` é a pilha de experts
   do módulo de draft; é streamada, logo gatilha o veredito.
4. **Falha fechada.** Uma convenção desconhecida cai no bucket `generic` e
   conta como **routed** — melhor um falso positivo barulhento que um modelo
   aprovado sem nunca ter sido classificado de fato.

## BF16 → F16: por que é lossy

BF16 tem 8 bits de expoente e 7 de mantissa; F16 tem 5 e 10. A conversão
**ganha** mantissa mas **perde faixa**: qualquer `|x| > 65504` vira infinito.
Só é lossless se todos os valores caberem em F16 — o que não dá para saber só
pelo header. O auditor reporta o volume candidato e deixa o **range check para
o conversor** (item 6).

## BF16 → F16: medido nos checkpoints reais (item 6)

O `needs_conversion` deste relatório era um **teto**, não um plano. O conversor
(`scripts/convert_expert_bf16_to_f16.py`) mediu o quanto é de fato conversível.
Dry-run nos dois checkpoints:

| checkpoint | alvo `needs_conversion` | conversível | recusado | convertido |
|---|---|---|---|---|
| Qwen3.6-35B-A3B-oQ4e-mtp | 1.92 GiB | 1.36 GiB (71%) | 72 tensores | — (dry run) |
| Tiel-Coder-35B-A3B-MLX-oQ4e | 1.88 GiB | 1.33 GiB (71%) | 70 tensores | — (dry run) |

**Split por projeção — é tudo ou nada:**

| projeção | convertidos | recusados |
|---|---|---|
| `down_proj.scales` | 82 / 80 | **0** |
| `gate_proj.scales` | 46 / 45 | 36 / 35 |
| `up_proj.scales` | 46 / 45 | 36 / 35 |

### Causa: faixa subnormal do F16, não overflow

Amostra real (`layers.0.mlp.switch_mlp.gate_proj.scales`, 4.19 M elementos):

```
min |x| != 0 = 1.001e-07   mediana = 2.335e-03   max = 1.489e-02
abaixo de 6.10e-5 (subnormal em F16): 1.736.256  (41.4%)
flushes to zero (< 5.96e-8):                  0  (0.0%)
overflow (> 65504):                           0  (0.0%)
```

`down_proj.scales` (min 1.268e-04) fica inteiro acima do limiar → 100% seguro.
`gate_proj`/`up_proj` são escalas de quantização 4-bit com dinâmica que desce a
1e-7; o F16 (menor normal 6.10e-5) não as representa. **Zero overflow** — a
intuição de "o perigo é o 65504" está errada para estes modelos.

### Consequência

Os 71% conversíveis são inúteis isolados: um expert precisa das três projeções,
e `gate`/`up` não convertem. **Não há conversão viável BF16→F16 para estes dois
checkpoints.** O caminho real para torná-los diretamente carregáveis é aceitar
BF16 no loader (cast em load, custo de cópia), não reescrever o modelo.

O conversor fica: o valor entregue é a **prova de recusa**, que impede uma
conversão cega que degradaria 41% das escalas sem ninguém notar.

## Uso

```bash
.venv/bin/python scripts/audit_expert_streaming_layout.py MODEL_DIR [MODEL_DIR ...]
.venv/bin/python scripts/audit_expert_streaming_layout.py DIR --json out.json
.venv/bin/python scripts/audit_expert_streaming_layout.py DIR --strict   # exit 1 se não limpo
```

## Testes

`tests/test_expert_streaming_layout_audit.py` — 15 testes. As fixtures escrevem
safetensors **sem payload nenhum** (só prefixo + JSON), o que é ao mesmo tempo a
fixture mais barata e uma prova viva do contrato header-only.
