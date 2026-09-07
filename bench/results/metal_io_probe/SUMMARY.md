# Trilha D — Fast Resource Loading (Metal IO): veredicto medido

**Data:** 2026-09-06 · **Hardware:** MacBook Pro M4 Pro 48 GB · macOS 27.0 · mlx 0.32.0 · metal-cpp do wheel
**Modelo:** Qwen3.8-Flash-Next-JANG_4M (26 shards, 800 KiB por fatia de expert)

## TL;DR

**Não construir a Trilha D.** O caminho atual (`os.preadv` com fila de 16 threads)
já satura o SSD. Metal IO empata no frio e ganha ~25% só no morno — e o decode
real é ~100% frio (hit rate de expert medido: 1,9%).

| padrão (decode real) | frio | morno |
|---|---|---|
| `preadv` 16 threads + cópia (produção) | **5,68 ms** | 0,52 ms |
| `preadv` 24 threads + cópia | 5,50 ms | 0,51 ms |
| Metal IO qd4 (**= D2, publicação direta**) | 5,65 ms | **0,39 ms** |
| Metal IO qd1 | 5,71 ms | 0,41 ms |
| Metal IO qd16 | 5,60 ms | 0,49 ms |

- **Frio: empate estatístico** (5,50–5,71 ms; spread 3,8%, dentro do ruído).
- **Morno: Metal IO ~25% mais rápido** (0,39 vs 0,52 ms).
- **D1 (staging) é estritamente pior que D2**: o número de Metal IO acima já é o
  caso otimista (carga direto no buffer de destino, zero cópia). D1 adiciona uma
  cópia em cima disso.

## Por que o ganho do PR #3359 (+21% TG) não se transfere

O baseline deles era um `pread` serial por expert. O nosso já tem a fila de
`preadv` paralela da Fase K F6 (QD=16). Medido aqui:

|baseline serial → produção| 9,28 ms → 5,68 ms = **1,63×**|

Ou seja: **o ganho que o PR #3359 atribui ao Fast Resource Loading já foi
capturado por F6.** Metal IO em cima disso vale ~0% no frio.

## O teto é o disco, não o caminho de IO

- Disco sequencial real (medido com `F_NOCACHE`, 2 GiB): **3,3 GB/s**.
- Produção atual no padrão de decode (18,8 MiB espalhados em 5,68 ms): **3,5 GB/s**.

Estamos em ~105% do teto sequencial do dispositivo. **Não há headroom para um
caminho de IO mais rápido.**

## Confirmação: decode é limitado por banda de disco

Geometria real do JANG_4M (`switch_mlp.*.weight`, U32, shape `[512, 2560, 80]`):
**819.200 bytes = 800 KiB por fatia de expert.**

```
48 camadas × 8 experts × 3 projeções × 800 KiB = 900 MiB por token
900 MiB / 3,5 GB/s ≈ 269 ms/token ≈ 3,7 tok/s
```

Observado no bench e2e: **2,7 tok/s**. A diferença é compute + escalas/vieses.
O modelo fecha — decode é limitado por bytes lidos do disco, não por cópias.

## Consequência para o plano

A única alavanca real de decode é **ler menos bytes**, não ler mais rápido:

1. **Residência do cache** — o achado mais forte desta sessão: 1440/1521 slots
   ocupados, **zero evictions**, e ainda assim hit rate de 1,9%. O seeder
   popula ~30 experts/camada que **não são os que o decode demanda**. A pergunta
   certa é *o que* fica residente, não *o que* é evictado — o que explica por que
   `route_frequency` (Trilha A) não moveu a agulha: sem eviction, toda política
   é o mesmo caminho de código.
2. Prefetch/predição (já existe: `warmer.py`, `prefetch.py`).
3. Quantização do tier frio.

## Metodologia (e os erros que a invalidaram antes)

Medições "frias" anteriores eram falsas por três bugs, todos corrigidos:

1. **Eviction por alocação não funciona** — `malloc`+touch+`free` de 24 GiB não
   derruba as páginas do UBC nesta máquina. 64 MiB "frios" saíam a 17–50 GB/s.
2. **Região sacrificial aquecia a região medida** — o run de descarte rodava no
   mesmo offset que a medição "fria".
3. **Cursor passava do EOF** — o arquivo tem 5,1 GB e o cursor avançava 728 MiB
   por medição × 40 medições = 29 GiB. A maioria dos braços lia além do fim
   (`pread` retornava 0 em 0,1 ms) e reportava "MISMATCH".

Versão final (`bench/metal_io_probe/metal_io_probe.cpp`):

- **Cursor frio multi-shard**: 26 shards × 5,1 GB = 133 GB de dados intocados;
  cada medição consome uma região que nenhuma outra tocou, partindo do shard 8.
- **Braços intercalados por repetição** (não em bloco) — elimina viés de deriva
  temporal.
- **Verificação byte-a-byte contra `pread` em toda medição**; short read ⇒ falha
  dura. Todas as 50 medições: `OK`.
- **Rep 0 descartada da leitura** (primeiro toque de um shard novo tem outlier
  sistemático); usa-se a mediana das reps 1..4.

## Artefatos

| arquivo | conteúdo |
|---|---|
| `bench/metal_io_probe/metal_io_probe.cpp` | A/B frio/morno, 10 braços |
| `bench/metal_io_probe/feasibility_probe.cpp` | exatidão byte-a-byte vs `pread` |
| `bench/metal_io_probe/run.sh` | build + execução standalone (sem build do omlx) |
| `bench/results/metal_io_probe/metal_io_ab.log` | saída do A/B |
| `bench/results/metal_io_probe/feasibility.log` | saída do probe de exatidão |

## Dados brutos

### Run 1 — shards 8.., mediana das reps 1..4

```
arm                cold(ms)    GB/s warm(ms)    GB/s
pread 1t               9.42    2.30     0.92   19.91
pread 1t + copy        9.28    2.19     1.20   15.24
pread 8t + copy        6.17    3.40     0.63   29.02
pread 16t + copy       5.68    3.92     0.52   34.90
pread 24t + copy       5.50    4.03     0.51   35.56
mtlio qd1              5.71    2.55     0.41   44.93
mtlio qd4              5.65    3.22     0.39   47.37
mtlio qd8              5.60    3.14     0.42   43.77
mtlio qd16             5.60    3.25     0.49   37.60
mtlio qd24             5.61    3.26     0.60   30.55
```

### Run 2 (confirmação independente) — shards 16.., rep limpa

```
arm                cold(ms) warm(ms)
pread 1t              21.73     0.84
pread 1t + copy        9.53     1.16
pread 8t + copy        5.78     0.69
pread 16t + copy       5.63     0.62
pread 24t + copy       5.79     0.52
mtlio qd1              5.60     0.42
mtlio qd4              5.58     0.42
mtlio qd8              5.57     0.49
mtlio qd16             5.58     0.50
mtlio qd24             5.56     0.64
```

As duas runs convergem: **frio ≈ 5,6 ms para todos os braços com QD ≥ 8
(empate); morno, Metal IO ~0,42 ms vs `preadv` 0,52–0,62 ms (−19% a −32%).**

> Nota: cada run consome ~33 GB de dados frios (10 braços × 5 reps × 664 MiB).
> Escolha `start_file` de modo que ainda restem ~8 shards (~36 GB) à frente,
> senão o pool esgota e as últimas reps saem sem dados.

## Notas técnicas do Metal IO (caso voltemos a ele)

Funciona e é **byte-exact**, incluindo tamanhos/offsets não alinhados (usa o
scratch buffer interno): 64 B, 1000 B e 4097 B com offset +3 conferem com `pread`.

- `MTL::Device::newIOCommandQueue` + `newIOHandle(NS::URL*)` são as únicas
  fábricas expostas no metal-cpp do mlx 0.32 (não há `newIOHandle(const char*)`).
- Metal IO **não** ignora o page cache: a segunda leitura da mesma região sai a
  ~0,3 ms (≈ 50 GB/s), ou seja, vem do UBC.
- `maxCommandsInFlight` alto não ajuda; qd4–qd8 é o ponto doce no frio e qd1–qd4
  no morno.
- Para publicar direto num array MLX: `allocator::Buffer::ptr()` carrega o
  `MTL::Buffer*` no backend Metal (`raw_ptr()` devolve `contents()`). Isso
  precisa de verificação em runtime antes de confiar.
