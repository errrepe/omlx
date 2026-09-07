# Trilha B — Prefault do PLE em SSD (adoção PR #3359)

**Data:** 2026-09-06 · **Máquina:** M4 Pro 48 GB · **Modelo:** Qwen3.8-Flash-Next-JANG_4S
**Artefatos:** `bench/bench_ple_prefault_ab.py`, `bench/results/ple_prefault/ab.json`

## Pergunta

O PLE (tabela N-gram, 14,9 GiB, 320 M linhas × 50 B, 128 shards) fica em mmap com
`MADV_RANDOM`. Um prefill de 8k tokens espalha ~131 k linhas pela tabela inteira:
uma page fault fria de 16 KiB por linha, serial. O PR #3359 resolveu isso com
prefault paralelo (**3,6–10 s → 3,6–3,9 s** no M5 Pro). Isso transfere para o nosso
hardware?

## Protocolo

- **Braços adjacentes** `off` (`OMLX_QWEN4_PLE_PREFAULT=0`) e `on` (`=1`), ordem
  alternada por repetição, **3 repetições**, mediana.
- **Bandas disjuntas e espalhadas:** cada amostra sorteia linhas uniformemente
  dentro de uma fatia própria da tabela (18 fatias). Reproduz o espalhamento real
  (ids N-gram são hash-uniformes) e mantém cada amostra fria.
- **Evicção obrigatória:** sem `purge` (sudo indisponível), uma rodada anterior
  deixa os 14,9 GiB inteiros em page cache e o bench degrada para medir só
  overhead de syscall. Antes da rodada final: alocação de 24 GiB tocada e liberada.
- **Controle de regime:** `cold_over_warm = off_p50 / warm_floor`. Rodadas com
  razão < 2 são marcadas `warm-dominated` e não valem como evidência. A rodada
  final cravou 71× (8k), 186× (2k), 323× (512) → frio de verdade.
- `warm_floor` = re-gather das mesmas linhas já residentes (só compute).

## Resultado

| Prefill | off p50 | on p50 | Ganho | p90/p50 on | warm floor |
|---|---|---|---|---|---|
| 512  | 1858,8 ms | 1074,1 ms | **1,73×** | 1,027 | 5,8 ms |
| 2048 | 4150,7 ms | 1293,8 ms | **3,21×** | 1,022 | 22,3 ms |
| 8192 | 6133,8 ms |  784,1 ms | **7,82×** | 1,020 | 86,6 ms |

Gate do handoff — `p90/p50 ≤ 1,15` no braço `on` e `on ≤ off`: **PASS nos três
tamanhos**, com folga (1,02–1,03). A bimodalidade do braço `off` é o próprio
número: 6,1 s contra um piso computacional de 87 ms — 70× acima do inevitável.

## O que realmente faz o ganho (e o que não faz)

1. **Alinhamento por página é obrigatório.** A primeira versão emitia um `pread`
   por linha (50 B). Numa página de 16 KiB cabem ~327 linhas, então 131 k linhas
   viravam 131 k syscalls sobre ~50 k páginas: **2k e 8k ficaram 2–3× piores que
   off**. Quantizando cada span para página antes do merge, o mesmo gather cai
   para ~50 k preads. Teste de regressão:
   `test_rows_sharing_a_page_collapse_into_one_read`.
2. **O ganho vem da ordem crescente, não do paralelismo.** Emitir os preads em
   offset ascendente deixa o readahead do kernel transformar leitura aleatória em
   bandwidth sequencial (~2,2 GB/s). Comparado no mesmo hardware, 8k frio:
   1 worker = 622 ms, 16 workers = 649 ms; no regime misto, 1 worker = 386 ms vs
   16 workers = 682 ms (**contenção de GIL**). `madvise(MADV_WILLNEED)` deu
   616–690 ms — equivalente ao pread, sem vantagem. `preadv` com buffer
   reutilizado não mudou nada (1,94 µs/span, idêntico ao `pread`): o custo é
   syscall, não alocação. **Default de workers = 1** por medição.
3. **Custo quando a tabela já está residente:** ~2 µs por span, ~0,4 s num
   prefill de 8k, contra 0,14 s sem warm-up. É o preço do lever hoje; a tabela de
   14,9 GiB não cabe em page cache junto do modelo, então frio é o caso comum —
   mas **Trilha D** (publicação direta, zero syscall) é o conserto dos dois lados.

## Veredito

**Transferido.** O item B do PR não era ponto de operação: era amplificação de
leitura aleatória, e ela existe igual no nosso hardware. 8k prefill: **6,1 s →
0,78 s** na contribuição do PLE.

Knobs (todos opt-out, leitura por chamada, bit-exact — prefault nunca altera
valores, só o tempo):

| Env | Default | Efeito |
|---|---|---|
| `OMLX_QWEN4_PLE_PREFAULT` | `1` | liga/desliga o warm-up |
| `OMLX_QWEN4_PLE_PREFAULT_WORKERS` | `1` | clamp 1–64; >1 mediu pior aqui |
| `OMLX_QWEN4_PLE_PREFAULT_MIN_ROWS` | `64` | abaixo disso não prefaulta (protege decode) |

## Próximos passos

- Item 3/4 (Trilha A, `RouteFrequencyCache`) segue na ordem aprovada.
- Reavaliar este lever depois da Trilha D: com publicação direta o custo do
  regime quente desaparece e o default pode subir para mais workers.
- Medir TTFT fim-a-fim (não só o gather) quando o 4S estiver em serving.
