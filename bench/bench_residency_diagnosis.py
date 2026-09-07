"""Fase M4 — por que o cache de experts tem hit rate de 1.9%?

Separa as tres explicacoes candidatas, que tem remedios completamente
diferentes:

  1. **Budget pequeno demais** — 31 slots/layer de 512 experts e' pouco.
     Remedio: mais memoria. (Falso, se o oraculo for alto.)
  2. **Conjunto errado** — o cache esta' cheio, mas com experts que o
     decode nunca pede. Remedio: trocar a politica de seeding.
  3. **Cache congelado** — o cache nunca admite demanda de decode.
     Remedio: consertar o caminho de insercao.

Mede, por layer, sobre a trace de roteamento real:

  - **SCH(K)**: hit rate de Belady (oraculo que ve o futuro) com K slots.
    E' o teto fisico de qualquer politica. Responde (1).
  - **seed_prefill(K) + LRU(K)**: o que o codigo faz hoje — semeia com o
    top-K por frequencia de *prefill* e roda LRU. Responde (2).
  - **seed_decode(K) + LRU(K)**: semeia com o top-K por frequencia dos
    primeiros 25% do decode. Limite do que um seeder honesto conseguiria.
  - **lru_cold(K)**: LRU partindo do zero, sem seed. Mostra quanto o seed
    atual ajuda (ou atrapalha).

Uso:
    .venv/bin/python bench/bench_residency_diagnosis.py \
        --trace bench/results/lrc/jang4m_trace.jsonl --cap 31

A trace e' o JSONL de `OMLX_EXPERT_STREAMING_TRACE` (uma linha por chamada
MoE: `{"call", "layer", "positions", "uniq"}`). Linhas com
`positions > --prefill-positions` sao tratadas como prefill.
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
from collections import Counter, OrderedDict
from pathlib import Path


def load_trace(path: str, prefill_positions: int):
    """-> (prefill_freq, decode_seq) por layer.

    prefill_freq[layer] = Counter(expert -> n de chamadas de prefill em que aparece)
    decode_seq[layer]   = [frozenset(uniq), ...] em ordem de chamada
    """
    prefill_freq: dict[int, Counter] = {}
    decode_rows: dict[int, list[tuple[int, frozenset]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            layer = int(r["layer"])
            uniq = frozenset(int(e) for e in r["uniq"])
            if int(r.get("positions", 0)) > prefill_positions:
                prefill_freq.setdefault(layer, Counter()).update(uniq)
            else:
                decode_rows.setdefault(layer, []).append((int(r["call"]), uniq))
    for layer in decode_rows:
        decode_rows[layer].sort(key=lambda t: t[0])
    return prefill_freq, {k: [u for _, u in v] for k, v in decode_rows.items()}


def _occurrences(seq):
    """expert -> lista ordenada de indices de chamada onde e' necessario."""
    occ: dict[int, list[int]] = {}
    for i, s in enumerate(seq):
        for e in s:
            occ.setdefault(e, []).append(i)
    return occ


def sch(seq, cache_size: int) -> float:
    """Belady classico com demanda em lote — mesma regra do lrc_analysis.py.

    Na chamada i, os experts presentes no cache sao acerto; os faltantes sao
    inseridos evictando o expert cacheado cujo proximo uso esta' MAIS DISTANTE
    no futuro (e so' se o entrante for estritamente mais proximo — quem nunca
    mais sera' usado nao desloca ningaum).
    """
    if cache_size <= 0 or not seq:
        return 0.0
    occ = _occurrences(seq)
    horizon = len(seq)

    def next_use(e: int, i: int) -> int:
        positions = occ.get(e)
        if not positions:
            return horizon
        j = bisect.bisect_right(positions, i)
        return positions[j] if j < len(positions) else horizon

    cache: set[int] = set()
    hits = 0
    total = 0
    for i, need in enumerate(seq):
        hits += len(cache & need)
        total += len(need)
        for e in sorted(need - cache):
            if len(cache) >= cache_size:
                victim = max(cache, key=lambda c: next_use(c, i))
                if next_use(victim, i) > next_use(e, i):
                    cache.discard(victim)
                    cache.add(e)
            else:
                cache.add(e)
    return hits / total if total else 0.0


def sim_lru(seq, cache_size: int, seed: list[int] | None = None) -> float:
    """LRU por layer com capacidade `cache_size`, opcionalmente pre-sembrado.

    `seed` e' inserido primeiro (na ordem dada), depois a demanda de decode
    roda com LRU normal. `seed` alem da capacidade e' truncado.
    """
    if cache_size <= 0 or not seq:
        return 0.0
    store: OrderedDict[int, None] = OrderedDict()
    for e in (seed or [])[:cache_size]:
        store[e] = None
    hits = 0
    total = 0
    for needed in seq:
        for e in sorted(needed):
            total += 1
            if e in store:
                hits += 1
                store.move_to_end(e)
            else:
                store[e] = None
                if len(store) > cache_size:
                    store.popitem(last=False)
    return hits / total if total else 0.0


def topk(counter: Counter, k: int) -> list[int]:
    """Top-K por frequencia, com desempate determinista (ordem crescente de id)."""
    return [e for e, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="bench/results/lrc/jang4m_trace.jsonl")
    ap.add_argument("--cap", default="4,8,10,16,31,64,128")
    ap.add_argument("--prefill-positions", type=int, default=64)
    ap.add_argument("--learn-frac", type=float, default=0.25,
                   help="fracao inicial do decode usada para aprender o hot set")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    caps = [int(c) for c in args.cap.split(",") if c.strip()]
    prefill_freq, decode = load_trace(args.trace, args.prefill_positions)
    layers = sorted(decode)
    if not layers:
        print(f"nenhuma linha de decode em {args.trace}")
        return 1

    n_calls = {l: len(decode[l]) for l in layers}
    union = {l: len(set().union(*decode[l])) for l in layers}
    print(f"trace={args.trace}")
    print(f"layers={len(layers)}  decode calls/layer: min={min(n_calls.values())} "
          f"med={statistics.median(n_calls.values())}  max={max(n_calls.values())}")
    print(f"working set (união de experts por layer): min={min(union.values())} "
          f"med={statistics.median(union.values())}  max={max(union.values())}")
    print(f"prefill layers com frequencia: {len(prefill_freq)}")
    if prefill_freq:
        pf = prefill_freq[layers[0]]
        print(f"  layer 0: {len(pf)} experts com freq; distribuicao de contagens = "
              f"{dict(sorted(Counter(pf.values()).items()))}")
        tot = sum(pf.values())
        top10 = sum(c for _, c in sorted(pf.items(), key=lambda kv: (-kv[1], kv[0]))[:10])
        print(f"  layer 0: top-10 por freq de prefill cobre "
              f"{top10 / tot:.1%} das aparicoes de prefill "
              f"(baseline uniforme = {10 / len(pf):.1%})")
        # cobertura do top-10 de prefill sobre a demanda real de decode
        d0 = Counter()
        for s in decode[layers[0]]:
            d0.update(s)
        dtot = sum(d0.values())
        dtop = sum(d0.get(e, 0) for e in topk(pf, 10))
        print(f"  layer 0: esse mesmo top-10 de prefill cobre "
              f"{dtop / dtot:.1%} da demanda de DECODE  <-- o que o seeder escolhe")

    print()
    hdr = (f"{'cap':>5} {'SCH(oracle)':>12} {'seed_decode+LRU':>16} "
           f"{'seed_prefill+LRU':>18} {'LRU frio':>10}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for c in caps:
        schs, sdec, spref, cold = [], [], [], []
        for l in layers:
            seq = decode[l]
            schs.append(sch(seq, c))
            cold.append(sim_lru(seq, c))
            dc = Counter()
            for s in seq[: max(1, int(len(seq) * args.learn_frac))]:
                dc.update(s)
            sdec.append(sim_lru(seq, c, topk(dc, c)))
            spref.append(sim_lru(seq, c, topk(prefill_freq.get(l, Counter()), c)))
        row = {
            "cap": c,
            "sch": statistics.mean(schs),
            "seed_decode": statistics.mean(sdec),
            "seed_prefill": statistics.mean(spref),
            "lru_cold": statistics.mean(cold),
        }
        rows.append(row)
        print(f"{c:>5} {row['sch']:>12.4f} {row['seed_decode']:>16.4f} "
              f"{row['seed_prefill']:>18.4f} {row['lru_cold']:>10.4f}")

    out = args.out or str(Path(args.trace).with_name("residency_diagnosis.json"))
    Path(out).write_text(json.dumps(
        {
            "trace": args.trace,
            "layers": len(layers),
            "decode_calls_per_layer_median": statistics.median(n_calls.values()),
            "working_set_median": statistics.median(union.values()),
            "rows": rows,
        },
        indent=1,
    ))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
