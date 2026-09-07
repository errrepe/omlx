#!/usr/bin/env python3
"""Por que o hit rate medido (0,062) e' 10x menor que o teto offline (0,637)?

O sistema real: seed por frequencia de prefill, depois o LRU **congela**
(sem write-back, evictions=0, puts==size). O simulador de sizing roda
`sim_lru(seq, cap, seed)` — LRU que **continua atualizando**. A hipotese e'
que a diferenca entre os dois e' o proprio gap.

Calcula, por layer, em cap experts:
  belady      - oracle que comeca vazio
  seed_static - cache congelado no seed  <-- o que a producao faz
  seed_lru    - seed + LRU atualizando   <-- o que o sizing previu (0,637)
  cold_lru    - LRU do zero
  overlap     - fracao do seed que o decode realmente toca
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, OrderedDict
from pathlib import Path

DEFAULT_TRACE = Path(__file__).resolve().parent / "results/lrc/jang4m_trace.jsonl"


def load(threshold: int, trace: Path):
    prefill_freq: dict[int, Counter] = {}
    decode: dict[int, list[tuple[int, frozenset]]] = {}
    pos_hist: Counter = Counter()
    with open(trace) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lay = int(r["layer"])
            pos = int(r["positions"])
            uniq = frozenset(int(x) for x in r["uniq"])
            pos_hist[pos] += 1
            if pos > threshold:
                prefill_freq.setdefault(lay, Counter()).update(uniq)
            else:
                decode.setdefault(lay, []).append((int(r["call"]), uniq))
    for lay in decode:
        decode[lay].sort(key=lambda t: t[0])
    return prefill_freq, {k: [u for _, u in v] for k, v in decode.items()}, pos_hist


def topk(freq: Counter, k: int) -> list[int]:
    return [e for e, _ in freq.most_common(k)]


def seed_static(seq, cap: int, seed: list[int]) -> float:
    """Cache congelado: so' o seed esta' la', nada entra nem sai."""
    cache = set(seed[:cap])
    hits = total = 0
    for need in seq:
        hits += len(cache & need)
        total += len(need)
    return hits / total if total else 0.0


def sim_lru(seq, cap: int, seed: list[int] | None = None) -> float:
    store: OrderedDict[int, None] = OrderedDict()
    for e in (seed or [])[:cap]:
        store[e] = None
    hits = total = 0
    for need in seq:
        for e in sorted(need):
            total += 1
            if e in store:
                hits += 1
                store.move_to_end(e)
            else:
                store[e] = None
                if len(store) > cap:
                    store.popitem(last=False)
    return hits / total if total else 0.0


def belady(seq, cap: int) -> float:
    occ: dict[int, list[int]] = {}
    for i, s in enumerate(seq):
        for e in s:
            occ.setdefault(e, []).append(i)
    horizon = len(seq)

    def next_use(e: int, i: int) -> int:
        for p in occ.get(e, ()):  # ja' ordenado
            if p > i:
                return p
        return horizon

    cache: set[int] = set()
    hits = total = 0
    for i, need in enumerate(seq):
        hits += len(cache & need)
        total += len(need)
        for e in sorted(need - cache):
            if len(cache) >= cap:
                victim = max(cache, key=lambda c: next_use(c, i))
                if next_use(victim, i) > next_use(e, i):
                    cache.discard(victim)
                    cache.add(e)
            else:
                cache.add(e)
    return hits / total if total else 0.0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--caps", type=int, nargs="+", default=[31])
    args = ap.parse_args()
    trace = args.trace
    _, _, pos_hist = load(10**9, trace)
    print("distribuicao de `positions`:")
    for p, n in sorted(pos_hist.items()):
        print(f"   positions={p:<6} x{n}")

    for threshold in (64, 128, 256):
        pf, dec, _ = load(threshold, trace)
        if not pf or not dec:
            print(f"\nthreshold {threshold}: sem split util "
                  f"(prefill={len(pf)} layers, decode={len(dec)} layers)")
            continue
        print(f"\n{'='*62}\nthreshold positions>{threshold}: "
              f"{len(pf)} layers prefill, {len(dec)} layers decode")

        for cap in args.caps:
            b, ss, sl, cl, ov, ws = [], [], [], [], [], []
            for lay, seq in sorted(dec.items()):
                if not seq:
                    continue
                seed = topk(pf.get(lay) or Counter(), cap)
                work = set()
                for s in seq:
                    work |= s
                b.append(belady(seq, cap))
                ss.append(seed_static(seq, cap, seed))
                sl.append(sim_lru(seq, cap, seed))
                cl.append(sim_lru(seq, cap))
                ws.append(len(work))
                ov.append(len(work & set(seed)) / max(1, len(seed)))
            print(f"\n  cap={cap} experts/layer (budget 4 GiB, n_proj=3)")
            print(f"    Belady            {statistics.mean(b):.4f}")
            print(f"    seed ESTATICO     {statistics.mean(ss):.4f}   <-- producao")
            print(f"    seed + LRU        {statistics.mean(sl):.4f}   <-- sizing")
            print(f"    cold LRU          {statistics.mean(cl):.4f}")
            print(f"    working set/layer mediana {statistics.median(ws):.0f} "
                  f"(max {max(ws)})")
            print(f"    overlap seed^work {statistics.mean(ov):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
