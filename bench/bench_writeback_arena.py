"""Fase M4 — sera que admitir *views* no LRU prende o banco inteiro vivo?

`_read_expert_banks` (streaming_switch.py:2199-2207) devolve as linhas como
`np.frombuffer(banks[k][i], ...)` — **views** zero-copy dentro de um banco
contiguo `(n_missing, per_slot)` alocado por (projecao, chamada de layer).
Quando o write-back faz `put(bundle_key(eid), raw)` com essas views, cada
entrada do LRU segura uma referencia ao banco *inteiro*.

Consequencia possivel: a contabilidade do cache (`size * per_slot`) so' bate
com a memoria real se TODAS as linhas de um banco continuarem vivas. Quando a
eviccao leva parte delas, o banco inteiro sobrevive por causa das que
ficaram — amplificacao de memoria nao contabilizada.

Este bench mede exatamente isso, sem engine: constroi o LRU com a geometria
real do JANG_4M e simula admissoes de decode.

Uso:
    .venv/bin/python bench/bench_writeback_arena.py --tokens 40
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np

from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

# Geometria real do Qwen3.8-Flash-Next-JANG_4M com o fix do n_proj.
PER_SLOT = 940_800
CAPACITY = 4565
LAYERS = 48
N_PROJ = 3
MISSING_PER_PROJ = 10  # decode: ~top-k experts faltando por projecao


def _rss_gib() -> float:
    # ru_maxrss is bytes on darwin
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def _root(arr) -> object:
    b = arr
    seen = 0
    while getattr(b, "base", None) is not None and seen < 8:
        b = b.base
        seen += 1
    return b


def retained_bytes(cache) -> int:
    """Bytes realmente retidos: soma dos bancos-root alcancaveis do cache."""
    seen: set[int] = set()
    total = 0
    for value in cache._store.values():
        for arr in value:
            if arr is None:
                continue
            root = _root(arr)
            if id(root) not in seen:
                seen.add(id(root))
                total += int(getattr(root, "nbytes", 0) or 0)
    return total


def accounted_bytes(cache) -> int:
    return cache.size * PER_SLOT


def run(arm: str, tokens: int) -> dict:
    """Simula `tokens` passos de decode admitindo tudo que falta."""
    # budget = capacity * per_slot reconstrói a capacidade que o runtime usa
    # (4565 slots x 940800 B a 4 GiB no JANG_4M, pós-fix do n_proj).
    cache = ExpertLRUCache(
        budget_bytes=CAPACITY * PER_SLOT,
        per_expert_bytes=PER_SLOT,
        num_layers=LAYERS,
    )
    admitted = 0
    peak_retained = 0
    peak_accounted = 0
    t0 = time.perf_counter()
    eid = 0
    for _tok in range(tokens):
        for layer in range(LAYERS):
            for _proj in range(N_PROJ):
                # O banco e alocado e lido de qualquer forma (caminho union).
                bank = np.empty((MISSING_PER_PROJ, PER_SLOT), dtype=np.uint8)
                bank[:] = 1  # simula o custo do read (mesmo nos dois bracos)
                if arm == "views":
                    rows = [
                        np.frombuffer(bank[i], dtype=np.uint8) for i in range(MISSING_PER_PROJ)
                    ]
                elif arm == "copies":
                    rows = [bank[i].copy() for i in range(MISSING_PER_PROJ)]
                else:  # baseline: sem write-back, banco morre aqui
                    rows = None
                if rows is not None:
                    for i, raw in enumerate(rows):
                        cache.put((layer, eid + i, f"w{_proj}"), (raw, None, None))
                        admitted += 1
                eid += MISSING_PER_PROJ
        if _tok % 5 == 0:
            gc.collect()
            peak_retained = max(peak_retained, retained_bytes(cache))
            peak_accounted = max(peak_accounted, accounted_bytes(cache))
    gc.collect()
    dt = time.perf_counter() - t0
    ret = retained_bytes(cache)
    acc = accounted_bytes(cache)
    out = {
        "arm": arm,
        "tokens": tokens,
        "admitted": admitted,
        "size": cache.size,
        "capacity": cache.capacity,
        "evictions": cache.stats.evictions,
        "accounted_gib": acc / 1024**3,
        "retained_gib": ret / 1024**3,
        "peak_retained_gib": peak_retained / 1024**3,
        "amplification": (ret / acc) if acc else 0.0,
        "wall_s": dt,
        "us_per_admission": (dt / admitted * 1e6) if admitted else 0.0,
        "rss_max_gib": _rss_gib(),
    }
    del cache
    gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--out", default="bench/results/writeback_arena.json")
    args = ap.parse_args()

    results = [run(arm, args.tokens) for arm in ("baseline", "views", "copies")]

    print(
        f"{'arm':9} {'admitted':>9} {'size':>6} {'evict':>8} "
        f"{'acc GiB':>8} {'retained GiB':>13} {'amplif':>7} "
        f"{'us/admit':>9} {'rss GiB':>8}"
    )
    for r in results:
        print(
            f"{r['arm']:9} {r['admitted']:>9} {r['size']:>6} {r['evictions']:>8} "
            f"{r['accounted_gib']:>8.2f} {r['retained_gib']:>13.2f} "
            f"{r['amplification']:>7.2f}x {r['us_per_admission']:>9.2f} "
            f"{r['rss_max_gib']:>8.2f}"
        )

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
