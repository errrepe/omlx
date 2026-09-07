"""Trilha A: offline route-trace A/B of the expert-cache eviction policies.

Why offline
-----------
The end-to-end bench (``bench_expert_streaming.py``) cannot discriminate the
policies at our operating point: on qwen-jang4m @ 4 GiB it logs
``evict=0 size=1440/1521`` — the hotness seeder fills the cache once and the
demand path never pushes it past its per-layer cap, so LRU, S3-FIFO and route
frequency are literally the same code path. This harness replays a routing
trace directly against the three caches so eviction actually happens, and so
the ranking is not a hostage of free-memory noise (the box sits at ~20 GB
free, which is below the bench's own 22 GB starvation guard).

Trace model
-----------
MoE routing is skewed and two-phased, and that is all the policy cares about:
  * **prefill**  — a broad scan: every layer streams through a wide slice of
    its experts once. This is the "one-hit wonder" traffic LRU handles badly.
  * **decode**   — a narrow reuse loop: ~80% of routes land on a small hot
    pool (Zipf), ~20% on the cold tail.
  * **phase change** — halfway through decode the hot pool rotates (new
    topic), which is what the counter decay has to track.
Topologies are read from the real checkpoints (qwen4_exp 48x512/top-10,
glm5_next 45x288/top-8, deepseek_v4 43x256/top-6). The *shape* is synthetic;
the sizes are not.

Usage
-----
    .venv/bin/python bench/bench_cache_policy_offline.py \
        --out bench/results/route_freq_ab/offline.json

Every arm sees the identical trace (same seed, same list) and the arms are
adjacent, so a difference is the policy and nothing else.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

# (label, layers, experts_per_layer, top_k, hot_pool)
# Sizes come from the checkpoints' config.json; hot_pool=30 matches the
# 1440 seeded slices / 48 layers the e2e bench logs for a 2k prompt.
TRACES = {
    "qwen": ("Qwen3.8-Flash-Next-JANG_4M", 48, 512, 10, 30),
    "glm": ("GLM-5.3-Flash-JANG-MTP", 45, 288, 8, 30),
    "dsv4": ("DeepSeek-V4-Flash-0731-JANG", 43, 256, 6, 30),
}
# Per-layer capacities: budget 4 GiB / 2.69 MB-per-expert / 48 layers ~= 31.
CAPS = (8, 16, 31, 63)
DEFAULT_TOKENS = 256
PROJ = "w"


def build_trace(
    layers: int,
    experts: int,
    top_k: int,
    hot_pool: int,
    tokens: int,
    scan: int,
    phase_change: bool,
    seed: int,
):
    """Return (prefill_events, decode_events) as lists of (layer, eid).

    Deterministic: same seed -> same list, so all arms are comparable.
    """
    import random

    rng = random.Random(seed)
    # Per-layer hot pool, plus a second pool for the phase change.
    hot = [set(rng.sample(range(experts), hot_pool)) for _ in range(layers)]
    hot2 = [set(rng.sample(range(experts), hot_pool)) for _ in range(layers)]

    def route(layer, pool_sorted, cold):
        picks = []
        for _ in range(top_k):
            if rng.random() < 0.8:
                picks.append(pool_sorted[int(len(pool_sorted) ** rng.random()) - 1])
            else:
                picks.append(cold[rng.randrange(len(cold))])
        return picks

    prefill = []
    cold_all = [e for e in range(experts)]
    for layer in range(layers):
        pool_sorted = sorted(hot[layer])
        cold = [e for e in cold_all if e not in hot[layer]]
        # Scan: each "chunk" of the prefill touches top_k fresh-ish experts.
        for _ in range(scan):
            for eid in route(layer, pool_sorted, cold or cold_all):
                prefill.append((layer, eid))

    decode = []
    decode_is_hot = []
    half = tokens // 2 if phase_change else tokens
    for t in range(tokens):
        pool = hot2 if (phase_change and t >= half) else hot
        for layer in range(layers):
            pool_sorted = sorted(pool[layer])
            cold = [e for e in cold_all if e not in pool[layer]]
            for eid in route(layer, pool_sorted, cold or cold_all):
                decode.append((layer, eid))
                decode_is_hot.append(eid in pool[layer])
    return prefill, decode, decode_is_hot


def replay(cache_cls, trace, layers, cap, per_expert_bytes=4096):
    """Drive one cache over (prefill, decode); return decode-phase stats.

    ``hot_hit_rate`` is the number that maps to decode speed: a route into
    the cold tail of a 256-512 expert layer is a miss under *every* policy
    at these budgets, so it only adds a constant to the comparison. What
    separates the policies is whether the reusable hot set stays resident.
    """
    prefill, decode, is_hot = trace
    cache = cache_cls(cap * layers * per_expert_bytes, per_expert_bytes, num_layers=layers)
    # Prefill: demand-fill (miss -> read -> put), same as the demand path.
    for layer, eid in prefill:
        key = (layer, eid, PROJ)
        if cache.get(key) is None:
            cache.put(key, object())
    # Decode: measure only this phase.
    hits = misses = 0
    hot_seen = hot_hits = 0
    t0 = time.perf_counter()
    for (layer, eid), hot_flag in zip(decode, is_hot):
        key = (layer, eid, PROJ)
        if cache.get(key) is not None:
            hits += 1
            if hot_flag:
                hot_hits += 1
        else:
            misses += 1
            cache.put(key, object())
        if hot_flag:
            hot_seen += 1
    elapsed = time.perf_counter() - t0
    total = hits + misses
    return {
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "hot_hit_rate": round(hot_hits / hot_seen, 4) if hot_seen else 0.0,
        "hits": hits,
        "misses": misses,
        "evictions": cache.stats.evictions,
        "replay_ms": round(elapsed * 1000.0, 1),
        "us_per_lookup": round(elapsed * 1e6 / total, 3) if total else 0.0,
        "size": cache.size,
        "capacity": cache.capacity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/route_freq_ab/offline.json")
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    ap.add_argument("--scan", type=int, default=64,
                    help="prefill chunks per layer (scan pressure)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--caps", default=",".join(str(c) for c in CAPS))
    ap.add_argument("--models", default="qwen,glm,dsv4")
    ap.add_argument("--no-phase-change", action="store_true")
    args = ap.parse_args()

    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        RouteFrequencyCache,
        S3FIFOExpertCache,
    )

    arms = {
        "lru": ExpertLRUCache,
        "s3fifo": S3FIFOExpertCache,
        "route_frequency": RouteFrequencyCache,
    }
    caps = [int(c) for c in args.caps.split(",") if c.strip()]
    out = {"traces": {}, "runs": [], "meta": {
        "tokens": args.tokens,
        "scan": args.scan,
        "repeats": args.repeats,
        "phase_change": not args.no_phase_change,
        "platform": platform.platform(),
        "note": ("synthetic trace shape, real MoE topology; per-layer caps "
                 "approximate budgets: 8~1G, 16~2G, 31~4G, 63~8G"),
    }}

    for mkey in [m.strip() for m in args.models.split(",") if m.strip()]:
        label, layers, experts, top_k, hot_pool = TRACES[mkey]
        out["traces"][mkey] = {
            "model": label, "layers": layers, "experts": experts,
            "top_k": top_k, "hot_pool": hot_pool,
        }
        for cap in caps:
            for rep in range(args.repeats):
                trace = build_trace(
                    layers, experts, top_k, hot_pool, args.tokens,
                    args.scan, not args.no_phase_change, seed=1000 + rep,
                )
                row = {"model": mkey, "cap": cap, "rep": rep}
                # Discarded warm-up: the first arm otherwise absorbs page
                # faults, dict resizing and branch-cache cold start, which
                # measured as a fake 8x "cost" for whatever ran first.
                for name in arms:
                    replay(arms[name], trace, layers, cap)
                # Alternating order so no arm gets a systematically warmer box.
                order = list(arms) if rep % 2 == 0 else list(reversed(arms))
                for name in order:
                    row[name] = replay(arms[name], trace, layers, cap)
                out["runs"].append(row)
                print(
                    f"{mkey:5s} cap={cap:3d} rep={rep}  "
                    + "  ".join(
                        f"{p}={row[p]['hot_hit_rate']:.4f}" for p in arms
                    )
                )

    # Aggregate: median hit rate per (model, cap, policy), and the ranking.
    summary = []
    for mkey in out["traces"]:
        for cap in caps:
            rows = [r for r in out["runs"] if r["model"] == mkey and r["cap"] == cap]
            entry = {"model": mkey, "cap": cap, "policies": {}}
            for p in arms:
                hrs = [r[p]["hit_rate"] for r in rows]
                hot = [r[p]["hot_hit_rate"] for r in rows]
                entry["policies"][p] = {
                    "p50_hit_rate": round(statistics.median(hrs), 4),
                    "min_hit_rate": round(min(hrs), 4),
                    "p50_hot_hit_rate": round(statistics.median(hot), 4),
                    "p50_us_per_lookup": round(
                        statistics.median([r[p]["us_per_lookup"] for r in rows]), 3
                    ),
                    "p50_evictions": round(
                        statistics.median([r[p]["evictions"] for r in rows]), 1
                    ),
                }
            ranked = sorted(
                arms, key=lambda p: -entry["policies"][p]["p50_hot_hit_rate"]
            )
            entry["best"] = ranked[0]
            entry["rf_beats_all"] = ranked[0] == "route_frequency"
            summary.append(entry)
    out["summary"] = summary

    rf_wins = sum(1 for e in summary if e["rf_beats_all"])
    out["verdict"] = {
        "route_frequency_wins": rf_wins,
        "of": len(summary),
        "gate": "route_frequency >= s3fifo AND >= lru in >= 2 of 3 models",
        "passed": None,  # filled below per model
    }
    per_model = {}
    for e in summary:
        d = per_model.setdefault(e["model"], {"wins": 0, "total": 0})
        d["total"] += 1
        if e["rf_beats_all"]:
            d["wins"] += 1
    out["verdict"]["per_model"] = per_model
    out["verdict"]["models_where_rf_wins"] = sum(
        1 for d in per_model.values() if d["wins"] > d["total"] / 2
    )
    out["verdict"]["passed"] = out["verdict"]["models_where_rf_wins"] >= 2

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {path}")
    print(json.dumps(out["verdict"], indent=2))
    for e in summary:
        print(
            f"{e['model']:5s} cap={e['cap']:3d} best={e['best']:16s} "
            + " ".join(
                f"{p}=hot{e['policies'][p]['p50_hot_hit_rate']:.4f}"
                f"/all{e['policies'][p]['p50_hit_rate']:.4f}"
                f"/{e['policies'][p]['p50_us_per_lookup']:.2f}us"
                for p in arms
            )
        )


if __name__ == "__main__":
    main()
