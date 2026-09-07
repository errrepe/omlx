"""Compare write-back A/B runs produced by bench_expert_streaming.py.

Usage:
    .venv/bin/python bench/compare_wb.py bench/results/wb_remeasure

Pairs ``off_r*.json`` against ``on_r*.json`` and prints tok_s, hit rate,
eviction counts, sys_cpu and *absolute* disk bytes (rate x decode_s) —
the absolute-bytes correction matters: a lower KiB/s over a longer decode
can still be MORE bytes.
"""

import json
import sys
from pathlib import Path

GIB = 1024**3


def _dig(d: dict, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _disk_gib(run: dict) -> float:
    """Absolute disk read in GiB for the decode phase.

    Prefers a direct byte counter when the harness records one; otherwise
    reconstructs from the average rate times the phase duration.
    """
    direct = _dig(run, "resources", "phases", "decode", "disk_read_bytes")
    if direct:
        return direct / GIB
    kibs = _dig(run, "resources", "phases", "decode", "disk_read_kib_s_avg")
    # Fall back to a top-level key some harness revisions used.
    if kibs is None:
        kibs = run.get("disk_read_kib_s_avg")
    if kibs is None:
        return float("nan")
    return kibs * 1024 * run.get("decode_s", 0.0) / GIB


def _row(run: dict) -> dict:
    cs = run.get("cache_stats") or {}
    return {
        "tok_s": run.get("tok_s", float("nan")),
        "decode_s": run.get("decode_s", float("nan")),
        "ttft_s": run.get("ttft_s", float("nan")),
        "hit_rate": cs.get("hit_rate", float("nan")),
        "hits": cs.get("hits", 0),
        "misses": cs.get("misses", 0),
        "puts": cs.get("puts", 0),
        "evictions": cs.get("evictions", 0),
        "size": cs.get("size", 0),
        "capacity": cs.get("capacity", run.get("cache_per_expert_cap", 0)),
        "disk_gib": _disk_gib(run),
        "sys_cpu": _dig(
            run, "resources", "phases", "decode", "sys_cpu_pct_avg", default=float("nan")
        ),
        "rss_max_gib": run.get("phys_lifetime_max_gib", float("nan")),
    }


def _fmt(v, spec=".4f"):
    if isinstance(v, float) and v != v:
        return "-"
    return format(v, spec)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    directory = Path(sys.argv[1])
    runs = {}
    for path in sorted(directory.glob("*.json")):
        stem = path.stem
        if "_r" not in stem:
            continue
        arm, _, rep = stem.rpartition("_r")
        runs.setdefault(arm, {})[rep] = _row(json.loads(path.read_text()))

    if not runs:
        print(f"no *_rN.json runs found in {directory}")
        sys.exit(1)

    keys = [
        ("tok_s", ".4f"),
        ("decode_s", ".2f"),
        ("hit_rate", ".4f"),
        ("size", ".0f"),
        ("puts", ".0f"),
        ("evictions", ".0f"),
        ("disk_gib", ".1f"),
        ("sys_cpu", ".2f"),
        ("rss_max_gib", ".2f"),
    ]
    arms = sorted(runs)
    header = f"{'metric':<14}" + "".join(f"{a:>16}" for a in arms)
    print(header)
    print("-" * len(header))
    for key, spec in keys:
        cells = []
        for arm in arms:
            vals = [r[key] for r in runs[arm].values()]
            good = [v for v in vals if v == v]
            if not good:
                cells.append("-")
                continue
            mean = sum(good) / len(good)
            suffix = "" if len(good) == 1 else f" (n={len(good)})"
            cells.append(_fmt(mean, spec) + suffix)
        print(f"{key:<14}" + "".join(f"{c:>16}" for c in cells))

    print()
    for arm in arms:
        for rep, row in sorted(runs[arm].items()):
            print(f"  {arm}_r{rep}: tok_s={_fmt(row['tok_s'])} hit={_fmt(row['hit_rate'])}")

    # Headline delta when both arms are present and single-valued-ish.
    if "off" in runs and "on" in runs:
        off = [r["tok_s"] for r in runs["off"].values()]
        on = [r["tok_s"] for r in runs["on"].values()]
        mo, mn = sum(off) / len(off), sum(on) / len(on)
        print()
        print(f"  write-back tok/s delta: {mn - mo:+.4f} ({(mn / mo - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
