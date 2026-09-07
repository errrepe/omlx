"""Fase M4 / Trilha E: honest expert-streaming sizing table for a 48 GB box.

Answers one question: *given model X on a 48 GB unified-memory machine, what
expert-cache budget should I set, and what do I actually get for it?*

Two halves, deliberately separated:

1. **Analytic** (deterministic, zero noise) — geometry from the checkpoint
   headers, converted to slots with the *corrected* arithmetic
   ``per_slot = per_expert // n_proj``. This is where the ``n_proj`` bug
   (see bench/results/residency_fix/SUMMARY.md) previously produced a 3x
   under-fill, so the table reports both the broken and fixed columns.

2. **Measured** (harvested, noisy) — real tok/s, hit rate and disk bytes from
   bench runs. Read ``--measured`` JSONs; nothing is simulated here.

Usage:
    .venv/bin/python bench/bench_sizing_48g.py --out bench/results/sizing
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from omlx.patches.expert_streaming.residency import (  # noqa: E402
    _MODEL_OVERHEAD_FACTOR,
    expert_streaming_estimate,
)

GIB = 1024**3

# Real routing traces (OMLX_EXPERT_STREAMING_TRACE jsonl) captured per model.
# Hit-rate curves are only reported for models that have one — nothing is
# extrapolated across architectures.
TRACES = {
    "Qwen3.8-Flash-Next-JANG_4M": "bench/results/lrc/jang4m_trace.jsonl",
    "Qwen3.8-Flash-Next-JANG_4S": "bench/results/lrc/jang4s_trace.jsonl",
}
PREFILL_POSITIONS = 64

# 48 GB M4 Pro. Unified memory is shared with the GPU, so the usable ceiling
# is well below the sticker number. We report three bands instead of one
# invented constant:
#   - OS + desktop + browser baseline: ~6 GiB wired before omlx starts
#   - MLX peak transient (one layer bank + prefill activations): ~4 GiB
#   - the rest is the streaming budget
TOTAL_GIB = 48.0
BANDS = {
    "conservative": 34.0,  # leaves ~14 GiB for OS/page cache/other apps
    "nominal": 38.0,  # omlx alone on the box
    "aggressive": 42.0,  # risks memory pressure on the page cache
}

# Budgets worth tabulating: from page-cache-only to "half the machine".
BUDGETS_GIB = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]

_MODEL_DIRS = [
    "/Volumes/SSD 4TB/AI Models",
    str(Path.home() / ".cache/huggingface/hub"),
]


@dataclass
class Geometry:
    model: str
    model_type: str
    supported: bool
    reason: str
    num_moe_layers: int
    experts_per_layer: int
    per_expert_bytes: int
    n_proj: int
    per_slot_bytes: int
    expert_gib: float
    dense_gib: float
    ple_gib: float
    resident_gib: float  # no streaming, no PLE mmap
    floor_gib: float  # dense resident with PLE mmap, budget 0


@dataclass
class BudgetRow:
    budget_gib: float
    # --- fixed arithmetic (post n_proj fix) ---
    capacity_slots: int
    slots_per_layer: int
    experts_per_layer_resident: int
    expert_coverage: float  # fraction of experts_per_layer resident
    cache_bytes_real: int
    resident_gib: float  # floor + cache
    # --- what the n_proj bug gave for the SAME budget ---
    broken_capacity_slots: int
    broken_experts_per_layer: int
    broken_cache_bytes_real: int
    broken_resident_gib: float
    underfill_x: float  # fixed / broken bytes
    # Offline hit-rate curves at this budget's experts/layer (None = no trace).
    sch: float | None = None  # Belady oracle, starts empty
    seeded_lru: float | None = None  # prefill top-k seed + LRU (today's policy)
    cold_lru: float | None = None  # LRU from empty
    pct_of_ceiling: float | None = None  # seeded_lru / trace-intrinsic max


def _index_files(model_path: Path) -> list[Path]:
    return sorted(model_path.glob("*.safetensors.index.json")) + sorted(
        model_path.glob("model*.safetensors.index.json")
    )


def _weight_map(model_path: Path) -> dict:
    for idx in _index_files(model_path):
        try:
            return json.loads(idx.read_text()).get("weight_map", {}) or {}
        except Exception:
            continue
    return {}


def _detect_n_proj(weight_map: dict) -> int:
    """Projections per expert: 2 for fused gate_up+down, 3 for split.

    Mirrors the (fixed) discrimination in
    omlx/patches/expert_streaming/__init__.py::_convert_switch_mlp_module:
    decide on ``gate_up_proj`` presence, never on list truthiness --
    ``down_proj`` exists in both layouts.
    """
    has_fused = False
    has_split = False
    for k in weight_map:
        if ".ngram_embedding." in k or ".ple." in k:
            continue
        if "shared_expert" in k:
            continue
        if "gate_up_proj" in k:
            has_fused = True
        elif "gate_proj" in k or "up_proj" in k:
            has_split = True
    if has_fused and not has_split:
        return 2
    if has_split and not has_fused:
        return 3
    # Both (mixed checkpoint) or neither: the runtime takes the majority over
    # converted GLUs; 3 is the conservative (smallest per_slot) choice.
    return 3 if (has_split or not has_fused) else 2


def _ple_gib(model_path: Path) -> float:
    try:
        from omlx.patches.mlx_vlm_qwen4_exp_compat.residency import (
            qwen4_exp_residency_estimate,
        )

        est = qwen4_exp_residency_estimate(str(model_path))
        if est.supported:
            return est.ple_bytes / GIB
    except Exception:
        pass
    return 0.0


def _sizing(
    geom: Geometry, budget_gib: float, assumed_n_proj: int
) -> tuple[int, int, int, int]:
    """(capacity_slots, slots_per_layer, experts_per_layer, cache_bytes).

    ``assumed_n_proj`` is what the *sizing code* believes; the byte size of a
    slot and the slots-per-expert conversion always use the true
    ``geom.n_proj``. Conflating the two is exactly the n_proj bug: the broken
    code reserved ``budget // per_expert`` slots, but every slot still held
    one projection (``per_expert // 3`` bytes), so the cache filled to a
    third of the budget it was promised while the *reported* capacity looked
    fine.
    """
    if not geom.supported or budget_gib <= 0 or geom.per_slot_bytes <= 0:
        return 0, 0, 0, 0
    true_n_proj = geom.n_proj
    per_slot_assumed = max(1, geom.per_expert_bytes // assumed_n_proj)
    budget = int(budget_gib * GIB)
    max_slots = geom.num_moe_layers * geom.experts_per_layer * true_n_proj
    capacity = min(budget // per_slot_assumed, max_slots)
    if capacity <= 0:
        return 0, 0, 0, 0
    per_layer_cap = capacity // geom.num_moe_layers
    # A layer cannot hold more than every projection of every expert.
    per_layer_cap = min(per_layer_cap, geom.experts_per_layer * true_n_proj)
    capacity = per_layer_cap * geom.num_moe_layers
    experts = per_layer_cap // true_n_proj
    # Real bytes: one slot really holds one projection.
    return capacity, per_layer_cap, experts, capacity * geom.per_slot_bytes


def budget_row(geom: Geometry, budget_gib: float) -> BudgetRow:
    cap, per_layer, experts, cache_bytes = _sizing(geom, budget_gib, geom.n_proj)
    bcap, bper_layer, bexperts, bcache_bytes = _sizing(geom, budget_gib, 1)
    return BudgetRow(
        budget_gib=budget_gib,
        capacity_slots=cap,
        slots_per_layer=per_layer,
        experts_per_layer_resident=experts,
        expert_coverage=(experts / geom.experts_per_layer) if geom.experts_per_layer else 0.0,
        cache_bytes_real=cache_bytes,
        resident_gib=geom.floor_gib + cache_bytes / GIB,
        broken_capacity_slots=bcap,
        broken_experts_per_layer=bexperts,
        broken_cache_bytes_real=bcache_bytes,
        broken_resident_gib=geom.floor_gib + bcache_bytes / GIB,
        underfill_x=(cache_bytes / bcache_bytes) if bcache_bytes else 0.0,
    )


def _hit_rates(trace_rel: str, caps: list[int]) -> tuple[dict[int, tuple[float, float, float]], float]:
    """(per-cap curves, trace ceiling).

    Curves are mean (sch, seeded_lru, cold_lru) over all layers.
    ``ceiling`` is the hit rate when *every* expert a layer ever touches is
    resident — i.e. only first touches miss. No policy can beat it; it is a
    property of the trace, not of the cache.

    Reuses the exact simulators from bench/bench_residency_diagnosis.py so the
    curve here and the curve in the residency report cannot drift apart.
    """
    import bench_residency_diagnosis as diag

    prefill_freq, decode = diag.load_trace(str(Path(trace_rel)), PREFILL_POSITIONS)
    layers = sorted(decode)

    # Intrinsic ceiling: give each layer a cache as large as its own union.
    ceil_pts = []
    for lay in layers:
        seq = decode[lay]
        if not seq:
            continue
        ceil_pts.append(diag.sim_lru(seq, len(set().union(*seq))))
    ceiling = statistics.mean(ceil_pts) if ceil_pts else 0.0

    out: dict[int, tuple[float, float, float]] = {}
    for cap in sorted(set(caps)):
        if cap <= 0:
            out[cap] = (0.0, 0.0, 0.0)
            continue
        sch_vals: list[float] = []
        seeded_vals: list[float] = []
        cold_vals: list[float] = []
        for lay in layers:
            seq = decode[lay]
            if not seq:
                continue
            sch_vals.append(diag.sch(seq, cap))
            seed = diag.topk(prefill_freq.get(lay) or {}, cap)
            seeded_vals.append(diag.sim_lru(seq, cap, seed))
            cold_vals.append(diag.sim_lru(seq, cap))
        out[cap] = (
            statistics.mean(sch_vals) if sch_vals else 0.0,
            statistics.mean(seeded_vals) if seeded_vals else 0.0,
            statistics.mean(cold_vals) if cold_vals else 0.0,
        )
    return out, ceiling


def geometry_of(model_path: Path) -> Geometry:
    est = expert_streaming_estimate(str(model_path))
    wm = _weight_map(model_path)
    n_proj = _detect_n_proj(wm)
    ple = _ple_gib(model_path)
    # With PLE mmap active the ngram table leaves resident memory entirely
    # (it is served by the page cache), so the streaming floor drops by it.
    dense_gib = est.dense_bytes / GIB
    floor = max(0.0, (dense_gib - ple)) * _MODEL_OVERHEAD_FACTOR
    return Geometry(
        model=_name(model_path),
        model_type=est.model_type,
        supported=est.supported,
        reason=est.reason or "",
        num_moe_layers=est.num_moe_layers,
        experts_per_layer=est.experts_per_layer,
        per_expert_bytes=est.per_expert_bytes,
        n_proj=n_proj,
        per_slot_bytes=max(1, est.per_expert_bytes // n_proj) if est.per_expert_bytes else 0,
        expert_gib=est.expert_bytes / GIB,
        dense_gib=dense_gib,
        ple_gib=ple,
        resident_gib=est.resident_bytes / GIB,
        floor_gib=floor,
    )


class _Model(Path):  # type: ignore[misc]
    """A Path carrying the friendly name for HF-cache snapshot dirs."""

    display: str = ""


def discover() -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for root in _MODEL_DIRS:
        base = Path(root)
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if not p.is_dir() or p.name.startswith("."):
                continue
            cands: list[tuple[Path, str]] = [(p, p.name)]
            if not (p / "config.json").exists():
                # HF cache layout: models--org--name/snapshots/<sha>/config.json.
                # The snapshot dir is a content hash; display the repo name.
                hits = [
                    q
                    for q in sorted(p.rglob("config.json"))
                    if "snapshots" in q.parts
                ][:1]
                cands = [(q.parent, p.name.replace("--", "/", 1)[7:]) for q in hits]
            for c, name in cands:
                if not (c / "config.json").exists():
                    continue
                if not list(c.glob("*.safetensors")) and not _index_files(c):
                    continue
                if name in seen:
                    continue
                seen.add(name)
                m = _Model(c)
                m.display = name
                found.append(m)
    return found


def _name(p: Path) -> str:
    return getattr(p, "display", "") or p.name


def _harvest_measured(paths: list[str]) -> list[dict]:
    """Real measured points. No simulation: only fields present in the JSON."""
    out: list[dict] = []
    for raw in paths:
        p = Path(raw)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            cs = d.get("cache_stats") or {}
            if "tok_s" not in d and "decode_tok_s" not in d:
                continue
            out.append(
                {
                    "artifact": f.name,
                    "model": d.get("model"),
                    "budget_gib": d.get("budget_gib"),
                    "tok_s": d.get("tok_s") or d.get("decode_tok_s"),
                    "decode_s": d.get("decode_s"),
                    "hit_rate": cs.get("hit_rate") or cs.get("lru_hit"),
                    "per_layer_cap": d.get("cache_per_layer_cap"),
                    "per_expert_cap": d.get("cache_per_expert_cap"),
                    "phys_gib": d.get("phys_after_decode_gib"),
                }
            )
    return out


def render(
    geoms: list[Geometry],
    rows: dict[str, list[BudgetRow]],
    measured: list[dict],
    ceilings: dict[str, float] | None = None,
    knees: dict[str, float] | None = None,
) -> str:
    ceilings = ceilings or {}
    knees = knees or {}
    lines: list[str] = []
    lines.append("# Expert streaming sizing — 48 GB unified memory\n")
    lines.append(
        f"Machine: M4 Pro, {TOTAL_GIB:.0f} GB unified. Usable bands: "
        + ", ".join(f"{k} {v:.0f} GiB" for k, v in BANDS.items())
        + ".\n"
    )
    lines.append(
        "Arithmetic: `per_slot = per_expert // n_proj`, "
        "`capacity = budget // per_slot`, `per_layer_cap = capacity // num_moe_layers`. "
        "The `broken_*` columns are what the `n_proj` short-circuit bug "
        "(`__init__.py:744`) delivered for the same budget.\n"
    )

    lines.append("\n## 1. Geometry\n")
    lines.append(
        "| model | type | L | E/layer | n_proj | per_expert | per_slot | expert GiB | dense GiB | PLE GiB | floor GiB (PLE mmap, budget 0) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in geoms:
        if not g.supported:
            lines.append(f"| {g.model} | {g.model_type} | — | — | — | — | — | — | — | — | unsupported: {g.reason} |")
            continue
        lines.append(
            f"| {g.model} | {g.model_type} | {g.num_moe_layers} | {g.experts_per_layer} | {g.n_proj} "
            f"| {g.per_expert_bytes/1024/1024:.2f} MB | {g.per_slot_bytes/1024/1024:.2f} MB "
            f"| {g.expert_gib:.2f} | {g.dense_gib:.2f} | {g.ple_gib:.2f} | {g.floor_gib:.2f} |"
        )

    for g in geoms:
        if not g.supported:
            continue
        lines.append(f"\n## 2. Budget sweep — {g.model}\n")
        has_curve = any(r.sch is not None for r in rows[g.model])
        head = (
            "| budget GiB | slots total | slots/layer | experts/layer | coverage "
            "| cache GiB (fixed) | resident GiB | broken experts/layer | broken cache GiB "
            "| underfill |"
        )
        if has_curve:
            head = (
                "| budget GiB | slots total | slots/layer | experts/layer | coverage "
                "| cache GiB (fixed) | resident GiB | broken experts/layer | broken cache GiB "
                "| underfill | Belady | seed+LRU | cold LRU | % of cold ceil |"
            )
        lines.append(head)
        lines.append("|" + "---|" * (10 + (4 if has_curve else 0)))
        for r in rows[g.model]:
            row = (
                f"| {r.budget_gib:g} | {r.capacity_slots} | {r.slots_per_layer} "
                f"| {r.experts_per_layer_resident} | {r.expert_coverage*100:.1f}% "
                f"| {r.cache_bytes_real/GIB:.2f} | {r.resident_gib:.2f} "
                f"| {r.broken_experts_per_layer} | {r.broken_cache_bytes_real/GIB:.2f} "
                f"| {r.underfill_x:.2f}x |"
            )
            if has_curve:
                f3 = lambda v: f"{v:.3f}" if v is not None else "—"  # noqa: E731
                pct = f"{r.pct_of_ceiling*100:.0f}%" if r.pct_of_ceiling is not None else "—"
                row += f" {f3(r.sch)} | {f3(r.seeded_lru)} | {f3(r.cold_lru)} | {pct} |"
            lines.append(row)
        if has_curve:
            lines.append("")
            lines.append(
                f"Cold-start ceiling: **{ceilings.get(g.model, 0.0):.3f}** — the hit rate "
                "when every expert a layer ever touches is resident, so only first touches "
                "miss. This is the max for any policy that starts empty.\n"
            )
            lines.append(
                "Belady = oracle that starts empty. seed+LRU = today's policy: prefill "
                "top-k seed, then LRU. cold LRU = no seed. Mean over layers of the "
                "captured decode trace.\n"
            )
            lines.append(
                "**seed+LRU can exceed both** (values >100% in the last column): the seed "
                "is built from *prefill* frequencies, which are future information relative "
                "to decode. That is real and achievable in production — the seeder runs "
                "before the first decode token — not a simulator artifact.\n"
            )
            knee = knees.get(g.model)
            if knee is not None and knee == knee:  # NaN guard
                lines.append(
                    f"**Knee: {knee:g} GiB** already reaches 95% of the ceiling. "
                    "More budget buys no hit rate.\n"
                )
            elif knee is not None:
                lines.append(
                    "No budget in the sweep reaches 95% of the ceiling for this model.\n"
                )
        lines.append("")
        lines.append(
            f"Full residency (streaming becomes pointless) needs "
            f"{g.expert_gib:.2f} GiB of cache — {'reachable' if g.expert_gib <= BANDS['aggressive'] - g.floor_gib else 'NOT reachable on this box'}.\n"
        )
        lines.append("Fit against the 48 GB bands (floor + budget <= usable):\n")
        lines.append(
            "| band | usable GiB | headroom GiB | max budget GiB | slots/layer | experts/layer | coverage |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for band, usable in BANDS.items():
            headroom = usable - g.floor_gib
            if headroom <= 0:
                lines.append(f"| {band} | {usable:.0f} | {headroom:.1f} | — | — | — | — |")
                continue
            _, per_layer, experts, _ = _sizing(g, headroom, g.n_proj)
            lines.append(
                f"| {band} | {usable:.0f} | {headroom:.1f} | {headroom:.1f} | {per_layer} "
                f"| {experts} | {experts / g.experts_per_layer * 100:.1f}% |"
            )

    if measured:
        lines.append("\n## 3. Measured points (real runs, 1 rep each — +/-4% noise)\n")
        lines.append(
            "| artifact | model | budget GiB | tok/s | hit rate | per-layer cap | phys GiB |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for m in sorted(measured, key=lambda x: (x.get("budget_gib") or 0)):
            hr = m.get("hit_rate")
            lines.append(
                f"| {m['artifact']} | {m.get('model')} | {m.get('budget_gib')} "
                f"| {m.get('tok_s', 0):.3f} "
                f"| {format(hr, '.4f') if hr is not None else '—'} | {m.get('per_layer_cap')} "
                f"| {m.get('phys_gib')} |"
            )

    lines.append("\n## 4. Recommended defaults (48 GB)\n")
    lines.append(
        "Rule used: take the knee (smallest budget at >=95% of the cold-start ceiling) "
        "when a trace exists; otherwise 8 GiB; then clamp to the conservative band's "
        "headroom so the box never swaps.\n"
    )
    lines.append("| model | floor GiB | knee GiB | recommended budget GiB | resident GiB | experts/layer |")
    lines.append("|---|---|---|---|---|---|")
    for g in geoms:
        if not g.supported:
            continue
        headroom = BANDS["conservative"] - g.floor_gib
        knee = knees.get(g.model)
        knee_val = knee if (knee is not None and knee == knee) else None
        want = knee_val if knee_val is not None else 8.0
        rec = max(0.0, min(want, headroom))
        _, per_layer, experts, _ = _sizing(g, rec, g.n_proj)
        knee_txt = f"{knee_val:g}" if knee_val is not None else "no trace"
        lines.append(
            f"| {g.model} | {g.floor_gib:.2f} | {knee_txt} | {rec:g} "
            f"| {g.floor_gib + rec:.2f} | {experts} |"
        )

    lines.append("\n## 5. What this table does *not* say\n")
    lines.append(
        "\n".join(
            [
                "- **Budget is not the binding constraint on throughput.** On JANG_4M at "
                "4 GiB the offline policy ceiling is 0.637 hit rate but the measured e2e "
                "hit rate was 0.062 — 10% of attainable. The decode path resolves experts "
                "through the layer-context union and never writes back to the LRU "
                "(`puts == size`, `evict == 0` in the run logs), so the cache is frozen "
                "after seeding. See bench/results/residency_fix/SUMMARY.md.",
                "- **More budget can be slower.** The legacy qwen sweep measured 1.029 tok/s "
                "at 0.5 GiB and 0.343 tok/s at 8 GiB on the same prompt. A user-space cache "
                "competes with the kernel page cache for the same unified memory; the "
                "near-empty-cache arm (admission filter rejecting the seeder) had the "
                "lowest disk read of every run at 681,773 KB/s.",
                "- **Single rep.** Every measured point is one run. Three runs of identical "
                "code spanned 2.647-2.867 tok/s (+/-4%), so no effect below ~8% is "
                "resolvable here.",
                "- **Coverage is not hit rate.** 49% of experts resident (32 GiB on 4M) buys "
                "exactly 0% more hit rate than 12% (8 GiB), because the trace's per-layer "
                "working set is ~103 experts median / 193 max, not 512.",
                "- **Models without a trace** (DeepSeek-V4-Flash-0731-JANG, "
                "GLM-5.3-Flash-JANG-MTP) have no hit-rate curve. Their 8 GiB default is a "
                "placeholder copied from the JANG geometry, not a measurement.",
            ]
        )
        + "\n"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="explicit model dirs")
    ap.add_argument(
        "--measured",
        nargs="*",
        default=[
            "bench/results/residency_fix",
            "bench/results/jang_4M_b4.json",
            "bench/results/qwen_0p5g.json",
            "bench/results/qwen_1g.json",
            "bench/results/qwen_2g.json",
            "bench/results/qwen_4g.json",
            "bench/results/qwen_8g.json",
        ],
        help="JSON artifacts (or dirs) to harvest real measured points from",
    )
    ap.add_argument("--out", default="bench/results/sizing")
    args = ap.parse_args()

    paths = [Path(p) for p in args.models] if args.models else discover()
    geoms = [geometry_of(p) for p in paths]
    geoms.sort(key=lambda g: (-g.supported, g.model))
    rows = {g.model: [budget_row(g, b) for b in BUDGETS_GIB] for g in geoms if g.supported}

    # Attach offline hit-rate curves where a real trace exists.
    ceilings: dict[str, float] = {}
    knees: dict[str, float] = {}
    for g in geoms:
        trace = TRACES.get(g.model)
        if not trace or not Path(trace).exists() or g.model not in rows:
            continue
        caps = [r.experts_per_layer_resident for r in rows[g.model]]
        try:
            curves, ceiling = _hit_rates(trace, caps)
        except Exception as exc:  # a missing/broken trace must not kill the table
            print(f"hit-rate curve skipped for {g.model}: {exc}")
            continue
        ceilings[g.model] = ceiling
        knee: float | None = None
        for r in rows[g.model]:
            sch, seeded, cold = curves.get(
                r.experts_per_layer_resident, (None, None, None)
            )
            r.sch, r.seeded_lru, r.cold_lru = sch, seeded, cold
            r.pct_of_ceiling = (seeded / ceiling) if (ceiling and seeded is not None) else None
            if knee is None and r.pct_of_ceiling is not None and r.pct_of_ceiling >= 0.95:
                knee = r.budget_gib
        knees[g.model] = knee if knee is not None else float("nan")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "machine_gib": TOTAL_GIB,
        "bands_gib": BANDS,
        "geometry": [asdict(g) for g in geoms],
        "sweep": {k: [asdict(r) for r in v] for k, v in rows.items()},
        "measured": _harvest_measured(args.measured),
    }
    (out / "sizing.json").write_text(json.dumps(payload, indent=2))

    # Scrub the non-deterministic bits out of the docstring echo.
    md = render(geoms, rows, payload["measured"], ceilings, knees)
    (out / "SUMMARY.md").write_text(md)

    for g in geoms:
        if g.supported:
            print(
                f"{g.model}: L={g.num_moe_layers} E={g.experts_per_layer} n_proj={g.n_proj} "
                f"per_slot={g.per_slot_bytes} floor={g.floor_gib:.2f} GiB expert={g.expert_gib:.2f} GiB"
            )
    print(f"\nwrote {out/'sizing.json'} and {out/'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
