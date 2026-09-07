#!/usr/bin/env python3
"""Trilha C, item 6: convert expert BF16 scales/biases to F16 — opt-in.

The direct-compatible layout (Trilha D) wants expert ``scales`` and
``biases`` as F16 and ``weight`` as U32. The audit
(``scripts/audit_expert_streaming_layout.py``) reports ``needs_conversion``
for checkpoints whose side tensors are BF16: Qwen3.6-35B-A3B-oQ4e-mtp and
Tiel-Coder-35B-A3B-MLX-oQ4e, ~3.8 GiB combined.

Why this is not a blind rewrite
-------------------------------
BF16 has 8 exponent bits and 7 mantissa bits; F16 has 5 and 10. Converting
**gains** mantissa but **loses range**: anything above 65504 becomes Inf, and
anything below the smallest F16 subnormal (5.96e-8) flushes toward zero.
So the conversion is only admissible when it is *exactly* value-preserving.

This tool decides per tensor with a round-trip test rather than a magnitude
heuristic: BF16 bits are widened to F32 (exact — BF16 is a prefix of F32),
cast to F16, cast back, and compared. Equal means the F16 representation is
exact, which automatically covers overflow, subnormals and zero. A tensor
with even one lossy element is **refused**, never silently degraded.

Safety
------
Default is a **dry run**: it plans and reports, writes nothing. Writing
requires an explicit ``--in-place`` or ``--out-dir``. ``--in-place`` rewrites
via a sibling temp file and ``os.replace``, so a crash cannot leave a
half-converted shard. Both dtypes are 2 bytes wide, so tensor sizes — and
therefore the index — are unchanged; only the shard's header and the payload
of converted tensors differ.

Usage
-----
    .venv/bin/python scripts/convert_expert_bf16_to_f16.py MODEL_DIR [MODEL_DIR ...]
    ... --in-place                 # rewrite shards (temp file + rename)
    ... --out-dir /path/to/new     # write converted shards beside a copy
    ... --include-shared           # also convert dense shared experts
    ... --json report.json
    ... --strict                   # exit 1 unless every routed expert tensor is F16/U32
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import audit_expert_streaming_layout as audit  # noqa: E402
except ModuleNotFoundError:
    # Only needed for `python -m scripts.convert_...`; a direct run already
    # has this directory as sys.path[0]. APPEND, never insert(0): scripts/
    # contains bench.py, which would otherwise shadow the top-level bench/
    # package for every later `from bench.X import ...` in the process.
    sys.path.append(str(Path(__file__).resolve().parent))
    import audit_expert_streaming_layout as audit  # noqa: E402

# Largest finite F16. Not used as a threshold — the round-trip test is the
# authority — but reported so a refusal is legible.
F16_MAX_FINITE = 65504.0

# Read/convert granularity. Bounds memory for very large tensors: two passes
# (analyse, then write) so nothing big is ever buffered.
SLICE_ELEMS = 8 << 20
COPY_CHUNK = 8 << 20

BF16 = "BF16"
F16 = "F16"


@dataclass
class TensorPlan:
    """A routed expert side tensor stored as BF16."""

    key: str
    role: str
    arch: str
    begin: int
    end: int
    n_elems: int


@dataclass
class TensorResult:
    key: str = ""
    role: str = ""
    n_elems: int = 0
    converted: bool = False
    reason: str = ""
    n_overflow: int = 0
    n_underflow: int = 0
    n_other_lossy: int = 0


@dataclass
class ShardResult:
    shard: str = ""
    written: str = ""
    results: list = field(default_factory=list)

    @property
    def n_converted(self) -> int:
        return sum(1 for r in self.results if r.converted)

    @property
    def n_refused(self) -> int:
        return sum(1 for r in self.results if not r.converted)

    @property
    def bytes_converted(self) -> int:
        return sum(r.n_elems * 2 for r in self.results if r.converted)


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Widen BF16 bits (uint16) to float32. Exact: BF16 is a prefix of F32."""
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)


def analyse(raw: np.ndarray) -> tuple[bytes, int, int, int]:
    """Return (f16_bytes, n_overflow, n_underflow, n_other_lossy).

    ``f16_bytes`` is empty when any element is not exactly representable in
    F16 — the caller must then leave the tensor alone.
    """
    f32 = bf16_to_f32(raw)
    # Overflow is the expected, *detected* outcome here, not an accident:
    # values past 65504 are exactly what makes a tensor inadmissible. Let the
    # cast saturate quietly and classify afterwards.
    with np.errstate(over="ignore", invalid="ignore"):
        f16 = f32.astype(np.float16)
        back = f16.astype(np.float32)
    # NaN != NaN, so a NaN is neither "equal" nor a value we can corrupt.
    same = (back == f32) | np.isnan(f32)
    if bool(same.all()):
        return f16.tobytes(), 0, 0, 0

    bad = f32[~same]
    finite = np.isfinite(bad)
    n_overflow = int(np.count_nonzero(finite & (np.abs(bad) > F16_MAX_FINITE)))
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = bad.astype(np.float16).astype(np.float32)
    n_underflow = int(np.count_nonzero((bad != 0) & (rounded == 0)))
    n_other = int(bad.size - n_overflow - n_underflow)
    return b"", n_overflow, n_underflow, max(0, n_other)


def _payload_base(path: Path) -> int:
    """Absolute file offset where tensor payloads start (8 + header bytes)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
    return 8 + n


def _slices(begin: int, end: int, width: int = 2):
    span = SLICE_ELEMS * width
    off = begin
    while off < end:
        nxt = min(off + span, end)
        yield off, nxt
        off = nxt


def _read_slice(fh, begin: int, end: int) -> np.ndarray:
    fh.seek(begin)
    return np.frombuffer(fh.read(end - begin), dtype="<u2")


def plan_shard(shard: Path, include_shared: bool) -> list[TensorPlan]:
    """Routed expert side tensors in *shard* that are stored as BF16."""
    try:
        header = audit.read_header(shard)
    except Exception as exc:
        print(f"  ! {shard.name}: {exc}", file=sys.stderr)
        return []
    base = _payload_base(shard)
    plans: list[TensorPlan] = []
    for key, meta in header.items():
        if key == "__metadata__" or "data_offsets" not in meta:
            continue
        got = audit.classify(key)
        if got is None:
            continue
        arch, role, _proj = got
        if arch in audit.SHARED_ARCHS and not include_shared:
            continue
        if role not in ("scales", "biases"):
            continue
        if str(meta.get("dtype", "")).upper() != BF16:
            continue
        b0, b1 = meta["data_offsets"]
        plans.append(
            TensorPlan(
                key=key,
                role=role,
                arch=arch,
                begin=base + int(b0),
                end=base + int(b1),
                n_elems=(int(b1) - int(b0)) // 2,
            )
        )
    plans.sort(key=lambda p: p.begin)
    return plans


def convert_shard(
    shard: Path,
    plans: list[TensorPlan],
    out_path: Path | None,
) -> ShardResult:
    """Analyse then rewrite *shard*. ``out_path`` None means analysis only."""
    result = ShardResult(shard=str(shard))
    header = audit.read_header(shard)

    # Pass 1 — decide, per tensor, whether the conversion is exact.
    verdicts: dict[str, TensorResult] = {}
    with open(shard, "rb") as fh:
        for plan in plans:
            res = TensorResult(key=plan.key, role=plan.role, n_elems=plan.n_elems)
            ok = True
            for b0, b1 in _slices(plan.begin, plan.end):
                _bits, ov, un, ot = analyse(_read_slice(fh, b0, b1))
                res.n_overflow += ov
                res.n_underflow += un
                res.n_other_lossy += ot
                if ov or un or ot:
                    ok = False
            if ok:
                res.converted = True
            else:
                res.reason = _refusal_reason(res)
            verdicts[plan.key] = res
            result.results.append(res)

    if out_path is None:
        return result

    # Pass 2 — write. Sizes are identical (both dtypes are 2 bytes), so only
    # the declared dtype and the payload bytes of converted tensors change.
    new_header = {}
    for key, meta in header.items():
        if key in verdicts and verdicts[key].converted:
            meta = dict(meta)
            meta["dtype"] = F16
        new_header[key] = meta
    blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")

    tmp = out_path.with_suffix(out_path.suffix + ".omlx-f16.tmp")
    with open(shard, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        base = _payload_base(shard)
        ordered = sorted(
            (k for k in header if "data_offsets" in header[k]),
            key=lambda k: header[k]["data_offsets"][0],
        )
        pos = base
        for key in ordered:
            b0 = base + int(header[key]["data_offsets"][0])
            b1 = base + int(header[key]["data_offsets"][1])
            if b0 > pos:
                _copy_range(fi, fo, pos, b0)
            if key in verdicts and verdicts[key].converted:
                for s0, s1 in _slices(b0, b1):
                    fo.write(_to_f16_bytes(_read_slice(fi, s0, s1)))
            else:
                _copy_range(fi, fo, b0, b1)
            pos = b1
        fi.seek(0, os.SEEK_END)
        if pos < fi.tell():
            _copy_range(fi, fo, pos, fi.tell())

    # Atomic: a crash mid-write leaves the original shard intact.
    os.replace(tmp, out_path)
    result.written = str(out_path)
    return result


def _to_f16_bytes(raw: np.ndarray) -> bytes:
    return bf16_to_f32(raw).astype(np.float16).tobytes()


def _copy_range(fi, fo, start: int, end: int) -> None:
    fi.seek(start)
    remaining = end - start
    while remaining > 0:
        chunk = fi.read(min(COPY_CHUNK, remaining))
        if not chunk:
            raise OSError(f"unexpected EOF copying {start}:{end}")
        fo.write(chunk)
        remaining -= len(chunk)


def _refusal_reason(res: TensorResult) -> str:
    parts = []
    if res.n_overflow:
        parts.append(f"{res.n_overflow} over F16 max {F16_MAX_FINITE:g}")
    if res.n_underflow:
        parts.append(f"{res.n_underflow} flushed to zero")
    if res.n_other_lossy:
        parts.append(f"{res.n_other_lossy} lose precision")
    return "; ".join(parts) or "not exactly representable"


def verify_shard(shard: Path, results: list[TensorResult]) -> list[str]:
    """Re-read converted tensors and confirm F16 decodes to the same value."""
    problems: list[str] = []
    converted = [r for r in results if r.converted]
    if not converted:
        return problems
    header = audit.read_header(shard)
    base = _payload_base(shard)
    with open(shard, "rb") as fh:
        for res in converted:
            meta = header.get(res.key)
            if meta is None:
                problems.append(f"{res.key}: missing after conversion")
                continue
            if str(meta.get("dtype", "")).upper() != F16:
                problems.append(f"{res.key}: dtype still {meta.get('dtype')}")
                continue
            b0 = base + int(meta["data_offsets"][0])
            b1 = base + int(meta["data_offsets"][1])
            fh.seek(b0)
            got = np.frombuffer(fh.read(b1 - b0), dtype="<f2").astype(np.float32)
            if got.size != res.n_elems:
                problems.append(
                    f"{res.key}: size changed ({got.size} != {res.n_elems})"
                )
                continue
            if not bool(np.isfinite(got).all()):
                problems.append(f"{res.key}: non-finite value after conversion")
    return problems


def convert_model(
    model_dir: Path,
    out_dir: Path | None,
    in_place: bool,
    include_shared: bool,
    do_verify: bool,
) -> dict:
    shards = audit.shard_files(model_dir)
    out: list[dict] = []
    total_conv = total_ref = total_bytes = 0
    wrote_any = False

    for shard in shards:
        plans = plan_shard(shard, include_shared)
        if not plans:
            continue
        if in_place:
            target: Path | None = shard
        elif out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / shard.name
        else:
            target = None
        res = convert_shard(shard, plans, target)
        wrote_any = wrote_any or target is not None
        if do_verify and target is not None:
            problems = verify_shard(target, res.results)
            for p in problems:
                print(f"  ! verify {shard.name}: {p}", file=sys.stderr)
        out.append({
            "shard": shard.name,
            "written": res.written,
            "n_converted": res.n_converted,
            "n_refused": res.n_refused,
            "bytes_converted": res.bytes_converted,
            "tensors": [
                {
                    "key": r.key,
                    "role": r.role,
                    "n_elems": r.n_elems,
                    "converted": r.converted,
                    "reason": r.reason,
                    "n_overflow": r.n_overflow,
                    "n_underflow": r.n_underflow,
                    "n_other_lossy": r.n_other_lossy,
                }
                for r in res.results
            ],
        })
        total_conv += res.n_converted
        total_ref += res.n_refused
        total_bytes += res.bytes_converted

    return {
        "model": str(model_dir),
        "name": model_dir.name,
        "dry_run": not wrote_any,
        "shards_touched": len(out),
        "tensors_converted": total_conv,
        "tensors_refused": total_ref,
        "bytes_converted": total_bytes,
        "shards": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dirs", nargs="+", type=Path)
    ap.add_argument("--in-place", action="store_true",
                    help="rewrite shards via temp file + rename")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write converted shards here (original untouched)")
    ap.add_argument("--include-shared", action="store_true",
                    help="also convert dense shared experts (resident, not streamed)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 unless every routed expert side tensor is F16")
    args = ap.parse_args()

    if args.in_place and args.out_dir is not None:
        print("--in-place and --out-dir are mutually exclusive", file=sys.stderr)
        return 2

    reports = []
    for model_dir in args.model_dirs:
        if not model_dir.is_dir():
            print(f"  ! not a directory: {model_dir}", file=sys.stderr)
            continue
        print(f"== {model_dir.name}")
        rep = convert_model(
            model_dir,
            args.out_dir,
            args.in_place,
            args.include_shared,
            not args.no_verify,
        )
        reports.append(rep)
        mode = "DRY RUN (nothing written)" if rep["dry_run"] else "written"
        print(
            f"   {mode}: {rep['tensors_converted']} converted "
            f"({rep['bytes_converted'] / 2**30:.2f} GiB), "
            f"{rep['tensors_refused']} refused, "
            f"{rep['shards_touched']} shard(s)"
        )
        for shard in rep["shards"]:
            for t in shard["tensors"]:
                if not t["converted"]:
                    print(f"   ! refused {t['key']}: {t['reason']}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2))

    if not (args.in_place or args.out_dir is not None):
        print(
            "\nDry run only. Re-run with --in-place or --out-dir DIR to write.",
            file=sys.stderr,
        )

    if args.strict:
        bad = [r for r in reports if r["tensors_refused"]]
        if bad:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
