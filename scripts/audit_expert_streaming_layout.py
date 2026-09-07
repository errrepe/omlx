"""Trilha C: audit MoE checkpoints for the direct-compatible expert layout.

Why this exists
---------------
Trilha D (native Fast Resource Loading) wants to publish an expert's bytes to
the GPU straight from the mapped file, with no staging copy and no dtype
conversion on the hot path. That is only possible when the bytes on disk are
already the bytes the kernel expects:

    weight   -> U32   (quantized words packed into uint32)
    scales   -> F16
    biases   -> F16

Anything else needs a copy-and-convert step, which is exactly the cost Trilha
D is trying to remove. This script answers "how much of the corpus is already
there?" before a line of Metal is written.

It is **read-only and header-only**: it reads the 8-byte length prefix and the
JSON header of every shard. It never touches tensor payloads, so it is safe to
run against a model volume that is mounted read-only, and it costs milliseconds
even on a 103-shard checkpoint.

Usage
-----
    .venv/bin/python scripts/audit_expert_streaming_layout.py MODEL_DIR [MODEL_DIR ...]
    .venv/bin/python scripts/audit_expert_streaming_layout.py DIR --json out.json
    .venv/bin/python scripts/audit_expert_streaming_layout.py DIR --strict   # exit 1 if not clean

Verdicts
--------
    direct_compatible  every expert tensor already has the target dtype
    needs_conversion   only BF16 scales/biases (F16-widening; see the caveat)
    incompatible       at least one tensor has no path to the target dtype

BF16 -> F16 caveat
------------------
BF16 has 8 exponent bits and 7 mantissa bits; F16 has 5 and 10. So the
conversion *gains* mantissa precision but **loses range**: any |x| > 65504
overflows to infinity. It is therefore only lossless when every value fits in
F16 range, which this script cannot know from headers alone — it reports the
candidate byte volume and leaves the range check to the converter (item 6).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

# Target layout for direct publication (Trilha D).
TARGET_DTYPES = {"weight": "U32", "scales": "F16", "biases": "F16"}

DTYPE_BYTES = {
    "F16": 2, "BF16": 2, "F32": 4, "U32": 4, "I32": 4,
    "I64": 8, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1,
}

# Architectures, in match order. Both real conventions put the role last and
# the projection second-to-last; they differ in where the expert index lives
# (qwen4_exp/glm5_next fold it into the leading tensor dim, deepseek_v4 makes
# it an explicit path component).
_ARCH_PATTERNS = (
    (
        "switch_mlp",
        re.compile(
            r"(?:^|\.)layers\.(?P<layer>\d+)\.mlp\.switch_mlp\."
            r"(?P<proj>[A-Za-z0-9_]+)\.(?P<role>weight|scales|biases)$"
        ),
    ),
    (
        "ffn_experts",
        re.compile(
            r"(?:^|\.)layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
            r"(?P<proj>w1|w2|w3)\.(?P<role>weight|scales|biases)$"
        ),
    ),
    # MTP draft module: its own expert stack, same layout rules. Without an
    # explicit pattern these fall through to the generic bucket and look
    # like an unknown convention in the report.
    (
        "mtp_experts",
        re.compile(
            r"(?:^|\.)mtp\.layers\.(?P<layer>\d+)\.mlp\.experts\."
            r"(?P<proj>[A-Za-z0-9_]+)\.(?P<role>weight|scales|biases)$"
        ),
    ),
    # Shared (dense) experts run on every token, so they are resident rather
    # than streamed and evicted. Reported, but excluded from the verdict:
    # their layout is not what gates Trilha D's hot path.
    (
        "shared_experts",
        re.compile(
            r"(?:^|\.)(?:mlp|ffn)\.shared_experts\."
            r"(?P<proj>[A-Za-z0-9_]+)\.(?P<role>weight|scales|biases)$"
        ),
    ),
)

# Only these are excluded from the verdict. Everything else — including the
# generic fallback for a convention we have never seen — counts as routed.
# That is deliberate: an unclassifiable expert tensor must fail the audit
# loudly rather than be silently ignored as "somebody else's problem".
SHARED_ARCHS = frozenset({"shared_experts"})

# Fallback for an unseen convention: a real expert index or a switch_mlp
# block, ending in a known role. Deliberately does NOT match ``ffn`` alone --
# that also catches the router (``layers.N.ffn.gate.weight``, F32 by design),
# which is not an expert weight and must not be flagged.
_GENERIC_EXPERT = re.compile(r"(switch_mlp|experts\.\d+)", re.IGNORECASE)
_GENERIC_ROLE = re.compile(r"(?P<role>weight|scales|biases)$")


def read_header(path: Path) -> dict:
    """Parse a safetensors header (length prefix + JSON). Header-only."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError(f"{path.name}: too short to be safetensors")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 512 * 1024 * 1024:
            raise ValueError(f"{path.name}: implausible header length {n}")
        return json.loads(f.read(n))


def shard_files(model_dir: Path) -> list[Path]:
    """Shards of the model itself, honouring the index when one exists.

    A checkpoint directory can carry unrelated safetensors (calibration
    blobs, AWQ scales, adapter dumps). When an index exists it is the
    authority; otherwise we take the ``model-*.safetensors`` convention.
    """
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            names = sorted(set(weight_map.values()))
            files = [model_dir / n for n in names if (model_dir / n).is_file()]
            if files:
                return files
        except Exception:
            pass  # fall through to the glob
    return sorted(model_dir.glob("model-*.safetensors"))


def classify(key: str) -> tuple[str, str, str] | None:
    """Return (arch, role, proj) for an expert tensor key, else None."""
    for arch, rx in _ARCH_PATTERNS:
        m = rx.search(key)
        if m:
            return arch, m.group("role"), m.group("proj")
    if _GENERIC_EXPERT.search(key):
        m = _GENERIC_ROLE.search(key)
        if m:
            return "generic", m.group("role"), "?"
    return None


def _nbytes(shape, dtype: str) -> int:
    size = DTYPE_BYTES.get(dtype)
    if size is None:
        return 0
    total = 1
    for d in shape or ():
        total *= int(d)
    return total * size


def audit_model(model_dir: Path) -> dict:
    """Audit one checkpoint directory. Header-only, never reads payloads."""
    files = shard_files(model_dir)
    counts: dict[tuple[str, str, str, str], int] = {}
    bytes_by_group: dict[tuple[str, str, str, str], int] = {}
    layers: set[int] = set()
    unknown_roles: set[str] = set()

    for path in files:
        try:
            header = read_header(path)
        except Exception as exc:  # unreadable shard must not silence the rest
            print(f"  ! {path.name}: {exc}", file=sys.stderr)
            continue
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            got = classify(key)
            if got is None:
                continue
            arch, role, proj = got
            dtype = str(meta.get("dtype", "?")).upper()
            group = (arch, proj, role, dtype)
            counts[group] = counts.get(group, 0) + 1
            bytes_by_group[group] = (
                bytes_by_group.get(group, 0) + _nbytes(meta.get("shape"), dtype)
            )
            lm = re.search(r"layers\.(\d+)", key)
            if lm:
                layers.add(int(lm.group(1)))

    groups = []
    for (arch, proj, role, dtype), n in sorted(counts.items(), key=lambda kv: kv[0]):
        target = TARGET_DTYPES.get(role)
        if dtype == target:
            status = "ok"
        elif role in ("scales", "biases") and dtype == "BF16":
            # Widening in mantissa, narrowing in range: convertible, but the
            # range check is the converter's job (see module docstring).
            status = "convertible"
        else:
            status = "incompatible"
            if role not in TARGET_DTYPES:
                unknown_roles.add(role)
        groups.append({
            "arch": arch, "proj": proj, "role": role, "dtype": dtype,
            "count": n, "bytes": bytes_by_group.get((arch, proj, role, dtype), 0),
            "target": target, "status": status,
            "scope": "shared" if arch in SHARED_ARCHS else "routed",
        })

    # The verdict follows the routed experts only; shared experts are
    # resident, so their dtype does not gate the streaming read path.
    statuses = {g["status"] for g in groups if g["scope"] == "routed"}
    if not groups:
        verdict = "no_expert_tensors"
    elif "incompatible" in statuses:
        verdict = "incompatible"
    elif "convertible" in statuses:
        verdict = "needs_conversion"
    else:
        verdict = "direct_compatible"

    return {
        "model": str(model_dir),
        "name": model_dir.name,
        "shards": len(files),
        "layers": len(layers),
        "verdict": verdict,
        "tensors": sum(g["count"] for g in groups),
        "bytes": sum(g["bytes"] for g in groups),
        "needs_conversion_bytes": sum(
            g["bytes"] for g in groups if g["status"] == "convertible"
        ),
        "incompatible_bytes": sum(
            g["bytes"] for g in groups if g["status"] == "incompatible"
        ),
        "groups": groups,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path,
                    help="checkpoint directories (or single .safetensors files)")
    ap.add_argument("--json", type=Path, default=None, help="write the full report")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 unless every model is direct_compatible")
    args = ap.parse_args()

    reports = []
    for p in args.paths:
        model_dir = p.parent if p.is_file() else p
        if not model_dir.is_dir():
            print(f"  ! not a directory: {model_dir}", file=sys.stderr)
            continue
        rep = audit_model(model_dir)
        reports.append(rep)
        print(f"{rep['name']}: {rep['verdict']}  "
              f"({rep['tensors']} expert tensors, {rep['shards']} shards, "
              f"{rep['layers']} MoE layers)")
        for g in rep["groups"]:
            flag = {"ok": "  ", "convertible": "~ ", "incompatible": "X "}[g["status"]]
            print(f"   {flag}{g['proj']:12s} {g['role']:8s} "
                  f"{g['dtype']:6s} -> {g['target']:4s}  "
                  f"x{g['count']:<7d} {g['bytes'] / 1024**2:9.1f} MiB"
                  f"{'  [shared]' if g['scope'] == 'shared' else ''}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"models": reports}, indent=2))
        print(f"\nsaved {args.json}")

    if args.strict and any(r["verdict"] != "direct_compatible" for r in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
