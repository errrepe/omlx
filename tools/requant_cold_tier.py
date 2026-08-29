"""Requantize a streaming model's expert banks into a cold precision tier (Fase I5).

Reads a checkpoint's `switch_mlp` stacked banks (oQ4e etc.), dequantizes and
requantizes them at a lower bit width (same group size / mode), and writes a
parallel shard set under `<model>/expert_cold/` with the SAME shard filenames
and key names. At runtime, `expert_streaming_cold_tier` makes the backing
store route expert reads to those files — cutting 25% (3-bit) or half
(2-bit) of the bytes per token that pin decode to the NVMe's I/O floor.

The requantized tensors keep the source group size; bits and group size are
recorded in each shard's `__metadata__` (`omlx_cold_bits` /
`omlx_cold_group_size`) so the runtime can build the gather_qmm without
guessing. Source bits/group size/mode come from config.json's `quantization`
block (per-tensor overrides honored). Only affine-mode banks with a `.bias`
key are converted — the affine bias term must ride along or the runtime's
dequantize would reconstruct shifted values.

Quality is NOT decided here — bench/ppl_expert_streaming.py (I4) is the
gate. 3-bit is the conservative default; 2-bit is the max-bytes option.

Usage:
    .venv/bin/python tools/requant_cold_tier.py \
        --model "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e" --bits 3
    # report what would be written (no writes)
    .venv/bin/python tools/requant_cold_tier.py --model ... --bits 3 --check
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx

EXPERT_BANK_MARKERS = (
    ".switch_mlp.gate_proj.",
    ".switch_mlp.up_proj.",
    ".switch_mlp.down_proj.",
    ".switch_mlp.gate_up_proj.",
)

META_BITS = "omlx_cold_bits"
META_GS = "omlx_cold_group_size"


def bank_prefixes_from_index(weight_map: dict[str, str]) -> set[str]:
    """Stacked-bank key prefixes (…switch_mlp.<proj>) that carry .weight keys."""
    prefixes: set[str] = set()
    for key in weight_map:
        if not key.endswith(".weight"):
            continue
        for marker in EXPERT_BANK_MARKERS:
            idx = key.find(marker)
            if idx > 0:
                prefixes.add(key[: idx + len(marker) - 1])
                break
    return prefixes


def _read_header(path: Path) -> dict:
    with path.open("rb") as f:
        hsize = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hsize))


def quant_cfg_for(key: str, quant_cfg: dict) -> tuple[int, int, str]:
    """(group_size, bits, mode) for *key* — per-tensor override or defaults."""
    override = quant_cfg.get(key)
    if isinstance(override, dict):
        return (
            int(override.get("group_size", quant_cfg["group_size"])),
            int(override.get("bits", quant_cfg["bits"])),
            str(override.get("mode", quant_cfg.get("mode", "affine"))),
        )
    return (
        int(quant_cfg["group_size"]),
        int(quant_cfg["bits"]),
        str(quant_cfg.get("mode", "affine")),
    )


def requant_shard(
    src: Path,
    dst_dir: Path,
    quant_cfg: dict,
    bits: int,
    check_only: bool = False,
) -> dict:
    """Requantize every convertible expert bank of *src* into dst_dir/src.name."""
    out_path = dst_dir / src.name
    header = _read_header(src)
    bank_prefixes = {
        key[: -len(".weight")]
        for key in header
        if key.endswith(".weight") and any(s in key for s in EXPERT_BANK_MARKERS)
    }
    convertible = [
        p for p in sorted(bank_prefixes)
        if f"{p}.biases" in header
        and quant_cfg_for(f"{p}.weight", quant_cfg)[2] == "affine"
    ]
    skipped = sorted(bank_prefixes - set(convertible))
    if not bank_prefixes:
        return {"shard": src.name, "status": "no expert banks", "src_mib": 0, "dst_mib": 0}
    if skipped:
        return {
            "shard": src.name,
            "status": "skipped (banks without .bias or non-affine)",
            "src_mib": 0,
            "dst_mib": 0,
        }

    gs0 = quant_cfg_for(f"{convertible[0]}.weight", quant_cfg)[0]
    if out_path.exists():
        try:
            old = _read_header(out_path)
            meta = old.get("__metadata__") or {}
            if (
                int(meta.get(META_BITS, -1)) == bits
                and int(meta.get(META_GS, -1)) == gs0
                and all(f"{p}.weight" in old for p in convertible)
            ):
                return {"shard": src.name, "status": "already matches", "src_mib": 0, "dst_mib": 0}
        except Exception:
            pass
    if check_only:
        return {
            "shard": src.name,
            "status": "missing (would write)",
            "src_mib": 0,
            "dst_mib": 0,
        }

    loaded = mx.load(str(src))
    arrays: dict[str, mx.array] = {}
    src_bytes = dst_bytes = 0
    max_err = 0.0
    for prefix in convertible:
        w_key = f"{prefix}.weight"
        gs, src_bits, mode = quant_cfg_for(w_key, quant_cfg)
        w = loaded[w_key]
        scales = loaded[f"{prefix}.scales"]
        biases = loaded[f"{prefix}.biases"]
        dense = mx.dequantize(w, scales, biases, group_size=gs, bits=src_bits)
        w2, s2, b2 = mx.quantize(dense, group_size=gs, bits=bits, mode=mode)
        err = mx.abs(mx.dequantize(w2, s2, b2, group_size=gs, bits=bits) - dense).max().item()
        max_err = max(max_err, err)
        arrays[w_key] = w2
        arrays[f"{prefix}.scales"] = s2.astype(mx.bfloat16)
        arrays[f"{prefix}.biases"] = b2.astype(mx.bfloat16)
        src_bytes += w.nbytes
        dst_bytes += w2.nbytes

    dst_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(out_path),
        arrays,
        metadata={META_BITS: str(bits), META_GS: str(gs0)},
    )
    return {
        "shard": src.name,
        "status": "written",
        "banks": len(convertible),
        "max_requant_err": max_err,
        "src_mib": src_bytes / 2**20,
        "dst_mib": dst_bytes / 2**20,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument(
        "--bits", type=int, default=3, choices=(2, 3), help="cold tier bit width"
    )
    ap.add_argument("--check", action="store_true", help="report only; skip writes")
    args = ap.parse_args()

    model = Path(args.model).expanduser().resolve()
    cfg_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    if not index_path.is_file() or not cfg_path.is_file():
        print(
            "checkpoint needs config.json + model.safetensors.index.json",
            file=sys.stderr,
        )
        sys.exit(2)
    quant_cfg = json.loads(cfg_path.read_text()).get("quantization") or {}
    if not quant_cfg:
        print("config.json has no quantization block", file=sys.stderr)
        sys.exit(2)
    weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
    prefixes = bank_prefixes_from_index(weight_map)
    if not prefixes:
        print("no switch_mlp expert banks found in the index", file=sys.stderr)
        sys.exit(2)
    shards = sorted({weight_map[f"{p}.weight"] for p in prefixes})
    print(
        f"{len(prefixes)} expert bank(s) across {len(shards)} shard(s); "
        f"cold bits = {args.bits}, check={'yes' if args.check else 'no'}"
    )
    totals = {"src_mib": 0.0, "dst_mib": 0.0}
    for fname in shards:
        res = requant_shard(model / fname, model / "expert_cold", quant_cfg, args.bits, args.check)
        totals["src_mib"] += res["src_mib"]
        totals["dst_mib"] += res["dst_mib"]
        extra = (
            f"  requant_err<={res['max_requant_err']:.4f}"
            if "max_requant_err" in res
            else ""
        )
        print(
            f"  {res['shard']}: {res['status']}"
            + (f" ({res['banks']} banks)" if "banks" in res else "")
            + extra
        )
    if totals["src_mib"]:
        ratio = totals["dst_mib"] / totals["src_mib"]
        print(
            f"expert banks: {totals['src_mib'] / 1024:.1f} GiB -> "
            f"{totals['dst_mib'] / 1024:.1f} GiB ({ratio:.2f}x)"
        )


if __name__ == "__main__":
    main()
