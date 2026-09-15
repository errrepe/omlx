# SPDX-License-Identifier: Apache-2.0
"""Expert-packed bank converter: per-expert contiguous MoE records.

Why this exists (measured, bench 2026-09-10, qwen-jang4m, telemetry armed):
decode demand averaged 0.322 MiB per preadv -- 92.6% of runs carried a
single expert and each expert-miss cost ~4 separate commands (weight,
scales and biases live in different stacked tensors). On that volume the
per-command latency is fixed (~0.5-0.75 ms cold; 1 MB preadv = 1.62 GB/s
serial, 2 MB = 3.14 GB/s), so command SIZE is throughput. Repacking each
projection's components into one record per expert lets the demand path
issue ONE preadv per expert per projection, ~3x larger, with no 50 KB
stragglers -- the demand unit finally matches the storage unit.

Layout: safetensors-compatible container files that _ShardReader parses
unmodified. Per group (one projection of one layer)::

    <model>/.omlx/expert_bank/bank_<seq>_<layer>_<proj>.sfbank
        u64 LE header size + JSON header:
          "<prefix>.packed" -> {"dtype": "U8", "shape": [N, record_bytes],
                                "data_offsets": [0, N * record_bytes]}
          "__packed__"      -> {"components": [{key, offset, bytes}]}
    <model>/.omlx/expert_bank/manifest.json   (the commit point)

Record = [weight][pad][scales][pad][biases][pad]; every component offset is
8-byte aligned so typed views never straddle an itemsize boundary. Records
are byte-identical to the source slices -- same bytes, same expert order,
only the layout moves -- which is what makes the runtime A/B a pure
token-ID bit-exactness gate.

v1 contract (deliberate): single-tier, SOURCE packing only. pack refuses
models without stacked expert tensors; the runtime attach (shard_bank.
ExpertBackingStore.attach_expert_bank) refuses stale manifests -- source
shards changed -- never silently. The HOBBIT split and dsv4 spill stacking
keep the source layout (see convert_model_to_streaming for the gating).

CLI::

    .venv/bin/python -m omlx.patches.expert_streaming.expert_bank_pack \\
        "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-JANG_4M"
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import re
import struct
import time
from pathlib import Path

import numpy as np

BANK_DIRNAME = ".omlx/expert_bank"
MANIFEST_NAME = "manifest.json"
BANK_FORMAT = 3
_BANK_FORMAT_V2 = 2
_BANK_FORMAT_V1 = 1
_ALIGN = 8  # covers every safetensors itemsize (1/2/4/8)

# v2 (Cherenkov steal): one bank per LAYER, every projection fused into a
# single record per expert. The demand unit of a MoE layer-call is the
# expert: gate/up/down of one expert are always read together, so one
# full-expert preadv (1 command per miss) replaces the per-projection
# commands (3 after v1, up to 9 before it). Command size is throughput on
# NVMe (fixed ~0.5-0.75 ms latency per command; 1 MB = 1.62 GB/s serial,
# 2 MB = 3.14 GB/s — measured, see the module docstring).

_COMPONENT_ORDER = ("weight", "scales", "biases")
# Three stacked layouts in the wild:
#   ...mlp.experts.<proj>.{weight,scales,biases}          (fused-gate_up models)
#   ...mlp.switch_mlp.<proj>.{weight,scales,biases}       (qwen4_exp JANG checkpoints)
#   ...ffn.switch_mlp.<proj>.{weight,scales,biases}       (DeepSeek V4 checkpoints)
_GROUP_RE = re.compile(
    r"^(?P<prefix>.*(?:mlp|ffn)\.(?:experts|switch_mlp)\.(?P<proj>[^.]+))\.(?P<kind>weight|scales|biases)$"
)
_LAYER_RE = re.compile(r"(?:^|\.)(?:mtp\.)?layers\.(\d+)(?:\.|$)")


def bank_dir_for(model_dir: str | Path) -> Path:
    return Path(model_dir).expanduser() / BANK_DIRNAME


def _read_st_header(path: Path) -> tuple[dict, int]:
    """(header, data_start) of a safetensors-format file."""
    with open(path, "rb") as f:
        hsize = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hsize))
    return header, 8 + hsize


def _source_fingerprint(model_dir: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        st = f.stat()
        out[f.name] = [st.st_size, st.st_mtime_ns]
    return out


def _layer_of(prefix: str) -> tuple[str, int] | None:
    """(scope, layer_idx) of a group prefix, e.g. ("main", 12) / ("mtp0", 1)."""
    m = _LAYER_RE.search(prefix)
    if m is None:
        return None
    scope = "main"
    mp = re.match(r"^mtp\.", prefix)
    if mp is not None:
        stage = re.search(r"mtp\.?(\d+)?", prefix)
        scope = "mtp" + (stage.group(1) or "x")
    return (scope, int(m.group(1)))


def _discover_groups(
    model_dir: Path, num_experts: int
) -> list[dict]:
    """Stacked expert-tensor groups of the checkpoint, one per LAYER (v2).

    Every projection of a layer (.gate/.up/.down or fused .gate_up plus
    .down) is fused into ONE bank file: the runtime reads one record per
    expert-miss instead of one command per projection. Groups are ordered
    by (scope, layer) so the bank filenames sort by decode order. Per-
    expert-key models (.experts.<id>.) and non-stacked layouts are
    rejected: their keys do not carry the [num_experts, ...] leading axis
    this bank format is built on.
    """
    index = model_dir / "model.safetensors.index.json"
    files: list[Path]
    if index.is_file():
        try:
            wmap = json.loads(index.read_text()).get("weight_map") or {}
            files = sorted({model_dir / v for v in wmap.values()})
        except Exception:
            files = sorted(model_dir.glob("*.safetensors"))
    else:
        files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no safetensors shards under {model_dir}")

    groups: dict[str, dict] = {}
    for path in files:
        header, data_start = _read_st_header(path)
        for key, entry in header.items():
            if key.startswith("__"):
                continue
            m = _GROUP_RE.match(key)
            if m is None:
                continue
            shape = tuple(int(d) for d in entry.get("shape") or ())
            if len(shape) < 2 or shape[0] != num_experts:
                # Per-expert-key models land here (their leading axis is a
                # hidden dim, not the expert count) -- reject loudly below.
                continue
            start, end = entry["data_offsets"]
            lay = _layer_of(m.group("prefix"))
            lay_key = m.group("prefix")
            if lay is not None:
                # strip the trailing .<proj> so every projection of the
                # layer lands in the same bucket
                lay_key = m.group("prefix")[: m.group("prefix").rfind(".")]
            g = groups.setdefault(
                lay_key,
                {"prefix": lay_key, "layer": lay, "num_experts": shape[0],
                 "projections": {}},
            )
            pg = g["projections"].setdefault(
                m.group("proj"), {}
            )
            pg[m.group("kind")] = {
                "key": key,
                "file": path.name,
                "abs_off": data_start + int(start),
                "row_bytes": (int(end) - int(start)) // shape[0],
                "shape": shape,
                "dtype": str(entry.get("dtype") or ""),
            }
    out: list[dict] = []
    for lay_key in sorted(groups, key=lambda k: (
        groups[k]["layer"] or ("zzz", 0) if groups[k]["layer"] is None else groups[k]["layer"]
    )):
        g = groups[lay_key]
        if not g["projections"]:
            continue
        for proj, comps in sorted(g["projections"].items()):
            if "weight" not in comps:
                raise RuntimeError(
                    f"projection {lay_key}.{proj} has no weight tensor "
                    f"(kinds: {sorted(comps)})"
                )
            for c in comps.values():
                if c["row_bytes"] <= 0:
                    raise RuntimeError(f"empty row for {c['key']}")
        out.append(g)
    if not out:
        raise RuntimeError(
            "no stacked expert tensors found (per-expert-key or non-MoE "
            "layouts are outside the v1 bank contract)"
        )
    return out


def _align_up(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _bank_filename(seq: int, g: dict) -> str:
    lay = g["layer"]
    if lay is not None:
        scope, li = lay
        tag = ("mtp_l" if scope != "main" else "l") + f"{li:02d}"
    else:
        tag = f"g{seq:03d}"
    return f"bank_{seq:03d}_{tag}_full.sfbank"


def pack_model(
    model_dir: str | Path,
    *,
    force: bool = False,
    verify: int = 3,
    progress=None,
) -> dict:
    """Write the expert bank for *model_dir*; returns the manifest.

    Idempotent: an existing, fresh manifest is returned untouched (pass
    force=True to rebuild). The manifest is written LAST and atomically --
    a half-converted bank is never attached (bank_status_for refuses on a
    missing manifest, and attach refuses on a stale one).
    """
    from .residency import expert_streaming_estimate

    model_dir = Path(model_dir).expanduser().resolve()
    est = expert_streaming_estimate(model_dir)
    if not est.supported:
        raise RuntimeError(f"streaming estimate refused this model: {est.reason}")
    num_experts = int(est.experts_per_layer)
    if num_experts <= 0:
        raise RuntimeError("estimate carries no experts_per_layer")

    bdir = bank_dir_for(model_dir)
    manifest_path = bdir / MANIFEST_NAME
    if manifest_path.is_file() and not force:
        ok, why, existing = bank_status_for(model_dir)
        if ok and existing is not None:
            return existing
        # stale/partial: rebuild below rather than trust it

    groups = _discover_groups(model_dir, num_experts)
    fingerprint = _source_fingerprint(model_dir)

    bdir.mkdir(parents=True, exist_ok=True)
    mms: dict[str, mmap.mmap] = {}

    def _mm_for(fname: str) -> mmap.mmap:
        mm = mms.get(fname)
        if mm is None:
            f = open(model_dir / fname, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            f.close()  # the mmap keeps the fd alive
            mms[fname] = mm
        return mm

    manifest_groups: list[dict] = []
    try:
        for seq, g in enumerate(groups):
            prefix = g["prefix"]  # layer prefix, projections stripped
            n = g["num_experts"]
            # Flatten: every (proj, kind, component) in a fixed order —
            # projection name ascending, then weight/scales/biases. The
            # record carries ALL of them; per-projection offsets are
            # published in the manifest so the runtime can slice views.
            flat: list[tuple[str, str, dict]] = []
            for proj, comps in sorted(g["projections"].items()):
                for kind in _COMPONENT_ORDER:
                    c = comps.get(kind)
                    if c is not None:
                        flat.append((proj, kind, c))
            if not flat:
                raise RuntimeError(f"layer {prefix!r} has no components")

            offsets: list[int] = []
            off = 0
            for _p, _k, c in flat:
                offsets.append(off)
                off = _align_up(off + c["row_bytes"])
            record_bytes = off

            packed = np.empty((n, record_bytes), dtype=np.uint8)
            for (_p, _k, c), c_off in zip(flat, offsets):
                row_src = np.frombuffer(
                    _mm_for(c["file"]), dtype=np.uint8,
                    count=n * c["row_bytes"], offset=c["abs_off"],
                ).reshape(n, c["row_bytes"])
                packed[:, c_off:c_off + c["row_bytes"]] = row_src

            fname = _bank_filename(seq, g)
            packed_key = f"{prefix}.full.packed"
            header = {
                packed_key: {
                    "dtype": "U8",
                    "shape": [n, record_bytes],
                    "data_offsets": [0, n * record_bytes],
                },
                "__packed__": {
                    "layout": "fused",
                    "num_experts": n,
                    "record_bytes": record_bytes,
                    "components": [
                        {"key": c["key"], "proj": proj, "kind": kind,
                         "offset": c_off, "bytes": c["row_bytes"]}
                        for (proj, kind, c), c_off in zip(flat, offsets)
                    ],
                },
            }
            hbytes = json.dumps(header, separators=(",", ":")).encode()
            fpath = bdir / fname
            tmp = bdir / (fname + ".tmp")
            with open(tmp, "wb") as f:
                f.write(struct.pack("<Q", len(hbytes)))
                f.write(hbytes)
                f.write(packed)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, fpath)

            # Verify: a few random records must be byte-identical to source.
            if verify > 0:
                rng = random.Random(0x5EED + seq)
                for _ in range(min(verify, n)):
                    eid = rng.randrange(n)
                    for (_p, _k, c), c_off in zip(flat, offsets):
                        expect = np.frombuffer(
                            _mm_for(c["file"]), dtype=np.uint8,
                            count=c["row_bytes"],
                            offset=c["abs_off"] + eid * c["row_bytes"],
                        )
                        got = packed[eid, c_off:c_off + c["row_bytes"]]
                        if not np.array_equal(expect, got):
                            raise RuntimeError(
                                f"verify failed: {c['key']} expert {eid}"
                            )

            manifest_groups.append(
                {
                    "prefix": prefix,
                    "layout": "fused",
                    "packed_key": packed_key,
                    "file": fname,
                    "num_experts": n,
                    "record_bytes": record_bytes,
                    "file_bytes": 8 + len(hbytes) + n * record_bytes,
                    "components": [
                        {"key": c["key"], "proj": proj, "kind": kind,
                         "offset": c_off, "bytes": c["row_bytes"]}
                        for (proj, kind, c), c_off in zip(flat, offsets)
                    ],
                }
            )
            if progress is not None:
                try:
                    progress(seq + 1, len(groups), prefix)
                except Exception:
                    pass
    finally:
        for mm in mms.values():
            try:
                mm.close()
            except Exception:
                pass

    manifest = {
        "format": _BANK_FORMAT_V2,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_dir": str(model_dir),
        "num_experts": num_experts,
        "source": fingerprint,
        "groups": manifest_groups,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, manifest_path)
    return manifest


def bank_status_for(
    model_dir: str | Path, bank_dir: str | Path | None = None
) -> tuple[bool, str, dict | None]:
    """(ok, reason, manifest) for the fused bank of *model_dir*.

    *bank_dir* overrides the default <model>/.omlx/expert_bank location —
    used when the user points a model at an existing bank kept elsewhere
    (a read-only volume, a second SSD). Relative paths resolve against
    the model dir.

    ok requires: manifest present, format current, every contributing
    source shard unchanged (size + mtime), and every bank file present at
    its recorded size. Anything else is a refusal with a reason -- attach
    never silently serves a stale layout.
    """
    model_dir = Path(model_dir).expanduser().resolve()
    bdir = (
        Path(bank_dir).expanduser()
        if bank_dir is not None
        else bank_dir_for(model_dir)
    )
    if not bdir.is_absolute():
        bdir = (model_dir / bdir).resolve()
    else:
        bdir = bdir.resolve()
    manifest_path = bdir / MANIFEST_NAME
    if not manifest_path.is_file():
        return (False, f"no manifest at {manifest_path}", None)
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        return (False, f"manifest unreadable: {e}", None)
    fmt = int(manifest.get("format") or 0) if isinstance(manifest, dict) else 0
    if fmt == _BANK_FORMAT_V1:
        return (False, "v1 per-projection bank present (rebuild with --force for the fused layout)", None)
    if fmt not in (_BANK_FORMAT_V2, BANK_FORMAT):
        return (False, "manifest format mismatch (rebuild the bank)", None)
    current = _source_fingerprint(model_dir)
    recorded = manifest.get("source") or {}
    if set(current) != set(recorded):
        return (False, "source shard set changed (rebuild the bank)", None)
    for name, rec in recorded.items():
        cur = current.get(name)
        if cur is None or int(cur[0]) != int(rec[0]) or int(cur[1]) != int(rec[1]):
            return (False, f"source shard {name} changed (rebuild the bank)", None)
    for g in manifest.get("groups") or []:
        fpath = bdir / str(g.get("file") or "")
        if not fpath.is_file():
            return (False, f"bank file missing: {fpath.name}", None)
        try:
            if fpath.stat().st_size != int(g.get("file_bytes") or -1):
                return (False, f"bank file size drift: {fpath.name}", None)
        except OSError as e:
            return (False, f"bank file stat failed: {fpath.name}: {e}", None)
    return (True, "ok", manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when a fresh manifest exists")
    ap.add_argument("--verify", type=int, default=3,
                    help="random records per group to byte-compare (default 3)")

    args = ap.parse_args()

    def progress(done: int, total: int, prefix: str) -> None:
        print(f"  [{done:3d}/{total}] {prefix}", flush=True)

    manifest = pack_model(args.model_dir, force=args.force,
                          verify=args.verify,
                          progress=progress)
    total = sum(g["file_bytes"] for g in manifest["groups"])
    print(
        f"expert bank ready: {len(manifest['groups'])} groups, "
        f"{total / 1024**3:.1f} GiB under "
        f"{bank_dir_for(args.model_dir)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
