#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repack MoE expert (switch_mlp) tensors into a page-aligned fullbank
artifact for zero-copy external wrapping (Fase M).

Artifact layout (format authority lives here; loader is
omlx/patches/expert_streaming/fullbank.py):
  [8-byte LE manifest_len][manifest JSON, zero-pad to PAGE]
  [tensor 0 data, zero-pad to PAGE][tensor 1 data, zero-pad to PAGE]...

Manifest: {"page_size": P, "fingerprint": {...}, "tensors":
           {key: {offset,length,dtype,shape,bits,group_size,mode}}}

Every tensor data region starts at a PAGE boundary so the native wrapper
(custom_kernels/expert_bank_wrap) can create a page-aligned
newBufferWithBytesNoCopy MTLBuffer over it.

Unlike the PR this is adapted from, the manifest carries a **content
fingerprint** binding the artifact to its source checkpoint (config sha256 +
shard filenames/sizes/mtimes). The loader refuses to engage on mismatch, so a
checkpoint update can never be served stale expert bytes silently.

Adapted from jundot/omlx PR #3437 (qwen4_moe_stream, repack.py) by
@alytaphoenix, Apache-2.0. Differences: --model-dir is required (no
hardcoded user path), quant params are read from config.json per module
(never inferred from tensor shapes), and the fingerprint block is added.

Usage:
  python tools/repack_fullbank.py <out_path> --model-dir DIR [--verify]
  OMLX_FULLBANK_PAGE (default 16384) overrides the page size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

DEFAULT_PAGE = 16384
ARTIFACT_NAME = "fullbank_experts.artifact"


def _align(n: int, a: int = DEFAULT_PAGE) -> int:
    return (n + a - 1) // a * a


def _shard_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _load_quant_config(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    q = cfg.get("quantization")
    if not isinstance(q, dict):
        raise SystemExit("config.json has no quantization block")
    return q


def _fingerprint(model_dir: Path, shard_names: set[str]) -> dict:
    """Content binding: config hash + shard file stats. Cheap (stat only)."""
    cfg_path = model_dir / "config.json"
    fp: dict = {
        "config_sha": hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16],
        "shards": {},
    }
    for name in sorted(shard_names):
        st = (model_dir / name).stat()
        fp["shards"][name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return fp


def _check_fingerprint(model_dir: Path, manifest_fp: dict) -> bool:
    """True when the model dir matches the manifest fingerprint today."""
    try:
        now = _fingerprint(model_dir, set(manifest_fp.get("shards", {})))
    except Exception:
        return False
    return now == manifest_fp


def enumerate_experts(model_dir: Path) -> list[tuple[str, Path, int, int, str, list[int]]]:
    """(key, shard_path, abs_offset, length, dtype, shape) for every
    switch_mlp tensor, sorted for deterministic layout. Generic over all
    SUPPORTED_TYPES families: any key containing ".switch_mlp." qualifies
    (covers layers.*.mlp.switch_mlp.*, ffn.switch_mlp.*, mtp stages)."""
    idx = model_dir / "model.safetensors.index.json"
    if not idx.is_file():
        raise SystemExit("model dir has no model.safetensors.index.json")
    weight_map = json.loads(idx.read_text()).get("weight_map") or {}
    headers: dict[str, tuple[dict, int]] = {}
    out = []
    for key, fname in sorted(weight_map.items()):
        if ".switch_mlp." not in key:
            continue
        path = model_dir / fname
        if fname not in headers:
            headers[fname] = _shard_header(path)
        hdr, data_start = headers[fname]
        entry = hdr.get(key)
        if entry is None:
            raise SystemExit(f"index references missing tensor: {key}")
        s, e = entry["data_offsets"]
        out.append((key, path, data_start + s, e - s, entry["dtype"], entry["shape"]))
    out.sort(key=lambda t: t[0])
    return out


def _quant_params(qcfg: dict, key: str) -> tuple[int, int, str]:
    """(bits, group_size, mode) for a tensor from config -- never inferred
    from shape (lesson from PR #3437: a checkpoint can be 2-bit in the
    trunk and 4-bit in the MTP block, making shape inference ambiguous)."""
    gb = int(qcfg.get("bits", 4))
    gg = int(qcfg.get("group_size", 64))
    gm = str(qcfg.get("mode", "affine"))
    override = qcfg.get(key.rsplit(".", 1)[0])
    if isinstance(override, dict):
        return int(override["bits"]), int(override["group_size"]), str(
            override.get("mode", gm)
        )
    return gb, gg, gm


def repack(out_path: Path, model_dir: Path, page: int, verify: bool) -> dict:
    tensors = enumerate_experts(model_dir)
    if not tensors:
        raise SystemExit("no switch_mlp expert tensors found in the index")
    qcfg = _load_quant_config(model_dir)

    shard_names = {t[1].name for t in tensors}
    manifest: dict = {
        "page_size": page,
        "fingerprint": _fingerprint(model_dir, shard_names),
        "tensors": {},
    }

    # Two-pass layout: reserve a generous manifest region (data offsets
    # depend on its page-aligned size).
    def _entry(cur: int, key: str, length: int, dtype: str, shape: list[int]) -> dict:
        bits, gs, mode = _quant_params(qcfg, key)
        return {
            "offset": cur,
            "length": length,
            "dtype": dtype,
            "shape": shape,
            "bits": bits,
            "group_size": gs,
            "mode": mode,
        }

    est_manifest = 8 + len(
        json.dumps({
            "page_size": page,
            "fingerprint": manifest["fingerprint"],
            "tensors": {
                k: _entry(0, k, ln, dt, sh) for (k, _p, _o, ln, dt, sh) in tensors
            },
        }).encode()
    ) + 4096
    data_start = _align(est_manifest, page)

    cur = data_start
    for (key, _path, _off, length, dtype, shape) in tensors:
        manifest["tensors"][key] = _entry(cur, key, length, dtype, shape)
        cur += _align(length, page)
    total = cur

    mb = json.dumps(manifest).encode()
    assert 8 + len(mb) <= data_start, "manifest bigger than reserved region"

    written = 0
    with open(out_path, "wb") as w:
        w.write(struct.pack("<Q", len(mb)))
        w.write(mb)
        w.write(b"\x00" * (data_start - 8 - len(mb)))
        for (key, path, off, length, dtype, shape) in tensors:
            tgt = manifest["tensors"][key]["offset"]
            assert w.tell() == tgt, (w.tell(), tgt, key)
            with open(path, "rb") as src:
                src.seek(off)
                remaining = length
                while remaining:
                    chunk = src.read(min(1 << 22, remaining))
                    if not chunk:
                        raise SystemExit(f"short read on {key}")
                    w.write(chunk)
                    remaining -= len(chunk)
            pad = _align(length, page) - length
            if pad:
                w.write(b"\x00" * pad)
            written += 1
            if written % 45 == 0:
                print(f"  {written}/{len(tensors)} ({w.tell() / 1024**3:.1f} GiB target)")
    print(f"wrote {total / 1024**3:.2f} GiB, {len(tensors)} tensors -> {out_path}")

    if verify:
        import mmap
        import numpy as np

        npd = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16, "U16": np.uint16, "U8": np.uint8}
        with open(out_path, "rb") as af:
            mm = mmap.mmap(af.fileno(), 0, access=mmap.ACCESS_READ)
        mlen = struct.unpack("<Q", mm[:8])[0]
        man = json.loads(mm[8 : 8 + mlen])
        bad = 0
        for (key, path, off, length, dtype, shape) in tensors:
            t = man["tensors"][key]
            assert t["offset"] % page == 0, f"{key} not page-aligned"
            a = np.frombuffer(mm[t["offset"] : t["offset"] + length], dtype=npd[dtype])
            with open(path, "rb") as src:
                src.seek(off)
                b = np.frombuffer(src.read(length), dtype=npd[dtype])
            if not np.array_equal(a, b):
                bad += 1
                print(f"  MISMATCH {key}")
        mm.close()
        print(f"verify: {len(tensors) - bad}/{len(tensors)} tensors byte-identical, all page-aligned")
        if bad:
            raise SystemExit(f"{bad} tensors mismatched")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_path", nargs="?", help=f"artifact path (default: <model-dir>/{ARTIFACT_NAME})")
    ap.add_argument("--model-dir", required=True, help="checkpoint directory (required)")
    ap.add_argument("--page", type=int, default=int(os.environ.get("OMLX_FULLBANK_PAGE", DEFAULT_PAGE)))
    ap.add_argument("--verify", action="store_true", help="byte-compare the artifact against source shards")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model dir not found: {model_dir!r}")
    if args.page < 16384 or (args.page & (args.page - 1)) != 0:
        raise SystemExit("page must be a power of two >= 16384")
    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else model_dir / ARTIFACT_NAME
    repack(out_path, model_dir, args.page, args.verify)


if __name__ == "__main__":
    main()
