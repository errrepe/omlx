# SPDX-License-Identifier: Apache-2.0
"""Bench: A/B of the SSD PLE prefault warm-up (Trilha B).

Measures the real gather path of `DiskBackedShardedEmbedding` with the
prefault on and off, at three prefill sizes, on the actual checkpoint.

Protocol
  * Rows are scattered uniformly inside a **band** of the PLE table, and
    every sample gets its own disjoint band. That reproduces the real
    access pattern (ngram ids spread over the whole table, ~1 row per
    16 KiB page) while keeping samples cold: `purge` needs root, which is
    unavailable here, so band disjointness is what stands in for it.
  * Arms are adjacent and their order alternates per repeat; medians over
    repeats. `OMLX_QWEN4_PLE_PREFAULT` is flipped in-process (the reader
    reads the env on every call, so no reload is needed).
  * A warm control re-gathers one already-touched sample per size to show
    the cold/warm gap that makes the prefill tail bimodal.

Usage:
    .venv/bin/python bench/bench_ple_prefault_ab.py \
        --model "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-JANG_4S" \
        --repeats 5 --out bench/results/ple_prefault/ab.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-JANG_4S"
PREFAULT_ENV = "OMLX_QWEN4_PLE_PREFAULT"
DEFAULT_SIZES = (512, 2048, 8192)
DEFAULT_REPEATS = 5
WARM_REPEATS = 3


def _read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size))


def detect_ple(model_path: Path) -> dict:
    """Find the PLE base, shard count, row geometry and rows/token."""
    weight_map = json.loads(
        (model_path / "model.safetensors.index.json").read_text()
    )["weight_map"]

    grouped: dict[str, list[int]] = {}
    pattern = re.compile(r"^(.*)\.shards?\.(\d+)\.weight$")
    for key in weight_map:
        match = pattern.match(key)
        if match and "ple" in match.group(1):
            grouped.setdefault(match.group(1), []).append(int(match.group(2)))
    if not grouped:
        raise SystemExit(f"no sharded PLE tensors under {model_path}")
    base = max(grouped, key=lambda name: len(grouped[name]))
    shard_ids = sorted(grouped[base])
    if shard_ids != list(range(len(shard_ids))):
        raise SystemExit(f"non-contiguous PLE shards for {base}: {shard_ids}")
    num_shards = len(shard_ids)

    key_fmt = f"{base}.shards.{{}}.weight"
    first_file = model_path / weight_map[key_fmt.format(0)]
    if not first_file.exists():
        key_fmt = f"{base}.shard_{{}}.weight"
        first_file = model_path / weight_map[key_fmt.format(0)]
    header = _read_header(first_file)
    rows_per_shard = int(header[key_fmt.format(0)]["shape"][0])
    packed = int(header[key_fmt.format(0)]["shape"][1])
    scales_shape = tuple(
        header[key_fmt.format(0).replace(".weight", ".scales")]["shape"]
    )
    row_bytes = packed * 4 + 2 * math.prod(scales_shape[1:])

    config = json.loads((model_path / "config.json").read_text())
    text = config.get("text_config", config)
    bits = None
    group_size = None
    for key, spec in (config.get("jang_config") or {}).get("bit_map", {}).items():
        if "ple" in key:
            bits = int(spec["bits"])
            group_size = int(spec["group_size"])
            break
    ngram_heads = int(text["heads_per_ngram"]) * (int(text["ngram_size"]) - 1)
    dims = int(text["ple_embed_dim"]) // ngram_heads
    if bits is None:
        bits = packed * 32 // dims
    if group_size is None:
        group_size = dims // int(scales_shape[1])

    return {
        "base": base,
        "num_shards": num_shards,
        "rows_per_shard": rows_per_shard,
        "num_embeddings": rows_per_shard * num_shards,
        "dims": dims,
        "bits": bits,
        "group_size": group_size,
        "row_bytes": row_bytes,
        "ngram_heads": ngram_heads,
        "table_gib": rows_per_shard * num_shards * row_bytes / 1024**3,
    }


def _candidate_prefixes(base: str) -> list[str]:
    """Prefixes the loader would pass to resolve this checkpoint's base."""
    with_embedding = base
    if ".ple_embedding" not in base:
        with_embedding = base.replace(".ple.", ".ple.ple_embedding.")
    candidates = []
    for stem in (with_embedding, "model." + with_embedding):
        if stem not in candidates:
            candidates.append(stem)
    return candidates


def _percentiles(samples: list[float]) -> dict:
    arr = np.array(samples)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    return {
        "n": len(samples),
        "samples_ms": [round(value, 2) for value in samples],
        "p50_ms": round(p50, 3),
        "p90_ms": round(p90, 3),
        "mean_ms": round(float(arr.mean()), 3),
        "p90_over_p50": round(p90 / p50, 3) if p50 else None,
    }


class BandAllocator:
    """Hands out disjoint, scattered row sets: one cold band per sample."""

    def __init__(self, num_rows: int, total_samples: int, seed: int = 0):
        self.num_rows = num_rows
        self.band = max(1, num_rows // max(1, total_samples))
        self.cursor = 0
        self.rng = np.random.default_rng(seed)

    def take(self, rows: int) -> np.ndarray:
        start = self.cursor
        if start + self.band > self.num_rows:
            raise SystemExit(
                f"PLE table exhausted after {start // self.band} bands; "
                "lower --repeats or --sizes"
            )
        self.cursor += self.band
        return self.rng.integers(start, start + self.band, size=rows, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--sizes", default=",".join(str(size) for size in DEFAULT_SIZES)
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--out", default="bench/results/ple_prefault/ab.json")
    args = parser.parse_args()

    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import (
        DiskBackedShardedEmbedding,
        _prefault_page_bytes,
    )

    sizes = [int(size) for size in args.sizes.split(",") if size.strip()]
    page = _prefault_page_bytes()

    model_path = Path(args.model).expanduser().resolve()
    info = detect_ple(model_path)
    print(
        f"PLE {info['base']}: {info['num_shards']} shards x "
        f"{info['rows_per_shard']} rows x {info['dims']}d "
        f"({info['bits']}b/g{info['group_size']}, {info['row_bytes']} B/row, "
        f"{info['table_gib']:.2f} GiB, {info['ngram_heads']} rows/token)",
        flush=True,
    )

    table = None
    prefix = None
    for candidate in _candidate_prefixes(info["base"]):
        try:
            table = DiskBackedShardedEmbedding(
                model_path,
                candidate,
                info["num_embeddings"],
                info["dims"],
                info["num_shards"],
            )
            prefix = candidate
            break
        except (KeyError, ValueError) as exc:
            print(f"  prefix {candidate!r} did not resolve: {exc}", flush=True)
    if table is None:
        raise SystemExit(f"could not resolve a PLE prefix for {info['base']}")
    print(f"resolved prefix: {prefix}", flush=True)

    rows_per_page = max(1, page // info["row_bytes"])
    allocator = BandAllocator(
        info["num_embeddings"], args.repeats * 2 * len(sizes)
    )
    results = {
        "model": model_path.name,
        "ple": info,
        "page_bytes": page,
        "rows_per_page": rows_per_page,
        "repeats": args.repeats,
        "protocol": (
            "disjoint scattered bands, adjacent arms, alternating order, "
            "median over repeats"
        ),
        "samples": {},
    }

    for size in sizes:
        rows = size * info["ngram_heads"]
        per_arm: dict[str, list[float]] = {"off": [], "on": []}
        first_ids: np.ndarray | None = None
        for repeat in range(args.repeats):
            arms = ["off", "on"] if repeat % 2 == 0 else ["on", "off"]
            values = {}
            for arm in arms:
                os.environ[PREFAULT_ENV] = "0" if arm == "off" else "1"
                ids = allocator.take(rows)
                first_ids = first_ids if first_ids is not None else ids
                start = time.perf_counter()
                out = table(mx.array(ids))
                mx.eval(out)
                elapsed = (time.perf_counter() - start) * 1000
                per_arm[arm].append(elapsed)
                values[arm] = elapsed
                mx.clear_cache()
            print(
                f"repeat {repeat} prefill {size:>5}: "
                + "  ".join(f"{arm}={values[arm]:.1f} ms" for arm in arms)
                + f"  (pages~{len(np.unique(ids // rows_per_page))})",
                flush=True,
            )

        stats = {arm: _percentiles(values) for arm, values in per_arm.items()}
        speedup = stats["off"]["p50_ms"] / stats["on"]["p50_ms"]
        stats["speedup_on_vs_off"] = round(speedup, 3)
        stats["rows_per_call"] = rows
        stats["distinct_pages_approx"] = int(len(np.unique(first_ids // rows_per_page)))
        stats["cold_read_mb_approx"] = round(
            stats["distinct_pages_approx"] * page / 1e6, 1
        )
        results["samples"][f"prefill_{size}"] = stats

        warm: dict[str, list[float]] = {"off": [], "on": []}
        for arm in ("off", "on"):
            os.environ[PREFAULT_ENV] = "0" if arm == "off" else "1"
            for _ in range(WARM_REPEATS):
                start = time.perf_counter()
                mx.eval(table(mx.array(first_ids)))
                warm[arm].append((time.perf_counter() - start) * 1000)
                mx.clear_cache()
        warm_floor = float(np.median(warm["off"]))
        results["samples"][f"warm_repeat_{size}"] = {
            arm: {"p50_ms": round(float(np.median(values)), 3), "n": len(values)}
            for arm, values in warm.items()
        }
        # Without root there is no `purge`, and a 14.9 GiB table read by a
        # previous run stays cached. The cold/warm ratio says which regime
        # this run actually measured: ~1 means "warm-dominated", i.e. the
        # arms below are only measuring syscall overhead, not SSD reads.
        ratio = stats["off"]["p50_ms"] / warm_floor if warm_floor else 0.0
        stats["cold_over_warm"] = round(ratio, 2)
        stats["regime"] = "cold" if ratio >= 2.0 else "warm-dominated"
        print(
            f"  prefill {size:>5}: off p50 {stats['off']['p50_ms']:.1f} ms "
            f"(p90/p50 {stats['off']['p90_over_p50']}) | "
            f"on p50 {stats['on']['p50_ms']:.1f} ms "
            f"(p90/p50 {stats['on']['p90_over_p50']}) | "
            f"speedup {speedup:.2f}x | warm floor {warm_floor:.1f} ms | "
            f"regime {stats['regime']} (cold/warm {ratio:.1f}x)",
            flush=True,
        )

    table.close()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== verdict (gate: on p90/p50 <= 1.15 and on p50 <= off p50) ===")
    for size in sizes:
        stats = results["samples"][f"prefill_{size}"]
        spread_ok = (stats["on"]["p90_over_p50"] or 9) <= 1.15
        faster = stats["speedup_on_vs_off"] >= 1.0
        print(
            f"prefill {size:>5}: on p90/p50={stats['on']['p90_over_p50']} "
            f"{'PASS' if spread_ok else 'FAIL'} | "
            f"speedup={stats['speedup_on_vs_off']}x {'PASS' if faster else 'FAIL'}"
        )
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
