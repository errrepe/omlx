#!/usr/bin/env python3
"""Offline dry-run of the ANE tuner's o_proj calibration dimension.

The tuner normally runs behind the admin server (`POST /api/bench/ane-tune/start`).
This script drives the o_proj slice directly against a local checkpoint so we can
confirm the 6th tuner dimension is wired and produces a sane recommendation without
standing up the server.

Usage:
    python benchmarks/ane_tuner_dryrun.py /path/to/Qwen3.8-27B-oQ4e-mtp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

from omlx.admin import ane_tuning
from omlx.custom_kernels.qwen35_prefill import fast
from omlx.patches import qwen35_ane_prefill as patch
from omlx.patches.qwen35_q4_mlp import apply_qwen35_q4_mlp_patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to a local Qwen3.5/3.6/3.8 checkpoint")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--dual-ane",
        action="store_true",
        default=True,
        help="Calibrate the dual-ANE o_proj path (default: on)",
    )
    args = parser.parse_args()

    if not fast.qwen35_ane_available():
        raise SystemExit("Private ANE runtime is unavailable on this machine.")

    print(f"Loading {args.model}", flush=True)
    from mlx_vlm.utils import load_model

    model = load_model(Path(args.model), lazy=False, strict=False)
    apply_qwen35_q4_mlp_patch()

    request = ane_tuning.ANETuningRequest(
        model_id="dryrun-27b",
        sequence_length=args.sequence_length,
        allow_ane_oproj=True,
    )
    run = ane_tuning.create_run(request)
    print(
        f"Planned {run.total} tuning points; o_proj slot = {ane_tuning._OPROJ_SLOT}",
        flush=True,
    )

    ok, fraction, latency = ane_tuning._calibrate_oproj_sync(
        run,
        model,
        base_settings=None,
        fast=fast,
        patch=patch,
        dual_ane=args.dual_ane,
    )
    mx.eval()
    oproj_result = run.results[ane_tuning._OPROJ_SLOT]
    print("o_proj calibration:", oproj_result, flush=True)
    print(
        f"DRYRUN {'PASS' if ok else 'SKIP'} "
        f"(enabled={oproj_result.get('oproj_enabled')}, "
        f"fraction={oproj_result.get('oproj_fraction')}, "
        f"latency_ms={oproj_result.get('latency_ms')})",
        flush=True,
    )
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
