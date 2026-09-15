"""Hidden-state capture for offline prerouter training (V2-6).

``OMLX_EXPERT_STREAMING_HSTRACE=/path/prefix`` turns it on; unset it
costs a single env lookup per MoE call. Each record stores the router
input row (fp16) plus the token's routed expert set — everything the
offline trainer needs to learn ``hidden(t) -> routed(t+delta)`` without
a second instrumented run.

Output files:
  <prefix>.bin    fp16 rows, row-major (n_records x dim)
  <prefix>.jsonl  {"layer", "dim", "experts"} per row, in call order

Single-position calls only: the call-site guard is
``x.size == x.shape[-1]`` — one router input row. Multi-position calls
(prefill chunks, verify blocks, batched decode) are skipped. A 1-token
prefill tail chunk IS recorded (the guard sees shape, not phase), so
offline consumers that need decode-pure traces filter on their own
position index. Memory is bounded: ~dim*2 bytes per (token, layer) —
a few hundred decode tokens is tens of MB, flushed at exit via atexit.
"""

import atexit
import json
import os
import sys

import numpy as np

_PATH = os.environ.get("OMLX_EXPERT_STREAMING_HSTRACE") or ""
_rows = []
_index = []
_flushed = False


def enabled() -> bool:
    return bool(_PATH)


def record(layer: int, x_row, experts) -> None:
    """Append one (layer, hidden-row, routed-set) record.

    ``x_row`` is a single token's router input (1-D over hidden dim);
    callers pass ``x[last_position]`` or the decode row. Cast to fp16
    keeps the capture cheap and is plenty for a trained head.
    """
    if not _PATH:
        return
    try:
        import mlx.core as mx

        vec = np.asarray(mx.stop_gradient(x_row).astype(mx.float16)).reshape(-1)
        _rows.append(vec)
        _index.append(
            {
                "layer": int(layer),
                "dim": int(vec.size),
                "experts": sorted({int(e) for e in experts}),
            }
        )
    except Exception:
        # Capture must never perturb inference.
        pass


def _flush() -> None:
    global _flushed
    if _flushed or not _PATH or not _index:
        return
    _flushed = True
    try:
        mat = np.stack(_rows)
        with open(_PATH + ".bin", "wb") as f:
            mat.tofile(f)
        with open(_PATH + ".jsonl", "w") as f:
            f.write(
                json.dumps({"rows": int(mat.shape[0]), "dim": int(mat.shape[1])})
                + "\n"
            )
            for entry in _index:
                f.write(json.dumps(entry) + "\n")
        print(
            f"hstrace: {mat.shape[0]} rows x {mat.shape[1]} dims -> {_PATH}.bin",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"hstrace flush failed: {exc}", file=sys.stderr)


atexit.register(_flush)
