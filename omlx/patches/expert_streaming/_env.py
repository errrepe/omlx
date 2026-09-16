# SPDX-License-Identifier: Apache-2.0
"""Tolerant env-knob parsing for module-level constants.

A malformed value (``OMLX_EXPERT_STREAMING_QD=abc``) must degrade to the
default, not raise ValueError at import and take the whole module down.
"""

from __future__ import annotations

import os


def env_int(name: str, default: int, lo: int | None = None) -> int:
    try:
        v = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        v = int(default)
    return v if lo is None else max(lo, v)


def env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        v = float(default)
    return v


def env_bool(name: str, default: bool) -> bool:
    """Tri-state bool knob: only an explicit "0"/"1" overrides *default*.

    The default argument reproduces both legacy spellings: opt-out knobs
    (``!= "0"``, on unless disabled) pass ``default=True`` — anything but
    "0" stays on — while opt-in knobs (``== "1"``, off unless enabled)
    pass ``default=False``.
    """
    v = os.environ.get(name, "").strip()
    return default if v not in ("0", "1") else v == "1"
