# SPDX-License-Identifier: Apache-2.0
"""Fase 1: auto-derived verdict parameters (derivators + adapter counters)."""

import json
from pathlib import Path

import pytest

from omlx.utils.storage_roofline import (
    _decode_bytes,
    derive_bytes_per_token_base,
    derive_tok_per_cycle,
    derive_verify_mult,
    load_auto_params,
    update_auto_params,
)


def _run(model="qwen-fake", mtp=False, depth=None, bytes=1_000_000, tokens=96, accept=None):
    run = {
        "model": model,
        "mtp": mtp,
        "decode_tokens": tokens,
        "read_stats": {"decode": {"bytes": bytes}} if bytes else None,
    }
    if mtp:
        run["mtp_depth"] = depth
    if accept is not None:
        run["mtp_accept_stats"] = accept
    return run


def _write_runs(tmp_path: Path, runs: list[dict]) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    for i, run in enumerate(runs):
        (d / f"run_{i:02d}.json").write_text(json.dumps(run))
    return d


MDIR = "/models/qwen-fake"
HINTS = ("qwen-fake",)


def test_decode_bytes_shapes():
    assert _decode_bytes({"decode": {"bytes": 5}}) == 5
    assert _decode_bytes({"decode": {"bytes": 0}}) is None
    assert _decode_bytes({"prefill": {"bytes": 5}}) is None
    assert _decode_bytes(None) is None
    assert _decode_bytes("junk") is None


def test_derive_verify_mult_pair(tmp_path):
    runs = [
        _run(bytes=1_000_000),
        _run(mtp=True, bytes=2_300_000),
    ]
    rdir = _write_runs(tmp_path, runs)
    out = derive_verify_mult([rdir], MDIR, HINTS)
    assert out is not None
    assert out["verify_byte_mult"] == pytest.approx(2.3)
    assert out["bytes_base"] == 1_000_000


def test_derive_verify_mult_prefers_newest(tmp_path):
    # Older base first, newer base last: the pair must use the newer one.
    runs = [
        _run(bytes=500_000),
        _run(bytes=1_000_000),
        _run(mtp=True, bytes=2_000_000),
    ]
    rdir = _write_runs(tmp_path, runs)
    out = derive_verify_mult([rdir], MDIR, HINTS)
    assert out["verify_byte_mult"] == pytest.approx(2.0)


def test_derive_verify_mult_no_data(tmp_path):
    runs = [_run(bytes=None), _run(mtp=True, bytes=None)]
    rdir = _write_runs(tmp_path, runs)
    assert derive_verify_mult([rdir], MDIR, HINTS) is None


def test_derive_verify_mult_implausible_ratio_rejected(tmp_path):
    # Different decode lengths would poison the ratio; bounded check drops it.
    runs = [_run(bytes=1_000_000), _run(mtp=True, bytes=20_000_000)]
    rdir = _write_runs(tmp_path, runs)
    assert derive_verify_mult([rdir], MDIR, HINTS) is None


def test_derive_skips_other_models(tmp_path):
    runs = [
        _run(model="glm-fake", bytes=1_000_000),
        _run(model="glm-fake", mtp=True, bytes=2_300_000),
    ]
    rdir = _write_runs(tmp_path, runs)
    assert derive_verify_mult([rdir], MDIR, HINTS) is None


def test_derive_tok_per_cycle_from_tokens(tmp_path):
    runs = [
        _run(mtp=True, accept={"cycles": 54, "accepted": 40,
                                "drafted": 54, "fallbacks": 2}),
    ]
    rdir = _write_runs(tmp_path, runs)
    out = derive_tok_per_cycle([rdir], MDIR, HINTS)
    # 96 tokens over 54 cycles
    assert out["tok_per_cycle"] == pytest.approx(96 / 54, abs=0.01)


def test_derive_tok_per_cycle_fallback_accept_rate(tmp_path):
    runs = [
        _run(mtp=True, tokens=None,
             accept={"cycles": 100, "accepted": 79,
                     "drafted": 100, "fallbacks": 0}),
    ]
    rdir = _write_runs(tmp_path, runs)
    out = derive_tok_per_cycle([rdir], MDIR, HINTS)
    assert out["tok_per_cycle"] == pytest.approx(1.79, abs=0.01)


def test_derive_tok_per_cycle_none_without_stats(tmp_path):
    runs = [_run(mtp=True)]
    rdir = _write_runs(tmp_path, runs)
    assert derive_tok_per_cycle([rdir], MDIR, HINTS) is None


def test_derive_bytes_per_token_base(tmp_path):
    runs = [_run(bytes=96_000_000, tokens=96)]
    rdir = _write_runs(tmp_path, runs)
    out = derive_bytes_per_token_base([rdir], MDIR, HINTS)
    assert out["bytes_per_token"] == pytest.approx(1_000_000)


def test_update_and_load_auto_params(tmp_path, monkeypatch):
    import omlx.utils.storage_roofline as sr

    # Redirect the persistence dir into tmp_path.
    monkeypatch.setattr(sr, "_results_dir", lambda: tmp_path / "sr")
    runs = [
        _run(bytes=1_000_000),
        _run(mtp=True, bytes=2_300_000,
             accept={"cycles": 54, "accepted": 40, "drafted": 54, "fallbacks": 0}),
    ]
    rdir = _write_runs(tmp_path, runs)
    out = update_auto_params(MDIR, HINTS, results_dirs=[rdir])
    assert out is not None
    assert out["verify_byte_mult"] == pytest.approx(2.3)
    assert out["tok_per_cycle"] == pytest.approx(96 / 54, abs=0.01)
    assert out["bytes_per_token_base"] == pytest.approx(1_000_000 / 96)
    # Round-trip through disk.
    loaded = load_auto_params(MDIR)
    assert loaded["verify_byte_mult"] == out["verify_byte_mult"]
    assert "derived_at" in loaded


def test_update_auto_params_none_when_no_data(tmp_path, monkeypatch):
    import omlx.utils.storage_roofline as sr

    monkeypatch.setattr(sr, "_results_dir", lambda: tmp_path / "sr")
    rdir = _write_runs(tmp_path, [_run(bytes=None)])
    assert update_auto_params(MDIR, HINTS, results_dirs=[rdir]) is None
    assert load_auto_params(MDIR) is None


# ---------------------------------------------------------------------------
# arm_read_telemetry + adapter counters
# ---------------------------------------------------------------------------


def test_arm_read_telemetry_flips_default_and_backings():
    from omlx.patches.expert_streaming.shard_bank import (
        ExpertBackingStore,
        arm_read_telemetry,
    )

    prev = arm_read_telemetry(True)
    try:
        # Backings created after arming inherit the armed default.
        store = ExpertBackingStore(model_path=".")
        assert store.read_telemetry.enabled is True
        # Flipping again updates live instances in place.
        arm_read_telemetry(False)
        assert store.read_telemetry.enabled is False
    finally:
        arm_read_telemetry(prev)


def test_adapter_mtp_stats_counters():
    from omlx.models.vlm import VLMModelAdapter

    class _InnerNoHooks:
        pass

    class _VLM:
        language_model = _InnerNoHooks()

    adapter = VLMModelAdapter.__new__(VLMModelAdapter)
    adapter._language_model = _VLM.language_model
    adapter.mtp_stats = {"cycles": 0, "accepted": 0, "drafted": 0, "fallbacks": 0}

    # No-hook clamp path: returns accepted unchanged and records the cycle.
    got = adapter.mtp_clamp_accept(cache=None, accepted=1, num_drafts=1)
    assert got == 1
    assert adapter.mtp_stats == {"cycles": 1, "accepted": 1, "drafted": 1, "fallbacks": 0}

    # Partial accept on a later cycle.
    adapter.mtp_clamp_accept(cache=None, accepted=0, num_drafts=1)
    assert adapter.mtp_stats["cycles"] == 2
    assert adapter.mtp_stats["accepted"] == 1

    # No-hook rollback fallback counts a wasted draft.
    assert adapter.mtp_partial_rollback(caches=[], accepted=0, num_drafts=1) is False
    assert adapter.mtp_stats["fallbacks"] == 1


def test_adapter_clamp_hook_clamps_count():
    from omlx.models.vlm import VLMModelAdapter

    class _InnerClamps:
        def mtp_clamp_accept(self, cache, accepted, num_drafts):
            return min(accepted, 1)

    class _VLM:
        language_model = _InnerClamps()

    adapter = VLMModelAdapter.__new__(VLMModelAdapter)
    adapter._language_model = _VLM.language_model
    adapter.mtp_stats = {"cycles": 0, "accepted": 0, "drafted": 0, "fallbacks": 0}

    got = adapter.mtp_clamp_accept(cache=None, accepted=2, num_drafts=3)
    assert got == 1
    # recorded 2 then subtracted the 1 clamped away
    assert adapter.mtp_stats["accepted"] == 1
    assert adapter.mtp_stats["drafted"] == 3
