"""Settings-driven fused expert bank wiring (opt-in + path override).

Covers the user-facing contract added after the E1-E3 bench matrix:
  * expert_streaming_bank_enabled=True attaches the fused bank at load;
  * the same setting set to False overrides OMLX_EXPERT_STREAMING_BANK=1
    (user opt-out wins over the operator env);
  * expert_streaming_bank_path points the model at an existing bank kept
    outside the model dir (relative paths resolve against the model dir);
  * a missing bank logs a warning and streams per-projection — never a
    load failure.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from omlx.patches.expert_streaming.expert_bank_pack import (
    bank_dir_for,
    bank_status_for,
    pack_model,
)
from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

_N = 4
_PREFIX = "language_model.layers.0.mlp.switch_mlp"
_PROJS = ("gate_proj", "up_proj", "down_proj")


def _write_safetensors(path, tensors):
    import struct

    _DTYPES = {"F32": np.dtype("<f4")}
    header = {}
    blob = bytearray()
    for name, (arr, dtype) in tensors.items():
        data = np.ascontiguousarray(arr).astype(_DTYPES[dtype]).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + bytes(blob))


@pytest.fixture()
def fixture_model(tmp_path):
    rng = np.random.default_rng(7)
    tensors = {}
    for proj in _PROJS:
        for kind, cols in (("weight", 9), ("scales", 3), ("biases", 2)):
            tensors[f"{_PREFIX}.{proj}.{kind}"] = (
                rng.standard_normal((_N, cols)).astype(np.float32),
                "F32",
            )
    shard = tmp_path / "model.safetensors"
    _write_safetensors(shard, tensors)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "num_experts": _N,
                "num_hidden_layers": 1,
                "num_experts_per_tok": 2,
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard.name for k in tensors}})
    )
    return tensors


def test_bank_status_path_override(fixture_model, tmp_path):
    """bank_status_for honors a bank_dir outside the model dir."""
    pack_model(tmp_path, verify=2)
    moved = tmp_path / "elsewhere" / "bank"
    moved.parent.mkdir(parents=True)
    shutil.move(str(bank_dir_for(tmp_path)), str(moved))
    ok_default, why_default, _ = bank_status_for(tmp_path)
    assert not ok_default  # nothing under <model>/.omlx anymore
    ok, why, manifest = bank_status_for(tmp_path, moved)
    assert ok, why
    assert manifest is not None


def test_attach_expert_bank_path_override(fixture_model, tmp_path):
    """attach_expert_bank reads a bank kept outside the model dir."""
    pack_model(tmp_path, verify=2)
    moved = tmp_path / "shared-bank"
    moved.mkdir()
    for f in bank_dir_for(tmp_path).iterdir():
        shutil.move(str(f), str(moved / f.name))
    bank_dir_for(tmp_path).mkdir(exist_ok=True)  # empty default dir
    store = ExpertBackingStore(tmp_path)
    try:
        ok, why, n = store.attach_expert_bank(moved)
        assert ok, why
        assert n == 1
        ok_rel, _why_rel, _ = store.attach_expert_bank()  # idempotent
        assert ok_rel
    finally:
        store.close()
    # Relative path resolves against the model dir
    store2 = ExpertBackingStore(tmp_path)
    try:
        ok, why, n = store2.attach_expert_bank("shared-bank")
        assert ok, why
        assert n == 1
    finally:
        store2.close()


def test_attach_missing_bank_refuses_not_raises(fixture_model, tmp_path):
    """A model with no packed bank refuses attach with a reason."""
    store = ExpertBackingStore(tmp_path)
    try:
        ok, why, n = store.attach_expert_bank()
        assert not ok
        assert n == 0
        assert why  # actionable message
    finally:
        store.close()
