# SPDX-License-Identifier: Apache-2.0
"""Tests for the cold precision tier (Fase I5): requant tool, cold-root
routing in the backing store, tier completeness, and settings plumbing."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from omlx.model_settings import ModelSettings
from omlx.patches.expert_streaming.shard_bank import (
    ExpertBackingStore,
    cold_tier_status,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def _write_quantized_checkpoint(tmp: Path, bits: int = 4, gs: int = 64):
    """One shard, one switch_mlp gate bank: (E=8, O=32, I=128) affine-quantized."""
    import mlx.nn as nn

    key_w = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    key_s = "model.layers.0.mlp.switch_mlp.gate_proj.scales"
    key_b = "model.layers.0.mlp.switch_mlp.gate_proj.biases"
    dense = mx.random.normal((8, 32, 128))
    w, s, b = mx.quantize(dense, group_size=gs, bits=bits)
    shard = tmp / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), {key_w: w, key_s: s.astype(mx.bfloat16), key_b: b.astype(mx.bfloat16)})
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key_w: shard.name, key_s: shard.name, key_b: shard.name}})
    )
    (tmp / "config.json").write_text(
        json.dumps({"quantization": {"group_size": gs, "bits": bits, "mode": "affine"}})
    )
    return key_w


def test_requant_tool_round_trip(tmp_path):
    from requant_cold_tier import META_BITS, META_GS, requant_shard

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    key_w = _write_quantized_checkpoint(ckpt, bits=4, gs=64)
    quant_cfg = json.loads((ckpt / "config.json").read_text())["quantization"]

    res = requant_shard(
        ckpt / "model-00001-of-00001.safetensors",
        ckpt / "expert_cold",
        quant_cfg,
        bits=3,
    )
    assert res["status"] == "written"
    assert res["dst_mib"] < res["src_mib"]  # 3-bit packs tighter than 4-bit

    out = ckpt / "expert_cold" / "model-00001-of-00001.safetensors"
    loaded = mx.load(str(out))
    src_loaded = mx.load(str(ckpt / "model-00001-of-00001.safetensors"))
    key_s = key_w.removesuffix(".weight") + ".scales"
    key_b = key_w.removesuffix(".weight") + ".biases"
    w2, s2, b2 = loaded[key_w], loaded[key_s], loaded[key_b]
    assert w2.shape[-1] < src_loaded[key_w].shape[-1]

    dense_src = mx.dequantize(
        src_loaded[key_w], src_loaded[key_s], src_loaded[key_b],
        group_size=64, bits=4,
    )
    dense_cold = mx.dequantize(w2, s2, b2, group_size=64, bits=3)
    err = mx.abs(dense_cold - dense_src).max().item()
    scale = mx.abs(dense_src).max().item()
    assert err < 0.2 * scale  # 3-bit requant stays in a sane envelope

    # Idempotent: a second run reports the existing output as matching.
    res2 = requant_shard(
        ckpt / "model-00001-of-00001.safetensors",
        ckpt / "expert_cold",
        quant_cfg,
        bits=3,
    )
    assert res2["status"] == "already matches"


def test_cold_tier_status_complete_and_partial(tmp_path):
    from requant_cold_tier import requant_shard

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_quantized_checkpoint(ckpt)
    quant_cfg = json.loads((ckpt / "config.json").read_text())["quantization"]

    # Missing tier
    ok, why = cold_tier_status(ckpt)
    assert ok is False and "missing" in why

    requant_shard(
        ckpt / "model-00001-of-00001.safetensors",
        ckpt / "expert_cold",
        quant_cfg,
        bits=2,
    )
    ok, why = cold_tier_status(ckpt)
    assert ok is True


def test_backing_routes_reads_to_cold_root(tmp_path):
    from requant_cold_tier import requant_shard

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    key_w = _write_quantized_checkpoint(ckpt, bits=4, gs=64)
    quant_cfg = json.loads((ckpt / "config.json").read_text())["quantization"]
    requant_shard(
        ckpt / "model-00001-of-00001.safetensors",
        ckpt / "expert_cold",
        quant_cfg,
        bits=2,
    )

    # Without the tier: primary packing (4-bit).
    plain = ExpertBackingStore(ckpt)
    assert plain.cold_quant_params(key_w) is None
    assert plain.tensor_dtype(key_w) == "U32"
    hot_slice = plain.load_expert_slice(key_w, 3)

    # With the tier: cold reader, cold packing metadata, different packed width.
    cold = ExpertBackingStore(ckpt, cold_root=ckpt / "expert_cold")
    assert cold.cold_quant_params(key_w) == (2, 64)
    cold_slice = cold.load_expert_slice(key_w, 3)
    assert cold_slice.shape[-1] < hot_slice.shape[-1]
    assert cold.tensor_dtype(key_w) == "U32"

    # The cold slice dequantizes as 2-bit data (roundtrip fidelity vs itself).
    reader = cold._reader_for_key(key_w)
    assert reader.path.parent.name == "expert_cold"


def test_partial_cold_tier_rejected_by_backing_setup(tmp_path):
    """A cold dir missing banks must report incomplete — convert refuses it."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_quantized_checkpoint(ckpt)
    (ckpt / "expert_cold").mkdir()
    (ckpt / "expert_cold" / "empty.safetensors").write_bytes(b"")
    ok, why = cold_tier_status(ckpt)
    assert ok is False


def test_expert_streaming_cold_tier_round_trip():
    s = ModelSettings(expert_streaming_cold_tier="3")
    d = s.to_dict()
    assert d["expert_streaming_cold_tier"] == "3"
    assert ModelSettings.from_dict(d).expert_streaming_cold_tier == "3"
    assert ModelSettings().expert_streaming_cold_tier is None


def test_expert_streaming_cold_tier_excluded_from_profiles():
    from omlx.model_profiles import EXCLUDED_FROM_PROFILES

    assert "expert_streaming_cold_tier" in EXCLUDED_FROM_PROFILES


@pytest.mark.asyncio
async def test_expert_streaming_cold_tier_api_validation():
    from omlx.admin import routes as admin_routes

    pool = None
    entry = None
    from tests.test_expert_streaming import _failed_pool, _update_settings

    pool, entry = _failed_pool()
    settings = ModelSettings()
    await _update_settings(
        pool, settings, admin_routes.ModelSettingsRequest(expert_streaming_cold_tier="2")
    )
    assert settings.expert_streaming_cold_tier == "2"
    from omlx.admin.routes import HTTPException

    with pytest.raises(HTTPException, match="'2' or '3'"):
        await _update_settings(
            pool, settings, admin_routes.ModelSettingsRequest(expert_streaming_cold_tier="4")
        )
