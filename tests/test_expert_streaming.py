# SPDX-License-Identifier: Apache-2.0
"""Tests for MoE expert streaming (SSD) — residency, settings, and forced logic."""

import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from omlx.admin import routes as admin_routes
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings
from omlx.patches.expert_streaming.residency import expert_streaming_estimate


def _failed_pool() -> tuple[EnginePool, EngineEntry]:
    pool = EnginePool()
    entry = EngineEntry(
        model_id="glm-moe",
        model_path="/tmp/glm-moe",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        load_failed=True,
        load_failure_message="failed",
        load_failure_at=1.0,
    )
    pool._entries[entry.model_id] = entry
    return pool, entry


async def _update_settings(pool, settings, request):
    manager = MagicMock()
    manager.get_settings.return_value = settings
    state = MagicMock()
    with (
        patch("omlx.admin.routes._get_engine_pool", return_value=pool),
        patch("omlx.admin.routes._get_settings_manager", return_value=manager),
        patch("omlx.admin.routes._get_server_state", return_value=state),
    ):
        result = await admin_routes.update_model_settings("glm-moe", request, is_admin=True)
    manager.set_settings.assert_called_once()
    return result


def _write_fake_glm_checkpoint(tmp: Path, num_layers=4, experts=8, hidden=32, moe_hidden=16):
    """Create a minimal fake GLM MoE DSA checkpoint with headers only (no real weights)."""
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": num_layers,
        "n_routed_experts": experts,
        "hidden_size": hidden,
        "moe_intermediate_size": moe_hidden,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
    }
    (tmp / "config.json").write_text(json.dumps(config))
    # Build a single safetensors file with stacked expert banks per layer
    # We write valid safetensors header + dummy data.
    import numpy as np

    # helper to write one safetensors file with a set of tensors
    # Use safetensors library if available, otherwise manual header
    tensors = {}
    for layer in range(num_layers):
        for proj, shape in [
            ("gate_proj", (experts, moe_hidden, hidden)),
            ("up_proj", (experts, moe_hidden, hidden)),
            ("down_proj", (experts, hidden, moe_hidden)),
        ]:
            key = f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight"
            # BF16: 2 bytes per element
            size = int(np.prod(shape)) * 2
            tensors[key] = (shape, "BF16", size)
        # shared experts etc not needed

    # Also add a dense tensor to have dense_bytes
    tensors["model.embed_tokens.weight"] = ((32000, hidden), "BF16", 32000 * hidden * 2)
    tensors["lm_head.weight"] = ((32000, hidden), "BF16", 32000 * hidden * 2)

    # Build header
    header = {}
    offset = 0
    for k, (shape, dtype, size) in tensors.items():
        header[k] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    header_bytes = json.dumps(header).encode()
    # Write file
    fname = tmp / "model.safetensors"
    with fname.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)
    # Also write index
    weight_map = {k: fname.name for k in tensors}
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_residency_supported_for_glm():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=8)
        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        assert est.num_moe_layers == 4
        assert est.experts_per_layer == 8
        assert est.per_expert_bytes > 0
        assert est.expert_bytes > 0
        assert est.dense_bytes > 0
        assert est.resident_bytes > est.streaming_bytes
        # budget 2 GiB should give some slots
        assert est.slots_for_budget(2 * 1024**3) == 8  # all experts fit


def test_residency_unsupported_for_dense_model():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 2}))
        # minimal safetensors with no expert keys
        header = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [100, 32], "data_offsets": [0, 6400]}}
        hb = json.dumps(header).encode()
        with (tmp / "model.safetensors").open("wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            f.write(b"\x00" * 6400)
        est = expert_streaming_estimate(tmp)
        assert est.supported is False


def test_residency_force_streaming():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=16, hidden=64, moe_hidden=32)
        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        # streaming should be smaller than resident
        assert est.resident_bytes > est.streaming_bytes
        ceiling = (est.resident_bytes + est.streaming_bytes) // 2
        assert est.force_streaming(ceiling) is True
        assert est.force_streaming(est.resident_bytes + 1) is False
        assert est.force_streaming(max(0, est.streaming_bytes - 1)) is False


def test_model_settings_round_trip():
    s = ModelSettings(expert_streaming_enabled=True, expert_streaming_budget_gib=3.5)
    d = s.to_dict()
    assert d["expert_streaming_enabled"] is True
    assert d["expert_streaming_budget_gib"] == 3.5
    s2 = ModelSettings.from_dict(d)
    assert s2.expert_streaming_enabled is True
    assert s2.expert_streaming_budget_gib == 3.5


def test_model_settings_default_not_streaming():
    s = ModelSettings()
    assert s.expert_streaming_enabled is False
    assert s.expert_streaming_budget_gib is None


@pytest.mark.asyncio
async def test_expert_streaming_persists_via_api():
    pool, entry = _failed_pool()
    entry.config_model_type = "glm_moe_dsa"
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_enabled=True, expert_streaming_budget_gib=2.0),
    )
    assert settings.expert_streaming_enabled is True
    assert settings.expert_streaming_budget_gib == 2.0


@pytest.mark.asyncio
async def test_expert_streaming_budget_validation():
    pool, _ = _failed_pool()
    settings = ModelSettings()
    with pytest.raises(admin_routes.HTTPException, match="between 0 and 64"):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(expert_streaming_budget_gib=100),
        )


@pytest.mark.asyncio
async def test_expert_streaming_budget_cleared_with_null():
    pool, _ = _failed_pool()
    settings = ModelSettings(expert_streaming_budget_gib=4.0)
    await _update_settings(pool, settings, admin_routes.ModelSettingsRequest(expert_streaming_budget_gib=None))
    assert settings.expert_streaming_budget_gib is None


@pytest.mark.asyncio
async def test_expert_streaming_change_triggers_reload_signature():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=2, experts=4, hidden=32, moe_hidden=16)
        pool, entry = _failed_pool()
        entry.model_path = str(tmp)
        entry.config_model_type = "glm_moe_dsa"
        entry.engine = MagicMock()
        entry.load_failed = False
        pool._unload_engine = AsyncMock()
        result = await _update_settings(pool, ModelSettings(), admin_routes.ModelSettingsRequest(expert_streaming_enabled=True))
        assert result["requires_reload"] is True
        assert result["auto_unloaded"] is True


def test_expert_streaming_excluded_from_profiles():
    from omlx.model_profiles import EXCLUDED_FROM_PROFILES

    assert "expert_streaming_enabled" in EXCLUDED_FROM_PROFILES
    assert "expert_streaming_budget_gib" in EXCLUDED_FROM_PROFILES


def test_engine_pool_forced_streaming_status(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=8, hidden=64, moe_hidden=32)
        pool = EnginePool()
        entry = EngineEntry(
            model_id="m",
            model_path=str(tmp),
            model_type="llm",
            engine_type="batched",
            estimated_size=1,
            config_model_type="glm_moe_dsa",
        )
        pool._entries["m"] = entry

        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        assert est.resident_bytes > est.streaming_bytes
        ceiling = (est.resident_bytes + est.streaming_bytes) // 2
        monkeypatch.setattr(pool, "_fallback_admission_ceiling", lambda: ceiling)
        monkeypatch.setattr(pool, "_current_ceiling", lambda: ceiling)
        enabled, forced, _ = pool._expert_streaming_status(entry, ModelSettings())
        assert forced is True
        assert enabled is True
        eff = pool._effective_expert_streaming_settings(entry, ModelSettings())
        assert eff.expert_streaming_enabled is True


def test_engine_pool_dflash_blocked_by_expert_streaming(monkeypatch):
    """DFlash selection must yield to expert streaming (requested or forced):
    DFlash's pipeline loads the target fully resident and never applies the
    streaming patches, so the two settings are mutually exclusive."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=8, hidden=64, moe_hidden=32)
        pool = EnginePool()
        entry = EngineEntry(
            model_id="m",
            model_path=str(tmp),
            model_type="llm",
            engine_type="batched",
            estimated_size=1,
            config_model_type="glm_moe_dsa",
        )

        # Requested streaming: the explicit setting wins over DFlash.
        dflash_settings = ModelSettings(
            dflash_enabled=True, dflash_draft_model="/tmp/draft"
        )
        assert pool._dflash_blocked_by_expert_streaming(entry, dflash_settings) is False
        dflash_settings.expert_streaming_enabled = True
        assert pool._dflash_blocked_by_expert_streaming(entry, dflash_settings) is True

        # Forced streaming: memory ceiling forces it without an explicit setting.
        est = expert_streaming_estimate(tmp)
        ceiling = (est.resident_bytes + est.streaming_bytes) // 2
        monkeypatch.setattr(pool, "_fallback_admission_ceiling", lambda: ceiling)
        monkeypatch.setattr(pool, "_current_ceiling", lambda: ceiling)
        unrequested = ModelSettings(
            dflash_enabled=True, dflash_draft_model="/tmp/draft"
        )
        assert pool._dflash_blocked_by_expert_streaming(entry, unrequested) is True

        # Streaming that cannot fit streamed (streaming_bytes > ceiling) does
        # not force, so DFlash stays eligible.
        tiny_ceiling = est.streaming_bytes // 2
        monkeypatch.setattr(pool, "_fallback_admission_ceiling", lambda: tiny_ceiling)
        monkeypatch.setattr(pool, "_current_ceiling", lambda: tiny_ceiling)
        assert pool._dflash_blocked_by_expert_streaming(entry, unrequested) is False


# ---------------------------------------------------------------------------
# DeepSeek V4 (ffn-nested MoE + MTP/DSpark stages)
# ---------------------------------------------------------------------------


def _write_fake_dsv4_checkpoint(
    tmp: Path,
    num_layers=2,
    experts=8,
    hidden=32,
    moe_hidden=16,
    mtp_stages=2,
):
    """Fake DeepSeek V4 checkpoint: ffn-nested switch_mlp + mtp.<stage> banks."""
    import numpy as np

    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": num_layers,
        "n_routed_experts": experts,
        "hidden_size": hidden,
        "moe_intermediate_size": moe_hidden,
    }
    (tmp / "config.json").write_text(json.dumps(config))

    tensors = {}

    def _add_bank(prefix: str) -> None:
        for proj, shape in [
            ("gate_proj", (experts, moe_hidden, hidden)),
            ("up_proj", (experts, moe_hidden, hidden)),
            ("down_proj", (experts, hidden, moe_hidden)),
        ]:
            key = f"{prefix}.{proj}.weight"
            tensors[key] = (shape, "BF16", int(np.prod(shape)) * 2)

    for layer in range(num_layers):
        _add_bank(f"model.layers.{layer}.ffn.switch_mlp")
    for stage in range(mtp_stages):
        _add_bank(f"mtp.{stage}.ffn.switch_mlp")

    tensors["model.embed_tokens.weight"] = ((32000, hidden), "BF16", 32000 * hidden * 2)
    tensors["lm_head.weight"] = ((32000, hidden), "BF16", 32000 * hidden * 2)

    header = {}
    offset = 0
    for k, (shape, dtype, size) in tensors.items():
        header[k] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    header_bytes = json.dumps(header).encode()
    fname = tmp / "model.safetensors"
    with fname.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)
    weight_map = {k: fname.name for k in tensors}
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_residency_supported_for_deepseek_v4_with_mtp():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        num_layers, experts, stages = 3, 8, 2
        _write_fake_dsv4_checkpoint(tmp, num_layers=num_layers, experts=experts, mtp_stages=stages)
        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        # MTP/DSpark stages count as streamable MoE layers
        assert est.num_moe_layers == num_layers + stages
        assert est.experts_per_layer == experts
        assert est.per_expert_bytes > 0
        assert est.expert_bytes > 0
        assert est.resident_bytes > est.streaming_bytes
        assert est.slots_for_budget(64 * 1024**3) == experts  # all fit


def test_residency_mtp_excluded_when_absent():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_dsv4_checkpoint(tmp, num_layers=2, experts=8, mtp_stages=0)
        est = expert_streaming_estimate(tmp)
        assert est.supported is True
        assert est.num_moe_layers == 2


def test_stacked_key_candidates_cover_ffn_and_mtp():
    from omlx.patches.expert_streaming import (
        _candidate_stacked_keys,
        _mtp_candidate_stacked_keys,
        _resolve_stacked_key,
    )

    cands = _candidate_stacked_keys(5, "gate_proj", "weight")
    assert cands[0] == "model.layers.5.mlp.switch_mlp.gate_proj.weight"
    assert "model.layers.5.ffn.switch_mlp.gate_proj.weight" in cands

    mtp_cands = _mtp_candidate_stacked_keys(2, "up_proj", "scales")
    assert mtp_cands[0] == "mtp.2.ffn.switch_mlp.up_proj.scales"
    assert mtp_cands[1] == "mtp.2.block.ffn.switch_mlp.up_proj.scales"

    class _MapFFN:
        _weight_map = {"model.layers.5.ffn.switch_mlp.gate_proj.weight": "a.safetensors"}

    assert (
        _resolve_stacked_key(cands, "gate_proj", "weight", _MapFFN(), "layers.5.")
        == "model.layers.5.ffn.switch_mlp.gate_proj.weight"
    )

    class _MapScan:
        _weight_map = {"model.layers.5.ffn.switch_mlp.up_proj.weight": "b.safetensors"}

    assert (
        _resolve_stacked_key(
            _candidate_stacked_keys(5, "up_proj", "weight"),
            "up_proj",
            "weight",
            _MapScan(),
            "layers.5.",
        )
        == "model.layers.5.ffn.switch_mlp.up_proj.weight"
    )

    # no weight map (RAM dict / single-file) -> legacy mlp default
    assert (
        _resolve_stacked_key(cands, "gate_proj", "weight", None, "layers.5.")
        == "model.layers.5.mlp.switch_mlp.gate_proj.weight"
    )


class _FakeProj:
    def __init__(self, shape):
        import mlx.core as mx

        self.weight = mx.ones(shape)


class _FakeSwitchGLU:
    def __init__(self, e, o, i, activation=None):
        self.gate_proj = _FakeProj((e, o, i))
        self.up_proj = _FakeProj((e, o, i))
        self.down_proj = _FakeProj((e, i, o))
        if activation is not None:
            self.activation = activation


class _FakeMoE:
    def __init__(self, e, o, i, activation=None):
        self.switch_mlp = _FakeSwitchGLU(e, o, i, activation)


class _FakeLayer:
    def __init__(self, e, o, i):
        self.ffn = _FakeMoE(e, o, i)


class _FakeDSparkStage:
    def __init__(self, e, o, i):
        self.ffn = _FakeMoE(e, o, i)


class _FakeTextModel:
    def __init__(self, layers, mtp, config):
        self.model = type("InnerModel", (), {})()
        self.model.layers = layers
        self.model.mtp = mtp
        self.config = config


def test_convert_walk_ffn_and_mtp():
    from omlx.patches.expert_streaming import convert_model_to_streaming
    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        num_layers, experts, hidden, moe_hidden, stages = 2, 4, 8, 4, 2
        _write_fake_dsv4_checkpoint(
            tmp,
            num_layers=num_layers,
            experts=experts,
            hidden=hidden,
            moe_hidden=moe_hidden,
            mtp_stages=stages,
        )
        marker = object()
        layers = [_FakeLayer(experts, moe_hidden, hidden) for _ in range(num_layers)]
        layers[0].ffn.switch_mlp.activation = marker
        mtp = [_FakeDSparkStage(experts, moe_hidden, hidden) for _ in range(stages)]
        model = _FakeTextModel(
            layers,
            mtp,
            {
                "model_type": "deepseek_v4",
                "hidden_size": hidden,
                "moe_intermediate_size": moe_hidden,
            },
        )

        out_model, backing = convert_model_to_streaming(
            model, str(tmp), None, use_file_backing=False
        )
        assert out_model is model
        # ram-dict backing is not returned (existing contract)
        assert backing is None

        for layer in layers:
            sm = layer.ffn.switch_mlp
            assert isinstance(sm, StreamingSwitchGLU)
            assert isinstance(sm.gate_proj, StreamingSwitchLinear)
            assert isinstance(sm.down_proj, StreamingSwitchLinear)
        # activation copied from the original switch_mlp
        assert layers[0].ffn.switch_mlp._activation is marker
        assert layers[1].ffn.switch_mlp._activation is None
        # MTP stages converted with distinct cache layer ids
        for stage in mtp:
            assert isinstance(stage.ffn.switch_mlp, StreamingSwitchGLU)


def test_convert_walk_mtpblock_layout():
    from omlx.patches.expert_streaming import convert_model_to_streaming
    from omlx.patches.expert_streaming.streaming_switch import StreamingSwitchGLU

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        num_layers, experts, hidden, moe_hidden = 2, 4, 8, 4
        _write_fake_dsv4_checkpoint(
            tmp,
            num_layers=num_layers,
            experts=experts,
            hidden=hidden,
            moe_hidden=moe_hidden,
            mtp_stages=0,
        )
        layers = [_FakeLayer(experts, moe_hidden, hidden) for _ in range(num_layers)]

        class _Block:
            def __init__(self, e, o, i):
                self.ffn = _FakeMoE(e, o, i)

        class _MTPBlockStage:
            def __init__(self, e, o, i):
                self.block = _Block(e, o, i)

        mtp = [_MTPBlockStage(experts, moe_hidden, hidden)]
        model = _FakeTextModel(
            layers,
            mtp,
            {
                "model_type": "deepseek_v4",
                "hidden_size": hidden,
                "moe_intermediate_size": moe_hidden,
            },
        )
        convert_model_to_streaming(model, str(tmp), None, use_file_backing=False)
        assert isinstance(layers[0].ffn.switch_mlp, StreamingSwitchGLU)
        assert isinstance(mtp[0].block.ffn.switch_mlp, StreamingSwitchGLU)


def test_streaming_glu_uses_custom_activation():
    import mlx.core as mx

    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    experts, moe_hidden, hidden = 4, 6, 8
    calls = []

    def act(up, gate):
        calls.append((tuple(up.shape), tuple(gate.shape)))
        return up + gate

    cache = ExpertLRUCache(1 << 24, 4096, num_layers=1)
    backing = {
        (0, "gate_proj"): mx.arange(experts * moe_hidden * hidden, dtype=mx.float32).reshape(
            experts, moe_hidden, hidden
        )
        / 100.0,
        (0, "up_proj"): mx.arange(experts * moe_hidden * hidden, dtype=mx.float32).reshape(
            experts, moe_hidden, hidden
        )
        / 200.0,
        (0, "down_proj"): mx.arange(experts * hidden * moe_hidden, dtype=mx.float32).reshape(
            experts, hidden, moe_hidden
        )
        / 300.0,
    }
    glu = StreamingSwitchGLU(
        input_dims=hidden,
        hidden_dims=moe_hidden,
        num_experts=experts,
        layer_idx=0,
        backing=backing,
        cache=cache,
        quantized=False,
        activation=act,
    )
    for name, out_dim, in_dim in [
        ("gate_proj", moe_hidden, hidden),
        ("up_proj", moe_hidden, hidden),
        ("down_proj", hidden, moe_hidden),
    ]:
        setattr(
            glu,
            name,
            StreamingSwitchLinear(
                layer_idx=0,
                proj_name=name,
                stacked_key=f"k.{name}",
                num_experts=experts,
                input_dims=in_dim,
                output_dims=out_dim,
                backing=backing,
                cache=cache,
            ),
        )

    x = mx.random.uniform(shape=(1, 3, hidden))
    idx = mx.array([[[0, 1], [2, 3], [0, 2]]])  # (1, 3, 2) — below sort threshold
    out = glu(x, idx)
    # custom activation used with the original call order (up, gate)
    assert len(calls) == 1
    assert glu._activation is act
    assert out.shape == (1, 3, 2, hidden)


def _write_shard(path: Path, tensors: dict[str, tuple[tuple[int, ...], str]]) -> None:
    """Write a minimal safetensors file: valid headers + zero data."""
    import numpy as np

    _DTY_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1}
    header: dict = {}
    offset = 0
    data: dict[str, int] = {}
    for key, (shape, dtype) in tensors.items():
        nbytes = int(np.prod(shape)) * _DTY_BYTES[dtype]
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
        data[key] = nbytes
    hb = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * offset)


def test_backing_store_multi_root_resolution():
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        primary = root / "model"
        extra = root / "stripe"
        extra.mkdir()
        primary.mkdir()

        # Primary shard: dense tensor only. Expert bank lives on the stripe root.
        _write_shard(primary / "model.safetensors", {"model.embed_tokens.weight": ((64, 32), "BF16")})
        _write_shard(extra / "experts.safetensors", {"model.layers.0.mlp.switch_mlp.gate_proj.weight": ((4, 16, 32), "BF16")})
        (primary / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.embed_tokens.weight": "model.safetensors",
                        "model.layers.0.mlp.switch_mlp.gate_proj.weight": "experts.safetensors",
                    }
                }
            )
        )

        backing = ExpertBackingStore(primary, extra_roots=[extra])
        assert backing._resolve_file("experts.safetensors") == (extra / "experts.safetensors").resolve()
        slc = backing.load_expert_slice("model.layers.0.mlp.switch_mlp.gate_proj.weight", 2)
        assert slc.shape == (16, 32)

        # Without the extra root the expert key cannot resolve.
        from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore as BS

        primary_only = BS(primary)
        with pytest.raises((KeyError, FileNotFoundError)):
            primary_only.load_expert_slice("model.layers.0.mlp.switch_mlp.gate_proj.weight", 0)


def test_backing_store_extra_root_wins_for_mirrored_shards():
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        primary = root / "model"
        extra = root / "stripe"
        primary.mkdir()
        extra.mkdir()
        _write_shard(primary / "model.safetensors", {"model.embed_tokens.weight": ((64, 32), "BF16")})
        _write_shard(extra / "model.safetensors", {"model.embed_tokens.weight": ((64, 32), "BF16")})
        (primary / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}})
        )
        backing = ExpertBackingStore(primary, extra_roots=[extra])
        # Mirrored shard is served from the stripe root (that copy is the
        # striped one the user placed there deliberately).
        assert backing._resolve_file("model.safetensors") == (extra / "model.safetensors").resolve()


def test_np_to_mx_bf16_preserves_bits():
    """BF16 promotion must reinterpret raw bits (like mx.load) — subnormals
    included. The old shift->f32->astype path flushed them to zero (FTZ)."""
    import mlx.core as mx
    import numpy as np

    from omlx.patches.expert_streaming.shard_bank import _np_to_mx

    bits = np.array([0x0002, 0x0032, 0x0070, 0x3F80, 0xC000, 0x7F80], dtype=np.uint16)
    out = _np_to_mx("k", bits, "BF16")
    assert out.dtype == mx.bfloat16
    assert np.array_equal(np.array(out.view(mx.uint16)), bits)


def test_streaming_glu_matches_reference():
    """Streaming SwitchGLU (shared routing plan across gate/up/down) must be
    bit-exact against mlx-lm's reference SwitchGLU."""
    import mlx.core as mx
    import numpy as np

    from mlx_lm.models.switch_layers import SwitchGLU

    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    E, H, D = 8, 32, 16
    ref = SwitchGLU(D, H, E)  # default SwiGLU activation
    mx.eval(ref.parameters())

    cache = ExpertLRUCache(1 << 26, 1 << 18, num_layers=1)
    backing: dict = {}
    glu = StreamingSwitchGLU(
        input_dims=D,
        hidden_dims=H,
        num_experts=E,
        layer_idx=0,
        backing=backing,
        cache=cache,
        quantized=False,
        activation=None,
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        ref_lin = getattr(ref, name)
        backing[(0, name)] = ref_lin.weight
        setattr(
            glu,
            name,
            StreamingSwitchLinear(
                layer_idx=0,
                proj_name=name,
                stacked_key=f"k.{name}",
                num_experts=E,
                input_dims=ref_lin.input_dims,
                output_dims=ref_lin.output_dims,
                backing=backing,
                cache=cache,
            ),
        )

    rng = np.random.default_rng(42)
    for trial in range(3):
        T = 40 if trial < 2 else 80  # second size crosses the sort threshold
        x = mx.array(rng.standard_normal((1, T, D)).astype(np.float32))
        idx = mx.array(rng.integers(0, E, size=(1, T, 1)).astype(np.int32))
        out_ref = ref(x, idx)
        out_s = glu(x, idx)
        assert out_s.shape == out_ref.shape
        assert mx.array_equal(out_ref, out_s), f"mismatch at trial {trial}"


def test_streaming_quantized_glu_matches_reference():
    """Quantized streaming GLU (f32 gate/up + qmm down, shared plan, sort
    path) must be bit-exact against the reference composition."""
    import mlx.core as mx
    import numpy as np

    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
        StreamingSwitchLinear,
    )

    E, H, D = 8, 32, 16
    ref = SwitchGLU(D, H, E)
    ref.down_proj = QuantizedSwitchLinear(H, D, E, bias=False, group_size=32, bits=4)
    mx.eval(ref.parameters())

    cache = ExpertLRUCache(1 << 26, 1 << 18, num_layers=1)
    backing: dict = {}
    glu = StreamingSwitchGLU(
        input_dims=D,
        hidden_dims=H,
        num_experts=E,
        layer_idx=0,
        backing=backing,
        cache=cache,
        quantized=True,
        activation=None,
    )
    for name in ("gate_proj", "up_proj"):
        ref_lin = getattr(ref, name)
        backing[(0, name)] = ref_lin.weight
        setattr(
            glu,
            name,
            StreamingSwitchLinear(
                layer_idx=0,
                proj_name=name,
                stacked_key=f"k.{name}",
                num_experts=E,
                input_dims=ref_lin.input_dims,
                output_dims=ref_lin.output_dims,
                backing=backing,
                cache=cache,
            ),
        )
    ref_down = ref.down_proj
    backing[(0, "down_proj", "weight")] = ref_down.weight
    backing[(0, "down_proj", "scales")] = ref_down.scales
    backing[(0, "down_proj", "biases")] = ref_down.biases
    setattr(
        glu,
        "down_proj",
        StreamingQuantizedSwitchLinear(
            layer_idx=0,
            proj_name="down_proj",
            stacked_weight_key="k.down_proj.weight",
            stacked_scales_key="k.down_proj.scales",
            stacked_biases_key="k.down_proj.biases",
            num_experts=E,
            input_dims=ref_down.input_dims,
            output_dims=ref_down.output_dims,
            backing=backing,
            cache=cache,
            group_size=32,
            bits=4,
            mode="affine",
        ),
    )

    rng = np.random.default_rng(7)
    for trial in range(3):
        T = 40 if trial < 2 else 80
        x = mx.array(rng.standard_normal((1, T, D)).astype(np.float32))
        idx = mx.array(rng.integers(0, E, size=(1, T, 1)).astype(np.int32))
        out_ref = ref(x, idx)
        out_s = glu(x, idx)
        assert out_s.shape == out_ref.shape
        assert mx.array_equal(out_ref, out_s), f"mismatch at trial {trial}"


def test_budget_zero_means_page_cache_only():
    """Explicit budget 0 = page-cache only: no LRU slots, admission charges
    dense only (file-cache pages are clean/evictable, not committed)."""
    import mlx.core as mx

    from omlx.patches.expert_streaming import _get_budget_bytes
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    assert _get_budget_bytes(ModelSettings(expert_streaming_budget_gib=0.0), None) == 0
    assert _get_budget_bytes(ModelSettings(expert_streaming_budget_gib=None), None) == 0
    assert _get_budget_bytes(ModelSettings(expert_streaming_budget_gib=2.0), None) == 2 << 30

    cache = ExpertLRUCache(0, 1 << 20, num_layers=4)
    assert cache.capacity == 0
    cache.put((0, 0, "k"), mx.zeros(4))
    assert cache.size == 0
    assert cache.get((0, 0, "k")) is None


def test_engine_pool_budget_zero_streaming_bytes():
    from omlx.patches.expert_streaming.residency import expert_streaming_estimate

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=2, experts=4, hidden=32, moe_hidden=16)
        est = expert_streaming_estimate(str(tmp))
        assert est.supported
        # 0 budget -> dense-only resident estimate (no cache term)
        assert est.streaming_bytes_for_budget(0) == int(est.dense_bytes * 1.05)
        assert est.slots_for_budget(0) == 0


def test_engine_pool_streaming_budget_helper():
    from omlx.engine_pool import EnginePool

    assert EnginePool._streaming_budget_bytes(None) == 0
    assert EnginePool._streaming_budget_bytes(ModelSettings()) == 0
    assert EnginePool._streaming_budget_bytes(ModelSettings(expert_streaming_budget_gib=0.0)) == 0
    assert EnginePool._streaming_budget_bytes(ModelSettings(expert_streaming_budget_gib=4)) == 4 << 30


@pytest.mark.asyncio
async def test_expert_streaming_budget_zero_persists():
    pool, _ = _failed_pool()
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_budget_gib=0),
    )
    assert settings.expert_streaming_budget_gib == 0.0


def test_adaptive_topk_configure_semantics():
    from omlx.patches.expert_streaming import adaptive_topk

    adaptive_topk.configure(None)
    assert adaptive_topk.current_threshold() is None
    adaptive_topk.configure(1.0)  # >= 1.0 normalizes to exact
    assert adaptive_topk.current_threshold() is None
    adaptive_topk.configure(0.85)
    assert adaptive_topk.current_threshold() == 0.85
    with pytest.raises(ValueError):
        adaptive_topk.configure(0.01)
    adaptive_topk.configure(None)
    assert adaptive_topk.configure_from_settings(ModelSettings()) is None
    assert (
        adaptive_topk.configure_from_settings(
            ModelSettings(expert_streaming_topk_threshold=0.9)
        )
        == 0.9
    )
    adaptive_topk.configure(None)


def test_adaptive_topk_truncate_math():
    import mlx.core as mx

    from omlx.patches.expert_streaming.adaptive_topk import truncate_topk_mass

    # sorted input, threshold 0.7: keep 0.5 + 0.3 (cum_before 0.8 crosses at 3rd)
    inds = mx.array([[7, 3, 5]], dtype=mx.uint32)
    scores = mx.array([[0.5, 0.3, 0.2]], dtype=mx.float32)
    i2, s2, keeps = truncate_topk_mass(inds, scores, 0.7, return_keeps=True)
    assert keeps == 2.0
    assert i2.tolist() == [[7, 3, 7]]  # dropped slot padded with top expert
    assert s2.shape == scores.shape
    # renormalized to the ORIGINAL total mass (1.0)
    total = mx.sum(s2, axis=-1)
    assert abs(float(total) - 1.0) < 1e-6
    assert s2[0][0].item() == pytest.approx(0.625, abs=1e-6)
    assert s2[0][1].item() == pytest.approx(0.375, abs=1e-6)
    assert s2[0][2].item() == 0.0

    # unsorted input: same kept SET, scores reordered descending
    inds3 = mx.array([[5, 7, 3]], dtype=mx.uint32)
    scores3 = mx.array([[0.2, 0.5, 0.3]], dtype=mx.float32)
    i3, s3 = truncate_topk_mass(inds3, scores3, 0.7)
    assert i3.tolist() == [[7, 3, 7]]

    # scale invariance (GLM routed_scaling_factor): x10 scores -> same split
    i4, s4 = truncate_topk_mass(inds, scores * 10.0, 0.7)
    assert i4.tolist() == [[7, 3, 7]]
    assert s4[0][0].item() == pytest.approx(6.25, abs=1e-5)

    # threshold >= total mass keeps everything (order changes only)
    i5, s5 = truncate_topk_mass(inds3, scores3, 0.999)
    assert sorted(i5[0].tolist()) == [3, 5, 7]
    assert abs(float(mx.sum(s5, axis=-1)) - 1.0) < 1e-6


def test_adaptive_topk_qwen_block_patch():
    """The patched Qwen3_5MoeSparseMoeBlock must bypass at exact threshold
    and produce finite, same-shape output when truncation engages."""
    import mlx.core as mx
    from types import SimpleNamespace

    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
    from omlx.patches.expert_streaming import adaptive_topk

    cfg = SimpleNamespace(
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
    )
    block = Qwen3_5MoeSparseMoeBlock(cfg)
    mx.eval(block.parameters())

    applied = adaptive_topk.apply_qwen35_moe_topk_patch()
    if not applied:
        pytest.skip("qwen3_5_moe language module not patchable here")

    rng = __import__("numpy").random.default_rng(3)
    x = mx.array(rng.standard_normal((1, 4, 32)).astype("float32"))

    adaptive_topk.configure(None)
    out_exact = block(x)
    mx.eval(out_exact)

    adaptive_topk.configure(0.6)
    out_trunc = block(x)
    mx.eval(out_trunc)

    assert out_trunc.shape == out_exact.shape
    assert bool(mx.isfinite(out_trunc).all())
    adaptive_topk.configure(None)


def test_shard_bank_pin_expert_mlock():
    """pin_expert locks the page-aligned file range and dedupes repeats."""
    import mlx.core as mx  # noqa: F401

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_shard(
            tmp / "model.safetensors",
            {"model.layers.0.mlp.switch_mlp.gate_proj.weight": ((4, 8, 16), "BF16")},
        )
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.layers.0.mlp.switch_mlp.gate_proj.weight": "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        locked = backing.pin_expert(key, 0)
        assert locked > 0
        assert backing.pinned_bytes == locked
        assert backing.pinned_count == 1
        # duplicate pin skipped
        assert backing.pin_expert(key, 0) == 0
        assert backing.pinned_count == 1
        # range math sanity: page-rounded >= slice bytes
        off, end = backing._reader_for_key(key).expert_byte_range(key, 1)
        assert end - off == 8 * 16 * 2


def test_page_cache_warmer_flow():
    from omlx.patches.expert_streaming import warmer
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"model.layers.0.mlp.switch_mlp.{proj}.weight"] = ((4, 8, 16), "BF16")
            tensors[f"model.layers.0.mlp.switch_mlp.{proj}.scales"] = ((4, 8, 1), "U32")
            tensors[f"model.layers.0.mlp.switch_mlp.{proj}.biases"] = ((4, 8, 1), "U32")
        tensors["model.layers.1.mlp.switch_mlp.gate_proj.weight"] = ((4, 8, 16), "BF16")
        tensors["model.layers.1.mlp.switch_mlp.gate_proj.scales"] = ((4, 8, 1), "U32")
        tensors["model.layers.1.mlp.switch_mlp.gate_proj.biases"] = ((4, 8, 1), "U32")
        _write_shard(tmp / "model.safetensors", tensors)
        wm = {k: "model.safetensors" for k in tensors}
        (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))

        backing = ExpertBackingStore(tmp)
        cache = ExpertLRUCache(0, 1 << 12, num_layers=2)

        def mk_lin(layer, proj):
            return StreamingQuantizedSwitchLinear(
                layer_idx=layer,
                proj_name=proj,
                stacked_weight_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight",
                stacked_scales_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.scales",
                stacked_biases_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.biases",
                num_experts=4,
                input_dims=16,
                output_dims=8,
                backing=backing,
                cache=cache,
            )

        linears = {0: [mk_lin(0, p) for p in ("gate_proj", "up_proj", "down_proj")]}
        linears[1] = [mk_lin(1, "gate_proj")]
        w = warmer.PageCacheWarmer(linears)
        hook = warmer.WarmPinHook(w, None)
        # token 1: layer 0 records its set; layer 1 records its own
        hook.on_layer_start(0, 8)
        hook.on_layer_plan(0, [1, 2, 3], 8)
        hook.on_layer_start(1, 8)
        hook.on_layer_plan(1, [2], 8)
        # token 2: layer 0 fires warm for layer 1's token-1 set
        hook.on_layer_start(0, 8)
        hook.on_layer_plan(0, [0, 1], 8)
        # warm pool is async; give it a moment and confirm no crash + reads done
        import time

        deadline = time.time() + 5
        while time.time() < deadline and w.warmed == 0:
            time.sleep(0.05)
        assert w.warmed > 0
        # big-prefetch guard: positions > _MAX_WARM_ROWS records empty
        hook.on_layer_start(0, 4096)
        hook.on_layer_plan(0, [0, 1, 2, 3], 4096)
        assert w.last_uniq[0] == []


def _ra_warmer_setup(budget_bytes: int = 0):
    from omlx.patches.expert_streaming import warmer
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
    )

    tmp = tempfile.mkdtemp()
    tmpp = Path(tmp)
    tensors = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        tensors[f"model.layers.0.mlp.switch_mlp.{proj}.weight"] = ((4, 8, 16), "BF16")
        tensors[f"model.layers.0.mlp.switch_mlp.{proj}.scales"] = ((4, 8, 1), "U32")
        tensors[f"model.layers.0.mlp.switch_mlp.{proj}.biases"] = ((4, 8, 1), "U32")
    tensors["model.layers.1.mlp.switch_mlp.gate_proj.weight"] = ((4, 8, 16), "BF16")
    tensors["model.layers.1.mlp.switch_mlp.gate_proj.scales"] = ((4, 8, 1), "U32")
    tensors["model.layers.1.mlp.switch_mlp.gate_proj.biases"] = ((4, 8, 1), "U32")
    _write_shard(tmpp / "model.safetensors", tensors)
    wm = {k: "model.safetensors" for k in tensors}
    (tmpp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))

    backing = ExpertBackingStore(tmpp)
    cache = ExpertLRUCache(budget_bytes, 1 << 12, num_layers=2)

    def mk_lin(layer, proj):
        return StreamingQuantizedSwitchLinear(
            layer_idx=layer,
            proj_name=proj,
            stacked_weight_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight",
            stacked_scales_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.scales",
            stacked_biases_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.biases",
            num_experts=4,
            input_dims=16,
            output_dims=8,
            backing=backing,
            cache=cache,
        )

    linears = {0: [mk_lin(0, p) for p in ("gate_proj", "up_proj", "down_proj")]}
    linears[1] = [mk_lin(1, "gate_proj")]
    return tmp, warmer, backing, linears


def test_readahead_warmer_advises_contiguous_runs():
    """advise-only warmer: contiguous expert ids collapse into one F_RDADVISE
    run per projection key, and no discarded reads happen (Fase G)."""
    from unittest.mock import patch

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    tmp, warmer, backing, linears = _ra_warmer_setup()
    try:
        advised = []
        w = warmer.PageCacheWarmer(linears, advise_only=True)
        hook = warmer.WarmPinHook(w, None)
        hook.on_layer_plan(1, [2], 8)
        hook.on_layer_plan(0, [1, 2, 3], 8)

        import time

        # The submit is async: the spy must be in place before the trigger.
        with patch.object(
            ExpertBackingStore,
            "advise_expert_run",
            side_effect=lambda key, first, count: advised.append((key, first, count)) or True,
        ):
            hook.on_layer_start(0, 8)  # fires advises for layer 1's [2]
            deadline = time.time() + 5
            while time.time() < deadline and not advised:
                time.sleep(0.05)

        assert advised, "advise jobs were not submitted"
        # layer 1 has one linear (gate_proj family: weight+scales+biases keys),
        # ids [2] -> single (2, 1) run per key.
        assert all(count == 1 and first == 2 for _, first, count in advised)
        assert all("model.layers.1." in key for key, _, _ in advised)
        assert w.warmed == 0  # advise-only: no reads
        assert w.advised == len(advised)
    finally:
        backing.close()


def test_backing_advise_expert_run_real_range():
    """Real F_RDADVISE against a temp shard: True on macOS, and a missing key
    degrades to False instead of raising."""
    import sys

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    tmp, warmer, backing, linears = _ra_warmer_setup()
    try:
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        got = backing.advise_expert_run(key, 1, 2)  # experts 1..2 contiguous
        if sys.platform == "darwin":
            assert got is True
        else:
            assert got is False
        assert backing.advise_expert_run("nope.missing.key", 0, 1) is False
    finally:
        backing.close()


def test_expert_lru_retain_hot():
    """retain_hot keeps only the (layer, expert) pairs given, rebuilding
    per-layer counts — the hotness seeder's cache-swap primitive."""
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    cache = ExpertLRUCache(1 << 20, 4096, num_layers=2)
    for eid in range(6):
        cache.put((0, eid, "w"), ("w", "s", None))
        cache.put((1, eid, "w"), ("w", "s", None))
    evicted = cache.retain_hot({(0, 1), (0, 2), (1, 5)})
    assert evicted == 9
    assert cache.size == 3
    assert (0, 1, "w") in cache
    assert (0, 2, "w") in cache
    assert (1, 5, "w") in cache
    assert cache._layer_counts == {0: 2, 1: 1}
    # Budget-0 caches are no-ops.
    empty = ExpertLRUCache(0, 4096, num_layers=2)
    assert empty.retain_hot({(0, 0)}) == 0


def test_prefill_hotness_recorder_seeds_lru():
    """Budget>0: the recorder swaps the cache to the prompt's hot set —
    non-hot entries evicted, missing hot bundles loaded from the backing."""
    tmp, warmer, backing, linears = _ra_warmer_setup(budget_bytes=4 << 20)
    try:
        cache = linears[0][0].cache
        wkey0 = linears[0][0].stacked_weight_key
        wkey1 = linears[1][0].stacked_weight_key
        # Pre-fill with non-hot entries (what the last prefill chunks left).
        cache.put((0, 0, wkey0), ("w", "s", None))
        cache.put((1, 0, wkey1), ("w", "s", None))

        rec = warmer.PrefillHotnessRecorder(linears, backing, cache, per_expert_bytes=0)
        rec.on_layer_plan(0, [1, 2, 0, 3], 4096)  # prefill-sized: all experts seen
        rec.on_layer_plan(1, [3, 2], 4096)
        assert rec.saw_prefill is True
        assert rec.seeded is False

        # Decode-sized call triggers the one-shot seed.
        rec.maybe_seed(0, 8)
        assert rec.seeded is True
        assert rec.seeded_experts > 0

        # Non-hot entries evicted (layer 0 expert 0 was seen but 1,2 rank higher
        # only if counts differ — with equal counts most_common is stable, so
        # assert the hot bundles are present and junk-only layers are gone).
        assert (1, 0, wkey1) not in cache
        assert (1, 3, wkey1) in cache
        assert cache.get((1, 3, wkey1)) is not None

        # One-shot: a second decode call does not re-seed.
        seeded_experts = rec.seeded_experts
        rec.maybe_seed(1, 8)
        assert rec.seeded_experts == seeded_experts
    finally:
        backing.close()


def test_prefill_hotness_recorder_budget0_page_cache_seed():
    """Budget-0: the seed issues bounded discarded reads (page-cache only),
    async on the warm pool, and never touches the LRU."""
    from unittest.mock import patch

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    tmp, warmer, backing, linears = _ra_warmer_setup(budget_bytes=0)
    try:
        cache = linears[0][0].cache
        assert cache.capacity == 0
        rec = warmer.PrefillHotnessRecorder(linears, backing, cache, per_expert_bytes=0)
        rec.on_layer_plan(0, [0, 1, 2, 3], 4096)
        rec.on_layer_plan(1, [1, 2], 4096)

        reads = []
        import time

        with patch.object(
            ExpertBackingStore,
            "load_expert_slice",
            side_effect=lambda key, eid: reads.append((key, eid)) or b"\0" * 8,
        ):
            rec.maybe_seed(0, 8)
            deadline = time.time() + 5
            while time.time() < deadline and not reads:
                time.sleep(0.05)

        assert reads, "page-cache seed burst was not submitted"
        assert rec.seeded is True
        assert cache.size == 0  # budget 0: nothing enters the LRU
    finally:
        backing.close()


def test_prefill_hotness_recorder_ignores_decode_only():
    """Without any prefill-sized call, the seed never fires (cheap no-op on
    every decode call — saw_prefill stays False)."""
    tmp, warmer, backing, linears = _ra_warmer_setup(budget_bytes=0)
    try:
        rec = warmer.PrefillHotnessRecorder(linears, backing, None)
        rec.on_layer_plan(0, [0, 1], 8)  # decode-sized rows
        rec.maybe_seed(0, 8)
        rec.maybe_seed(1, 8)
        assert rec.saw_prefill is False
        assert rec.seeded is False
        assert rec.seeded_experts == 0
    finally:
        backing.close()


def test_pin_controller_pins_within_budget():
    from omlx.patches.expert_streaming import warmer
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tensors = {}
        for layer in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                tensors[f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight"] = ((4, 8, 16), "BF16")
                tensors[f"model.layers.{layer}.mlp.switch_mlp.{proj}.scales"] = ((4, 8, 1), "U32")
                tensors[f"model.layers.{layer}.mlp.switch_mlp.{proj}.biases"] = ((4, 8, 1), "U32")
        _write_shard(tmp / "model.safetensors", tensors)
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
        )
        backing = ExpertBackingStore(tmp)

        def _stub_lin(layer, proj):
            from types import SimpleNamespace

            return SimpleNamespace(
                stacked_weight_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight",
                stacked_scales_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.scales",
                stacked_biases_key=f"model.layers.{layer}.mlp.switch_mlp.{proj}.biases",
                backing=backing,
            )

        pc = warmer.PinController(
            {0: [_stub_lin(0, p) for p in ("gate_proj", "up_proj", "down_proj")],
             1: [_stub_lin(1, p) for p in ("gate_proj", "up_proj", "down_proj")]},
            backing,
            budget_bytes=1 << 20,
            observe_calls=2,
            per_expert_bytes=(8 * 16 * 2) + (8 * 4) + (8 * 4),
        )
        for token in range(4):
            for layer in range(2):
                pc.on_layer_plan(layer, [token % 4, (token + 1) % 4], 8)
        # pin pass scheduled after 2*2 layer reports
        import time

        deadline = time.time() + 5
        while time.time() < deadline and not pc.pinned:
            time.sleep(0.05)
        assert pc.pinned
        deadline = time.time() + 5
        while time.time() < deadline and backing.pinned_count == 0:
            time.sleep(0.05)
        assert backing.pinned_count > 0
        assert 0 < backing.pinned_bytes <= (1 << 20) + (1 << 12)


class _GuardBackingStub:
    """Minimal backing exposing only streaming_guard_info for the scheduler."""

    def __init__(self, num_moe_layers, experts_per_layer, per_expert_bytes):
        self.streaming_guard_info = {
            "num_moe_layers": num_moe_layers,
            "experts_per_layer": experts_per_layer,
            "per_expert_bytes": per_expert_bytes,
        }


def _guard_scheduler(model):
    """A Scheduler via __new__ with only the transient-model attrs wired."""
    from omlx.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.model = model
    sched._prefill_transient_tracker = None
    sched.memory_monitor = None
    return sched


def test_streaming_bank_bytes_bound_and_saturation():
    back = _GuardBackingStub(48, 512, 2_500_000)
    model = MagicMock()
    model._expert_streaming_backing = back
    sched = _guard_scheduler(model)
    ratio = sched._STREAMING_BANK_TOKEN_RATIO

    # Small chunk: banks scale with tokens (uniq ~ ratio * n per layer).
    small = sched._streaming_bank_bytes(100)
    assert small == 48 * int(ratio * 100) * 2_500_000
    # Saturation at experts_per_layer.
    big = sched._streaming_bank_bytes(10_000)
    assert big == 48 * 512 * 2_500_000
    # Monotone in between.
    assert small < sched._streaming_bank_bytes(413) < big


def test_streaming_bank_bytes_absent_without_backing():
    model = MagicMock()
    del model._expert_streaming_backing  # MagicMock: raise AttributeError
    sched = _guard_scheduler(model)
    assert sched._streaming_bank_bytes(413) == 0
    assert sched._predicted_chunk_transient(413, 0) == 0.0


def test_predicted_chunk_transient_includes_bank_term():
    back = _GuardBackingStub(48, 512, 2_500_000)
    model = MagicMock()
    model._expert_streaming_backing = back
    sched = _guard_scheduler(model)

    pred = sched._predicted_chunk_transient(413, 0)
    uniq = min(512, int(sched._STREAMING_BANK_TOKEN_RATIO * 413))
    expected = 48 * uniq * 2_500_000 * sched._PREFILL_TRANSIENT_SAFETY
    assert pred == pytest.approx(expected)


def test_admission_floor_scales_down_for_streaming_chunks():
    """A big chunk's measured transient must not floor-limit smaller chunks
    when streaming banks dominate (they scale ~linearly with tokens)."""
    from types import SimpleNamespace

    from omlx.scheduler import Scheduler
    from omlx.prefill_transient_tracker import PrefillTransientTracker

    back = _GuardBackingStub(48, 512, 2_500_000)
    model = MagicMock()
    model._expert_streaming_backing = back
    sched = _guard_scheduler(model)
    tracker = PrefillTransientTracker()
    tracker.update(1897, int(17_385_500))  # measured 17.4MB/token at 1.9k chunk
    tracker._observed_max_bytes = float(1897 * 17_385_500)  # ~32GB
    tracker._last_n_tokens = 1897
    sched._prefill_transient_tracker = tracker

    n_small = 512
    bound = Scheduler._admission_transient_bound(sched, n_small, 0)
    # The floor is discounted linearly (never the raw ~32GB)...
    floor = tracker.observed_max_bytes * n_small / 1897
    assert floor < tracker.observed_max_bytes
    # ...so the bound is the bank-aware predicted term, which itself stays
    # below the raw observed max.
    uniq = min(512, int(sched._STREAMING_BANK_TOKEN_RATIO * n_small))
    expected_pred = 48 * uniq * 2_500_000 * sched._PREFILL_TRANSIENT_SAFETY
    assert bound == pytest.approx(max(expected_pred, floor))
    assert bound < tracker.observed_max_bytes


def test_admission_floor_kept_without_streaming():
    """Non-streaming keeps the conservative size-invariant observed_max floor."""
    from omlx.scheduler import Scheduler
    from omlx.prefill_transient_tracker import PrefillTransientTracker

    sched = _guard_scheduler(MagicMock())
    del sched.model._expert_streaming_backing
    tracker = PrefillTransientTracker()
    tracker.update(1897, int(17_385_500))
    tracker._observed_max_bytes = 32.0 * 1024**3
    tracker._last_n_tokens = 1897
    sched._prefill_transient_tracker = tracker

    bound = Scheduler._admission_transient_bound(sched, 512, 0)
    assert bound == pytest.approx(32.0 * 1024**3)


# ---------------------------------------------------------------------------
# Fase H: per-model IO overrides (autotune)
# ---------------------------------------------------------------------------


def test_io_overrides_resolution_and_clamping():
    """Settings resolution: unset keys stay None, depth clamps to 1..64."""
    from omlx.model_settings import ModelSettings
    from omlx.patches.expert_streaming import _io_overrides

    # All defaults → every override None
    ov = _io_overrides(ModelSettings())
    assert ov == {
        "expert_streaming_io_depth": None,
        "expert_streaming_coalesce": None,
        "expert_streaming_readahead": None,
        "expert_streaming_seed": None,
        "expert_streaming_pilot": None,
        "expert_streaming_per_layer_eval": None,
        "expert_streaming_pins": None,
        "expert_streaming_hot_fraction": None,
        "expert_streaming_pin_gib": None,
    }

    # Depth clamping and boolean pass-through
    ov = _io_overrides(
        ModelSettings(
            expert_streaming_io_depth=999,
            expert_streaming_coalesce=False,
            expert_streaming_readahead=False,
            expert_streaming_seed=False,
            expert_streaming_pilot=True,
            expert_streaming_per_layer_eval=False,
            expert_streaming_pins=True,
            expert_streaming_pin_gib=3.0,
        )
    )
    assert ov["expert_streaming_io_depth"] == 64
    assert ov["expert_streaming_coalesce"] is False
    assert ov["expert_streaming_readahead"] is False
    assert ov["expert_streaming_seed"] is False
    assert ov["expert_streaming_pilot"] is True
    assert ov["expert_streaming_per_layer_eval"] is False
    assert ov["expert_streaming_pins"] is True
    assert ov["expert_streaming_pin_gib"] == 3.0

    # Invalid depth → None (env default); None settings object → all None
    ov = _io_overrides(ModelSettings(expert_streaming_io_depth=0))
    assert ov["expert_streaming_io_depth"] is None
    assert _io_overrides(None)["expert_streaming_io_depth"] is None


def test_convert_applies_io_overrides():
    """convert_model_to_streaming wires per-model io_depth/coalesce to linears."""
    from omlx.model_settings import ModelSettings  # noqa: F811
    from omlx.patches.expert_streaming import convert_model_to_streaming
    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingSwitchLinear,
        io_pool_for,
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        num_layers, experts, hidden, moe_hidden, stages = 2, 4, 8, 4, 1
        _write_fake_dsv4_checkpoint(
            tmp,
            num_layers=num_layers,
            experts=experts,
            hidden=hidden,
            moe_hidden=moe_hidden,
            mtp_stages=stages,
        )
        layers = [_FakeLayer(experts, moe_hidden, hidden) for _ in range(num_layers)]
        mtp = [_FakeDSparkStage(experts, moe_hidden, hidden) for _ in range(stages)]
        model = _FakeTextModel(
            layers,
            mtp,
            {
                "model_type": "deepseek_v4",
                "hidden_size": hidden,
                "moe_intermediate_size": moe_hidden,
            },
        )
        convert_model_to_streaming(
            model,
            str(tmp),
            ModelSettings(expert_streaming_io_depth=8, expert_streaming_coalesce=False),
            use_file_backing=False,
        )

        pool = io_pool_for(8)
        assert pool._max_workers == 8
        # Registry dedup: same depth → same executor instance
        assert io_pool_for(8) is pool
        assert io_pool_for(None)._max_workers >= 1

        for lyr in layers:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(lyr.ffn.switch_mlp, proj)
                assert isinstance(lin, StreamingSwitchLinear)
                assert lin._io_pool_override is pool
                assert lin._coalesce_override is False
        for stage in mtp:
            lin = stage.ffn.switch_mlp.down_proj
            assert lin._io_pool_override is pool
            assert lin._coalesce_override is False


def test_convert_without_io_overrides_keeps_defaults():
    """No settings overrides → linears keep the env-default IO behavior."""
    from omlx.model_settings import ModelSettings
    from omlx.patches.expert_streaming import convert_model_to_streaming

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_dsv4_checkpoint(
            tmp, num_layers=2, experts=4, hidden=8, moe_hidden=4, mtp_stages=0
        )
        layers = [_FakeLayer(4, 4, 8) for _ in range(2)]
        model = _FakeTextModel(
            layers,
            [],
            {
                "model_type": "deepseek_v4",
                "hidden_size": 8,
                "moe_intermediate_size": 4,
            },
        )
        convert_model_to_streaming(model, str(tmp), ModelSettings(), use_file_backing=False)
        for lyr in layers:
            lin = lyr.ffn.switch_mlp.down_proj
            assert lin._io_pool_override is None
            assert lin._coalesce_override is None


def test_io_pool_for_clamps_and_caches():
    """io_pool_for clamps invalid depths and caches one executor per depth."""
    from omlx.patches.expert_streaming.streaming_switch import io_pool_for

    fallback = io_pool_for(None)
    assert io_pool_for("bogus") is fallback
    assert io_pool_for(-3) is fallback  # invalid depth → env-default pool
    pool = io_pool_for(3)
    assert pool._max_workers == 3
    assert io_pool_for(3) is pool
    assert io_pool_for("5") is io_pool_for(5)
    assert io_pool_for(999)._max_workers == 64


# --- Qwen per-layer eval boundary (G4 / qwen35_stream_eval) -----------------


def _stream_eval_recorder(monkeypatch):
    """Swap the module's mx for a recorder so the wrapper's eval/clear calls
    are observable without touching the real allocator."""
    import types

    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    calls = {"eval": 0, "clear": 0}

    def _bump(key):
        def _f(*args, **kwargs):
            calls[key] += 1

        return _f

    monkeypatch.setattr(
        qse, "mx", types.SimpleNamespace(eval=_bump("eval"), clear_cache=_bump("clear"))
    )
    # The clear routes through omlx.utils.metal_sync (Fase J Etapa C/D — a
    # pool clear must never run unsynchronized), so that helper is patched
    # too. Whichever route the wrapper takes is counted exactly once.
    monkeypatch.setattr(
        "omlx.utils.metal_sync._sync_and_clear_cache", _bump("clear")
    )
    return qse, calls


def test_qwen_stream_eval_fires_on_prefill(monkeypatch):
    import numpy as np

    qse, calls = _stream_eval_recorder(monkeypatch)
    assert qse.configure_from_settings(True) is True

    class Layer:
        _stream_eval = True

    wrapped = qse._wrap_call(lambda self, x, *a, **k: x + 1)
    out = wrapped(Layer(), np.zeros((1, 8, 4), dtype=np.float32))
    assert calls == {"eval": 1, "clear": 1}
    # The passthrough result is untouched.
    assert out.shape == (1, 8, 4)


def test_qwen_stream_eval_skips_decode_verify_and_gating(monkeypatch):
    import numpy as np

    qse, calls = _stream_eval_recorder(monkeypatch)

    class Layer:
        _stream_eval = True

    wrapped = qse._wrap_call(lambda self, x, *a, **k: x + 1)
    qse.configure_from_settings(True)

    # Decode shape [1, 1, H]: forced syncs/token would erode the QD16 win.
    wrapped(Layer(), np.zeros((1, 1, 4), dtype=np.float32))
    assert calls == {"eval": 0, "clear": 0}

    # MTP verify passes stay lazy for the same reason.
    wrapped(Layer(), np.zeros((1, 6, 4), dtype=np.float32), target_verify=True)
    assert calls == {"eval": 0, "clear": 0}

    # A layer without the converter's _stream_eval flag is untouched
    # (non-streaming loads never see the boundary).
    class Plain:
        pass

    wrapped(Plain(), np.zeros((1, 6, 4), dtype=np.float32))
    assert calls == {"eval": 0, "clear": 0}

    # Knob off disables the boundary entirely.
    assert qse.configure_from_settings(False) is False
    wrapped(Layer(), np.zeros((1, 6, 4), dtype=np.float32))
    assert calls == {"eval": 0, "clear": 0}


def test_qwen_stream_eval_configure_none_restores_env_default(monkeypatch):
    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    # The env default is captured at import; simulate both captured values and
    # confirm None defers to them while explicit values override.
    monkeypatch.setattr(qse, "_PER_LAYER_EVAL_DEFAULT", False)
    assert qse.configure_from_settings(None) is False
    monkeypatch.setattr(qse, "_PER_LAYER_EVAL_DEFAULT", True)
    assert qse.configure_from_settings(None) is True
    assert qse.configure_from_settings(False) is False
    assert qse.configure_from_settings(True) is True
    qse.configure_from_settings(None)


def test_qwen_stream_eval_patch_idempotent():
    try:
        from mlx_vlm.models.qwen3_5_moe import language as q35
    except ImportError:
        pytest.skip("mlx_vlm not installed")
    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    cls = q35.Qwen3_5MoeDecoderLayer
    orig_call = cls.__call__
    try:
        assert qse.apply_qwen35_moe_stream_eval() is True
        wrapped_once = cls.__call__
        # Second apply must not stack a second wrapper.
        assert qse.apply_qwen35_moe_stream_eval() is True
        assert cls.__call__ is wrapped_once
        assert getattr(cls, "_omlx_stream_eval_wrapped", False) is True
    finally:
        cls.__call__ = orig_call
        if hasattr(cls, "_omlx_stream_eval_wrapped"):
            del cls._omlx_stream_eval_wrapped


def test_expert_streaming_per_layer_eval_round_trip():
    s = ModelSettings(expert_streaming_per_layer_eval=False)
    d = s.to_dict()
    assert d["expert_streaming_per_layer_eval"] is False
    assert ModelSettings.from_dict(d).expert_streaming_per_layer_eval is False
    assert ModelSettings().expert_streaming_per_layer_eval is None


def test_expert_streaming_per_layer_eval_excluded_from_profiles():
    from omlx.model_profiles import EXCLUDED_FROM_PROFILES

    assert "expert_streaming_per_layer_eval" in EXCLUDED_FROM_PROFILES


@pytest.mark.asyncio
async def test_expert_streaming_per_layer_eval_persists_via_api():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen4_exp"
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_per_layer_eval=False),
    )
    assert settings.expert_streaming_per_layer_eval is False


# --- Learned pin store server integration (I2) -------------------------------


def test_pin_controller_profile_round_trip(tmp_path):
    """An explicitly wired per-model profile path persists observed frequencies
    and a fresh controller wires the learned hot set from token 1."""
    from collections import Counter
    from unittest.mock import MagicMock

    from omlx.patches.expert_streaming import warmer as warmer_mod

    profile = str(tmp_path / ".omlx" / "expert_pin_profile.json")
    linears = {0: [MagicMock()], 1: [MagicMock()]}
    ctl = warmer_mod.PinController(
        linears,
        MagicMock(),
        per_expert_bytes=1024,
        profile_path=profile,
    )
    ctl.freq = {0: Counter({3: 9, 5: 2}), 1: Counter({7: 4})}
    ctl.save_profile()

    import json
    import os

    assert os.path.exists(profile)
    data = json.loads(open(profile).read())
    assert data["freq"]["0"] == [[3, 9], [5, 2]]

    ctl2 = warmer_mod.PinController(
        linears,
        MagicMock(),
        per_expert_bytes=1024,
        profile_path=profile,
    )
    assert ctl2.freq == {0: Counter({3: 9, 5: 2}), 1: Counter({7: 4})}
    # Learned hot set available: pinned immediately, no observation window.
    assert ctl2.pinned is True
    # Env path (explicit opt-in) wins over the per-model derived path.
    assert warmer_mod.PinController(
        linears, MagicMock(), profile_path=profile
    ).profile_path == profile or warmer_mod.PIN_PROFILE_PATH == ""


def test_expert_streaming_pins_io_overrides():
    from omlx.model_settings import ModelSettings
    from omlx.patches.expert_streaming import _io_overrides

    ov = _io_overrides(
        ModelSettings(expert_streaming_pins=True, expert_streaming_pin_gib=3.0)
    )
    assert ov["expert_streaming_pins"] is True
    assert ov["expert_streaming_pin_gib"] == 3.0
    ov = _io_overrides(ModelSettings())
    assert ov["expert_streaming_pins"] is None
    assert ov["expert_streaming_pin_gib"] is None


@pytest.mark.asyncio
async def test_expert_streaming_pins_persist_via_api():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen4_exp"
    settings = ModelSettings()
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            expert_streaming_pins=True, expert_streaming_pin_gib=2.5
        ),
    )
    assert settings.expert_streaming_pins is True
    assert settings.expert_streaming_pin_gib == 2.5


@pytest.mark.asyncio
async def test_expert_streaming_pin_gib_validation():
    pool, _ = _failed_pool()
    settings = ModelSettings()
    with pytest.raises(admin_routes.HTTPException, match="between 0 and 64"):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(expert_streaming_pin_gib=99),
        )


def test_expert_streaming_pins_excluded_from_profiles():
    from omlx.model_profiles import EXCLUDED_FROM_PROFILES

    assert "expert_streaming_pins" in EXCLUDED_FROM_PROFILES
    assert "expert_streaming_pin_gib" in EXCLUDED_FROM_PROFILES


def test_save_expert_pin_profile_hook():
    """The engine stop() helper saves via the backing's attached controller
    and never raises, even when nothing is attached."""
    from omlx.patches.expert_streaming import save_expert_pin_profile

    engine = MagicMock()
    backing = MagicMock()
    engine._expert_streaming_backing = backing
    save_expert_pin_profile(engine)  # no controller attached -> no-op
    backing._pin_controller = None
    save_expert_pin_profile(engine)
    ctl = MagicMock()
    backing._pin_controller = ctl
    save_expert_pin_profile(engine)
    ctl.save_profile.assert_called_once()

    # Nothing attached anywhere: silent no-op (and the model-side holder path).
    bare = MagicMock(spec=object)
    save_expert_pin_profile(bare)

class _TieredDictBacking:
    """HOBBIT test backing: `load_expert` serves the source (4-bit) bank for
    hot experts and a requantized (3-bit) bank for the rest, keyed by the
    same (layer, proj) layout the dict fallback reads."""

    def __init__(self, e: int, hot: set, out_d: int, in_d: int, seed: int = 3):
        import mlx.core as mx
        import numpy as np
        mx.random.seed(seed)
        dense = mx.random.normal((e, out_d, in_d))
        w4, s4, b4 = mx.quantize(dense, group_size=32, bits=4)
        w3, s3, b3 = mx.quantize(dense, group_size=32, bits=3)
        self.banks = {
            "weight": {"hot": w4, "cold": w3},
            "scales": {"hot": s4, "cold": s3},
            "biases": {"hot": b4, "cold": b3},
        }
        self.hot = {int(h) for h in hot}

    def load_expert(self, key: str, expert_id: int):
        part = key.rsplit(".", 1)[1] if "." in key else key
        tier = "hot" if int(expert_id) in self.hot else "cold"
        bank = self.banks[part][tier]
        w = bank[int(expert_id)]
        return w


def test_hobbit_split_matches_tiered_reference():
    """Dual-tier gather (I6): the split output must equal composing each
    position's expert from ITS OWN tier's packing — hot positions at 4-bit
    source values, cold positions at the 3-bit tier values."""
    import mlx.core as mx
    import numpy as np

    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
    )

    E, D_in, D_out = 8, 64, 32
    hot = {1, 3}
    backing = _TieredDictBacking(E, hot, D_out, D_in)

    lin = StreamingQuantizedSwitchLinear(
        layer_idx=0,
        proj_name="down_proj",
        stacked_weight_key="k.down_proj.weight",
        stacked_scales_key="k.down_proj.scales",
        stacked_biases_key="k.down_proj.biases",
        num_experts=E,
        input_dims=D_in,
        output_dims=D_out,
        backing=backing,
        cache=ExpertLRUCache(1 << 26, 1 << 18, num_layers=1),
        group_size=32,
        bits=4,  # source (hot) packing
        mode="affine",
    )
    lin.set_hobbit_split(hot, cold_bits=3, cold_gs=32)

    rng = np.random.default_rng(11)
    T = 40
    # Linear contract (post _gather_sort): one x row per route, rhs flat ids.
    # x_exp is [N, 1, D] where N = len(indices.flat); rhs is [N].
    x_rows = rng.standard_normal((T, D_in)).astype(np.float32)
    x_rows_mx = mx.array(x_rows)
    flat_ids = rng.integers(0, E, size=T).astype(np.int32)
    x = mx.array(x_rows[:, None, :])  # [T, 1, D_in]
    idx = mx.array(flat_ids)  # [T]

    out = lin(x, idx)
    assert out.shape == (T, 1, D_out), out.shape

    # Reference: per position, dequantize that expert's tier and matmul.
    banks = backing.banks
    outs = np.zeros((T, 1, D_out), dtype=np.float32)
    for t in range(T):
        e = int(flat_ids[t])
        tier = "hot" if e in hot else "cold"
        w = banks["weight"][tier][e]
        s = banks["scales"][tier][e]
        b = banks["biases"][tier][e]
        dense = mx.dequantize(w, s, b, group_size=32, bits=4 if tier == "hot" else 3)
        r = x_rows_mx[t] @ dense.T
        mx.eval(r)
        outs[t, 0] = np.array(r).astype(np.float32)
    diff = np.abs(np.array(out.astype(mx.float32)) - outs).max()
    assert diff < 1e-3, f"split output mismatch: {diff}"

    # All-hot and all-cold degenerate arms must equal the uniform banks.
    # Each arm gets its OWN backing tier set — the runtime contract is that
    # the backing and the linear share one hot set (installed by the convert).
    all_hot_backing = _TieredDictBacking(E, set(range(E)), D_out, D_in)
    lin_all_hot = StreamingQuantizedSwitchLinear(
        layer_idx=1, proj_name="down_proj",
        stacked_weight_key="k.down_proj.weight",
        stacked_scales_key="k.down_proj.scales",
        stacked_biases_key="k.down_proj.biases",
        num_experts=E, input_dims=D_in, output_dims=D_out,
        backing=all_hot_backing, cache=ExpertLRUCache(1 << 26, 1 << 18, num_layers=1),
        group_size=32, bits=4, mode="affine",
    )
    lin_all_hot.set_hobbit_split(set(range(E)), cold_bits=3, cold_gs=32)
    out_all_hot = lin_all_hot(x, idx)
    ref4 = mx.gather_qmm(
        x, all_hot_backing.banks["weight"]["hot"], all_hot_backing.banks["scales"]["hot"], all_hot_backing.banks["biases"]["hot"],
        rhs_indices=idx.astype(mx.uint32), transpose=True, group_size=32,
        bits=4, mode="affine", sorted_indices=False,
    )
    assert mx.allclose(out_all_hot, ref4, atol=1e-3).item()

    # Uniform-cold arm (I5 contract): an empty hot set means split INACTIVE
    # and the converter overrides bits to the tier packing — here modeled by
    # a bits=3 linear reading the all-cold backing.
    all_cold_backing = _TieredDictBacking(E, set(), D_out, D_in)
    lin_all_cold = StreamingQuantizedSwitchLinear(
        layer_idx=2, proj_name="down_proj",
        stacked_weight_key="k.down_proj.weight",
        stacked_scales_key="k.down_proj.scales",
        stacked_biases_key="k.down_proj.biases",
        num_experts=E, input_dims=D_in, output_dims=D_out,
        backing=all_cold_backing, cache=ExpertLRUCache(1 << 26, 1 << 18, num_layers=1),
        group_size=32, bits=3, mode="affine",
    )
    out_all_cold = lin_all_cold(x, idx)
    ref3 = mx.gather_qmm(
        x, all_cold_backing.banks["weight"]["cold"], all_cold_backing.banks["scales"]["cold"], all_cold_backing.banks["biases"]["cold"],
        rhs_indices=idx.astype(mx.uint32), transpose=True, group_size=32,
        bits=3, mode="affine", sorted_indices=False,
    )
    assert mx.allclose(out_all_cold, ref3, atol=1e-3).item()


# --- Fase I6: hotness signal, profile cap, and fraction denominator ------------


def _i6_tmp_profile(pairs, layer="0"):
    import json
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"freq": {layer: pairs}}, f)
    return path


def test_pin_controller_counts_per_token_usage():
    """Fase I6: the per-token bincount payload makes the learned profile rank
    experts by USE — id 1 (5 of 6 routed positions) dominates; the old
    presence signal counted each uniq expert exactly once and flattened
    every frequency (the 42x64 all-count-4 profile)."""
    from collections import Counter

    import numpy as np

    from omlx.patches.expert_streaming import warmer

    flat = np.array([1, 1, 4, 1, 1, 1], dtype=np.int32)
    pc = warmer.PinController({0: []}, None, num_experts=8)
    pc.on_layer_plan(0, [1, 4], int(flat.size), np.bincount(flat, minlength=8))
    assert pc.freq[0] == Counter({1: 5, 4: 1})
    # Position-weighted prefill-sized calls accumulate too (never pin).
    pc.on_layer_plan(0, [1, 4], 4096, np.bincount(flat, minlength=8))
    assert pc.freq[0] == Counter({1: 10, 4: 2})
    assert pc.pinned is False

    # dict and list payloads coerce identically; oversized lists clamp to
    # the controller's expert width.
    pc2 = warmer.PinController({0: []}, None, num_experts=4)
    pc2.on_layer_plan(0, [0, 2], 8, {0: 3, 2: 1})
    assert pc2.freq[0] == Counter({0: 3, 2: 1})
    pc2.on_layer_plan(0, [1], 8, [5, 7, 0, 0, 9])
    # list clamped to the expert width (first 4): experts 0/1 gain 5/7.
    assert pc2.freq[0] == Counter({0: 8, 1: 7, 2: 1})
    # Unusable payloads fall back to the presence signal.
    pc3 = warmer.PinController({0: []}, None, num_experts=8)
    pc3.on_layer_plan(0, [2, 6], 8, "not-a-payload")
    assert pc3.freq[0] == Counter({2: 1, 6: 1})


def test_pin_controller_three_arg_fallback_compat():
    """Old wiring calls on_layer_plan(layer, uniq, positions) with no counts:
    the presence-based fallback must keep the pre-I6 behavior unchanged."""
    from collections import Counter

    from omlx.patches.expert_streaming import warmer

    pc = warmer.PinController({0: []}, None)
    pc.on_layer_plan(0, [3, 7], 8)
    pc.on_layer_plan(0, [3], 8)
    assert pc.freq[0] == Counter({3: 2, 7: 1})
    pc.on_layer_plan(0, [3, 7], 4096)  # prefill-sized: presence, no pin
    assert pc.freq[0] == Counter({3: 3, 7: 2})
    assert pc.pinned is False


def test_warm_pin_hook_forwards_counts_only_to_pin_and_recorder():
    """WarmPinHook repasse: counts reach the pinner and the recorder, never
    the readahead warmer (its contiguous-run F_RDADVISE grouping is
    set-based); a warmer-only hook asks for no histogram at all."""
    from unittest.mock import MagicMock

    from omlx.patches.expert_streaming import warmer

    counts = [1, 2, 0, 0]
    w = MagicMock(spec=["on_layer_plan", "on_layer_start"])
    p = MagicMock(spec=["on_layer_plan"])
    rec = MagicMock(spec=["on_layer_plan", "maybe_seed"])
    hook = warmer.WarmPinHook(w, p, rec)
    assert hook.wants_usage_counts is True
    hook.on_layer_plan(0, [0, 1], 8, counts)
    p.on_layer_plan.assert_called_once_with(0, [0, 1], 8, counts)
    rec.on_layer_plan.assert_called_once_with(0, [0, 1], 8, counts)
    # Warmer got the 3-arg contract: NO counts payload.
    w.on_layer_plan.assert_called_once_with(0, [0, 1], 8)
    hook2 = warmer.WarmPinHook(w, None, None)
    assert hook2.wants_usage_counts is False


def test_pin_profile_keep_cap_env_default_and_truncation(tmp_path, monkeypatch):
    """_PIN_PROFILE_KEEP defaults to 512 (OMLX_EXPERT_STREAMING_PIN_KEEP) and
    caps the per-layer record list in the saved JSON without changing the
    format — a 288-expert model must not be truncated to 64 entries."""
    import json

    from omlx.patches.expert_streaming import warmer as warmer_mod

    assert warmer_mod._PIN_PROFILE_KEEP == 512
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_PIN_KEEP", "300")
    import importlib
    from collections import Counter

    warmer2 = importlib.reload(warmer_mod)
    try:
        assert warmer2._PIN_PROFILE_KEEP == 300
        profile = str(tmp_path / "expert_pin_profile.json")
        linears = {0: [MagicMock()], 1: [MagicMock()]}
        ctl = warmer2.PinController(
            linears,
            MagicMock(),
            per_expert_bytes=1024,
            profile_path=profile,
        )
        # 400 distinct experts observed: only the top 300 survive the cap.
        # Counts are 400-e (expert 0 hottest), so the top 300 are 0..299.
        ctl.freq = {0: Counter({e: 400 - e for e in range(400)})}
        ctl.save_profile()
        data = json.loads(open(profile).read())
        assert len(data["freq"]["0"]) == 300
        assert {e for e, _ in data["freq"]["0"]} == set(range(300))
    finally:
        monkeypatch.delenv("OMLX_EXPERT_STREAMING_PIN_KEEP")
        importlib.reload(warmer_mod)


def test_hot_set_loader_real_denominator():
    """Fase I6 fraction semantics: ceil(fraction * num_experts) hot experts
    (clamped to available records); without num_experts the record count
    stays the denominator so legacy 2-arg calls are unchanged."""
    from omlx.patches.expert_streaming.shard_bank import load_hot_set_from_profile

    # Full-width profile on a 288-expert model: the fraction's denominator
    # is the model width, not the record count.
    pairs = [[i, 288 - i] for i in range(288)]  # descending counts
    p = _i6_tmp_profile(pairs)
    p2 = _i6_tmp_profile([[5, 9], [2, 4]])
    try:
        hot = load_hot_set_from_profile(p, 0.25, num_experts=288)
        assert len(hot["layer_0"]) == 72  # ceil(0.25 * 288)
        assert hot["layer_0"] == {e for e, _ in pairs[:72]}
        # Legacy 2-arg (no num_experts): denominator is the record count.
        hot_legacy = load_hot_set_from_profile(p, 0.25)
        assert len(hot_legacy["layer_0"]) == 72  # ceil(0.25 * 288 records)
        # Truncated profile (64 records, the pre-I6 keep cap) on a wide
        # model: the real denominator would ask 72 but only 64 exist.
        p_trunc = _i6_tmp_profile(pairs[:64])
        hot_trunc = load_hot_set_from_profile(p_trunc, 0.25, num_experts=288)
        assert len(hot_trunc["layer_0"]) == 64  # clamped to available
        # Fraction beyond the records clamps to what was observed.
        hot_full = load_hot_set_from_profile(p, 0.99, num_experts=288)
        assert len(hot_full["layer_0"]) == 286  # ceil(0.99 * 288)
        # Sparse profile on a wide model: cannot elect unseen experts.
        hot_sparse = load_hot_set_from_profile(p2, 0.5, num_experts=288)
        assert hot_sparse["layer_0"] == {5, 2}
        import os

        os.unlink(p_trunc)
    finally:
        import os

        os.unlink(p)
        os.unlink(p2)



# ---------------------------------------------------------------------------
# Fase K F1/F2/F3 — O2 advisor key fix, speculation guard, stash ring
# ---------------------------------------------------------------------------

class _AdviseRecorderBacking:
    """Backing double for the O2 advisor: records advise_expert_run calls
    and serves fake runs/slices for the stash ring."""

    def __init__(self, per_expert_bytes: int = 64, num_experts: int = 64):
        self.advised: list[tuple[str, int, int]] = []
        self.read_runs: list[tuple[str, int, int]] = []
        self.per_expert_bytes = per_expert_bytes
        self.num_experts = num_experts
        self.reader = _FakeReaderShim(self)

    def advise_expert_run(self, key: str, first_id: int, count: int) -> bool:
        self.advised.append((key, first_id, count))
        return True

    def _reader_for_key(self, key: str, expert_id: int | None = None):
        return self.reader

    def load_expert_run(self, key: str, first_id: int, count: int) -> list:
        import numpy as np

        self.read_runs.append((key, first_id, count))
        return [
            np.arange(self.per_expert_bytes, dtype=np.uint8).reshape(
                (self.per_expert_bytes,)
            )
            + (first_id + i)
            for i in range(count)
        ]

    def load_expert_slice(self, key: str, expert_id: int):
        import numpy as np

        return np.arange(self.per_expert_bytes, dtype=np.uint8)

    def tensor_dtype(self, key: str):
        return None


class _FakeReaderShim:
    def __init__(self, backing):
        self.backing = backing
        self.path = "/fake/shard.safetensors"

    def _rp_for(self, key: str):
        return None  # unused by the advisor


def _make_advise_linear(layer_idx: int, proj: str, backing, cache):
    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingQuantizedSwitchLinear,
    )

    return StreamingQuantizedSwitchLinear(
        layer_idx=layer_idx,
        proj_name=proj,
        stacked_weight_key=f"model.layers.{layer_idx}.mlp.switch_mlp.{proj}.weight",
        stacked_scales_key=f"model.layers.{layer_idx}.mlp.switch_mlp.{proj}.scales",
        stacked_biases_key=None,
        num_experts=backing.num_experts,
        input_dims=32,
        output_dims=32,
        backing=backing,
        cache=cache,
    )


class TestFaseKO2Advisor:
    def test_advise_uses_next_layer_key(self):
        """Fase K F1: the advisor must F_RDADVISE the NEXT layer's banks."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        backing = _AdviseRecorderBacking()
        lin0 = _make_advise_linear(0, "gate_proj", backing, cache)
        lin1_up = _make_advise_linear(1, "up_proj", backing, cache)
        lin1_down = _make_advise_linear(1, "down_proj", backing, cache)
        ss._STREAM_LINEARS_BY_LAYER.clear()
        ss.register_streaming_linears(1, [lin1_up, lin1_down])
        ss._PREV_UNIQ_BY_LAYER[1] = [3, 4, 9]
        ss._SPEC_STASH.clear()
        ss._SPEC_STASH_ORDER.clear()
        old_stats = dict(ss._ADVISE_STATS)
        try:
            with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", False):
                plan = ss._RemapPlan()
                lin0._advise_next_layer_prev_token(plan)
            keys = {k for k, _, _ in backing.advised}
            assert keys == {
                "model.layers.1.mlp.switch_mlp.up_proj.weight",
                "model.layers.1.mlp.switch_mlp.down_proj.weight",
            }, f"advised wrong banks: {keys}"
            # 2 targets x 3 experts (runs (3,2) + (9,1))
            assert ss._ADVISE_STATS["advised"] - old_stats["advised"] == 6
        finally:
            ss._STREAM_LINEARS_BY_LAYER.clear()
            ss._PREV_UNIQ_BY_LAYER.pop(1, None)
            ss._SPEC_STASH.clear()
            ss._SPEC_STASH_ORDER.clear()

    def test_advise_guard_skips_prefill_shaped_sets(self):
        """Fase K F2: > _MAX_ADVISE_ROWS experts is prefill-shaped, skip."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        backing = _AdviseRecorderBacking(num_experts=512)
        lin0 = _make_advise_linear(0, "gate_proj", backing, cache)
        lin1 = _make_advise_linear(1, "up_proj", backing, cache)
        ss._STREAM_LINEARS_BY_LAYER.clear()
        ss.register_streaming_linears(1, [lin1])
        ss._PREV_UNIQ_BY_LAYER[1] = list(range(200))
        try:
            with patch.object(ss, "_RA_ENV", True):
                plan = ss._RemapPlan()
                lin0._advise_next_layer_prev_token(plan)
            assert backing.advised == [], "prefill-shaped set must not be advised"
        finally:
            ss._STREAM_LINEARS_BY_LAYER.clear()
            ss._PREV_UNIQ_BY_LAYER.pop(1, None)

    def test_advise_dedupes_runs_per_layer_call(self):
        """Fase K F2: the 3 projections share one plan -> no double advise."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        backing = _AdviseRecorderBacking()
        lin0 = _make_advise_linear(0, "gate_proj", backing, cache)
        lin1 = _make_advise_linear(1, "up_proj", backing, cache)
        ss._STREAM_LINEARS_BY_LAYER.clear()
        ss.register_streaming_linears(1, [lin1])
        ss._PREV_UNIQ_BY_LAYER[1] = [3, 4]
        try:
            with patch.object(ss, "_RA_ENV", True):
                plan = ss._RemapPlan()
                lin0._advise_next_layer_prev_token(plan)
                lin0._advise_next_layer_prev_token(plan)
            assert len(backing.advised) == 1, f"expected 1 run, got {backing.advised}"
        finally:
            ss._STREAM_LINEARS_BY_LAYER.clear()
            ss._PREV_UNIQ_BY_LAYER.pop(1, None)


class TestFaseKO2StashRing:
    def test_stash_populate_serves_demand(self):
        """Fase K F3: speculated runs land in the ring under tier-aware
        bundle keys and a later demand get() hits them."""
        import time

        import numpy as np

        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        backing = _AdviseRecorderBacking()
        lin0 = _make_advise_linear(0, "gate_proj", backing, cache)
        lin1 = _make_advise_linear(1, "up_proj", backing, cache)
        ss._STREAM_LINEARS_BY_LAYER.clear()
        ss.register_streaming_linears(1, [lin1])
        ss._PREV_UNIQ_BY_LAYER[1] = [3, 4]
        ss._SPEC_STASH.clear()
        ss._SPEC_STASH_ORDER.clear()
        try:
            with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", True):
                plan = ss._RemapPlan()
                lin0._advise_next_layer_prev_token(plan)
                deadline = time.time() + 5.0
                while ss._ADVISE_STATS["stash_inserts"] < 2 and time.time() < deadline:
                    time.sleep(0.02)
            assert ss._ADVISE_STATS["stash_inserts"] == 2, "stash reads must land"
            for eid in (3, 4):
                key = lin1.bundle_key(eid)
                assert key in ss._SPEC_STASH, f"stash missing {key}"
                w, s, b = ss._SPEC_STASH[key]
                assert w.shape == (64,)
            # A demand get() against the LRU resolution path must hit the ring.
            with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", True):
                got = lin1._bundle_cached_or_staged(3)
            assert got is not None and got[0].shape == (64,)
            # FIFO ring: inserting beyond _STASH_MAX_ENTRIES evicts the oldest.
            for eid in range(10, 10 + ss._STASH_MAX_ENTRIES):
                key = (1, eid, lin1.stacked_weight_key)
                ss._SPEC_STASH[key] = (np.zeros(4, np.uint8), None, None)
                ss._SPEC_STASH_ORDER.append(key)
            while len(ss._SPEC_STASH) > ss._STASH_MAX_ENTRIES:
                old = ss._SPEC_STASH_ORDER.pop(0)
                ss._SPEC_STASH.pop(old, None)
            assert len(ss._SPEC_STASH) <= ss._STASH_MAX_ENTRIES
        finally:
            ss._STREAM_LINEARS_BY_LAYER.clear()
            ss._PREV_UNIQ_BY_LAYER.pop(1, None)
            ss._SPEC_STASH.clear()
            ss._SPEC_STASH_ORDER.clear()

    def test_stash_off_by_default_no_stash_reads(self):
        """Fase K F3: STASH=0 (default) issues advisory hints only."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(0, 4096, num_layers=2)
        backing = _AdviseRecorderBacking()
        lin0 = _make_advise_linear(0, "gate_proj", backing, cache)
        lin1 = _make_advise_linear(1, "up_proj", backing, cache)
        ss._STREAM_LINEARS_BY_LAYER.clear()
        ss.register_streaming_linears(1, [lin1])
        ss._PREV_UNIQ_BY_LAYER[1] = [3, 4]
        ss._SPEC_STASH.clear()
        ss._SPEC_STASH_ORDER.clear()
        try:
            with patch.object(ss, "_RA_ENV", True), patch.object(ss, "_STASH_ENV", False):
                plan = ss._RemapPlan()
                lin0._advise_next_layer_prev_token(plan)
            assert backing.read_runs == [], "stash disabled: no speculative reads"
            assert len(backing.advised) == 1, "F_RDADVISE still fires"
        finally:
            ss._STREAM_LINEARS_BY_LAYER.clear()
            ss._PREV_UNIQ_BY_LAYER.pop(1, None)
            ss._SPEC_STASH.clear()
            ss._SPEC_STASH_ORDER.clear()


class TestFaseKAdmissionFilter:
    def test_admission_engages_at_large_budget(self):
        """Fase K F9: the baggy capacity < 4096 cap is gone — the filter
        must engage at a 6 GiB budget (measured capacity ~6847)."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        cache = ss.ExpertLRUCache(6 * 1024**3, 920_000, num_layers=48)
        assert cache.capacity > 4096, "premise: 6 GiB budget exceeds the old cap"
        with patch.object(ss, "_ADMISSION_ENV", True):
            cache2 = ss.ExpertLRUCache(6 * 1024**3, 920_000, num_layers=48)
            assert cache2._admission_enabled is True
        with patch.object(ss, "_ADMISSION_ENV", False):
            cache3 = ss.ExpertLRUCache(6 * 1024**3, 920_000, num_layers=48)
            assert cache3._admission_enabled is False


class TestFaseKPriorPath:
    def test_prior_round_trip_via_model_path(self, tmp_path):
        """Fase K F10: prior persists next to the real model dir; load clamps
        the sample count to 1 so the first measured chunk dominates."""
        from unittest.mock import MagicMock

        from omlx import prefill_transient_tracker as pt

        model_dir = tmp_path / "models" / "Qwen3.8-Flash-Next-oQ4e-mtp"
        model_dir.mkdir(parents=True)
        with patch.object(
            pt, "_PRIOR_DIR", tmp_path / "cache"
        ), patch.object(
            pt, "logger", MagicMock()
        ):
            t = pt.PrefillTransientTracker("qwen", model_path=model_dir)
            t.update(1000, 200_000)
            for _ in range(10):
                t.update(1000, 200_000)
            assert t.samples == 11
            t.save_prior()
            prior_file = model_dir / ".omlx" / "prefill_transient_prior.json"
            assert prior_file.exists(), "prior must live next to the model"
            t2 = pt.PrefillTransientTracker("qwen", model_path=model_dir)
            assert t2.load_prior() is True
            assert t2.bytes_per_token == 200.0
            assert t2.samples == 0, "samples must be clamped to 0 on load"
            t2.update(1000, 400_000)
            assert t2.bytes_per_token == 400.0, "first measured chunk replaces prior"
            assert t2.samples == 1

    def test_prior_no_hardcoded_roots(self, tmp_path):
        """Fase K F10: without a model path the prior falls back to the
        env/cache dir — never a hardcoded device root."""
        from unittest.mock import MagicMock

        from omlx import prefill_transient_tracker as pt

        cache_dir = tmp_path / "omlx-cache"
        cache_dir.mkdir()
        with patch.object(pt, "_PRIOR_DIR", cache_dir), patch.object(
            pt, "logger", MagicMock()
        ):
            t = pt.PrefillTransientTracker("m1")
            t.update(100, 50_000)
            t.save_prior()
            assert (cache_dir / "prefill_prior_m1.json").exists()
            t2 = pt.PrefillTransientTracker("m1")
            assert t2.load_prior() is True

# ---------------------------------------------------------------------------
# Fase K F6 — read_expert_into, bank promotion (A1/A1b), rolling layer
# context (Etapa B), memtrace (ports of the faseJ pipeline tests)
# ---------------------------------------------------------------------------


def _write_shard_filled(path: Path, tensors: dict, seed: int = 1234) -> None:
    """Minimal safetensors file with deterministic non-zero payloads.

    _write_shard writes all-zero data, which would make any bit-exactness
    comparison vacuous — every byte pattern would match trivially.

    BF16 tensors get values from a small positive range rather than random
    bit patterns: random bf16 words decode to exponents around 1e30, the
    dequantized products overflow to inf/NaN, and then array_equal fails
    on NaN != NaN for outputs whose bit patterns are in fact identical.
    """
    import mlx.core as mx

    _DTY_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1}
    rng = np.random.default_rng(seed)
    header: dict = {}
    offset = 0
    blobs = []
    for key, (shape, dtype) in tensors.items():
        nbytes = int(np.prod(shape)) * _DTY_BYTES[dtype]
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        if dtype == "BF16":
            vals = rng.uniform(0.5, 2.0, size=int(np.prod(shape))).astype(np.float32)
            words = np.asarray(
                mx.array(vals).astype(mx.bfloat16).view(mx.uint16), dtype="<u2"
            )
            blobs.append(words.tobytes())
        else:
            blobs.append(
                rng.integers(1, 256, size=nbytes, dtype=np.uint8).tobytes()
            )
        offset += nbytes
    hb = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for blob in blobs:
            f.write(blob)


def _quant_linear(tmp: Path, n_experts: int = 4):
    """A StreamingQuantizedSwitchLinear backed by a real shard store."""
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
    )

    w_key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    s_key = "model.layers.0.mlp.switch_mlp.gate_proj.scales"
    b_key = "model.layers.0.mlp.switch_mlp.gate_proj.biases"
    # 4-bit packed in uint32 (8 values/word): 128 in / 16 out, group_size 64.
    # MLX requires scales and biases to share a shape, so biases are
    # per-group too, matching what a real affine-quantized checkpoint stores.
    _write_shard_filled(
        tmp / "model.safetensors",
        {
            w_key: ((n_experts, 16, 16), "U32"),
            s_key: ((n_experts, 16, 2), "BF16"),
            b_key: ((n_experts, 16, 2), "BF16"),
        },
    )
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {k: "model.safetensors" for k in (w_key, s_key, b_key)}}
        )
    )
    backing = ExpertBackingStore(tmp)
    cache = ExpertLRUCache(64 * 1024, 1024, num_layers=1)
    linear = StreamingQuantizedSwitchLinear(
        layer_idx=0,
        proj_name="gate_proj",
        stacked_weight_key=w_key,
        stacked_scales_key=s_key,
        stacked_biases_key=b_key,
        num_experts=n_experts,
        input_dims=128,
        output_dims=16,
        backing=backing,
        cache=cache,
        group_size=64,
        bits=4,
        mode="affine",
        has_bias=False,
    )
    return linear


def test_backing_read_expert_into_matches_load_expert_slice():
    """C2: the coalesced read_expert_into must be byte-identical to the
    per-expert load_expert_slice path (correctness contract for the miss
    path consolidation in the linear's bank reader).

    The payload is deterministic NON-zero (filled shard), so a reversed
    run->buffer pairing cannot slip through undetected.
    """
    import numpy as np

    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        per = 16 * 32 * 2
        _write_shard_filled(tmp / "model.safetensors", {key: ((4, 16, 32), "BF16")})
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        out1 = np.empty((1, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, [0])], [out1])
        assert out1.any(), "fixture payload is all zeros; this gate proves nothing"
        assert np.array_equal(
            out1[0], backing.load_expert_slice(key, 0).view(np.uint8).reshape(-1)
        )
        # Multi-expert, out of order: exercises the multi-run parallel branch.
        eids = [3, 0, 2]
        out3 = np.empty((3, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, eids)], [out3])
        for slot, eid in enumerate(eids):
            assert np.array_equal(
                out3[slot], backing.load_expert_slice(key, eid).view(np.uint8).reshape(-1)
            )
        # Wrong buffer shape -> False (safe fallback signal).
        bad = np.empty((2, per + 1), dtype=np.uint8)
        assert not backing.read_expert_into([(key, [0, 1])], [bad])
        empty = np.empty((0, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, [])], [empty])


def test_memtrace_disabled_by_default_is_noop():
    """With the env var unset the singleton is a null tracer: recording is
    a no-op and enabled is False, so the hot path stays free."""
    from omlx.patches.expert_streaming import streaming_switch as ss

    assert ss.memtrace.enabled is False
    assert ss.memtrace.record("linear.resolve", layer=0) is None


def test_memtracer_records_rows_and_tracks_peaks():
    """An armed tracer records one row per call with the memory counters
    and keeps a running per-metric peak."""
    import mlx.core as mx

    from omlx.patches.expert_streaming.memtrace import MemTracer

    tracer = MemTracer(path=None)
    tracer.record("linear.stack", layer=3, proj="up_proj", bank_bytes=1234)
    tracer.record("linear.stack", layer=3, proj="down_proj", bank_bytes=7)

    rows = tracer.rows()
    assert len(rows) == 2
    first = rows[0]
    for key in ("seq", "t", "event", "layer", "proj", "active", "cache", "peak",
                "footprint", "rss", "bank_bytes"):
        assert key in first, f"missing field {key}"
    assert first["event"] == "linear.stack"
    assert first["layer"] == 3
    assert first["proj"] == "up_proj"
    assert rows[1]["seq"] == 2

    peaks = tracer.peaks()
    assert peaks["bank_bytes"] == 1234
    assert peaks["active"] >= 0 and isinstance(peaks["active"], int)
    assert int(mx.get_active_memory()) >= 0


def test_memtracer_writes_jsonl():
    """A path-armed tracer appends valid JSONL rows flush-on-write."""
    import json

    from omlx.patches.expert_streaming.memtrace import MemTracer

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "trace.jsonl"
        tracer = MemTracer(path=str(path))
        tracer.record("glu.enter", layer=1, uniq=8)
        tracer.record("glu.exit", layer=1, uniq=8)
        tracer.flush()
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2
        rows = [json.loads(ln) for ln in lines]
        assert [r["event"] for r in rows] == ["glu.enter", "glu.exit"]
        assert tracer.rows() == []


def test_memtracer_scope_emits_enter_exit():
    """scope brackets a block with .enter/.exit rows, even on exception."""
    from omlx.patches.expert_streaming.memtrace import MemTracer

    tracer = MemTracer(path=None)
    with tracer.scope("ctx.ensure", layer=2):
        pass
    assert [r["event"] for r in tracer.rows()] == ["ctx.ensure.enter", "ctx.ensure.exit"]

    tracer.reset()
    try:
        with tracer.scope("ctx.ensure", layer=2):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert [r["event"] for r in tracer.rows()] == ["ctx.ensure.enter", "ctx.ensure.exit"]


def test_memtracer_summary_reports_gib_peaks():
    """summary() exposes both raw bytes and GiB-normalized peaks."""
    from omlx.patches.expert_streaming.memtrace import MemTracer

    tracer = MemTracer(path=None)
    tracer.record("linear.stack", bank_bytes=2 << 30)
    s = tracer.summary()
    assert s["enabled"] is True
    assert s["peaks_bytes"]["bank_bytes"] == 2 << 30
    assert s["peak_bank_bytes_gib"] == 2.0


def _ctx_modes_setup():
    """Build (tmp, linears, cache) for the _LayerLoadContext mode tests."""
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    tmp, _warmer, _backing, linears = _ra_warmer_setup()
    # budget 0 -> every expert misses, so all three projections must load.
    return tmp, linears, ExpertLRUCache(0, 1 << 12, num_layers=2)


def test_ctx_union_mode_traces_union_bank_bytes():
    """Legacy union mode (rolling=0) holds every projection's NumPy bank at
    once; the trace must report that union, not one projection."""
    import shutil

    from omlx.patches.expert_streaming import streaming_switch as ss
    from omlx.patches.expert_streaming.memtrace import MemTracer

    tmp, linears, cache = _ctx_modes_setup()
    tracer = MemTracer(path=None)
    original_trace, original_mode = ss.memtrace, ss._CTX_ROLLING_ENV
    ss.memtrace = tracer
    ss._CTX_ROLLING_ENV = False
    try:
        ctx = ss._LayerLoadContext(linears[0], cache)
        ctx.ensure(linears[0][0], [0, 1, 2, 3])

        events = [r["event"] for r in tracer.rows()]
        assert "ctx.ensure.enter" in events
        assert "ctx.ensure.exit" in events

        exit_row = next(r for r in tracer.rows() if r["event"] == "ctx.ensure.exit")
        assert exit_row["n_proj"] == 3
        assert exit_row["uniq"] == 4
        assert exit_row["n_loaded"] == 3
        assert exit_row["miss_per_proj"] == [4, 4, 4]
        expected = sum(lin._bank_bytes_for(4) for lin in linears[0])
        assert expected > 0
        assert exit_row["bank_bytes"] == expected
        assert exit_row["bank_bytes"] == 3 * linears[0][0]._bank_bytes_for(4)
        assert ctx.failed is False
        for lin in linears[0]:
            assert len(ctx.bundles[id(lin)]) == 4
    finally:
        ss.memtrace = original_trace
        ss._CTX_ROLLING_ENV = original_mode
        shutil.rmtree(tmp, ignore_errors=True)


def test_ctx_rolling_mode_resolves_one_projection_at_a_time():
    """Rolling mode (default): each call resolves only the asking projection
    and keeps at most _CTX_PREFETCH_AHEAD banks in flight — never the union."""
    import shutil

    from omlx.patches.expert_streaming import streaming_switch as ss
    from omlx.patches.expert_streaming.memtrace import MemTracer

    tmp, linears, cache = _ctx_modes_setup()
    tracer = MemTracer(path=None)
    original_trace, original_mode = ss.memtrace, ss._CTX_ROLLING_ENV
    ss.memtrace = tracer
    ss._CTX_ROLLING_ENV = True
    try:
        ctx = ss._LayerLoadContext(linears[0], cache)

        ctx.ensure(linears[0][0], [0, 1, 2, 3])
        assert len(ctx.bundles[id(linears[0][0])]) == 4
        assert len(ctx.bundles[id(linears[0][1])]) == 0
        assert len(ctx._futures) <= max(1, ss._CTX_PREFETCH_AHEAD)
        first_row = next(r for r in tracer.rows() if r["event"] == "ctx.ensure.exit")
        assert first_row["proj"] == linears[0][0].proj_name
        assert first_row["miss"] == 4
        assert first_row["bank_bytes"] == linears[0][0]._bank_bytes_for(4)
        assert first_row["bank_bytes"] * 3 == sum(
            lin._bank_bytes_for(4) for lin in linears[0]
        )

        ctx.ensure(linears[0][1], [0, 1, 2, 3])
        ctx.ensure(linears[0][2], [0, 1, 2, 3])
        for lin in linears[0]:
            assert len(ctx.bundles[id(lin)]) == 4
            assert ctx.hits[id(lin)] == 0
            assert ctx.misses[id(lin)] == 4
        assert ctx.failed is False
        assert ctx._futures == {}
        assert ctx._inflight_bytes == 0

        exits = [r for r in tracer.rows() if r["event"] == "ctx.ensure.exit"]
        assert [r["proj"] for r in exits] == [lin.proj_name for lin in linears[0]]
        before = len(tracer.rows())
        ctx.ensure(linears[0][0], [0, 1, 2, 3])
        assert len(tracer.rows()) == before
    finally:
        ss.memtrace = original_trace
        ss._CTX_ROLLING_ENV = original_mode
        shutil.rmtree(tmp, ignore_errors=True)


def test_ctx_rolling_mode_prefetch_ahead_zero_reads_on_demand():
    """With the prefetch window closed, each projection reads synchronously
    and no future is ever held in flight (still correct, just no overlap)."""
    import shutil

    from omlx.patches.expert_streaming import streaming_switch as ss

    tmp, linears, cache = _ctx_modes_setup()
    original_ahead, original_mode = ss._CTX_PREFETCH_AHEAD, ss._CTX_ROLLING_ENV
    ss._CTX_ROLLING_ENV = True
    ss._CTX_PREFETCH_AHEAD = 0
    try:
        ctx = ss._LayerLoadContext(linears[0], cache)
        for lin in linears[0]:
            ctx.ensure(lin, [0, 1, 2, 3])
            assert ctx._futures == {}
        assert ctx.failed is False
        for lin in linears[0]:
            assert len(ctx.bundles[id(lin)]) == 4
    finally:
        ss._CTX_PREFETCH_AHEAD = original_ahead
        ss._CTX_ROLLING_ENV = original_mode
        shutil.rmtree(tmp, ignore_errors=True)


def test_ctx_prefetch_ahead_default_keeps_io_depth_above_one():
    """The rolling path must keep more than one projection's read in flight.
    read_expert_into issues its preadv calls one at a time, so the prefetch
    window is the only thing giving the layer call an I/O queue depth above
    1. Pins the default so a revert to 1 fails here rather than as an
    unexplained throughput regression."""
    from omlx.patches.expert_streaming import streaming_switch as ss

    assert ss._CTX_PREFETCH_AHEAD >= 2


def test_ctx_rolling_mode_matches_union_mode_results():
    """Both modes must produce identical bundles — the rolling path is a
    scheduling change, not a data change. Bit-exactness of the loaded rows
    is what protects the token-ID gate downstream."""
    import shutil

    import numpy as np

    from omlx.patches.expert_streaming import streaming_switch as ss
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    tmp, _warmer, backing, linears = _ra_warmer_setup()
    original_mode = ss._CTX_ROLLING_ENV
    try:
        results = {}
        for mode in (False, True):
            cache = ExpertLRUCache(0, 1 << 12, num_layers=2)
            ss._CTX_ROLLING_ENV = mode
            ctx = ss._LayerLoadContext(linears[0], cache)
            for lin in linears[0]:
                ctx.ensure(lin, [3, 1, 0, 2])  # deliberately unsorted
            per_proj = {}
            for lin in linears[0]:
                bundles = ctx.bundles[id(lin)]
                per_proj[lin.proj_name] = {
                    eid: tuple(
                        np.asarray(part) if part is not None else None
                        for part in bundle
                    )
                    for eid, bundle in sorted(bundles.items())
                }
            results[mode] = per_proj
        assert set(results[False]) == set(results[True])
        for proj in results[False]:
            for eid in results[False][proj]:
                a, b = results[False][proj][eid], results[True][proj][eid]
                assert len(a) == len(b)
                for pa, pb in zip(a, b):
                    if pa is None or pb is None:
                        assert pa is pb
                    else:
                        assert np.array_equal(pa, pb)
    finally:
        ss._CTX_ROLLING_ENV = original_mode
        shutil.rmtree(tmp, ignore_errors=True)


def test_bank_bytes_for_scales_linearly_and_zeroes():
    """_bank_bytes_for is the tile-sizing primitive; it must be linear in
    the expert count and return 0 for a non-positive count."""
    import shutil

    from omlx.patches.expert_streaming import streaming_switch as ss

    tmp, linears, cache = _ctx_modes_setup()
    try:
        lin = linears[0][0]
        assert lin._bank_bytes_for(0) == 0
        one = lin._bank_bytes_for(1)
        assert one > 0
        assert lin._bank_bytes_for(3) == 3 * one
    finally:
        ss.memtrace = getattr(ss, "memtrace", None)
        shutil.rmtree(tmp, ignore_errors=True)

def test_bank_promote_is_bit_identical_to_per_expert_stack(tmp_path):
    """Etapa A1: promoting the whole demand bank in one mx.array must yield
    byte-for-byte the same tensors as promoting U per-expert slices and
    stacking them — so gather_qmm sees an identical bank."""
    import mlx.core as mx

    linear = _quant_linear(tmp_path, n_experts=5)

    ids = [4, 1, 3]  # deliberately out of order: row order must be preserved
    got = linear._load_expert_bank_mx(ids)
    assert got is not None
    segments, rows = got
    # Non-split linear: exactly one segment holding every asked expert.
    assert len(segments) == 1
    seg_ids, (w_fast, s_fast, b_fast) = segments[0]
    assert seg_ids == ids

    dt = linear._slice_dtypes_lazy()
    mini_w, mini_s, mini_b = [], [], []
    for w, s, b in rows:
        mini_w.append(linear._promote_np(w))
        mini_s.append(linear._promote_np(s, dt[0]))
        mini_b.append(linear._promote_np(b, dt[1]))

    assert w_fast.dtype == mx.stack(mini_w, axis=0).dtype
    assert mx.array_equal(w_fast, mx.stack(mini_w, axis=0))
    assert s_fast.dtype == mx.stack(mini_s, axis=0).dtype
    assert mx.array_equal(s_fast, mx.stack(mini_s, axis=0))
    assert mx.array_equal(b_fast, mx.stack(mini_b, axis=0))

    # Row k of the bank is expert ids[k], not file order.
    assert mx.array_equal(w_fast[0], mini_w[0])
    assert mx.array_equal(w_fast[2], mini_w[2])
    assert not mx.array_equal(w_fast[0], mini_w[1])


def test_quantized_call_identical_with_and_without_bank_promote(monkeypatch, tmp_path):
    """Etapa A1 end-to-end: the layer output must be bit-identical whether
    the single-promotion fast path is enabled or not. This is the gate that
    makes A1 lossless rather than near-lossless."""
    import mlx.core as mx

    from omlx.patches.expert_streaming import streaming_switch as ss

    x = mx.random.normal((6, 128)).astype(mx.float32)
    indices = mx.array([2, 0, 1, 2, 0, 1], dtype=mx.int32)

    def run(enabled):
        # A fresh linear per arm: reusing one would leave the LRU populated,
        # turning the second arm into a hit-only run and making the comparison
        # vacuous (neither arm would exercise a miss).
        linear = _quant_linear(tmp_path, n_experts=3)
        engaged: list[bool] = []
        real = linear._load_expert_bank_mx

        def spy(ids):
            out = real(ids)
            engaged.append(out is not None)
            return out

        monkeypatch.setattr(ss, "_BANK_PROMOTE_ENV", enabled)
        monkeypatch.setattr(linear, "_load_expert_bank_mx", spy)
        out = linear(x, indices)
        mx.eval(out)
        return out, engaged

    out_on, engaged_on = run(True)
    out_off, engaged_off = run(False)

    assert engaged_on == [True], engaged_on
    assert engaged_off == [], engaged_off

    assert out_on.dtype == out_off.dtype
    bits_on = np.ascontiguousarray(out_on).view(np.uint32).reshape(-1)
    bits_off = np.ascontiguousarray(out_off).view(np.uint32).reshape(-1)
    assert np.array_equal(bits_on, bits_off)
    assert bool(mx.all(mx.isfinite(out_on)))


def test_bank_promote_returns_none_without_bank_backing(tmp_path):
    """A dict backing cannot serve contiguous banks, so A1 must decline and
    let the legacy per-expert path run."""
    import mlx.core as mx

    linear = _quant_linear(tmp_path, n_experts=3)
    linear.backing = {(0, "gate_proj", "weight"): mx.zeros((3, 16, 16))}
    assert linear._load_expert_bank_mx([0, 1]) is None


def test_read_expert_banks_matches_legacy_bank_np(tmp_path):
    """The _read_expert_banks refactor must not change what
    _load_expert_bank_np returns to its callers."""
    linear = _quant_linear(tmp_path, n_experts=5)
    ids = [4, 1, 3]

    rows = linear._load_expert_bank_np(ids)
    got = linear._read_expert_banks(ids)
    assert rows is not None and got is not None
    segments, rows2 = got

    assert len(segments) == 1
    seg_ids, banks = segments[0]
    assert seg_ids == ids
    assert banks[0].shape == (len(ids), linear._slice_bytes(linear.stacked_weight_key))
    assert len(rows) == len(rows2) == len(ids)
    for (w1, s1, b1), (w2, s2, b2) in zip(rows, rows2):
        assert np.array_equal(w1, w2)
        assert np.array_equal(s1, s2)
        assert np.array_equal(b1, b2)


# ---------------------------------------------------------------------------
# Fase J — Etapa A1b: single-promotion bank build on the layer-context path
# ---------------------------------------------------------------------------


def _quant_glu(tmp: Path, n_experts: int = 3):
    """A quantized StreamingSwitchGLU whose projections sit on real shards.

    Built with quantized=True so StreamingSwitchGLU installs a
    _LayerLoadContext — the Etapa B path that runs by default and the one
    A1b targets. A1 itself never fires here: the context pre-resolves every
    expert, so missing is empty by the time the linear asks. That gap is
    exactly why A1b exists.
    """
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
    )

    hidden, moe_hidden = 128, 64
    specs = (
        ("gate_proj", moe_hidden, hidden),
        ("up_proj", moe_hidden, hidden),
        ("down_proj", hidden, moe_hidden),
    )
    tensors = {}
    for proj, o, i in specs:
        base = f"model.layers.0.mlp.switch_mlp.{proj}"
        tensors[f"{base}.weight"] = ((n_experts, o, i // 8), "U32")
        tensors[f"{base}.scales"] = ((n_experts, o, i // 64), "BF16")
        tensors[f"{base}.biases"] = ((n_experts, o, i // 64), "BF16")
    _write_shard_filled(tmp / "model.safetensors", tensors)
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )

    backing = ExpertBackingStore(tmp)
    cache = ExpertLRUCache(64 * 1024, 1024, num_layers=1)
    glu = StreamingSwitchGLU(
        input_dims=hidden,
        hidden_dims=moe_hidden,
        num_experts=n_experts,
        layer_idx=0,
        backing=backing,
        cache=cache,
        fused_gate_up=False,
        inverse_scatter=False,
        quantized=True,
        group_size=64,
        bits=4,
        mode="affine",
    )
    for proj, o, i in specs:
        base = f"model.layers.0.mlp.switch_mlp.{proj}"
        setattr(
            glu,
            proj,
            StreamingQuantizedSwitchLinear(
                layer_idx=0,
                proj_name=proj,
                stacked_weight_key=f"{base}.weight",
                stacked_scales_key=f"{base}.scales",
                stacked_biases_key=f"{base}.biases",
                num_experts=n_experts,
                input_dims=i,
                output_dims=o,
                backing=backing,
                cache=cache,
                group_size=64,
                bits=4,
                mode="affine",
                has_bias=False,
            ),
        )
    return glu


def test_ctx_bank_promote_is_bit_identical_on_default_path(monkeypatch, tmp_path):
    """Etapa A1b: with the Etapa B barrier on (the default), the GLU output
    must be bit-identical whether single-promotion is enabled or not.

    This is the gate that makes A1b lossless rather than near-lossless. It
    also proves A1b engages where A1 does not: A1 measured zero invocations
    on this path, and would leave the default configuration with no
    bank-promotion win.
    """
    import mlx.core as mx

    from omlx.patches.expert_streaming import streaming_switch as ss

    assert ss._LAYER_BARRIER_ENV  # the default path A1b is scoped to

    x = mx.random.normal((1, 6, 128)).astype(mx.float32)
    indices = mx.array([2, 0, 1, 2, 0, 1], dtype=mx.int32)

    def run(enabled):
        monkeypatch.setattr(ss, "_BANK_PROMOTE_CTX_ENV", enabled)
        glu = _quant_glu(tmp_path, n_experts=3)
        engaged: list[bool] = []
        real = ss.StreamingQuantizedSwitchLinear._promote_banks

        def spy(self, segments):
            out = real(self, segments)
            engaged.append(out is not None)
            return out

        monkeypatch.setattr(ss.StreamingQuantizedSwitchLinear, "_promote_banks", spy)
        out = glu(x, indices)
        mx.eval(out)
        return out, engaged

    out_on, engaged_on = run(True)
    out_off, engaged_off = run(False)

    # One call per projection (gate/up/down).
    assert engaged_on == [True, True, True], engaged_on
    assert engaged_off == [], engaged_off

    assert out_on.dtype == out_off.dtype
    bits_on = np.ascontiguousarray(out_on).view(np.uint32).reshape(-1)
    bits_off = np.ascontiguousarray(out_off).view(np.uint32).reshape(-1)
    assert np.array_equal(bits_on, bits_off)
    assert bool(mx.all(mx.isfinite(out_on)))


def test_ctx_bank_promote_declined_on_partial_demand(monkeypatch, tmp_path):
    """A bank covering only part of the demand set must not be promoted.

    The missing experts do live in one contiguous bank, but the cache hits
    would have to be promoted separately and concatenated — a different
    layout contract that A1b deliberately leaves on the legacy path.
    """
    import mlx.core as mx

    from omlx.patches.expert_streaming import streaming_switch as ss

    monkeypatch.setattr(ss, "_BANK_PROMOTE_CTX_ENV", True)
    glu = _quant_glu(tmp_path, n_experts=3)

    # Seed one expert so gate_proj's demand set is a partial miss.
    proj = glu.gate_proj
    rows = proj._load_expert_bank_np([0])
    assert rows is not None
    glu._cache.put((0, 0, proj.stacked_weight_key), rows[0])

    recorded: dict = {}
    real_ensure = ss._LayerLoadContext.ensure

    def spy(ctx, linear, expert_ids):
        real_ensure(ctx, linear, expert_ids)
        recorded[linear.proj_name] = (
            ctx.bank_raw.get(id(linear)),
            ctx.bank_ids.get(id(linear)),
            ctx.misses.get(id(linear)),
        )

    monkeypatch.setattr(ss._LayerLoadContext, "ensure", spy)

    x = mx.random.normal((1, 6, 128)).astype(mx.float32)
    indices = mx.array([2, 0, 1, 2, 0, 1], dtype=mx.int32)
    out = glu(x, indices)
    mx.eval(out)

    assert recorded["gate_proj"][2] == 2, recorded["gate_proj"]
    assert recorded["gate_proj"][0] is None
    assert recorded["gate_proj"][1] is None
    assert recorded["up_proj"][0] is not None
    assert recorded["up_proj"][1] == [0, 1, 2]
    assert bool(mx.all(mx.isfinite(out)))


class TestFaseKRunMerge:
    def test_group_runs_bridges_small_gap_within_tier(self):
        """Fase K F7: a run may bridge a <=2 gap of non-demanded ids (the
        gap bytes are read but never promoted); tier boundaries never merge.
        Bridging is active only while the HOBBIT split is on — single-tier
        demand is already contiguous, and bridging costs ~8% of 2k TTFT there.

        """
        from omlx.patches.expert_streaming import streaming_switch as ss

        lin = _make_advise_linear(0, "gate_proj", _AdviseRecorderBacking(), ss.ExpertLRUCache(0, 4096))
        # No split: bridging must NOT engage regardless of the env value.
        with patch.object(ss, "_RUN_MERGE_GAP", 2):
            runs = lin._group_runs([3, 4, 7, 8])
            assert runs == [(3, 2), (7, 2)], runs
            runs = lin._group_runs([3, 4, 8, 9])
            assert runs == [(3, 2), (8, 2)], runs
        # Split active, every demanded id hot: bridge engages (covers 3..8).
        lin.set_hobbit_split({3, 4, 5, 6, 7, 8, 9}, cold_bits=2, cold_gs=32)
        with patch.object(ss, "_RUN_MERGE_GAP", 2):
            runs = lin._group_runs([3, 4, 7, 8])
            assert runs == [(3, 6)], runs
            # Gap of 3 (ids 3,4 then 8,9) is beyond the bridge budget.
            runs = lin._group_runs([3, 4, 8, 9])
            assert runs == [(3, 2), (8, 2)], runs
            # Bridge disabled by env.
            with patch.object(ss, "_RUN_MERGE_GAP", 0):
                runs = lin._group_runs([3, 4, 7, 8])
                assert runs == [(3, 2), (7, 2)], runs
            # max_run still bounds the bridged run (bridge clamps to the cap).
            with patch.object(ss, "_RUN_MAX", 4):
                runs = lin._group_runs([3, 4, 7, 8])
                assert runs == [(3, 4), (7, 2)], runs

    def test_group_runs_bridge_never_crosses_tier(self):
        """Under the HOBBIT split the bridge stays inside the first tier."""
        from omlx.patches.expert_streaming import streaming_switch as ss

        backing = _AdviseRecorderBacking()
        lin = _make_advise_linear(0, "gate_proj", backing, ss.ExpertLRUCache(0, 4096))
        lin.set_hobbit_split({1, 2, 9}, cold_bits=2, cold_gs=32)  # hot: 1,2,9
        ids = [1, 2, 4, 5, 9]  # hot(1,2) gap cold(4,5): must NOT bridge to cold
        with patch.object(ss, "_RUN_MERGE_GAP", 2):
            runs = lin._group_runs(ids)
        assert runs[0] == (1, 2), f"hot run must stay intact: {runs}"
        assert (4, 2) in runs or (4, 1) in runs
