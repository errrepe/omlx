# SPDX-License-Identifier: Apache-2.0
"""Tests for MoE expert streaming (SSD) — residency, settings, and forced logic."""

import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
