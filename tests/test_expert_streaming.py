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
        from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

        primary_only = ExpertBackingStore(primary)
        with pytest.raises((KeyError, FileNotFoundError)):
            primary_only.load_expert_slice("model.layers.0.mlp.switch_mlp.gate_proj.weight", 0)


# --- Fase J, C1: preadv zero-copy + per-key read params -----------------------
def test_shard_reader_preadv_slice_shape_and_run_matches_slices():
    """C1: preadv zero-copy slice/run yield correct shapes and the run equals
    the independently-sliced per-expert views (no content regression)."""
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        _write_shard(tmp / "model.safetensors", {key: ((4, 16, 32), "BF16")})
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        assert backing.load_expert_slice(key, 0).shape == (16, 32)
        assert backing.load_expert_slice(key, 3).shape == (16, 32)
        run = backing.load_expert_run(key, 0, 4)
        assert len(run) == 4
        for i, s in enumerate(run):
            assert s.shape == (16, 32)
            assert np.array_equal(s, backing.load_expert_slice(key, i))


def test_shard_reader_rp_for_cached_and_zero_copy_buffer():
    """C1: _rp_for caches per-key params and the slice is a view into the
    writable preadv buffer (no bytearray double-copy)."""
    from omlx.patches.expert_streaming.shard_bank import _ShardReader

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key = "m.weight"
        _write_shard(tmp / "model.safetensors", {key: ((4, 16, 32), "BF16")})
        reader = _ShardReader(tmp / "model.safetensors")
        rp1 = reader._rp_for(key)
        rp2 = reader._rp_for(key)
        assert rp1 is rp2  # cached, no per-call recompute
        assert rp1.expert_bytes == (4 * 16 * 32 * 2) // 4
        assert rp1.per_shape == (16, 32)
        slc = reader.expert_slice(key, 2)
        assert slc.shape == (16, 32)
        assert slc.base is not None  # view into the preadv-filled buffer


def test_shard_reader_size_mismatch_raises_value_error():
    """C1: a header whose data_offsets disagree with n_elements*item raises
    ValueError (computed once, then cached)."""
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key = "m.switch_mlp.gate_proj.weight"
        shape = (4, 16, 32)
        item = 2  # BF16
        nbytes = int(np.prod(shape)) * item  # 4096
        # header claims MORE bytes than the tensor actually holds
        header = {key: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, nbytes + 904]}}
        hb = json.dumps(header).encode()
        with (tmp / "model.safetensors").open("wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            f.write(b"\x00" * (nbytes + 904))
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        with pytest.raises(ValueError):
            backing.load_expert_slice(key, 0)


def test_shard_reader_short_read_raises_os_error():
    """C1: reading an expert whose byte range runs past EOF surfaces OSError
    (the zero-copy preadv short read is not silently swallowed)."""
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key = "m.switch_mlp.gate_proj.weight"
        shape = (2, 16, 32)
        item = 2
        nbytes = int(np.prod(shape)) * item  # 2048
        # header claims the full 2048, but the file data is truncated by 100 bytes
        header = {key: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, nbytes]}}
        hb = json.dumps(header).encode()
        with (tmp / "model.safetensors").open("wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            f.write(b"\x00" * (nbytes - 100))
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        # expert 1's range extends 100 bytes past EOF -> short read -> OSError
        with pytest.raises(OSError):
            backing.load_expert_slice(key, 1)


def test_backing_read_expert_into_matches_load_expert_slice():
    """C2: the coalesced read_expert_into must be byte-identical to the
    per-expert load_expert_slice path (correctness contract for the miss
    path consolidation in _load_expert_bundle)."""
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        # 4 experts, BF16 -> 16*32*2 bytes per expert
        per = 16 * 32 * 2
        _write_shard(tmp / "model.safetensors", {key: ((4, 16, 32), "BF16")})
        (tmp / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors"}})
        )
        backing = ExpertBackingStore(tmp)
        # single-expert coalesced read
        out1 = np.empty((1, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, [0])], [out1])
        assert np.array_equal(out1[0], backing.load_expert_slice(key, 0).view(np.uint8).reshape(-1))
        # multi-expert (out of order) coalesced read, in id order in the buffer
        eids = [3, 0, 2]
        out3 = np.empty((3, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, eids)], [out3])
        for slot, eid in enumerate(eids):
            assert np.array_equal(
                out3[slot], backing.load_expert_slice(key, eid).view(np.uint8).reshape(-1)
            )
        # wrong buffer shape -> False (safe fallback signal)
        bad = np.empty((2, per + 1), dtype=np.uint8)
        assert not backing.read_expert_into([(key, [0, 1])], [bad])
        empty = np.empty((0, per), dtype=np.uint8)
        assert backing.read_expert_into([(key, [])], [empty])
        assert not backing.read_expert_into([(key, [])], [np.empty((0, per + 1), dtype=np.uint8)])


def test_expert_lru_per_layer_cap_evicts_oldest_same_layer():
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    cache = ExpertLRUCache(4 * 1024, 1024, num_layers=2)
    cache.put((0, 0, "w"), "e0")
    cache.put((0, 1, "w"), "e1")
    cache.put((0, 2, "w"), "e2")
    assert (0, 0, "w") not in cache
    assert (0, 1, "w") in cache
    assert (0, 2, "w") in cache
    assert cache._layer_counts == {0: 2}


def test_expert_lru_discard_keeps_layer_counts_consistent():
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    cache = ExpertLRUCache(8 * 1024, 1024, num_layers=2)
    keys = [(0, 0, "w"), (0, 1, "w"), (1, 0, "w")]
    for key in keys:
        cache.put(key, key)
    assert cache.discard(keys[1])
    assert not cache.discard(keys[1])
    assert cache._layer_counts == {0: 1, 1: 1}
    assert cache.size == 2


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
    glu.down_proj = StreamingQuantizedSwitchLinear(layer_idx=0, proj_name="down_proj", stacked_weight_key="k.down_proj.weight", stacked_scales_key="k.down_proj.scales", stacked_biases_key="k.down_proj.biases", num_experts=E, input_dims=ref_down.input_dims, output_dims=ref_down.output_dims, backing=backing, cache=cache, group_size=32, bits=4, mode="affine")

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
    from types import SimpleNamespace

    import mlx.core as mx

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
        assert rec.seed_done.wait(5)

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
            "load_expert_run",
            side_effect=lambda key, first, count: reads.append((key, first, count))
            or [b"\0" * 8] * count,
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

    from omlx.prefill_transient_tracker import PrefillTransientTracker
    from omlx.scheduler import Scheduler

    back = _GuardBackingStub(48, 512, 2_500_000)
    model = MagicMock()
    model._expert_streaming_backing = back
    sched = _guard_scheduler(model)
    tracker = PrefillTransientTracker()
    tracker.update(1897, 17_385_500)  # measured 17.4MB/token at 1.9k chunk
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
    from omlx.prefill_transient_tracker import PrefillTransientTracker
    from omlx.scheduler import Scheduler

    sched = _guard_scheduler(MagicMock())
    del sched.model._expert_streaming_backing
    tracker = PrefillTransientTracker()
    tracker.update(1897, 17_385_500)
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
    are observable without touching the real allocator.

    The clear goes through ``omlx.utils.metal_sync._sync_and_clear_cache``
    (Etapa C: a pool clear must never run unsynchronized — on M4 the driver
    panics with 'completeMemory() prepare count underflow'), so that helper is
    patched too. Whichever route the wrapper takes is counted exactly once.
    """
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


def test_streaming_glm_weighted_sum_kernel_module_imports():
    """C12: the streaming path resolves the shared GLM kernel module rather
    than swallowing a missing relative expert_streaming.kernels import."""
    from omlx.patches.glm_moe_dsa.kernels import fast

    assert fast is not None


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


# ---------------------------------------------------------------------------
# Fase J — per-layer / per-projection memory trace (memtrace)
# ---------------------------------------------------------------------------


def test_memtrace_disabled_by_default_is_noop():
    """With the env var unset the singleton is a null tracer: recording is a
    no-op and ``enabled`` is False, so the hot path stays free."""
    from omlx.patches.expert_streaming import streaming_switch as ss

    assert ss.memtrace.enabled is False
    assert ss.memtrace.record("linear.resolve", layer=0) is None
    with ss.memtrace.scope("glu", layer=0):
        pass
    assert ss.memtrace.summary() == {"enabled": False}
    assert ss.memtrace.rows() if hasattr(ss.memtrace, "rows") else True


def test_memtracer_records_rows_and_tracks_peaks():
    """An armed tracer records one row per call with the memory counters and
    keeps a running per-metric peak."""
    import mlx.core as mx

    from omlx.patches.expert_streaming.memtrace import MemTracer

    tracer = MemTracer(path=None)
    tracer.record("linear.stack", layer=3, proj="up_proj", bank_bytes=1234)
    # Force a *lower* second sample so the peak must come from the first.
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
    # Peaks are monotone maxima: the larger bank_bytes must win.
    assert peaks["bank_bytes"] == 1234
    # MLX counters must be whatever mlx reports (non-negative ints).
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
        # Writing to a path must not retain rows in memory (unbounded growth).
        assert tracer.rows() == []


def test_memtracer_scope_emits_enter_exit():
    """``scope`` brackets a block with .enter/.exit rows, even on exception."""
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
    # The exit row must still be emitted so the trace never looks truncated.
    assert [r["event"] for r in tracer.rows()] == ["ctx.ensure.enter", "ctx.ensure.exit"]


def test_memtracer_sample_every_decimates():
    """sample_every=N keeps 1 row in N without breaking peak tracking of the
    rows it did see."""
    from omlx.patches.expert_streaming.memtrace import MemTracer

    tracer = MemTracer(path=None, sample_every=4)
    for i in range(12):
        tracer.record("linear.resolve", uniq=i, bank_bytes=i * 10)
    rows = tracer.rows()
    # seq 4, 8, 12 survive.
    assert [r["seq"] for r in rows] == [4, 8, 12]
    assert tracer.summary()["samples"] == 12


def test_memtracer_summary_reports_gib_peaks():
    """summary() exposes both raw bytes and GiB-normalized peaks, which is the
    form the bench reports as the acceptance criterion."""
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
    """Legacy union mode (_CTX_ROLLING_ENV=0) holds every projection's NumPy
    bank at once; the trace must report that union, not one projection.

    This is the quantity the rolling path removes, so getting the accounting
    right is a precondition for claiming a win.
    """
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
        # One call resolves every projection (union semantics).
        ctx.ensure(linears[0][0], [0, 1, 2, 3])

        events = [r["event"] for r in tracer.rows()]
        assert "ctx.ensure.enter" in events
        assert "ctx.ensure.exit" in events

        exit_row = next(r for r in tracer.rows() if r["event"] == "ctx.ensure.exit")
        assert exit_row["n_proj"] == 3
        assert exit_row["uniq"] == 4
        assert exit_row["n_loaded"] == 3
        assert exit_row["miss_per_proj"] == [4, 4, 4]
        # Union across projections, not per-projection.
        expected = sum(lin._bank_bytes_for(4) for lin in linears[0])
        assert expected > 0
        assert exit_row["bank_bytes"] == expected
        # Sanity: the union is 3x one projection's bank (all same shape here).
        assert exit_row["bank_bytes"] == 3 * linears[0][0]._bank_bytes_for(4)
        assert ctx.failed is False
        # Every projection is fully resolved by the single call.
        for lin in linears[0]:
            assert len(ctx.bundles[id(lin)]) == 4
    finally:
        ss.memtrace = original_trace
        ss._CTX_ROLLING_ENV = original_mode
        shutil.rmtree(tmp, ignore_errors=True)


def test_ctx_rolling_mode_resolves_one_projection_at_a_time():
    """Rolling mode (default): each call resolves only the asking projection and
    keeps at most _CTX_PREFETCH_AHEAD banks in flight — never the union."""
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

        # First call resolves projection 0 only, and prefetches at most `ahead`.
        ctx.ensure(linears[0][0], [0, 1, 2, 3])
        assert len(ctx.bundles[id(linears[0][0])]) == 4
        assert len(ctx.bundles[id(linears[0][1])]) == 0
        # Nothing beyond the prefetch window is resident.
        assert len(ctx._futures) <= max(1, ss._CTX_PREFETCH_AHEAD)
        first_row = next(r for r in tracer.rows() if r["event"] == "ctx.ensure.exit")
        assert first_row["proj"] == linears[0][0].proj_name
        assert first_row["miss"] == 4
        # The trace reports this projection's bank, NOT the 3x union.
        assert first_row["bank_bytes"] == linears[0][0]._bank_bytes_for(4)
        assert first_row["bank_bytes"] * 3 == sum(
            lin._bank_bytes_for(4) for lin in linears[0]
        )

        # Remaining projections resolve on demand in call order.
        ctx.ensure(linears[0][1], [0, 1, 2, 3])
        ctx.ensure(linears[0][2], [0, 1, 2, 3])
        for lin in linears[0]:
            assert len(ctx.bundles[id(lin)]) == 4
            assert ctx.hits[id(lin)] == 0
            assert ctx.misses[id(lin)] == 4
        assert ctx.failed is False
        # All prefetches drained; nothing left in flight.
        assert ctx._futures == {}
        assert ctx._inflight_bytes == 0

        exits = [r for r in tracer.rows() if r["event"] == "ctx.ensure.exit"]
        assert [r["proj"] for r in exits] == [lin.proj_name for lin in linears[0]]
        # Idempotent: re-asking does not re-read or re-trace.
        before = len(tracer.rows())
        ctx.ensure(linears[0][0], [0, 1, 2, 3])
        assert len(tracer.rows()) == before
    finally:
        ss.memtrace = original_trace
        ss._CTX_ROLLING_ENV = original_mode
        shutil.rmtree(tmp, ignore_errors=True)


def test_ctx_rolling_mode_prefetch_ahead_zero_reads_on_demand():
    """With the prefetch window closed, each projection reads synchronously and
    no future is ever held in flight (still correct, just no overlap)."""
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

    `read_expert_into` issues its preadv calls one at a time, so the prefetch
    window is the *only* thing giving the layer call an I/O queue depth above
    1. At depth 1 the device idles between reads and decode throughput follows
    it down: measured on qwen4_exp (2k prompt, 48 decode), AHEAD=1 gave
    41% CPU / 0.46 GiB/s / 1.86 tok/s against the baseline's 85% / 0.92 GiB/s
    / 3.06 tok/s. AHEAD=3 recovers to 50% / 0.56 GiB/s / 2.22 tok/s with no
    memory cost. This pins the default so a revert to 1 fails here rather
    than as an unexplained throughput regression.
    """
    from omlx.patches.expert_streaming import streaming_switch as ss

    assert ss._CTX_PREFETCH_AHEAD >= 2


def test_ctx_rolling_mode_matches_union_mode_results():
    """Both modes must produce identical bundles — the rolling path is a
    scheduling change, not a data change. Bit-exactness of the loaded rows is
    what protects the token-ID gate downstream."""
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
        shutil.rmtree(tmp, ignore_errors=True)


def test_bank_bytes_for_scales_linearly_and_zeroes():
    """_bank_bytes_for is the tile-sizing primitive for Etapa A; it must be
    linear in the expert count and return 0 for a non-positive count."""
    import shutil

    from omlx.patches.expert_streaming.streaming_switch import (
        StreamingQuantizedSwitchLinear,
    )

    tmp, _warmer, backing, linears = _ra_warmer_setup()
    try:
        lin = linears[0][0]
        assert lin._bank_bytes_for(0) == 0
        assert lin._bank_bytes_for(-5) == 0
        one = lin._per_expert_bytes()
        assert one > 0
        assert lin._bank_bytes_for(1) == one
        assert lin._bank_bytes_for(7) == 7 * one
        # Per-expert bytes include weight + scales + biases for this projection.
        assert one == (
            lin._slice_bytes(lin.stacked_weight_key)
            + lin._slice_bytes(lin.stacked_scales_key)
            + lin._slice_bytes(lin.stacked_biases_key)
        )
        assert isinstance(lin, StreamingQuantizedSwitchLinear)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_glu_forward_emits_memtrace_events():
    """End-to-end: a quantized GLU forward emits glu/layer events and the
    per-projection linear events, with the real (null) tracer swapped out."""
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    from omlx.patches.expert_streaming import streaming_switch as ss
    from omlx.patches.expert_streaming.memtrace import MemTracer
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
    )

    # mx.quantize only supports group sizes 32/64/128 and requires the last
    # weight dim to be divisible by it -> D must be >= 32.
    E, H, D, GS = 8, 64, 32, 32
    cache = ExpertLRUCache(0, 1 << 18, num_layers=1)
    backing: dict = {}
    glu = StreamingSwitchGLU(
        input_dims=D, hidden_dims=H, num_experts=E, layer_idx=0,
        backing=backing, cache=cache, quantized=True, activation=None,
    )
    for name, in_dims, out_dims in (
        ("gate_proj", D, H), ("up_proj", D, H), ("down_proj", H, D),
    ):
        ref_lin = QuantizedSwitchLinear(in_dims, out_dims, E, bias=True, group_size=GS, bits=4)
        mx.eval(ref_lin.parameters())
        backing[(0, name, "weight")] = ref_lin.weight
        backing[(0, name, "scales")] = ref_lin.scales
        backing[(0, name, "biases")] = ref_lin.biases
        setattr(glu, name, StreamingQuantizedSwitchLinear(
            layer_idx=0, proj_name=name,
            stacked_weight_key=f"k.{name}.weight",
            stacked_scales_key=f"k.{name}.scales",
            stacked_biases_key=f"k.{name}.biases",
            num_experts=E, input_dims=in_dims, output_dims=out_dims,
            backing=backing, cache=cache, group_size=GS, bits=4, mode="affine",
        ))

    tracer = MemTracer(path=None)
    original = ss.memtrace
    ss.memtrace = tracer
    try:
        rng = np.random.default_rng(11)
        x = mx.array(rng.standard_normal((1, 80, D)).astype(np.float32))
        idx = mx.array(rng.integers(0, E, size=(1, 80, 1)).astype(np.int32))
        out = glu(x, idx)
        mx.eval(out)
    finally:
        ss.memtrace = original

    events = [r["event"] for r in tracer.rows()]
    assert "glu.enter" in events
    assert "glu.exit" in events
    # dict backing has no read_expert_into -> the bank load fails (traced as
    # ctx.ensure.fail in rolling mode, ctx.ensure.enter in union mode) and the
    # linears fall back to per-expert resolution, which must still be traced.
    assert any(e.startswith("ctx.ensure.") for e in events)
    assert "linear.resolve" in events
    assert "linear.stack" in events

    resolved = [r for r in tracer.rows() if r["event"] == "linear.resolve"]
    # One per projection (gate, up, down).
    assert {r["proj"] for r in resolved} == {"gate_proj", "up_proj", "down_proj"}
    for row in resolved:
        assert row["uniq"] > 0
        assert row["misses"] > 0
        assert row["from_ctx"] is False
        assert row["bank_bytes"] >= 0
    # Indices of shape (B, T, 1) keep the trailing singleton — stock SwitchGLU
    # returns the same rank here, so this is parity, not an artifact.
    assert out.shape == (1, 80, 1, D)


# ---------------------------------------------------------------------------
# Fase J — Etapa C: per-layer eval boundary (G2, qwen4_exp parity)
# ---------------------------------------------------------------------------


def test_stream_eval_applies_to_qwen_decoder_classes():
    """The boundary must install on every qwen decoder that ignores
    _stream_eval, and re-applying must not double-wrap (a double wrap would
    cost two syncs per layer and change nothing else)."""
    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    assert qse.apply_qwen35_moe_stream_eval() is True
    names = qse.wrapped_class_names()
    assert "Qwen3_5MoeDecoderLayer" in names

    classes = dict(qse._candidate_decoder_classes())
    before = {name: cls.__call__ for name, cls in classes.items()}
    assert qse.apply_qwen35_moe_stream_eval() is True
    # Identity check: a second apply short-circuits on the class flag.
    for name, cls in classes.items():
        assert cls.__call__ is before[name], f"{name} was re-wrapped"


def test_stream_eval_boundary_gating_and_bit_exactness():
    """The boundary fires only on prefill-shaped calls of streamed layers, and
    never changes the returned values.

    Prefill-shaped = _stream_eval set, not an MTP verify pass, and T > 1.
    Decode (T == 1) and verify passes stay lazy: 48 forced syncs per token
    would erase the QD16 decode win.
    """
    import mlx.core as mx

    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    class _Stub:
        def __init__(self, stream_eval: bool):
            self._stream_eval = stream_eval
            self.seen = 0

        def __call__(self, x, target_verify: bool = False):
            self.seen += 1
            return x * 3.0

    unwrapped = _Stub.__call__

    counters = {"eval": 0, "clear": 0}
    real_eval, real_clear = mx.eval, mx.clear_cache
    real_thresh, real_enabled = qse._cache_threshold_bytes, qse._per_layer_eval_enabled

    def counting_eval(*args, **kwargs):
        counters["eval"] += 1
        return real_eval(*args, **kwargs)

    def counting_clear():
        counters["clear"] += 1
        return real_clear()

    qse._per_layer_eval_enabled = True
    qse._cache_threshold_bytes = lambda: 0  # force the clear branch
    mx.eval = counting_eval
    mx.clear_cache = counting_clear
    try:
        wrapped = qse._wrap_call(unwrapped)

        # 1) prefill-shaped, streamed -> boundary fires
        stub = _Stub(True)
        prefill = mx.arange(16, dtype=mx.float32).reshape(1, 8, 2)
        out = wrapped(stub, prefill)
        mx.eval(out)
        base_eval, base_clear = counters["eval"], counters["clear"]
        assert base_eval >= 1
        assert base_clear >= 1
        # bit-exact: the wrapper returns the layer's own output untouched
        assert mx.array_equal(out, unwrapped(stub, prefill) * 1.0)

        # 2) decode-shaped (T == 1) -> stays lazy
        counters["eval"] = counters["clear"] = 0
        wrapped(stub, mx.zeros((1, 1, 2)))
        assert counters["eval"] == 0
        assert counters["clear"] == 0

        # 3) MTP verify pass -> stays lazy
        counters["eval"] = counters["clear"] = 0
        wrapped(stub, mx.zeros((1, 8, 2)), target_verify=True)
        assert counters["eval"] == 0
        assert counters["clear"] == 0

        # 4) layer without _stream_eval -> untouched
        counters["eval"] = counters["clear"] = 0
        plain = _Stub(False)
        wrapped(plain, mx.zeros((1, 8, 2)))
        assert counters["eval"] == 0
        assert counters["clear"] == 0

        # 5) knob off -> boundary disabled entirely
        qse._per_layer_eval_enabled = False
        counters["eval"] = counters["clear"] = 0
        wrapped(stub, mx.zeros((1, 8, 2)))
        assert counters["eval"] == 0
        assert counters["clear"] == 0
    finally:
        mx.eval = real_eval
        mx.clear_cache = real_clear
        qse._cache_threshold_bytes = real_thresh
        qse._per_layer_eval_enabled = real_enabled


def test_stream_eval_clear_goes_through_sync_helper():
    """Etapa C requires every pool clear to be synchronized first — an
    unsynchronized mx.clear_cache() panics the M4 driver. The helper must
    prefer _sync_and_clear_cache and still work if it cannot be imported."""
    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    # Prefer the locked helper when importable.
    qse._clear_cache_synced()
    # And degrade gracefully when it is not.
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("omlx.utils.metal_sync"):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked_import
    try:
        qse._clear_cache_synced()  # must not raise
    finally:
        builtins.__import__ = real_import


def test_qwen4_exp_decoder_is_a_boundary_candidate():
    """qwen4_exp's decoder does NOT honour _stream_eval, so it must be in the
    wrap list — this is the G2 gap that left the flag inert on qwen4_exp."""
    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    names = [name for name, _ in qse._candidate_decoder_classes()]
    # The installed qwen3_5_moe decoder is always present in this environment.
    assert "Qwen3_5MoeDecoderLayer" in names
    # qwen4_exp is vendored: present when the compat patch can register it.
    if "Qwen4ExpDecoderLayer" in names:
        classes = dict(qse._candidate_decoder_classes())
        q4 = classes["Qwen4ExpDecoderLayer"]
        q35 = classes["Qwen3_5MoeDecoderLayer"]
        # Independent hierarchies: wrapping both cannot double-eval.
        assert q4 is not q35
        assert not issubclass(q4, q35)


# ---------------------------------------------------------------------------
# Fase J — Etapa D (threshold-gated pool clears) and Etapa E (guard accounting)
# ---------------------------------------------------------------------------


def test_cache_clear_threshold_bytes_default_and_env(monkeypatch):
    """OMLX_EXPERT_STREAMING_CACHE_THRESH is in GiB; 2 GiB when unset, and an
    unparseable value must fall back rather than crash a running prefill."""
    from omlx.utils import metal_sync as ms

    monkeypatch.delenv(ms._CACHE_CLEAR_THRESH_ENV, raising=False)
    assert ms.cache_clear_threshold_bytes() == 2 * 1024**3

    monkeypatch.setenv(ms._CACHE_CLEAR_THRESH_ENV, "0.5")
    assert ms.cache_clear_threshold_bytes() == int(0.5 * 1024**3)

    # 0 disables the gate: every clear runs.
    monkeypatch.setenv(ms._CACHE_CLEAR_THRESH_ENV, "0")
    assert ms.cache_clear_threshold_bytes() == 0

    monkeypatch.setenv(ms._CACHE_CLEAR_THRESH_ENV, "not-a-number")
    assert ms.cache_clear_threshold_bytes() == 2 * 1024**3


def test_should_clear_cache_gates_on_pool_size(monkeypatch):
    """Below threshold the pool stays pooled; above it, the clear is
    justified. The helper returns only the decision — the caller keeps its
    own sync-before-clear call site."""
    import mlx.core as mx

    from omlx.utils import metal_sync as ms

    monkeypatch.setattr(mx, "get_cache_memory", lambda: 128 * 1024**2)
    assert ms.should_clear_cache() is False

    monkeypatch.setattr(mx, "get_cache_memory", lambda: 8 * 1024**3)
    assert ms.should_clear_cache() is True

    # threshold=0 -> always justified, even with an empty pool.
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 0)
    assert ms.should_clear_cache(threshold_bytes=0) is True

    # Boundary: exactly at the threshold counts as worth clearing.
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 2 * 1024**3)
    assert ms.should_clear_cache() is True


def test_should_clear_cache_without_cache_probe(monkeypatch):
    """get_cache_memory unavailable -> always clear. The pool cannot be
    measured, so the conservative (pre-Fase-J) branch must win."""
    import types

    from omlx.utils import metal_sync as ms

    # A stub mx with no get_cache_memory at all.
    monkeypatch.setattr(
        ms,
        "mx",
        types.SimpleNamespace(synchronize=lambda *a: None, clear_cache=lambda: None),
    )
    assert ms.should_clear_cache() is True


def test_scheduler_clear_cache_if_pool_large_keeps_sync_choke_point(monkeypatch):
    """Etapa D must not bypass the module's single patchable
    ``_sync_and_clear_cache`` — the prefill tests instrument it."""
    import mlx.core as mx

    from omlx import scheduler as sched_mod
    from omlx.scheduler import Scheduler

    calls = []
    sched = Scheduler.__new__(Scheduler)
    sched._stream = "ENGINE"
    monkeypatch.setattr(
        sched_mod, "_sync_and_clear_cache", lambda stream=None: calls.append(stream)
    )
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 128 * 1024**2)
    assert sched._clear_cache_if_pool_large() is False
    assert calls == []

    monkeypatch.setattr(mx, "get_cache_memory", lambda: 8 * 1024**3)
    assert sched._clear_cache_if_pool_large() is True
    assert calls == ["ENGINE"]


def test_stream_eval_threshold_delegates_to_shared_resolver(monkeypatch):
    """Etapa D: the knob has ONE home. The per-layer gate and the scheduler's
    chunk-boundary clears must not be able to drift apart."""
    from omlx.patches.expert_streaming import qwen35_stream_eval as qse
    from omlx.utils import metal_sync as ms

    monkeypatch.setenv(ms._CACHE_CLEAR_THRESH_ENV, "3")
    assert qse._cache_threshold_bytes() == 3 * 1024**3
    assert qse._cache_threshold_bytes() == ms.cache_clear_threshold_bytes()


def test_prefill_clears_are_gated_with_ungated_handoff():
    """Etapa D: chunk-boundary clears are gated on pool size, but each prefill
    path keeps a single unconditional clear at the handoff to decode so the
    pool never carries the prefill working set into the decode loop.

    Source-level via AST (driving either loop needs a fully wired scheduler).
    """
    import ast
    import inspect
    import textwrap

    from omlx.scheduler import Scheduler

    def clear_calls(fn):
        out = []
        src = textwrap.dedent(inspect.getsource(fn))
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in ("_sync_and_clear_cache", "_clear_cache_if_pool_large"):
                out.append((node.lineno, name))
        return sorted(out)

    ext = clear_calls(Scheduler._do_external_prefill)
    # First-chunk + between-chunk clears are both gated.
    assert [n for _, n in ext].count("_clear_cache_if_pool_large") >= 2
    # ...and the LAST clear the external path issues is unconditional.
    assert ext[-1][1] == "_sync_and_clear_cache"

    # Chunked path: gate inside the chunk step...
    chunk = [n for _, n in clear_calls(Scheduler._step_prefill_chunk)]
    assert chunk.count("_clear_cache_if_pool_large") >= 1
    # ...unconditional completion clear where the request is handed to decode.
    adv = [n for _, n in clear_calls(Scheduler._advance_chunked_prefills)]
    assert "_sync_and_clear_cache" in adv


def _boundary_backing(**over):
    """Guard info as the converter emits it, with Etapa E's keys."""
    info = {
        "num_moe_layers": 48,
        "experts_per_layer": 512,
        "per_expert_bytes": 2_500_000,
        "boundary_active": False,
        "projections": 3,
        "activation_bytes_per_token": 0,
    }
    info.update(over)
    back = MagicMock()
    back.streaming_guard_info = info
    return back


def _sched_with(etapa_e=True, **over):
    """Scheduler over a synthetic streaming backing.

    ``etapa_e`` opts into Etapa E explicitly. It is off by default in the
    product (it changes chunk size, hence the numerics), so these tests must
    ask for it rather than inherit it.
    """
    model = MagicMock()
    model._expert_streaming_backing = _boundary_backing(**over)
    sched = _guard_scheduler(model)
    sched._STREAMING_BANK_BOUNDARY_ACCOUNT = etapa_e
    return sched


def test_streaming_bank_bytes_defaults_to_per_layer_charge():
    """Etapa E must ship OFF.

    The collapsed charge is not an accounting detail: it changes the chunk
    size the guard admits, and chunk size changes the GEMM shapes and hence
    the reduction order. Measured on qwen4_exp with Etapa E on, the greedy
    decode diverges from baseline at token 3 of 48 (deterministic, 6/6 runs).
    The memory win does not depend on it — with E off, phys_footprint still
    drops 37.03 -> 11.09 GiB. So correctness wins and E becomes opt-in.
    """
    from omlx.scheduler import Scheduler

    assert Scheduler._STREAMING_BANK_BOUNDARY_ACCOUNT is False
    sched = _sched_with(etapa_e=False, boundary_active=True)
    tile = 512 * 2_500_000
    assert sched._streaming_bank_bytes(10_000) == 48 * tile


def test_streaming_bank_bytes_boundary_collapses_per_layer_charge():
    """Etapa E (opt-in): with a live per-layer eval boundary the lazy graph no
    longer retains one mini-bank per MoE layer, so the charge drops from 48
    banks to min(2, projections) — the difference between a guard that refuses
    to admit chunks and one that sizes them correctly. Buying that chunk size
    is exactly what costs bit-exactness, hence the opt-in default."""
    n = 10_000  # saturates uniq at experts_per_layer
    tile = 512 * 2_500_000

    off = _sched_with(etapa_e=False, boundary_active=False)
    assert off._streaming_bank_bytes(n) == 48 * tile

    on = _sched_with(boundary_active=True)
    assert on._streaming_bank_bytes(n) == 2 * tile
    assert on._streaming_bank_bytes(n) < off._streaming_bank_bytes(n) // 20


@pytest.mark.parametrize(
    "projections,mult", [(1, 1), (2, 2), (3, 2), (4, 2), (0, 1)]
)
def test_streaming_bank_bytes_projections_clamped_at_two(
    projections, mult, monkeypatch
):
    """min(2, projections): the bank under construction plus at most one
    rolling prefetch. A bogus 0/absent count must floor at one bank, never
    shrink the charge to zero."""
    sched = _sched_with(boundary_active=True, projections=projections)
    tile = 512 * 2_500_000
    assert sched._streaming_bank_bytes(10_000) == mult * tile


def test_streaming_bank_bytes_adds_one_layer_activation():
    """The boundary materializes the layer output, so one activation is live
    on top of the bank term."""
    sched = _sched_with(boundary_active=True, activation_bytes_per_token=4096 * 2)
    n = 100
    uniq = min(512, int(sched._STREAMING_BANK_TOKEN_RATIO * n))
    tile = uniq * 2_500_000
    assert sched._streaming_bank_bytes(n) == 2 * tile + 4096 * 2 * n


def test_streaming_bank_bytes_boundary_kill_switch(monkeypatch):
    """OMLX_STREAMING_BANK_BOUNDARY_ACCOUNT flips Etapa E both ways without a
    code change — the escape hatch, and the switch an operator pulls to trade
    bit-exactness for the 62% TTFT win."""
    sched = _sched_with(etapa_e=True, boundary_active=True)
    tile = 512 * 2_500_000
    assert sched._streaming_bank_bytes(10_000) == 2 * tile
    monkeypatch.setattr(sched, "_STREAMING_BANK_BOUNDARY_ACCOUNT", False)
    assert sched._streaming_bank_bytes(10_000) == 48 * tile
    monkeypatch.setattr(sched, "_STREAMING_BANK_BOUNDARY_ACCOUNT", True)
    assert sched._streaming_bank_bytes(10_000) == 2 * tile


def test_glu_projection_count():
    """Projections sharing one per-layer load context: 2 when the checkpoint
    fuses gate_up_proj, 3 when gate/up are split."""
    from omlx.patches.expert_streaming import _glu_projection_count

    fused = MagicMock()
    fused.switch_mlp.linears = [MagicMock(), MagicMock()]
    split = MagicMock()
    split.switch_mlp.linears = [MagicMock()] * 3

    assert _glu_projection_count([fused]) == 2
    assert _glu_projection_count([split]) == 3
    # No converted GLU reachable -> conservative split default.
    assert _glu_projection_count([MagicMock()]) == 3
    assert _glu_projection_count(None) == 3

    class Boom:
        @property
        def switch_mlp(self):
            raise RuntimeError("boom")

    assert _glu_projection_count([Boom()]) == 3


def test_boundary_active_requires_enabled_and_wrapped(monkeypatch):
    """The guard may only relax the bank charge when a boundary is really
    installed: knob on AND at least one decoder class carrying the wrapper."""
    from omlx.patches.expert_streaming import qwen35_stream_eval as qse

    monkeypatch.setattr(qse, "_per_layer_eval_enabled", True)
    monkeypatch.setattr(qse, "wrapped_class_names", lambda: [])
    assert qse.boundary_active() is False

    monkeypatch.setattr(
        qse, "wrapped_class_names", lambda: ["Qwen4ExpDecoderLayer"]
    )
    assert qse.boundary_active() is True

    monkeypatch.setattr(qse, "_per_layer_eval_enabled", False)
    assert qse.boundary_active() is False


def test_residency_estimate_exposes_hidden_size():
    """Etapa E sources the activation term (hidden_size x 2 bytes/token) from
    the checkpoint config."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=8, hidden=32)
        assert expert_streaming_estimate(tmp).hidden_size == 32


def test_residency_hidden_size_prefers_text_config():
    """VLMs expose the vision tower's width at the top level and the language
    model's under text_config — the activation term must use the latter."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_fake_glm_checkpoint(tmp, num_layers=4, experts=8, hidden=32)
        cfg = json.loads((tmp / "config.json").read_text())
        cfg["hidden_size"] = 999
        cfg["text_config"] = {"hidden_size": 4096}
        (tmp / "config.json").write_text(json.dumps(cfg))
        assert expert_streaming_estimate(tmp).hidden_size == 4096


# ---------------------------------------------------------------------------
# Fase J — Etapa A1: single-promotion bank build (bit-exact)
# ---------------------------------------------------------------------------


def _write_shard_filled(path: Path, tensors: dict, seed: int = 1234) -> None:
    """Minimal safetensors file with deterministic non-zero payloads.

    ``_write_shard`` writes all-zero data, which would make any bit-exactness
    comparison vacuous — every byte pattern would match trivially.

    BF16 tensors get values from a small positive range rather than random
    bit patterns: random bf16 words decode to exponents around 1e30, the
    dequantized products overflow to inf/NaN, and then ``array_equal`` fails
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
            # Packed weights: random bytes, no zero lanes.
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
    # MLX requires scales and biases to share a shape, so biases are per-group
    # too, matching what a real affine-quantized checkpoint stores.
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


def test_bank_promote_is_bit_identical_to_per_expert_stack(tmp_path):
    """Etapa A1: promoting the whole demand bank in one mx.array must yield
    byte-for-byte the same tensors as promoting U per-expert slices and
    stacking them — so gather_qmm sees an identical bank."""
    import mlx.core as mx

    linear = _quant_linear(tmp_path, n_experts=5)

    ids = [4, 1, 3]  # deliberately out of order: row order must be preserved
    got = linear._load_expert_bank_mx(ids)
    assert got is not None
    w_fast, s_fast, b_fast, rows = got

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
    """Etapa A1 end-to-end: the layer output must be bit-identical whether the
    single-promotion fast path is enabled or not. This is the gate that makes
    A1 lossless rather than near-lossless."""
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

    # Prove the fast path really ran in one arm and not the other — otherwise
    # identical outputs would prove nothing.
    assert engaged_on == [True], engaged_on
    assert engaged_off == [], engaged_off

    assert out_on.dtype == out_off.dtype

    # Compare bit patterns, not ==: a NaN payload would fail array_equal even
    # when the two arms produced byte-identical results.
    bits_on = np.ascontiguousarray(out_on).view(np.uint32).reshape(-1)
    bits_off = np.ascontiguousarray(out_off).view(np.uint32).reshape(-1)
    assert np.array_equal(bits_on, bits_off)
    # Guard against a vacuous pass: garbage inputs would make every bit 0/NaN.
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
    keys, banks, rows2 = got

    assert keys[0] == linear.stacked_weight_key
    assert len(rows) == len(rows2) == len(ids)
    assert banks[0].shape == (len(ids), linear._slice_bytes(keys[0]))
    for (w1, s1, b1), (w2, s2, b2) in zip(rows, rows2):
        assert np.array_equal(w1, w2)
        assert np.array_equal(s1, s2)
        assert np.array_equal(b1, b2)


# ---------------------------------------------------------------------------
# Fase J — Etapa A1b: single-promotion bank build on the layer-context path
# ---------------------------------------------------------------------------


def _quant_glu(tmp: Path, n_experts: int = 3):
    """A quantized StreamingSwitchGLU whose projections sit on real shards.

    Built with ``quantized=True`` so ``StreamingSwitchGLU`` installs a
    ``_LayerLoadContext`` — the Etapa B path that runs by default and the one
    A1b targets. A1 itself never fires here: the context pre-resolves every
    expert, so ``missing`` is empty by the time the linear asks. That gap is
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
    """Etapa A1b: with the Etapa B barrier on (the default), the GLU output must
    be bit-identical whether single-promotion is enabled or not.

    This is the gate that makes A1b lossless rather than near-lossless. It also
    proves A1b engages where A1 does not: A1 measured zero invocations on this
    path, and would leave the default configuration with no bank-promotion win.
    """
    import mlx.core as mx

    from omlx.patches.expert_streaming import streaming_switch as ss

    assert ss._LAYER_BARRIER_ENV  # the default path A1b is scoped to

    x = mx.random.normal((1, 6, 128)).astype(mx.float32)
    indices = mx.array([2, 0, 1, 2, 0, 1], dtype=mx.int32)

    def run(enabled):
        monkeypatch.setattr(ss, "_BANK_PROMOTE_CTX_ENV", enabled)
        # A fresh GLU per arm: reusing one would leave state behind and could
        # turn the second arm into a hit-only run, making the comparison
        # vacuous because neither arm would exercise a miss.
        glu = _quant_glu(tmp_path, n_experts=3)
        engaged: list[bool] = []
        real = ss.StreamingQuantizedSwitchLinear._promote_banks

        def spy(self, keys, banks, n):
            out = real(self, keys, banks, n)
            engaged.append(out is not None)
            return out

        monkeypatch.setattr(ss.StreamingQuantizedSwitchLinear, "_promote_banks", spy)
        out = glu(x, indices)
        mx.eval(out)
        return out, engaged

    out_on, engaged_on = run(True)
    out_off, engaged_off = run(False)

    # One call per projection (gate/up/down). Proves the fast path really ran
    # in one arm and not the other — otherwise identical outputs prove nothing.
    assert engaged_on == [True, True, True], engaged_on
    assert engaged_off == [], engaged_off

    assert out_on.dtype == out_off.dtype
    # Compare bit patterns, not ==: a NaN payload would fail array_equal even
    # when the two arms produced byte-identical results.
    bits_on = np.ascontiguousarray(out_on).view(np.uint32).reshape(-1)
    bits_off = np.ascontiguousarray(out_off).view(np.uint32).reshape(-1)
    assert np.array_equal(bits_on, bits_off)
    # Guard against a vacuous pass: garbage inputs would make every bit 0/NaN.
    assert bool(mx.all(mx.isfinite(out_on)))


def test_ctx_bank_promote_declined_on_partial_demand(monkeypatch, tmp_path):
    """A bank covering only part of the demand set must not be promoted.

    The missing experts do live in one contiguous bank, but the cache hits
    would have to be promoted separately and concatenated — a different layout
    contract that A1b deliberately leaves on the legacy path. Promoting a
    partial bank as if it were the whole demand set would mis-pair experts.
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

    # The seeded expert really was a hit: 2 of 3 missing, not 3 of 3.
    assert recorded["gate_proj"][2] == 2, recorded["gate_proj"]
    # ...so A1b must record nothing for gate_proj.
    assert recorded["gate_proj"][0] is None
    assert recorded["gate_proj"][1] is None
    # The other two projections were all-miss and stay eligible: the bank
    # describes exactly the demand set, which is what licenses promotion.
    assert recorded["up_proj"][0] is not None
    assert recorded["up_proj"][1] == [0, 1, 2]
    assert bool(mx.all(mx.isfinite(out)))
