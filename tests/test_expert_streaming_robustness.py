# SPDX-License-Identifier: Apache-2.0
"""Fase 3 audit fixes: legacy cache fetch-first install + serialization
marker, cold-reader cleanup, atomic transition-profile writes
(2.2, 2.4, 2.7-2.9, N5-N7). The DSv4.1 rollback/floor/compact cases live
with the deepseek_v41 adapter tests."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest


class TestLegacyCacheAtomicity:
    """2.2/2.4: fetch-first install + a real lock on the legacy cache."""

    def _cache(self, tmp_path, capacity=2, n=4):
        from mlx_lm.models.switch_layers import SwitchGLU
        import mlx.nn as nn

        from omlx.patches.moe_expert_offload import (
            CheckpointExpertStore,
            ExpertCache,
            _GLUStoreView,
        )

        glu = SwitchGLU(32, 32, n)
        nn.quantize(glu, group_size=32, bits=4)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(glu, proj)
            for field in ("weight", "scales", "biases"):
                if lin.get(field) is not None:
                    tensors[f"layers.0.mlp.switch_glu.{proj}.{field}"] = lin[field]
        mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
        store = CheckpointExpertStore(tmp_path)
        view = _GLUStoreView(store, "layers.0.mlp.switch_glu")
        return ExpertCache(glu, capacity, view)

    def test_failed_install_mutates_nothing(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.ensure(mx.array([0]))  # one resident
        before_slots = dict(cache.slot_of)
        before_free = list(cache.free)

        orig = cache.disk.fetch
        calls = {"n": 0}

        def _boom(proj, field, expert):
            calls["n"] += 1
            if calls["n"] > 2:
                raise OSError("injected")
            return orig(proj, field, expert)

        cache.disk.fetch = _boom
        with pytest.raises(OSError):
            cache._install(1)
        assert cache.slot_of == before_slots
        assert cache.free == before_free
        assert 1 not in cache.slot_of

    def test_legacy_apply_stamps_serialization_marker(self, tmp_path):
        import mlx.nn as nn
        from mlx_lm.models.switch_layers import SwitchGLU

        from omlx.patches.moe_expert_offload import _apply_legacy_adapter
        from omlx.scheduler import _model_uses_expert_streaming

        glu = SwitchGLU(32, 32, 4)
        nn.quantize(glu, group_size=32, bits=4)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(glu, proj)
            for field in ("weight", "scales", "biases"):
                if lin.get(field) is not None:
                    tensors[f"layers.0.mlp.switch_glu.{proj}.{field}"] = lin[field]
        mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)

        class _GLU(nn.Module):
            def __init__(self, g):
                super().__init__()
                self.switch_glu = g

        class _Layer(nn.Module):
            def __init__(self, g):
                super().__init__()
                self.mlp = _GLU(g)

        class _Model(nn.Module):
            def __init__(self, g):
                super().__init__()
                self.layers = [_Layer(g)]

        model = _Model(glu)
        wrapped = _apply_legacy_adapter(model, tmp_path, 0.5)
        assert wrapped == 1
        # Requests must serialize: the marker is what the scheduler checks.
        assert _model_uses_expert_streaming(model) is True


class TestShardBankClose:
    """2.9: the cold-tier key memo must not hand back closed readers."""

    def test_close_clears_cold_key_map(self, tmp_path):
        from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

        # Minimal checkpoint: the store only needs a readable header.
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "x"}))
        mx.save_safetensors(
            str(tmp_path / "model.safetensors"), {"w": mx.zeros((2, 2))}
        )
        store = ExpertBackingStore(tmp_path)
        store._cold_key_to_reader["some.key"] = SimpleNamespace()
        store.close()
        assert store._cold_key_to_reader == {}


class TestTransitionProfileAtomic:
    """N5: the profile write is tmp+replace, never a truncated dest."""

    def test_save_is_atomic(self, tmp_path):
        from omlx.patches.expert_streaming import save_transition_profile

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "x"}))

        class _Spec:
            trans_updates = 3

            def to_payload(self):
                return {"regimes": {}, "trans_updates": 3}

        backing = SimpleNamespace(spec_state=_Spec(), model_path=str(tmp_path))
        save_transition_profile(backing)
        dest = tmp_path / ".omlx" / "expert_transition.json"
        assert dest.is_file()
        assert not (tmp_path / ".omlx" / "expert_transition.json.tmp").exists()
        payload = json.loads(dest.read_text())
        assert payload["trans_updates"] == 3


class TestLegacyCacheResize:
    """Governor-facing resize/clear on the legacy per-layer cache."""

    def _cache(self, tmp_path, capacity=4, n=8):
        import mlx.nn as nn
        from mlx_lm.models.switch_layers import SwitchGLU

        from omlx.patches.moe_expert_offload import (
            CheckpointExpertStore,
            ExpertCache,
            _GLUStoreView,
        )

        glu = SwitchGLU(32, 32, n)
        nn.quantize(glu, group_size=32, bits=4)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(glu, proj)
            for field in ("weight", "scales", "biases"):
                if lin.get(field) is not None:
                    tensors[f"layers.0.mlp.switch_glu.{proj}.{field}"] = lin[field]
        mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
        store = CheckpointExpertStore(tmp_path)
        view = _GLUStoreView(store, "layers.0.mlp.switch_glu")
        return ExpertCache(glu, capacity, view)

    def test_shrink_evicts_lru_and_caps_installs(self, tmp_path):
        cache = self._cache(tmp_path)
        try:
            cache.ensure(mx.array([0, 1, 2, 3]))  # fill all 4 rooms
            cache.resize(2)
            assert cache.capacity == 2
            assert len(cache.slot_of) == 2
            assert 0 not in cache.slot_of and 1 not in cache.slot_of
            # Ceiling is a count, not a row range: a free row exists but
            # the install must still evict the LRU victim.
            cache.ensure(mx.array([4]))
            assert 4 in cache.slot_of
            assert len(cache.slot_of) == 2
            assert len(cache.slot_of) + len(cache.free) == cache.rooms
        finally:
            cache.disk._store.close()

    def test_grow_reallocs_and_installs_land(self, tmp_path):
        cache = self._cache(tmp_path, capacity=4, n=8)
        try:
            cache.ensure(mx.array([0, 1]))
            cache.resize(6)
            assert cache.rooms == 6
            assert cache.capacity == 6
            cache.ensure(mx.array([5, 6]))
            assert 5 in cache.slot_of and 6 in cache.slot_of
            assert len(cache.slot_of) + len(cache.free) == cache.rooms
            # Beyond n_experts clamps.
            cache.resize(64)
            assert cache.capacity == 8
            assert cache.rooms == 8
        finally:
            cache.disk._store.close()

    def test_clear_drops_residency(self, tmp_path):
        cache = self._cache(tmp_path)
        try:
            cache.ensure(mx.array([0, 1]))
            cache.clear()
            assert cache.slot_of == {}
            assert len(cache.free) == cache.rooms
            assert cache.warm is False
        finally:
            cache.disk._store.close()


class TestLegacyOffloadState:
    """The legacy aggregate presents the V4.1 governor duck-type."""

    def _module(self, tmp_path, capacity=4, n=8):
        import mlx.nn as nn
        from mlx_lm.models.switch_layers import SwitchGLU

        from omlx.patches.moe_expert_offload import (
            CheckpointExpertStore,
            OffloadSwitchGLU,
            _GLUStoreView,
        )

        glu = SwitchGLU(32, 32, n)
        nn.quantize(glu, group_size=32, bits=4)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(glu, proj)
            for field in ("weight", "scales", "biases"):
                if lin.get(field) is not None:
                    tensors[f"layers.0.mlp.switch_glu.{proj}.{field}"] = lin[field]
        mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
        store = CheckpointExpertStore(tmp_path)
        view = _GLUStoreView(store, "layers.0.mlp.switch_glu")
        return OffloadSwitchGLU(glu, capacity, view), store

    def test_governor_duck_and_stats(self, tmp_path):
        from omlx.patches.moe_expert_offload import LegacyOffloadState

        mod, store = self._module(tmp_path)
        try:
            state = LegacyOffloadState([mod.cache], dynamic=True, min_cap=2)
            assert state.governor is not None
            assert state.governor.min_cap == 2
            state.note_visit(0, True)
            state.note_visit(0, False)
            assert state.stats.decode_layers == 2
            assert state.stats.decode_layers_missed == 1
            assert state.stats.decode_misses_by_layer == {0: 1}
            state.resize(2, None)
            assert mod.cache.capacity == 2
            state.set_layer_caps({0: 3})
            assert mod.cache.capacity == 3
            assert state.layer_cap_overrides() == {0: 3}
            state.clear()
            assert mod.cache.slot_of == {}
            summary = state.summary()
            assert summary["layers"] == 1
            # Persistent slot buffers: no mini-bank transient term.
            assert state.streaming_guard_info is None
        finally:
            store.close()

    def test_decode_visit_feeds_stats(self, tmp_path):
        from omlx.patches.moe_expert_offload import LegacyOffloadState

        mod, store = self._module(tmp_path)
        try:
            state = LegacyOffloadState([mod.cache], dynamic=False)
            mod._state = state
            mod._layer = 0
            out = mod(mx.zeros((1, 32)), mx.array([[0]]))
            mx.eval(out)
            assert state.stats.decode_layers == 1
            assert state.stats.decode_layers_missed == 1  # cold miss
            # Multi-token call is not a decode visit.
            out = mod(mx.zeros((3, 32)), mx.array([[0], [1], [0]]))
            mx.eval(out)
            assert state.stats.decode_layers == 1
        finally:
            store.close()


class TestResolveBudgetBytes:
    """Public budget resolver for the admission path (2.6)."""

    def test_pin_auto_and_zero(self):
        from omlx.model_settings import ModelSettings
        from omlx.patches.expert_streaming import resolve_budget_bytes

        assert resolve_budget_bytes(
            ModelSettings(expert_streaming_budget_gib=1.5)
        ) == int(1.5 * 1024**3)
        assert (
            resolve_budget_bytes(
                ModelSettings(expert_streaming_budget_auto=False)
            )
            == 0
        )
        assert resolve_budget_bytes(None) == 0
        # MiB spellings exist only on dict-shaped/legacy settings objects.
        assert resolve_budget_bytes(
            SimpleNamespace(expert_streaming_budget_mib=512)
        ) == 512 * 1024 * 1024
        # Beyond the 64 GiB ceiling clamps (P2-16).
        assert resolve_budget_bytes(
            ModelSettings(expert_streaming_budget_gib=1000)
        ) == 64 * 1024**3


class TestSummaryMergesBacking:
    """Cache-less backings (V4.1/legacy) must not log a permanent 0."""

    def test_backing_summary_folds_into_lru_slots(self):
        from omlx.patches.expert_streaming import expert_streaming_summary

        backing = SimpleNamespace(
            summary=lambda: {
                "hits": 3,
                "misses": 1,
                "evictions": 2,
                "resident": 4,
                "capacity_per_layer": 8,
                "layers": 2,
                "governor": {},
            }
        )
        out = expert_streaming_summary(None, backing)
        assert out["lru_hits"] == 3
        assert out["lru_misses"] == 1
        assert out["lru_evictions"] == 2
        assert out["lru_size"] == 4
        assert out["lru_capacity"] == 16
        assert out["lru_hit_rate"] == 0.75
        assert out["backing"]["layers"] == 2


class TestFfnBankGroupRegex:
    """3.4: the bank pack must discover DeepSeek V4 ffn.switch_mlp keys."""

    def test_ffn_and_mlp_layouts_match(self):
        from omlx.patches.expert_streaming.expert_bank_pack import _GROUP_RE

        m = _GROUP_RE.match(
            "model.layers.0.ffn.switch_mlp.gate_proj.weight"
        )
        assert m is not None
        assert m.group("kind") == "weight"
        assert _GROUP_RE.match(
            "model.layers.0.mlp.experts.down_proj.scales"
        )
        assert _GROUP_RE.match(
            "language_model.model.layers.3.ffn.switch_mlp.up_proj.biases"
        )


class TestAliasPreservesSettings:
    """3.3: the alias path clones the user's settings instead of
    building a bare ModelSettings that drops every streaming tunable."""

    def test_tunables_survive_and_pin_wins(self, tmp_path, monkeypatch):
        import omlx.patches.expert_streaming as es
        import omlx.patches.expert_streaming.residency as res
        from omlx.model_settings import ModelSettings
        from omlx.patches.moe_expert_offload import _apply_via_streaming

        est = SimpleNamespace(
            supported=True, expert_bytes=4 * 1024**3, num_moe_layers=2
        )
        monkeypatch.setattr(
            res, "expert_streaming_estimate", lambda *a, **k: est
        )
        captured = {}

        def _conv(model, path, settings, **kw):
            captured["settings"] = settings
            return model, SimpleNamespace()

        monkeypatch.setattr(es, "convert_model_to_streaming", _conv)

        ms = ModelSettings(
            moe_expert_offload_enabled=True,
            expert_streaming_io_depth=24,
        )
        wrapped = _apply_via_streaming(SimpleNamespace(), tmp_path, 0.5, ms)
        assert wrapped == 2
        s = captured["settings"]
        assert s.expert_streaming_enabled is True
        assert s.expert_streaming_io_depth == 24  # tunable survived
        assert s.expert_streaming_dynamic is True  # alias default forced
        assert s.expert_streaming_budget_gib == pytest.approx(2.0)

        ms2 = ModelSettings(
            moe_expert_offload_enabled=True,
            expert_streaming_budget_gib=5.0,
            expert_streaming_dynamic=False,
        )
        _apply_via_streaming(SimpleNamespace(), tmp_path, 0.5, ms2)
        s2 = captured["settings"]
        assert s2.expert_streaming_budget_gib == 5.0  # pin wins over fraction
        assert s2.expert_streaming_dynamic is False  # explicit False kept


class TestSchedulerStreamingFloor:
    """2.6: the guard counts the LRU heap accounting + wired pin pages."""

    def _sched(self, info, cache, backing):
        from omlx.scheduler import Scheduler

        sched = Scheduler.__new__(Scheduler)
        sched._streaming_guard_info = info
        sched._streaming_lru_cache = cache
        sched._streaming_lru_bytes_last = None
        sched._streaming_backing = backing
        sched._last_mlx_active_memory_bytes = 0
        return sched

    def test_floor_covers_resident_plus_pins(self):
        sched = self._sched(
            {"x": 1},
            SimpleNamespace(resident_bytes=lambda: 3 * 1024**3),
            SimpleNamespace(pinned_bytes=512 * 1024**2),
        )
        used = sched._current_usage_bytes(refresh_mlx_active=False)
        assert used == 3 * 1024**3 + 512 * 1024**2

    def test_backing_without_guard_info_takes_streaming_branch(self):
        # V4.1/legacy report streaming_guard_info=None — resolved to {} —
        # but still stream through mmap; the branch must fire on the
        # backing, not the metadata dict (phys footprint would report the
        # page cache and return a real nonzero number here).
        sched = self._sched({}, None, SimpleNamespace(pinned_bytes=0))
        assert sched._current_usage_bytes(refresh_mlx_active=False) == 0
