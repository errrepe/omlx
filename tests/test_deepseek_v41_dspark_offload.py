"""DSpark verify under frozen expert residency (clean-split design).

Layer 3 only: verify_scope() suspends LRU reordering in _ExpertSlots
while the shared block kernels (layer 1) and plan builders (layer 2)
are untouched. Native MTP + offload validates for deepseek_v41 alone.
"""

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import write_checkpoint

from omlx.model_settings import validate_moe_expert_offload
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.deepseek_v41.moe_offload import verify_scope
from omlx.patches.deepseek_v41.streaming_backing import V41StreamingBacking


def _backed_disk(tmp_path, **kwargs):
    disk = _offloaded_disk(tmp_path)
    layers = [
        (i, layer.ffn.experts.slots)
        for i, layer in enumerate(disk.language_model.layers)
    ]
    backing = V41StreamingBacking(disk._moe_offload_plan, layers, **kwargs)
    # Engine parity: the hook chain resolves the backing off the model.
    disk._expert_streaming_backing = backing
    return disk, backing


def _offloaded_disk(tmp_path, subdir=None):
    base = tmp_path / subdir if subdir else tmp_path
    base.mkdir(parents=True, exist_ok=True)
    source, _ = write_checkpoint(
        base, vision=False, n_routed_experts=8, n_activated_experts=2
    )
    disk, _ = load(source, moe_expert_offload_resident_fraction=0.25)
    assert disk._moe_offload_plan.capacity == 2
    return disk


def _prime_recall(backing, layer, value=1.0):
    """Seed the per-layer staging recall gate.

    The EWMA needs ~4 matching prev->now routing observations to cross
    the 0.3 floor; tests set it directly so staging expectations measure
    the path under test, not gate warm-up.
    """
    backing.recall_ewma[int(layer)] = value


def test_verify_scope_freezes_slot_recency(tmp_path):
    """DSpark verify traffic must not disturb decode-hot LRU order."""
    disk = _offloaded_disk(tmp_path)
    try:
        slots = disk.language_model.layers[0].ffn.experts.slots
        slots.ensure(mx.array([0, 1]))
        assert list(slots.slot_of) == [0, 1]
        with verify_scope():
            # Frozen hit: served, order untouched.
            slots.ensure(mx.array([1]))
            assert list(slots.slot_of) == [0, 1]
            # Frozen miss: draft-only expert lands oldest-first, evicting
            # the oldest non-needed entry without reordering the rest.
            slots.ensure(mx.array([2]))
            assert list(slots.slot_of) == [2, 1]
            # Capacity guard still applies inside the scope.
            with pytest.raises(ValueError, match="exceeds resident capacity"):
                slots.ensure(mx.array([0, 1, 2]))
        # Outside the scope, normal LRU resumes.
        slots.ensure(mx.array([2]))
        assert list(slots.slot_of) == [1, 2]
    finally:
        disk.close()


def test_verify_scope_preserves_forward(tmp_path):
    """Frozen recency changes slot assignment, never the arithmetic."""
    disk = _offloaded_disk(tmp_path)
    try:
        ffn = disk.language_model.layers[0].ffn
        mx.random.seed(0)
        x = mx.random.normal((1, 4, 32))
        plain = ffn(x, None)
        mx.eval(plain)
        with verify_scope():
            scoped = ffn(x, None)
            mx.eval(scoped)
        np.testing.assert_allclose(
            np.array(plain), np.array(scoped), rtol=1e-5, atol=1e-6
        )
    finally:
        disk.close()


def test_offload_dspark_validation_matrix():
    """Native MTP + offload is allowed only for deepseek_v41 (DSpark)."""
    base = {
        "moe_expert_offload_enabled": True,
        "moe_expert_offload_resident_fraction": 0.125,
    }
    validate_moe_expert_offload(
        {**base, "mtp_enabled": True}, model_type="deepseek_v41"
    )
    for extra in ("vlm_mtp_enabled", "dflash_enabled"):
        with pytest.raises(ValueError, match="cannot be combined"):
            validate_moe_expert_offload(
                {**base, "mtp_enabled": True, extra: True},
                model_type="deepseek_v41",
            )
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_moe_expert_offload(
            {**base, "mtp_enabled": True}, model_type="qwen3_8_next"
        )
    # Unknown type stays strict: the exception needs a known model.
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_moe_expert_offload({**base, "mtp_enabled": True})


def test_settings_construction_scopes_by_model_type():
    """Post-init enforces the same per-type rule as load/save."""
    from omlx.model_settings import ModelSettings

    ModelSettings(
        moe_expert_offload_enabled=True,
        moe_expert_offload_resident_fraction=0.125,
        mtp_enabled=True,
        model_type="deepseek_v41",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        ModelSettings(
            moe_expert_offload_enabled=True,
            mtp_enabled=True,
            model_type="qwen3_8_next",
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        ModelSettings(moe_expert_offload_enabled=True, mtp_enabled=True)


def test_backing_resize_preserves_arithmetic(tmp_path):
    """Grow/shrink change rooms, never the math (bit-exact gate)."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        assert backing.governor is None
        slots = disk.language_model.layers[0].ffn.experts.slots
        assert backing.capacity == 2
        backing.resize(6, 6)
        assert slots.cap == 6 and slots.rooms == 2  # lazy growth
        slots.ensure(mx.array([0, 1, 2, 3]))
        assert slots.rooms == 4  # grown to need, not to ceiling
        assert list(slots.slot_of) == [0, 1, 2, 3]
        backing.resize(2, 2)
        assert list(slots.slot_of) == [2, 3]  # newest survive
        assert slots.rooms == 2
        mx.random.seed(1)
        x = mx.random.normal((1, 4, 32))
        resident_out = None
        ffn = disk.language_model.layers[0].ffn
        with verify_scope():
            pass
        out = ffn(x, None)
        mx.eval(out)
        assert out.shape == (1, 4, 32)
    finally:
        disk.close()


def test_backing_stats_count_decode_only(tmp_path):
    """Hunger sees decode visits; prefill and verify are invisible."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots = disk.language_model.layers[0].ffn.experts.slots
        slots.ensure(mx.array([[0, 1]]))  # decode-shaped: counted
        assert backing.stats.decode_layers == 1
        slots.ensure(mx.array([[0, 1], [0, 1]]))  # prefill: skipped
        assert backing.stats.decode_layers == 1
        with verify_scope():
            slots.ensure(mx.array([[4, 5]]))  # verify: skipped
        assert backing.stats.decode_layers == 1
        assert backing.stats.decode_layers_missed >= 1
    finally:
        disk.close()


def test_parallel_fetch_matches_serial(tmp_path, monkeypatch):
    """Worker fetch must be bit-identical: ordered join, same LRU."""
    import os

    from test_deepseek_v41 import write_checkpoint

    from omlx.patches.deepseek_v41.loading import load as _load

    base = tmp_path / "shared"
    base.mkdir(parents=True, exist_ok=True)
    source, _ = write_checkpoint(
        base, vision=False, n_routed_experts=8, n_activated_experts=2
    )

    def _run(name, threads):
        disk, _ = _load(source, moe_expert_offload_resident_fraction=0.25)
        try:
            if threads is None:
                monkeypatch.delenv("OMLX_V41_FETCH_THREADS", raising=False)
            else:
                monkeypatch.setenv("OMLX_V41_FETCH_THREADS", threads)
            mx.random.seed(7)
            x = mx.random.normal((1, 4, 32))
            out = disk.language_model.layers[0].ffn(x, None)
            mx.eval(out)
            slots = disk.language_model.layers[0].ffn.experts.slots
            return np.array(out), list(slots.slot_of.items()), slots.misses
        finally:
            disk.close()
            monkeypatch.delenv("OMLX_V41_FETCH_THREADS", raising=False)

    serial = _run("serial", None)
    parallel = _run("parallel", "4")
    assert np.array_equal(parallel[0], serial[0])
    assert parallel[1] == serial[1]
    assert parallel[2] == serial[2]
    # Degenerate env values fall back to serial without crashing.
    _run("degenerate", "banana")


def test_backing_observe_acts_end_to_end(tmp_path):
    """Regression: observe() must not idle-skip on the backing.

    The generic governor gates on cache.capacity; the backing used to
    expose only base_cap, so every observe returned '' (actions stayed 0
    on real runs despite 96% stall). With tiny thresholds, free memory is
    abundant and observe must grow the per-layer base."""
    disk, backing = _backed_disk(tmp_path)
    try:
        mx.random.seed(11)
        x = mx.random.normal((1, 4, 32))
        mx.eval(disk.language_model.layers[0].ffn(x, None))
        bk = disk._expert_streaming_backing
        if bk is None:
            bk = backing
        gov = bk.governor
        assert gov is not None
        gov.low_free_bytes = 1
        gov.target_free_bytes = 2
        gov.high_free_bytes = 3
        gov.cooldown_s = 0
        before = bk.base_cap
        action = gov.observe()
        assert action != "", "observe idle-skipped on V41 backing"
        assert gov.actions == 1
        assert bk.base_cap > before
    finally:
        disk.close()


def test_backing_governor_shrink_and_grow(tmp_path):
    """Forced observe steps retarget per-layer caps and stay exact."""
    disk, backing = _backed_disk(tmp_path)
    try:
        assert backing.governor is not None
        gov = backing.governor
        slots = disk.language_model.layers[0].ffn.experts.slots
        # Fixture-scale floors: drop the RAM-sized defaults so one slot
        # of pressure/hunger moves the needle deterministically.
        gov.min_budget_bytes = 0
        gov.min_cap = 1
        gov.grow_add_frac = 1.0
        # Pressure: shrink halves the per-layer base.
        gov.target_free_bytes = 10**18
        gov.observe(force=True)
        assert backing.capacity == 1
        assert "shrink" in gov.last_action
        # Hunger: decode misses + headroom grow it back.
        slots.ensure(mx.array([[4, 4]]))
        slots.ensure(mx.array([[5, 5]]))
        gov.target_free_bytes = 0
        gov.low_free_bytes = 0
        gov.min_window_layers = 1
        gov.stall_target = 0.0
        gov.observe(force=True)
        assert backing.capacity == 2
        assert "grow" in gov.last_action
        assert gov.actions == 2
        # Targeting overrides are per-layer ceilings.
        backing.set_layer_caps({0: 4})
        assert backing.layer_cap_overrides() == {0: 4}
        assert slots.cap == 4
        backing.clear()
        assert len(slots.slot_of) == 0
    finally:
        disk.close()


def test_backing_layer_cap_noop_dropped(tmp_path):
    """The governor's num_layers==1 retarget sends {layer: base_cap} — an
    override that lands on the applied base is not targeting; dropping it
    keeps layer_cap_overrides() truthful."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        base = backing.base_cap
        backing.set_layer_caps({0: base})
        assert backing.layer_cap_overrides() == {}
        assert backing.slots_of[0].cap == base
        # A real per-layer ceiling still applies and reports.
        backing.set_layer_caps({1: base + 1})
        assert backing.layer_cap_overrides() == {1: base + 1}
        assert backing.slots_of[1].cap == base + 1
        assert backing.slots_of[0].cap == base
    finally:
        disk.close()


def test_backing_clear_cancels_staged(tmp_path):
    """clear() drops residency AND pending staged fetches — an in-flight
    mispredict cannot commit after a desperate-clear."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        slots1.ensure(mx.array([[6, 7]]))
        backing.note_routing(1, {4, 5})
        _prime_recall(backing, 1)
        slots0.ensure(mx.array([[0, 1]]))
        assert set(slots1._staged) == {4, 5}
        backing.clear()
        assert not slots1._staged
        assert len(slots1.slot_of) == 0
    finally:
        disk.close()


def test_backing_stage_next_prefetches_and_drops(tmp_path):
    """P8 staging: predicted next-layer set reads ahead, joins via the
    same commit path, and mispredicts drop without touching numerics."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        slots0.ensure(mx.array([[0, 1]]))  # decode-shaped: records + stages
        assert not slots1._staged  # no predictor yet
        # Inject a predictor for layer 1 whose set is NOT resident:
        # ensure() leaves prev_uniq == resident, so a realistic miss
        # needs a predictor seeded by routing history directly.
        slots1.ensure(mx.array([[6, 7]]))
        backing.note_routing(1, {4, 5})
        _prime_recall(backing, 1)
        slots0.ensure(mx.array([[2, 3]]))  # stages {4,5} on slots1
        assert set(slots1._staged) == {4, 5}
        # Demand ensure: misses join the staged futures (same payload,
        # same commit order — staged_hits counts the prefetch wins).
        rows = slots1.ensure(mx.array([[4, 5]]))
        mx.eval(rows)
        assert slots1.staged_hits == 2
        assert list(slots1.slot_of) == [4, 5]
        # Mispredict: stage {0,1} but demand {4,5}-evicting set {6,7}.
        backing.note_routing(1, {0, 1})
        slots0.ensure(mx.array([[0, 1]]))
        assert set(slots1._staged) == {0, 1}
        slots1.ensure(mx.array([[6, 7]]))
        assert slots1.staged_drops == 2
        assert list(slots1.slot_of) == [6, 7]
    finally:
        disk.close()


def test_backing_stage_disabled_env(tmp_path, monkeypatch):
    """OMLX_V41_STAGE=0 leaves the legacy demand-only path untouched."""
    monkeypatch.setenv("OMLX_V41_STAGE", "0")
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        backing.note_routing(1, {4, 5})
        slots0.ensure(mx.array([[0, 1]]))
        assert not slots1._staged
        rows = slots1.ensure(mx.array([[4, 5]]))
        mx.eval(rows)
        assert slots1.staged_hits == 0
        assert list(slots1.slot_of) == [4, 5]
    finally:
        disk.close()


def test_backing_stage_suppressed_without_headroom(tmp_path):
    """Starved regime (measured: drops 270 > hits 200 on the 48GB/422GB
    run): at the capacity floor or inside the clear band, speculative
    reads only waste saturated disk — the gate skips them."""
    disk, backing = _backed_disk(tmp_path)
    try:
        gov = backing.governor
        assert gov is not None
        gov.min_budget_bytes = 0
        gov.min_cap = 2  # = plan capacity = the fixture working set
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        # First ensure populates gov._last_free_gib via tick->observe.
        slots0.ensure(mx.array([[0, 1]]))
        backing.note_routing(1, {4, 5})
        _prime_recall(backing, 1)  # suppression below is headroom-only
        # At the floor: no slot can hold a misprediction -> suppressed.
        slots0.ensure(mx.array([[0, 1]]))
        assert not ({4, 5} & set(slots1._staged))
        assert backing.staged_skips == 1
        # Spare capacity: staging resumes.
        backing.resize(4, 4)
        slots0.ensure(mx.array([[2, 3]]))
        assert set(slots1._staged) == {4, 5}
        rows = slots1.ensure(mx.array([[4, 5]]))
        mx.eval(rows)
        assert slots1.staged_hits == 2
        # Desperate-free band suppresses even with spare capacity.
        gov.low_free_bytes = 10**18
        backing.note_routing(1, {0, 1})
        slots0.ensure(mx.array([[6, 7]]))
        assert not ({0, 1} & set(slots1._staged))
        assert backing.staged_skips == 2
    finally:
        disk.close()


def test_backing_staged_payload_preserves_arithmetic(tmp_path):
    """Rows committed from staged payloads are bit-equal to the
    resident model's expert weights — prefetch changes WHEN the read
    happens, never WHAT lands in the slot."""
    base = tmp_path / "arith"
    base.mkdir(parents=True, exist_ok=True)
    source, _ = write_checkpoint(
        base, vision=False, n_routed_experts=8,
        n_activated_experts=2,
    )
    resident, _ = load(source)
    disk, _ = load(source, moe_expert_offload_resident_fraction=0.25)
    try:
        layers = [
            (i, layer.ffn.experts.slots)
            for i, layer in enumerate(disk.language_model.layers)
        ]
        backing = V41StreamingBacking(
            disk._moe_offload_plan, layers, dynamic=False
        )
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        # Prime layer-1 residency away from the staged set, then let a
        # layer-0 ensure stage {4,5} into layer 1.
        slots1.ensure(mx.array([[6, 7]]))
        backing.note_routing(1, {4, 5})
        _prime_recall(backing, 1)
        slots0.ensure(mx.array([[0, 1]]))
        assert set(slots1._staged) == {4, 5}
        slots1.ensure(mx.array([[4, 5]]))
        assert slots1.staged_hits == 2
        resident_experts = resident.language_model.layers[1].ffn.experts
        for expert in (4, 5):
            slot = slots1.slot_of[expert]
            for proj in ("w1", "w3", "w2"):
                np.testing.assert_array_equal(
                    np.array(getattr(slots1.expert, proj).weight[slot]),
                    np.array(getattr(resident_experts, proj).weight[expert]),
                )
    finally:
        resident.close()
        disk.close()


def test_backing_stage_gated_by_recall(tmp_path):
    """Per-layer recall gate (generic stage_gate parity): a layer whose
    prev-token prediction never proved itself does not earn speculative
    reads; once the EWMA crosses the floor staging resumes."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        slots1.ensure(mx.array([[6, 7]]))
        backing.note_routing(1, {4, 5})  # predictor set, recall unproven
        slots0.ensure(mx.array([[0, 1]]))
        assert not slots1._staged
        assert backing.staged_skips == 1
        _prime_recall(backing, 1)
        slots0.ensure(mx.array([[2, 3]]))
        assert set(slots1._staged) == {4, 5}
    finally:
        disk.close()


def test_prefill_tail_chunks_are_not_decode(tmp_path):
    """Phase is decided once per call from the pre-chunk route count.

    Fixture capacity 2 / top_k 2 makes every prefill chunk a single row —
    the old per-chunk shape check scored each as a decode visit: governor
    stats, the prev_uniq predictor and staging all saw phantom decode."""
    disk, backing = _backed_disk(tmp_path, dynamic=False)
    try:
        mx.random.seed(3)
        x = mx.random.normal((1, 5, 32))
        mx.eval(disk.language_model.layers[0].ffn(x, None))
        assert backing.stats.decode_layers == 0
        assert not backing.prev_uniq
    finally:
        disk.close()


def test_offload_preserves_dspark_tensors_and_matches(tmp_path):
    """Offload + preserve_mtp loads the draft head and stays exact."""
    source, _ = write_checkpoint(
        tmp_path,
        vision=False,
        n_routed_experts=8,
        n_activated_experts=2,
        preserve_mtp=True,
        n_mtp_layers=3,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_n_routed_experts=2,
        dspark_n_activated_experts=1,
        dspark_markov_rank=32,
        compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
        temperature=0,
    )
    resident, _ = load(source, preserve_mtp=True)
    try:
        disk, _ = load(
            source,
            preserve_mtp=True,
            moe_expert_offload_resident_fraction=0.25,
        )
        try:
            # Draft head survived the offloaded load (the fixture stores
            # bare `mtp.*` keys, so draft_bytes — which counts production
            # `language_model.mtp.*` keys — stays 0 here by design).
            assert getattr(disk.language_model, "mtp", None)
            mx.random.seed(0)
            x = mx.random.normal((1, 4, 32))
            for layer in range(2):
                out_r = resident.language_model.layers[layer].ffn(x, None)
                mx.eval(out_r)
                with verify_scope():
                    out_d = disk.language_model.layers[layer].ffn(x, None)
                    mx.eval(out_d)
                np.testing.assert_allclose(
                    np.array(out_r), np.array(out_d), rtol=2e-4, atol=2e-4
                )
        finally:
            disk.close()
    finally:
        resident.close()
