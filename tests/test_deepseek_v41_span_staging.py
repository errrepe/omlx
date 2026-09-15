"""V4.1 staged span reads — converted-checkpoint staging coverage.

Every other staging test (test_deepseek_v41_dspark_offload.py) runs on
SOURCE checkpoints, where ``stage_predicted`` submits one ``_stage_one``
future per expert. Converted checkpoints take the coalesced path:
``_span_groups`` merges the predicted set into runs, one ``_stage_span``
future covers each run, and every expert's ``_StagedSpan`` handle slices
its payload out of the shared block via ``block[expert - lo]``. These
tests exercise that fan-out — including a gap-merged run where an
off-by-one in the row offset would commit a different expert's bytes.
"""

import mlx.core as mx
import numpy as np
from test_deepseek_v41 import write_checkpoint

from omlx.patches.deepseek_v41.convert import convert
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.deepseek_v41.moe_offload import _stage_span
from omlx.patches.deepseek_v41.streaming_backing import V41StreamingBacking


def _converted_disk(tmp_path, fraction=0.25):
    """Tiny checkpoint -> convert() -> offload load (converted path)."""
    source, _ = write_checkpoint(
        tmp_path, vision=False, n_routed_experts=8, n_activated_experts=2
    )
    target = tmp_path / "converted"
    convert(source, target)
    disk, _ = load(target, moe_expert_offload_resident_fraction=fraction)
    assert disk._moe_offload_plan.converted is not None
    return disk


def _backing(disk):
    layers = [
        (i, layer.ffn.experts.slots)
        for i, layer in enumerate(disk.language_model.layers)
    ]
    return V41StreamingBacking(disk._moe_offload_plan, layers, dynamic=False)


def test_stage_span_payload_rows_match_per_expert_fetch(tmp_path):
    """``_stage_span`` row math directly: each expert's payload sliced
    out of the merged block must equal the per-expert fetch — expert 3
    reads block[0], expert 5 reads block[2] of the [3,6) span."""
    disk = _converted_disk(tmp_path)
    try:
        plan = disk._moe_offload_plan
        prefix = "language_model.layers.1.ffn.experts"
        got = _stage_span(plan, prefix, [3, 5])
        assert set(got) == {3, 5}
        for expert in (3, 5):
            for proj in ("w1", "w3", "w2"):
                want = plan.fetch(prefix, proj, expert)["weight"]
                np.testing.assert_array_equal(
                    np.asarray(
                        got[expert][proj]["weight"].astype(mx.float32)
                    ),
                    np.asarray(want.astype(mx.float32)),
                )
    finally:
        disk.close()


def test_backing_staged_span_preserves_arithmetic(tmp_path, monkeypatch):
    """Converted-checkpoint analogue of
    test_backing_staged_payload_preserves_arithmetic: rows committed
    from a gap-merged staged span are bit-equal to the resident model's
    expert weights."""
    monkeypatch.setenv("OMLX_V41_SPAN_GAP", "2")  # {3, 5} merges into one span
    disk = _converted_disk(tmp_path)
    resident, _ = load(tmp_path / "converted")
    try:
        backing = _backing(disk)
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        # Prime layer-1 residency away from the staged set, then let a
        # layer-0 ensure stage {3,5} into layer 1: one merged _stage_span
        # over rows [3,6), two _StagedSpan handles on a single future.
        slots1.ensure(mx.array([[6, 7]]))
        backing.note_routing(1, {3, 5})
        backing.recall_ewma[1] = 1.0  # skip EWMA warm-up (recall gate)
        slots0.ensure(mx.array([[0, 1]]))
        assert set(slots1._staged) == {3, 5}
        assert slots1._staged[3]._fut is slots1._staged[5]._fut
        slots1.ensure(mx.array([[3, 5]]))
        assert slots1.staged_hits == 2
        resident_experts = resident.language_model.layers[1].ffn.experts
        for expert in (3, 5):
            slot = slots1.slot_of[expert]
            for proj in ("w1", "w3", "w2"):
                np.testing.assert_array_equal(
                    np.array(getattr(slots1.expert, proj).weight[slot]),
                    np.array(getattr(resident_experts, proj).weight[expert]),
                )
    finally:
        resident.close()
        disk.close()


def test_staged_span_multi_run_drops_undemanded(tmp_path, monkeypatch):
    """A predicted set spanning several merged runs maps each run to one
    future; experts never demanded drop and cancel like the per-expert
    path."""
    monkeypatch.setenv("OMLX_V41_SPAN_GAP", "1")  # strictly-adjacent merge
    disk = _converted_disk(tmp_path)
    try:
        backing = _backing(disk)
        slots0 = disk.language_model.layers[0].ffn.experts.slots
        slots1 = disk.language_model.layers[1].ffn.experts.slots
        slots1.ensure(mx.array([[6, 7]]))
        # {0,1} and {4,5} are two separate runs under gap=1.
        backing.note_routing(1, {0, 1, 4, 5})
        backing.recall_ewma[1] = 1.0
        slots0.ensure(mx.array([[2, 3]]))
        assert set(slots1._staged) == {0, 1, 4, 5}
        # One future per merged run, shared by that run's handles.
        assert slots1._staged[0]._fut is slots1._staged[1]._fut
        assert slots1._staged[4]._fut is slots1._staged[5]._fut
        assert slots1._staged[0]._fut is not slots1._staged[4]._fut
        rows = slots1.ensure(mx.array([[0, 1]]))
        mx.eval(rows)
        assert slots1.staged_hits == 2
        assert slots1.staged_drops == 2  # {4, 5} predicted, not demanded
        assert list(slots1.slot_of) == [0, 1]
    finally:
        disk.close()
