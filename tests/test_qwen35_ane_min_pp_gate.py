"""Tests for the ANE prefill min-PP production gate (Phase 2).

Covers the MLX-free decision helper and the per-module latch toggle that the
engine uses to soft-gate the opt-in ANE prefill backends per request.
"""

import pytest

from omlx.ane_prefill_gate import ane_prefill_should_engage


# --------------------------------------------------------------------------- #
# Decision logic (no MLX dependency)
# --------------------------------------------------------------------------- #
def test_gate_disabled_when_threshold_zero():
    assert ane_prefill_should_engage(0, 0) is True
    assert ane_prefill_should_engage(10, 0) is True
    assert ane_prefill_should_engage(99999, 0) is True


def test_gate_disabled_when_threshold_none():
    assert ane_prefill_should_engage(5, None) is True


def test_gate_engages_at_and_above_threshold():
    threshold = 2048
    assert ane_prefill_should_engage(2047, threshold) is False
    assert ane_prefill_should_engage(2048, threshold) is True
    assert ane_prefill_should_engage(2049, threshold) is True
    assert ane_prefill_should_engage(8192, threshold) is True


def test_gate_coerces_string_tokens():
    # The engine may pass the raw token count; guard against accidental str.
    assert ane_prefill_should_engage("2048", 2048) is True
    assert ane_prefill_should_engage("100", 2048) is False


# --------------------------------------------------------------------------- #
# Latch toggle (needs the real patch, which imports mlx)
# --------------------------------------------------------------------------- #
class _FakeModule:
    def __init__(self, **state_attrs):
        for k, v in state_attrs.items():
            setattr(self, k, v)


class _FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


def _make_model():
    return _FakeModel(
        [
            _FakeModule(_omlx_ane_prefill_state=object()),   # -> _omlx_ane_prefill_failed
            _FakeModule(_omlx_ane_gdn_state=object()),       # -> _omlx_ane_gdn_failed
            _FakeModule(_omlx_ane_oproj_state=object()),     # -> _omlx_ane_oproj_failed
            _FakeModule(_omlx_ane_fused_down_state=object()),  # -> _omlx_ane_prefill_failed
            _FakeModule(),  # no ANE state -> untouched
        ]
    )


def test_set_skip_toggles_latches():
    try:
        from omlx.patches.qwen35_ane_prefill import set_qwen35_ane_prefill_skip
    except Exception:
        pytest.skip("qwen35_ane_prefill patch requires mlx (hardware)")

    model = _make_model()
    toggled = set_qwen35_ane_prefill_skip(model, True)
    assert toggled == 4
    assert model._modules[0]._omlx_ane_prefill_failed is True
    assert model._modules[1]._omlx_ane_gdn_failed is True
    assert model._modules[2]._omlx_ane_oproj_failed is True
    assert model._modules[3]._omlx_ane_prefill_failed is True
    # module with no ANE state is untouched
    assert not hasattr(model._modules[4], "_omlx_ane_prefill_failed")

    # Re-enabling restores ANE dispatch instantly (no recompile).
    toggled = set_qwen35_ane_prefill_skip(model, False)
    assert toggled == 4
    assert model._modules[0]._omlx_ane_prefill_failed is False
    assert model._modules[1]._omlx_ane_gdn_failed is False
    assert model._modules[2]._omlx_ane_oproj_failed is False
    assert model._modules[3]._omlx_ane_prefill_failed is False


def test_set_skip_idempotent():
    try:
        from omlx.patches.qwen35_ane_prefill import set_qwen35_ane_prefill_skip
    except Exception:
        pytest.skip("qwen35_ane_prefill patch requires mlx (hardware)")

    model = _make_model()
    set_qwen35_ane_prefill_skip(model, True)
    # Second call with same value must not double-count / must stay consistent.
    toggled = set_qwen35_ane_prefill_skip(model, True)
    assert toggled == 4
    assert model._modules[0]._omlx_ane_prefill_failed is True
