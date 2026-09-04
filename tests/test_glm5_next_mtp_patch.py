# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5.3 (glm5_next, mlx-vlm vendored) native MTP patch."""

import sys

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_glm5_next_compat as compat
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active


@pytest.fixture(scope="module")
def glm():
    compat.apply_mlx_vlm_glm5_next_compat_patch()
    apply_mlx_lm_mtp_patch()
    return sys.modules["mlx_vlm.models.glm5_next"]


@pytest.fixture()
def mtp_active():
    set_mtp_active(True)
    yield
    set_mtp_active(False)


def _headful_text_config():
    from tests.test_mlx_vlm_glm5_next_compat import _tiny_config

    text = _tiny_config(with_vision=False).text_config
    text.num_nextn_predict_layers = 1
    text.layer_types = [
        "linear_attention",
        "deepseek_sparse_attention",
        "deepseek_sparse_attention",
    ]
    text.mlp_layer_types = ["dense", "dense", "sparse"]
    text.first_k_dense_replace = 0
    text.n_routed_experts = 4
    text.n_shared_experts = 1
    return text


class TestPatchApply:
    def test_patch_applies_and_noops_without_module(self, glm):
        from omlx.patches.mlx_lm_mtp import glm5_next_model

        assert glm5_next_model.apply() is True
        # Marker present on the patched class.
        assert glm.LanguageModel.__dict__.get("_omlx_mtp_patched") == "patch"

    def test_headless_init_leaves_model_stock(self, glm):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        assert getattr(lm, "mtp", None) is None
        assert lm._omlx_mtp_decode_enabled is False
        assert not hasattr(lm, "_omlx_mtp_chain")


class TestHeadAttach:
    def test_active_load_attaches_chain_markers(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        assert lm.mtp is not None and len(lm.mtp) == 1
        assert lm._omlx_mtp_chain is True
        assert lm._omlx_mtp_depth >= 1
        assert lm._omlx_mtp_head_clone is False
        assert lm._omlx_mtp_head_prenorm is True
        assert lm._omlx_mtp_decode_enabled is True

    def test_block_layout_matches_nextn_surface(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        block = lm.mtp[0]
        assert hasattr(block, "enorm") and hasattr(block, "hnorm")
        assert hasattr(block, "eh_proj") and hasattr(block, "norm")
        # Draft runs the full sparse-attention decoder layer (DSA indexer
        # + 288-expert MoE in the real checkpoint).
        assert type(block.block).__name__ == "Glm5NextDecoderLayer"
        assert block.block.is_linear is False
        assert type(block.block.mlp).__name__ == "Glm5NextMoE"

    def test_mtp_cache_is_cache_list_with_pooling(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        caches = lm.make_mtp_cache()
        assert len(caches) == 1
        assert type(caches[0]).__name__ == "CacheList"
        assert len(caches[0].caches) == 2
        assert type(caches[0].caches[0]).__name__ == "KVCache"


class TestForward:
    def test_return_hidden_logits_bit_identical_to_stock(self, glm, mtp_active):
        text = _headful_text_config()
        mx.random.seed(11)
        lm = glm.LanguageModel(text)
        mx.eval(lm.parameters())
        tokens = mx.array([[1, 2, 3, 4, 5]])
        plain = lm(tokens, cache=None, num_logits_to_keep=2)
        rh = lm(tokens, cache=lm.make_cache(), return_hidden=True, num_logits_to_keep=2)
        assert mx.all(mx.equal(plain.logits, rh.logits)).item() is True
        # hidden is the pre-norm variant: applying the trunk's final norm
        # re-derives the post-norm hidden the stock path projects.
        post = lm.model.norm(rh.hidden_states)
        assert post.shape == rh.hidden_states.shape

    def test_mtp_forward_shapes_and_finiteness(self, glm, mtp_active):
        text = _headful_text_config()
        mx.random.seed(13)
        lm = glm.LanguageModel(text)
        mx.eval(lm.parameters())
        tokens = mx.array([[1, 2, 3, 4]])
        out = lm(tokens, cache=lm.make_cache(), return_hidden=True)
        h = out.hidden_states[:, -1:, :]
        caches = lm.make_mtp_cache()
        logits, head_hidden = lm.mtp_forward(
            h, mx.array([[5]]), caches, return_hidden=True
        )
        assert logits.shape == (1, 1, text.vocab_size)
        assert head_hidden.shape == (1, 1, text.hidden_size)
        assert mx.all(mx.isfinite(logits)).item() is True
        # Chained depth-2 step: the head's raw output feeds back as h.
        logits2, h2 = lm.mtp_forward(
            head_hidden, mx.array([[6]]), caches, return_hidden=True
        )
        assert logits2.shape == (1, 1, text.vocab_size)
        assert mx.all(mx.isfinite(logits2)).item() is True

    def test_get_mtp_module_via_adapter_contract(self, glm, mtp_active):
        text = _headful_text_config()
        lm = glm.LanguageModel(text)
        assert lm.get_mtp_module() is lm.mtp
        # get_mtp_module is a plain attr reader: delete the head and it
        # reports None (the adapter's VLMModelAdapter.mtp contract).
        del lm.mtp
        assert lm.get_mtp_module() is None


class TestQuantOverrideRemap:
    def test_nextn_overrides_copied_to_runtime_paths(self, glm, mtp_active):
        from omlx.patches.mlx_lm_mtp.glm5_next_model import (
            remap_mtp_quant_overrides,
        )

        three_bit = {"group_size": 128, "bits": 2}
        params = {
            "quantization": {
                "model.layers.2.mlp.switch_mlp.down_proj": dict(three_bit),
                "model.layers.2.eh_proj": dict(three_bit),
                "model.layers.2.self_attn.indexer.wk": {"group_size": 64, "bits": 8},
                "model.layers.2.shared_head.head": {"group_size": 64, "bits": 4},
                "model.layers.1.mlp.switch_mlp.down_proj": dict(three_bit),
            }
        }
        remap_mtp_quant_overrides(params, n_main=2, n_mtp=1)
        q = params["quantization"]
        assert q["mtp.0.block.mlp.switch_mlp.down_proj"] == three_bit
        assert q["mtp.0.eh_proj"] == three_bit
        assert q["mtp.0.block.self_attn.indexer.wk"] == {
            "group_size": 64,
            "bits": 8,
        }
        # Shared lm_head duplicate dropped; no runtime copy.
        assert not any("shared_head" in k for k in q if k.startswith("mtp."))
        # Backbone overrides untouched.
        assert "model.layers.1.mlp.switch_mlp.down_proj" in q
        assert not any("layers.1" in k for k in q if k.startswith("mtp."))

    def test_no_quant_block_is_inert(self, glm, mtp_active):
        from omlx.patches.mlx_lm_mtp.glm5_next_model import (
            remap_mtp_quant_overrides,
        )

        params = {"quantization": None}
        remap_mtp_quant_overrides(params, n_main=2, n_mtp=1)  # must not raise
        assert params["quantization"] is None


class TestPartialRollback:
    def test_trim_shortfall_refuses(self, glm, mtp_active):
        class _ShortCache:
            def is_trimmable(self):
                return True

            def trim(self, n):
                return 0

        lm = glm.LanguageModel(_headful_text_config())
        assert lm.mtp_partial_rollback([_ShortCache()], accepted=0, num_drafts=2) is False

    def test_no_rollback_needed_accepts(self, glm, mtp_active):
        lm = glm.LanguageModel(_headful_text_config())
        assert lm.mtp_partial_rollback([], accepted=2, num_drafts=2) is True

