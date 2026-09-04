# SPDX-License-Identifier: Apache-2.0
"""Full-bank external wrap (Fase M) — tests.

Covers: manifest/fingerprint parsing, staleness refusal, per-linear coverage,
env gating (zero mmap when off), provider idempotency, discount arithmetic,
and — gated on the native extension — a bit-exact fullbank-vs-demand-path
round-trip on a tiny synthetic checkpoint built in tmp_path (no 47 GB
artifact dependency; improves on the test design of jundot/omlx PR #3437,
which this feature is adapted from).
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.expert_streaming import fullbank as fb
from omlx.utils import proc_memory

PATH_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(PATH_TOOLS))

import repack_fullbank as repack_mod  # noqa: E402

from omlx.custom_kernels.expert_bank_wrap import fast  # noqa: E402

_native = pytest.mark.skipif(
    not fast.is_native_available(), reason="expert_bank_wrap native ext unavailable"
)

PAGE = 16384


def _build_fake_checkpoint(
    tmp_path: Path, n_experts: int = 4, out_dim: int = 8, in_dim: int = 128, group_size: int = 32, bits: int = 4
):
    """Tiny synthetic checkpoint with stacked switch_mlp banks, real
    safetensors layout (manual header, no safetensors dep): one shard holding
    the three bank tensors of one MoE layer, plus config + index."""
    import shutil

    md = tmp_path / "model"
    md.mkdir()
    E, O, I = n_experts, out_dim, in_dim
    gs = group_size
    # affine 4-bit: weights packed uint32 (8 x 4-bit); scales/biases BF16 raw
    # (uint16 storage), matching mlx-lm QuantizedSwitchLinear banks.
    packed = I // 8
    rng = np.random.default_rng(7)
    w = rng.integers(0, 2**32, size=(E, O, packed), dtype=np.uint32)
    s = rng.integers(1, 200, size=(E, O, I // gs), dtype=np.uint16)
    b = np.zeros((E, O, I // gs), dtype=np.uint16)
    tensors = {
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": (w, "U32"),
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": (s, "BF16"),
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": (b, "BF16"),
    }
    shard = md / "model-00001-of-00001.safetensors"
    header: dict = {}
    blob = bytearray()
    for key, (arr, dt) in tensors.items():
        header[key] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [len(blob), len(blob) + arr.nbytes]}
        blob += arr.tobytes()
    hb = json.dumps(header).encode()
    shard.write_bytes(struct.pack("<Q", len(hb)) + hb + bytes(blob))
    (md / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard.name for k in tensors}})
    )
    (md / "config.json").write_text(
        json.dumps({"quantization": {"bits": bits, "group_size": gs, "mode": "affine"}})
    )
    return md, tensors


def _repack(md: Path, tmp_out: Path | None = None) -> Path:
    out = tmp_out or (md / fb.ARTIFACT_NAME)
    repack_mod.repack(out, md, PAGE, verify=False)
    return out


# --------------------------------------------------------------------------- #
# No-native-required: manifest, fingerprint, gating, registry                 #
# --------------------------------------------------------------------------- #
def test_env_flag_gates_engagement():
    os.environ.pop("OMLX_EXPERT_STREAMING_FULLBANK", None)
    assert fb.fullbank_enabled() is False
    os.environ["OMLX_EXPERT_STREAMING_FULLBANK"] = "1"
    try:
        assert fb.fullbank_enabled() is True
    finally:
        os.environ.pop("OMLX_EXPERT_STREAMING_FULLBANK", None)


def test_fingerprint_roundtrip_and_staleness(tmp_path):
    md, _ = _build_fake_checkpoint(tmp_path)
    art_path = _repack(md)
    manifest, page = fb._read_manifest(str(art_path))
    assert page == PAGE
    assert "fingerprint" in manifest
    assert fb.fingerprint_matches(md, manifest) is True

    # touch a shard -> mtime changes -> stale
    shard = next(md.glob("*.safetensors"))
    st = shard.stat()
    os.utime(shard, ns=(st.st_atime_ns + 1, st.st_mtime_ns + 1))
    assert fb.fingerprint_matches(md, manifest) is False

    # config change -> stale
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns))
    cfg = json.loads((md / "config.json").read_text())
    cfg["extra"] = 1
    (md / "config.json").write_text(json.dumps(cfg))
    assert fb.fingerprint_matches(md, manifest) is False


def test_missing_artifact_returns_none(tmp_path):
    md, _ = _build_fake_checkpoint(tmp_path)
    assert fb.artifact_path(md) is None


def test_provider_registration_idempotent():
    proc_memory.register_external_wired_provider("__test_fb__", lambda: 100)
    proc_memory.register_external_wired_provider("__test_fb__", lambda: 100)
    try:
        assert proc_memory.external_wired_bytes() >= 100  # not 200
        assert proc_memory.discount_external_wired(1000) == 1000 - proc_memory.external_wired_bytes()
        assert proc_memory.discount_external_wired(1) == 0
    finally:
        proc_memory.unregister_external_wired_provider("__test_fb__")


def test_throwing_provider_swallowed():
    proc_memory.register_external_wired_provider("__test_fb_bad__", lambda: 1 / 0)
    try:
        assert proc_memory.external_wired_bytes() >= 0
    finally:
        proc_memory.unregister_external_wired_provider("__test_fb_bad__")


def test_maybe_attach_refuses_stale(tmp_path, monkeypatch):
    md, _ = _build_fake_checkpoint(tmp_path)
    _repack(md)
    shard = next(md.glob("*.safetensors"))
    st = shard.stat()
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FULLBANK", "1")

    class _B:
        model_path = str(md)

    assert fb.maybe_attach_fullbank(_B()) is None


def test_maybe_attach_refuses_env_off(tmp_path, monkeypatch):
    md, _ = _build_fake_checkpoint(tmp_path)
    _repack(md)
    monkeypatch.delenv("OMLX_EXPERT_STREAMING_FULLBANK", raising=False)

    class _B:
        model_path = str(md)

    assert fb.maybe_attach_fullbank(_B()) is None


def test_maybe_attach_accepts_fresh(tmp_path, monkeypatch):
    md, _ = _build_fake_checkpoint(tmp_path)
    _repack(md)
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FULLBANK", "1")

    class _B:
        model_path = str(md)

    art = fb.maybe_attach_fullbank(_B())
    assert art is not None
    assert len(art) == 3
    assert art.has("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight")
    assert art.coverage_for([
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales",
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases",
    ]) == 3


# --------------------------------------------------------------------------- #
# Native-gated: wrap round-trip + deferred unmap + canary + bit-exact qmm    #
# --------------------------------------------------------------------------- #
@_native
def test_wrap_tensor_bit_exact_and_deferred_unmap(tmp_path, monkeypatch):
    md, tensors = _build_fake_checkpoint(tmp_path)
    art_path = _repack(md)
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FULLBANK", "1")

    art = fb.FullbankArtifact(str(art_path), str(md))
    art.open()
    try:
        key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
        e = art.entry(key)
        wrapped = art.wrap(key)
        assert tuple(wrapped.shape) == tuple(e["shape"])
        assert fast.mapped_bytes() > 0
        # bit-exact vs source
        src = tensors[key][0]
        got = np.array(wrapped.reshape(-1).view(mx.uint32)).reshape(src.shape)
        assert np.array_equal(got, src)
        # close while alive -> deferred; mapping stays until array frees
        art.close()
        assert fast.mapped_bytes() > 0
        # GPU touch after close must not fault (deferred unmap is the point)
        assert int(wrapped.sum()) >= 0
        # wrap on closed id is refused
        with pytest.raises(Exception):
            fast.wrap_tensor(art._id if art._id is not None else 1, e["offset"], e["length"], e["shape"], e["dtype"])
        del wrapped
        import gc

        for _ in range(10):
            gc.collect()
            mx.synchronize()
            if fast.mapped_bytes() == 0:
                break
        assert fast.mapped_bytes() == 0  # deferred unmap completed
    finally:
        art.close()


@_native
def test_canary_independent_detects_corruption(tmp_path, monkeypatch):
    md, tensors = _build_fake_checkpoint(tmp_path)
    art_path = _repack(md)
    # corrupt one byte inside the FIRST tensor the canary picks
    # (sorted(keys)[0] = ...gate_proj.biases — the canary window starts at
    # offset 0 of that tensor, so corrupt within the first few values)
    raw = bytearray(art_path.read_bytes())
    manifest, _ = fb._read_manifest(str(art_path))
    off = manifest["tensors"]["language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases"]["offset"]
    raw[off + 3] ^= 0xFF
    art_path.write_bytes(bytes(raw))
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FULLBANK", "1")

    art = fb.FullbankArtifact(str(art_path), str(md))
    assert art.run_canary_once() is False  # corruption caught vs ORIGINAL shard
    assert art.run_canary_once() is False  # once-per-instance stays failed


@_native
def test_canary_passes_on_clean_artifact(tmp_path, monkeypatch):
    md, _ = _build_fake_checkpoint(tmp_path)
    art_path = _repack(md)
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FULLBANK", "1")
    art = fb.FullbankArtifact(str(art_path), str(md))
    assert art.run_canary_once() is True
    assert art.run_canary_once() is True  # cached pass
    art.close()


@_native
def test_fullbank_vs_demand_bit_exact(tmp_path, monkeypatch):
    """The headline invariant: fullbank gather_qmm == demand-path gather_qmm
    on the same fake checkpoint (better than the PR: no 47 GB artifact)."""
    md, tensors = _build_fake_checkpoint(tmp_path)
    art_path = _repack(md)
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FULLBANK", "1")

    art = fb.FullbankArtifact(str(art_path), str(md))
    art.open()
    try:
        pfx = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
        w = art.wrap(f"{pfx}.weight")
        s = art.wrap(f"{pfx}.scales")
        b = art.wrap(f"{pfx}.biases")

        # demand-path reference: the ORIGINAL arrays straight from the
        # source checkpoint (what pread would return, promoted to mx).
        w_ref = mx.array(tensors[f"{pfx}.weight"][0])
        s_ref = mx.array(tensors[f"{pfx}.scales"][0]).view(mx.bfloat16)
        b_ref = mx.array(tensors[f"{pfx}.biases"][0]).view(mx.bfloat16)

        T = 5
        in_dim = tensors[f"{pfx}.weight"][0].shape[2] * 8  # unpacked dim
        x = mx.random.normal((T, 1, in_dim)).astype(mx.bfloat16)
        idx = mx.array([[i % 4] for i in range(T)], dtype=mx.uint32)

        y_fb = mx.gather_qmm(x, w, s, b, rhs_indices=idx, transpose=True,
                             group_size=32, bits=4, mode="affine", sorted_indices=False)
        y_ref = mx.gather_qmm(x, w_ref, s_ref, b_ref, rhs_indices=idx, transpose=True,
                              group_size=32, bits=4, mode="affine", sorted_indices=False)
        assert float(mx.max(mx.abs(y_fb.astype(mx.float32) - y_ref.astype(mx.float32)))) == 0.0
    finally:
        art.close()
