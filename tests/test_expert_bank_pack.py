"""Expert-packed bank: pack -> attach -> read roundtrip (bit-exact).

Covers the v2 fused contract of omlx.patches.expert_streaming.expert_bank_pack:
records are byte-identical to the source component slices, ONE bank file
per layer carries every projection fused, the single-file layout parses
through the regular _ShardReader, and a stale source shard refuses attach
instead of serving a drifted layout.
"""

import json
import os
import struct

import numpy as np

from omlx.patches.expert_streaming.expert_bank_pack import (
    bank_status_for,
    pack_model,
)
from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore

_N = 4
_PREFIX = "language_model.layers.0.mlp.experts"
_LAYER_PREFIX = "language_model.layers.0.mlp.experts"
_DTYPE = {"F32": "<f4"}
# (weight_cols, scales_cols, biases_cols): 20/12-byte rows exercise the
# 8-byte record padding the typed views depend on.
_SHAPES = {"gate_up_proj": (10, 5, 3), "down_proj": (6, 3, 2)}


def _write_safetensors(path, tensors):
    header = {}
    blob = bytearray()
    for name, (arr, dtype) in tensors.items():
        data = np.ascontiguousarray(arr).astype(np.dtype(_DTYPE[dtype])).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + bytes(blob))


def _build_model(tmp_path):
    rng = np.random.default_rng(7)
    tensors = {}
    for proj, (w_cols, s_cols, b_cols) in _SHAPES.items():
        for kind, cols in (("weight", w_cols), ("scales", s_cols), ("biases", b_cols)):
            tensors[f"{_PREFIX}.{proj}.{kind}"] = (
                rng.standard_normal((_N, cols)).astype(np.float32),
                "F32",
            )
    # dense tensor: must stay out of the bank
    tensors["language_model.layers.0.mlp.shared_expert.gate_proj.weight"] = (
        rng.standard_normal((4, 4)).astype(np.float32),
        "F32",
    )
    shard = tmp_path / "model.safetensors"
    _write_safetensors(shard, tensors)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "num_experts": _N,
                "num_hidden_layers": 1,
                "num_experts_per_tok": 2,
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard.name for k in tensors}})
    )
    return tensors


def test_pack_read_roundtrip(tmp_path):
    _build_model(tmp_path)
    manifest = pack_model(tmp_path, verify=2)
    # v2: ONE group per LAYER, both projections fused in one file
    assert len(manifest["groups"]) == 1
    g = manifest["groups"][0]
    assert g["layout"] == "fused"
    assert g["prefix"] == _LAYER_PREFIX
    n_comps = sum(len(p) for p in [range(3), range(3)])  # 2 projs x 3 kinds
    assert len(g["components"]) == 6
    # dense/shared tensors never enter the bank
    for c in g["components"]:
        assert "shared_expert" not in c["key"]

    ok, why, _m = bank_status_for(tmp_path)
    assert ok, why

    store = ExpertBackingStore(tmp_path)
    try:
        ok, why, n_packed = store.attach_expert_bank()
        assert ok, why
        assert n_packed == 1

        pk, record_bytes = g["packed_key"], g["record_bytes"]
        reader = store._reader_for_key(pk)
        rp = reader._rp_for(pk)
        assert rp.num_experts == _N
        assert rp.expert_bytes == record_bytes

        # ONE fused read serves both projections of a 2-expert demand set
        ids = [1, 3]
        got = store.read_expert_bank_fused(_LAYER_PREFIX, ids)
        assert got is not None
        bank, rb = got
        assert rb == record_bytes
        assert bank.shape == (len(ids), record_bytes)

        for row_i, eid in enumerate(ids):
            for proj in _SHAPES:
                for kind in ("weight", "scales", "biases"):
                    key = f"{_PREFIX}.{proj}.{kind}"
                    _pk, off, nb = store.packed_component(key)
                    assert _pk == pk
                    src = store.load_expert_slice(key, eid)
                    view = np.frombuffer(
                        bank[row_i][off:off + nb], dtype=src.dtype
                    ).reshape(src.shape)
                    np.testing.assert_array_equal(view, src)

        # advisory through the FUSED packed key: one segment covering the
        # full records of a 2-expert run (source-projection keys keep
        # advising their original shard granularity by design).
        ok_adv, adv_bytes, adv_segs = store.advise_expert_run(pk, 0, 2)
        assert ok_adv
        assert adv_bytes == 2 * rp.expert_bytes
        assert adv_segs == 1
    finally:
        store.close()


def test_stale_source_refuses(tmp_path):
    _build_model(tmp_path)
    pack_model(tmp_path, verify=0)
    shard = tmp_path / "model.safetensors"
    st = shard.stat()
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    ok, why, _m = bank_status_for(tmp_path)
    assert not ok
    assert "changed" in why


def test_missing_bank_refuses(tmp_path):
    _build_model(tmp_path)
    ok, why, _m = bank_status_for(tmp_path)
    assert not ok
    assert "no manifest" in why


def test_idempotent_pack(tmp_path):
    _build_model(tmp_path)
    first = pack_model(tmp_path, verify=0)
    second = pack_model(tmp_path, verify=0)  # fresh manifest: no rewrite
    assert first == second


def test_repacked_checkpoint_refuses(tmp_path):
    """A checkpoint dir carrying expert_order.json is a repacked
    co-activation layout — serving it as identity would return permuted
    experts. The store refuses it loudly instead."""
    _build_model(tmp_path)
    (tmp_path / "expert_order.json").write_text(
        json.dumps({"format": 1, "layers": {"0": {"n": _N, "row_of": [3, 2, 1, 0]}}})
    )
    try:
        ExpertBackingStore(tmp_path)
        raise AssertionError("expected repacked checkpoint to be refused")
    except ValueError as e:
        assert "co-activation" in str(e)


def test_stale_ordered_bank_refuses(tmp_path):
    """A bank group written in co-activation order is a stale layout:
    attach refuses it rather than serve permuted rows as logical ids."""
    _build_model(tmp_path)
    manifest = pack_model(tmp_path, verify=0)
    bdir = tmp_path / ".omlx" / "expert_bank"
    mpath = bdir / "manifest.json"
    doc = json.loads(mpath.read_text())
    doc["format"] = 3
    doc["groups"][0]["row_of"] = [3, 2, 1, 0]
    mpath.write_text(json.dumps(doc))

    store = ExpertBackingStore(tmp_path)
    try:
        ok, why, _n = store.attach_expert_bank()
        assert not ok
        assert "co-activation" in why
    finally:
        store.close()
