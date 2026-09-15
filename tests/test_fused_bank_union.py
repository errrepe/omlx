"""End-to-end fused-bank integration test (Cherenkov steal, v2).

Packs a tiny fixture model, attaches the fused bank, runs the union
context (_LayerLoadContext._ensure_union) over a demand set, and proves:
  * the fused path engages (one read per layer, not one per projection),
  * rows are bit-identical to the source per-expert slices,
  * bundles carry the full demand set (cached + fused-read),
  * the writeback path stores fused-derived rows under the same keys.
"""

import json
import struct

import numpy as np
import pytest

from omlx.patches.expert_streaming.expert_bank_pack import pack_model
from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
from omlx.patches.expert_streaming.streaming_switch import (
    StreamingQuantizedSwitchLinear,
    _LayerLoadContext,
)

_N = 6
_PREFIX = "language_model.layers.0.mlp.switch_mlp"
_PROJS = ("gate_proj", "up_proj", "down_proj")
_DTYPE = {"F32": "<f4"}


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


@pytest.fixture()
def fixture_model(tmp_path):
    rng = np.random.default_rng(11)
    tensors = {}
    for proj in _PROJS:
        for kind, cols in (("weight", 9), ("scales", 3), ("biases", 2)):
            tensors[f"{_PREFIX}.{proj}.{kind}"] = (
                rng.standard_normal((_N, cols)).astype(np.float32),
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


class _TinyCache:
    """ExpertLRUCache stand-in with just the contract the context uses."""

    def __init__(self):
        self.store = {}
        self.stats = type("S", (), {
            "decode_hits": 0, "decode_misses": 0,
            "prefill_hits": 0, "prefill_misses": 0,
        })()
        self._fallbacks = {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, value):
        self.store[key] = value

    def _count_ctx_fallback(self, why):
        self._fallbacks[why] = self._fallbacks.get(why, 0) + 1


def test_ensure_union_fused_end_to_end(fixture_model, tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FUSED_BANK", "1")
    import omlx.patches.expert_streaming.streaming_switch as ss

    ss._FUSED_BANK_ENV = True
    pack_model(tmp_path, verify=2)
    store = ExpertBackingStore(tmp_path)
    cache = _TinyCache()
    linears = []
    try:
        ok, why, n = store.attach_expert_bank()
        assert ok, why
        assert n == 1

        for proj in _PROJS:
            linears.append(
                StreamingQuantizedSwitchLinear(
                    layer_idx=0,
                    proj_name=proj,
                    stacked_weight_key=f"{_PREFIX}.{proj}.weight",
                    stacked_scales_key=f"{_PREFIX}.{proj}.scales",
                    stacked_biases_key=f"{_PREFIX}.{proj}.biases",
                    num_experts=_N,
                    input_dims=4,
                    output_dims=8,
                    backing=store,
                    cache=cache,
                )
            )
        ctx = _LayerLoadContext(linears, cache, mode="union", positions=2)

        # 3 projections, 3 ids demanded: the fused path must produce a
        # complete bundle per projection in ONE pass.
        demand = [1, 4, 2]
        for _lin in linears:

            ctx._ensure_union(_lin, demand)

        assert not ctx.failed, "context must not fail"
        assert not ctx.declined, "union must not decline"
        for lin in linears:
            bundle = ctx.bundles.get(id(lin))
            assert bundle is not None and len(bundle) == len(demand)
            for eid in demand:
                w, s, b = bundle[int(eid)]
                src_w = store.load_expert_slice(lin.stacked_weight_key, int(eid))
                src_s = store.load_expert_slice(lin.stacked_scales_key, int(eid))
                src_b = store.load_expert_slice(lin.stacked_biases_key, int(eid))
                np.testing.assert_array_equal(w, src_w)
                np.testing.assert_array_equal(s, src_s)
                np.testing.assert_array_equal(b, src_b)

        # ONE fused read happened (telemetry): the read path produced rows
        # for the union of missing ids across all projections in one call.
        assert cache._fallbacks.get("fused_read_l0", 0) == 0, cache._fallbacks
        # the fused path genuinely engaged: prefix resolved on every
        # projection (a gate failure would have left it None and taken
        # the per-projection path silently).
        for lin in linears:
            assert lin._fused_prefix_cache == _PREFIX, lin._fused_prefix_cache
        assert getattr(ctx, "_demand_counted", False)

        # partial-cache: prime one expert, ensure the read skips it
        primed = 4
        cached_row = (
            store.load_expert_slice(linears[0].stacked_weight_key, primed),
            store.load_expert_slice(linears[0].stacked_scales_key, primed),
            store.load_expert_slice(linears[0].stacked_biases_key, primed),
        )
        cache.store[linears[0].bundle_key(primed)] = cached_row
        ctx2 = _LayerLoadContext(linears, cache, mode="union", positions=2)
        for _lin in linears:

            ctx2._ensure_union(_lin, [2, primed])
        assert not ctx2.failed
        b0 = ctx2.bundles[id(linears[0])]
        assert b0[int(primed)][0] is cached_row[0], "cached row reused"
        assert int(primed) in b0 and 2 in b0
    finally:
        store.close()


def test_ensure_union_fused_hit_aware(fixture_model, tmp_path, monkeypatch):
    """Partial-miss experts skip the fused record; full-miss experts use it.

    E1-E3 post-mortem policy: priming ONE projection of an expert makes it
    a partial miss, so the fused path must read only that projection's
    missing components via the per-projection bank read -- never the full
    2.64 MiB-style record -- while a fully-missing expert still rides the
    single fused record.
    """
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FUSED_BANK", "1")
    import omlx.patches.expert_streaming.streaming_switch as ss

    ss._FUSED_BANK_ENV = True
    pack_model(tmp_path, verify=2)
    store = ExpertBackingStore(tmp_path)
    cache = _TinyCache()
    linears = []
    try:
        ok, why, n = store.attach_expert_bank()
        assert ok, why
        assert n == 1

        for proj in _PROJS:
            linears.append(
                StreamingQuantizedSwitchLinear(
                    layer_idx=0,
                    proj_name=proj,
                    stacked_weight_key=f"{_PREFIX}.{proj}.weight",
                    stacked_scales_key=f"{_PREFIX}.{proj}.scales",
                    stacked_biases_key=f"{_PREFIX}.{proj}.biases",
                    num_experts=_N,
                    input_dims=4,
                    output_dims=8,
                    backing=store,
                    cache=cache,
                )
            )

        fused_reads: list[list[int]] = []
        per_proj_reads: list[tuple[str, list[int]]] = []
        lead = linears[0]

        orig_fused = lead._read_fused_rows

        def spy_fused_rows(expert_ids, keys=None):
            fused_reads.append(list(expert_ids))
            return orig_fused(expert_ids, keys)

        def make_np_spy(lin):
            orig = lin._load_expert_bank_np

            def spy(ids):
                per_proj_reads.append((lin.proj_name, list(ids)))
                return orig(ids)

            return spy

        # Expert 3 has its gate_proj cached -> partial miss; expert 5 is
        # fully missing -> must ride the fused record.
        partial_eid = 3
        full_eid = 5
        gate = linears[0]
        cached_row = (
            store.load_expert_slice(gate.stacked_weight_key, partial_eid),
            store.load_expert_slice(gate.stacked_scales_key, partial_eid),
            store.load_expert_slice(gate.stacked_biases_key, partial_eid),
        )
        cache.store[gate.bundle_key(partial_eid)] = cached_row

        ctx = _LayerLoadContext(linears, cache, mode="union", positions=2)
        orig_np = {id(l): l._load_expert_bank_np for l in linears}
        for lin in linears:
            lin._load_expert_bank_np = make_np_spy(lin)
        try:
            lead._read_fused_rows = spy_fused_rows
            for _lin in linears:

                ctx._ensure_union(_lin, [partial_eid, full_eid])
        finally:
            lead._read_fused_rows = orig_fused
            for lin in linears:
                lin._load_expert_bank_np = orig_np[id(lin)]

        assert not ctx.failed, "context must not fail"
        assert not ctx.declined, "union must not decline"

        # Fused record served EXACTLY the full-miss expert.
        assert fused_reads == [[full_eid]], fused_reads
        # Per-projection reads served the partial expert's OTHER two
        # projections only (gate came from cache).
        assert sorted(
            (proj, tuple(ids)) for proj, ids in per_proj_reads
        ) == [
            ("down_proj", (partial_eid,)),
            ("up_proj", (partial_eid,)),
        ], per_proj_reads

        # Bundles complete and bit-exact for BOTH experts on every proj.
        for lin in linears:
            bundle = ctx.bundles.get(id(lin))
            assert bundle is not None and len(bundle) == 2
            for eid in (partial_eid, full_eid):
                w, s, b = bundle[int(eid)]
                np.testing.assert_array_equal(
                    w, store.load_expert_slice(lin.stacked_weight_key, eid)
                )
                np.testing.assert_array_equal(
                    s, store.load_expert_slice(lin.stacked_scales_key, eid)
                )
                np.testing.assert_array_equal(
                    b, store.load_expert_slice(lin.stacked_biases_key, eid)
                )
        # cached gate row reused, not reread
        b_gate = ctx.bundles[id(gate)]
        assert b_gate[partial_eid][0] is cached_row[0]
    finally:
        store.close()


def test_fused_phase_gate_positions_one(fixture_model, tmp_path, monkeypatch):
    """Single-token calls (decode shape) must NOT ride the fused record.

    E3/E3h verdict: fused records win multi-token calls (prefill TTFT) but
    regress single-token decode -10..-12%; positions<=1 routes to the
    per-projection path even with the knob on.
    """
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FUSED_BANK", "1")
    import omlx.patches.expert_streaming.streaming_switch as ss

    ss._FUSED_BANK_ENV = True
    pack_model(tmp_path, verify=2)
    store = ExpertBackingStore(tmp_path)
    cache = _TinyCache()
    linears = []
    try:
        ok, why, n = store.attach_expert_bank()
        assert ok, why

        for proj in _PROJS:
            linears.append(
                StreamingQuantizedSwitchLinear(
                    layer_idx=0,
                    proj_name=proj,
                    stacked_weight_key=f"{_PREFIX}.{proj}.weight",
                    stacked_scales_key=f"{_PREFIX}.{proj}.scales",
                    stacked_biases_key=f"{_PREFIX}.{proj}.biases",
                    num_experts=_N,
                    input_dims=4,
                    output_dims=8,
                    backing=store,
                    cache=cache,
                )
            )

        # positions=1: decode-shaped call -> per-projection path.
        ctx = _LayerLoadContext(linears, cache, mode="union", positions=1)
        for _lin in linears:

            ctx._ensure_union(_lin, [0, 2])
        assert not ctx.failed
        assert not ctx.declined
        for lin in linears:
            bundle = ctx.bundles.get(id(lin))
            assert bundle is not None and len(bundle) == 2
            for eid in (0, 2):
                w, s, b = bundle[int(eid)]
                np.testing.assert_array_equal(
                    w, store.load_expert_slice(lin.stacked_weight_key, eid)
                )
        # The fused prefix cache stays unset for every projection when the
        # phase gate declined BEFORE gate 1 runs.
        for lin in linears:
            assert getattr(lin, "_fused_prefix_cache", None) is None

        # positions=2 (prefill/batch shape) still rides the fused record.
        ctx2 = _LayerLoadContext(linears, cache, mode="union", positions=2)
        for _lin in linears:

            ctx2._ensure_union(_lin, [0, 2])
        assert not ctx2.failed
        for lin in linears:
            assert lin._fused_prefix_cache == _PREFIX
    finally:
        store.close()


def test_fused_phase_gate_topk_decode(fixture_model, tmp_path, monkeypatch):
    """top_k>1 single-token decode must NOT ride the fused record.

    positions conflates routed rows with tokens: a GLM-5.3 JANG decode is
    1 token x top_k 8 = 8 rows, which passed the old positions<=1 gate and
    regressed decode -44% (E6). seq_len (indices.shape[-2]) is the
    authoritative signal: seq_len<=1 declines even when positions>1.
    """
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FUSED_BANK", "1")
    import omlx.patches.expert_streaming.streaming_switch as ss

    ss._FUSED_BANK_ENV = True
    pack_model(tmp_path, verify=2)
    store = ExpertBackingStore(tmp_path)
    cache = _TinyCache()
    linears = []
    try:
        ok, why, _n = store.attach_expert_bank()
        assert ok, why
        for proj in _PROJS:
            linears.append(
                StreamingQuantizedSwitchLinear(
                    layer_idx=0,
                    proj_name=proj,
                    stacked_weight_key=f"{_PREFIX}.{proj}.weight",
                    stacked_scales_key=f"{_PREFIX}.{proj}.scales",
                    stacked_biases_key=f"{_PREFIX}.{proj}.biases",
                    num_experts=_N,
                    input_dims=4,
                    output_dims=8,
                    backing=store,
                    cache=cache,
                )
            )
        # 8 routed rows but a single token: decode shape -> per-projection.
        ctx = _LayerLoadContext(
            linears, cache, mode="union", positions=8, seq_len=1
        )
        for _lin in linears:

            ctx._ensure_union(_lin, [0, 2])
        assert not ctx.failed
        assert not ctx.declined
        for lin in linears:
            assert getattr(lin, "_fused_prefix_cache", None) is None
        # Same 8 rows across many tokens: prefill/batch rides the record.
        for lin in linears:
            lin._fused_prefix_cache = None
        ctx2 = _LayerLoadContext(
            linears, cache, mode="union", positions=8, seq_len=8
        )
        for _lin in linears:

            ctx2._ensure_union(_lin, [0, 2])
        assert not ctx2.failed
        for lin in linears:
            assert lin._fused_prefix_cache == _PREFIX
    finally:
        store.close()


def test_fused_disabled_env(fixture_model, tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_EXPERT_STREAMING_FUSED_BANK", "0")
    import omlx.patches.expert_streaming.streaming_switch as ss

    ss._FUSED_BANK_ENV = False
    pack_model(tmp_path, verify=0)
    store = ExpertBackingStore(tmp_path)
    cache = _TinyCache()
    try:
        ok, _why, _n = store.attach_expert_bank()
        assert ok
        lin = StreamingQuantizedSwitchLinear(
            layer_idx=0,
            proj_name="gate_proj",
            stacked_weight_key=f"{_PREFIX}.gate_proj.weight",
            stacked_scales_key=f"{_PREFIX}.gate_proj.scales",
            stacked_biases_key=f"{_PREFIX}.gate_proj.biases",
            num_experts=_N,
            input_dims=4,
            output_dims=8,
            backing=store,
            cache=cache,
        )
        ctx = _LayerLoadContext([lin], cache, mode="union", positions=2)
        for _lin in [lin]:
            ctx._ensure_union(_lin, [0, 3])
        assert not ctx.failed
        # per-projection path still resolves everything
        bundle = ctx.bundles[id(lin)]
        assert len(bundle) == 2
    finally:
        store.close()


def test_chunked_bank_read_bit_exact(fixture_model, tmp_path, monkeypatch):
    """Fase 4: a demand set over _BANK_MAX_BYTES is read in cap-bounded
    row windows instead of declining to the per-expert fallback — and the
    rows stay bit-identical to per-expert slices."""
    import omlx.patches.expert_streaming.streaming_switch as ss

    monkeypatch.setattr(ss, "_BANK_CHUNK_ENV", True)
    # Fixture experts are 56 B/projection-set (36+12+8): a 112 B cap forces
    # exactly 3 chunk windows for a 6-expert demand set.
    monkeypatch.setattr(ss, "_BANK_MAX_BYTES", 112)
    store = ExpertBackingStore(tmp_path)
    cache = _TinyCache()
    calls = []
    orig = ExpertBackingStore.read_expert_into

    def _spy(self, components, outs, **kw):
        calls.append([len(ids) for _k, ids in components])
        return orig(self, components, outs, **kw)

    monkeypatch.setattr(ExpertBackingStore, "read_expert_into", _spy)
    try:
        lin = StreamingQuantizedSwitchLinear(
            layer_idx=0,
            proj_name="gate_proj",
            stacked_weight_key=f"{_PREFIX}.gate_proj.weight",
            stacked_scales_key=f"{_PREFIX}.gate_proj.scales",
            stacked_biases_key=f"{_PREFIX}.gate_proj.biases",
            num_experts=_N,
            input_dims=4,
            output_dims=8,
            backing=store,
            cache=cache,
        )
        demand = [0, 1, 2, 3, 4, 5]
        got = lin._load_expert_bank_np_full(demand)
        assert got is not None, "chunked bank must not decline over the cap"
        segments, rows = got
        assert len(rows) == len(demand)
        # 6 experts / 2-per-chunk -> 3 read_expert_into calls, each with
        # one 2-id component per stacked key (weight, scales, biases)
        assert calls == [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
        for eid, (w, s, b) in zip(demand, rows):
            np.testing.assert_array_equal(
                w, store.load_expert_slice(lin.stacked_weight_key, eid)
            )
            np.testing.assert_array_equal(
                s, store.load_expert_slice(lin.stacked_scales_key, eid)
            )
            np.testing.assert_array_equal(
                b, store.load_expert_slice(lin.stacked_biases_key, eid)
            )
        # chunks of one tier stay in one segment (promotion contract:
        # one bank per (key, tier), not one per chunk)
        assert len(segments) == 1 and len(segments[0][0]) == len(demand)
    finally:
        store.close()


def test_chunked_bank_disabled_declines(fixture_model, tmp_path, monkeypatch):
    """A/B kill switch: _BANK_CHUNK_ENV=0 restores the cap decline."""
    import omlx.patches.expert_streaming.streaming_switch as ss

    monkeypatch.setattr(ss, "_BANK_CHUNK_ENV", False)
    monkeypatch.setattr(ss, "_BANK_MAX_BYTES", 112)
    store = ExpertBackingStore(tmp_path)
    cache = _TinyCache()
    try:
        lin = StreamingQuantizedSwitchLinear(
            layer_idx=0,
            proj_name="gate_proj",
            stacked_weight_key=f"{_PREFIX}.gate_proj.weight",
            stacked_scales_key=f"{_PREFIX}.gate_proj.scales",
            stacked_biases_key=f"{_PREFIX}.gate_proj.biases",
            num_experts=_N,
            input_dims=4,
            output_dims=8,
            backing=store,
            cache=cache,
        )
        assert lin._load_expert_bank_np_full([0, 1, 2, 3, 4, 5]) is None
    finally:
        store.close()


def _drain(worker, pred, timeout=10.0):
    import time as _t

    t0 = _t.perf_counter()
    while _t.perf_counter() - t0 < timeout:
        if pred(worker):
            return True
        _t.sleep(0.01)
    return False


def test_detached_admission_worker(fixture_model, tmp_path):
    """P2: union demand misses admit off-path via copy + 2nd-touch filter.

    First demand: rows serve the call but the worker filter drops them
    (1st touch). Second demand of the same set: admitted — and the stored
    row must be a private copy, not a view into the shared bank.
    """
    import omlx.patches.expert_streaming.streaming_switch as ss
    from omlx.patches.expert_streaming.streaming_switch import ExpertLRUCache

    ss._FUSED_BANK_ENV = True
    pack_model(tmp_path, verify=2)
    store = ExpertBackingStore(tmp_path)
    cache = ExpertLRUCache(
        budget_bytes=1 << 26, per_expert_bytes=1 << 12, num_layers=1
    )
    assert cache.admission is not None, "detached worker must be armed"
    linears = []
    try:
        ok, why, n = store.attach_expert_bank()
        assert ok, why
        for proj in _PROJS:
            linears.append(
                StreamingQuantizedSwitchLinear(
                    layer_idx=0,
                    proj_name=proj,
                    stacked_weight_key=f"{_PREFIX}.{proj}.weight",
                    stacked_scales_key=f"{_PREFIX}.{proj}.scales",
                    stacked_biases_key=f"{_PREFIX}.{proj}.biases",
                    num_experts=_N,
                    input_dims=4,
                    output_dims=8,
                    backing=store,
                    cache=cache,
                )
            )
        demand = [1, 4, 2]
        ctx = _LayerLoadContext(linears, cache, mode="union", positions=2)
        for _lin in linears:

            ctx._ensure_union(_lin, demand)
        adm = cache.admission
        # 3 projections x 3 experts submitted; with room in the layer and
        # the global store the first sighting admits directly.
        assert _drain(adm, lambda w: w.submitted >= 9 and w.admitted >= 9)
        assert len(cache._store) >= 9
        for key, row in cache._store.items():
            for arr in row:
                if arr is not None:
                    assert getattr(arr, "base", None) is None, (
                        "admitted rows must be private copies, not bank views"
                    )
        # Under pressure the 2nd-touch filter still applies: shrink the
        # global store to exactly full (decode phase => `capacity` is the
        # active cap), then first sightings are filtered, second admit.
        cache.resize(9)
        assert len(cache._store) == 9
        k = (0, 99, "gate_proj")
        assert cache.admission_note(k) is False
        assert cache.admission_note(k) is True
    finally:
        store.close()
