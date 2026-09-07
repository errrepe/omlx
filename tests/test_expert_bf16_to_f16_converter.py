"""Trilha C, item 6: the BF16 -> F16 expert side-tensor converter.

The contract under test is *admissibility*, not just conversion: a tensor is
rewritten only when BF16 -> F16 is exactly value-preserving. Every refusal
path (overflow, underflow, subnormal precision loss) must leave the original
bytes untouched.
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the scripts by path, NOT by putting scripts/ on sys.path.
#
# scripts/ contains bench.py. The top-level bench/ directory is a *namespace*
# package (no __init__.py), and namespace packages lose to a regular module
# found ANYWHERE on sys.path — so the moment scripts/ is importable, every
# later `from bench.bench_expert_streaming import ...` (test_expert_streaming.py
# does it lazily inside test bodies) resolves to scripts/bench.py and explodes
# with "'bench' is not a package". Reordering does not help; only staying off
# sys.path does.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    # Register first: the converter does `import audit_expert_streaming_layout`
    # at module level and must find it here, not via a sys.path fallback.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_expert_streaming_layout")
conv = _load("convert_expert_bf16_to_f16")

# -- helpers ---------------------------------------------------------------


def bf16_bits(values) -> bytes:
    """Truncate float32 to BF16 (top 16 bits). Good enough for fixtures."""
    u32 = np.asarray(values, dtype=np.float32).view(np.uint32)
    return (u32 >> np.uint32(16)).astype(np.uint16).tobytes()


def f32_of_bf16(raw: bytes) -> np.ndarray:
    return conv.bf16_to_f32(np.frombuffer(raw, dtype="<u2"))


def _payload_len(path: Path) -> int:
    """Bytes of tensor payload in a safetensors file (excludes 8 + header)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
    return path.stat().st_size - 8 - n


def write_shard(path: Path, tensors: dict) -> None:
    """tensors: {name: (dtype, shape, payload_bytes)}."""
    header = {}
    off = 0
    payloads = []
    for name, (dtype, shape, blob) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [off, off + len(blob)],
        }
        off += len(blob)
        payloads.append(blob)
    blob = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for p in payloads:
            f.write(p)


def expert_key(role: str, proj: str = "gate_proj", layer: int = 0, expert: int = 3):
    return f"model.layers.{layer}.ffn.experts.{expert}.{proj}.{role}"


def shared_key(role: str, proj: str = "gate_proj"):
    return f"model.mlp.shared_experts.{proj}.{role}"


# -- dtype widening --------------------------------------------------------


def test_bf16_to_f32_is_exact_for_known_values():
    raw = np.frombuffer(bf16_bits([1.0, -2.0, 0.5, 3.0]), dtype="<u2")
    got = conv.bf16_to_f32(raw)
    assert got.dtype == np.float32
    # BF16 keeps 8 significant bits: 1.0/-2.0/0.5/3.0 survive truncation.
    assert got[0] == 1.0
    assert got[1] == -2.0
    assert got[2] == 0.5
    assert got[3] == 3.0


def test_bf16_to_f32_keeps_large_exponent():
    values = np.array([1e30, 1e-30], dtype=np.float32)
    raw = np.frombuffer(bf16_bits(values), dtype="<u2")
    got = conv.bf16_to_f32(raw)
    assert got[0] == pytest.approx(1e30, rel=1e-2)
    assert got[1] == pytest.approx(1e-30, rel=1e-2)


# -- admissibility ---------------------------------------------------------


def test_analyse_accepts_typical_scales_and_round_trips():
    values = [0.0, 1.0, 0.5, 0.01, 123.25, -0.75, 1024.0]
    raw = np.frombuffer(bf16_bits(values), dtype="<u2")
    bits, ov, un, ot = conv.analyse(raw)
    assert (ov, un, ot) == (0, 0, 0)
    assert bits  # non-empty means accepted
    decoded = np.frombuffer(bits, dtype="<f2").astype(np.float32)
    assert np.array_equal(decoded, f32_of_bf16(bf16_bits(values)))


def test_analyse_refuses_overflow_past_f16_max():
    raw = np.frombuffer(bf16_bits([1.0, 70000.0]), dtype="<u2")
    bits, ov, un, ot = conv.analyse(raw)
    assert bits == b""
    assert ov == 1
    assert un == 0


def test_analyse_refuses_value_that_flushes_to_zero():
    # 1e-8 is below the smallest F16 subnormal (5.96e-8): it becomes 0.
    raw = np.frombuffer(bf16_bits([1.0, 1e-8]), dtype="<u2")
    bits, ov, un, ot = conv.analyse(raw)
    assert bits == b""
    assert un == 1


def test_analyse_refuses_subnormal_precision_loss():
    # 1e-7 is inside the F16 subnormal range but not a multiple of 2^-24.
    raw = np.frombuffer(bf16_bits([1.0, 1e-7]), dtype="<u2")
    bits, ov, un, ot = conv.analyse(raw)
    assert bits == b""
    assert un + ot >= 1


def test_analyse_accepts_zero_and_negative_zero():
    raw = np.frombuffer(bf16_bits([0.0, -0.0, 1.0]), dtype="<u2")
    bits, ov, un, ot = conv.analyse(raw)
    assert (ov, un, ot) == (0, 0, 0)
    assert len(bits) == 6


def test_analyse_treats_nan_as_admissible():
    nan_bits = np.array([0x7FC0], dtype=np.uint16)  # quiet NaN in BF16
    bits, ov, un, ot = conv.analyse(nan_bits)
    assert (ov, un, ot) == (0, 0, 0)
    assert bits


def test_analyse_infinity_is_not_counted_as_lossy():
    inf_bits = np.array([0x7F80], dtype=np.uint16)  # +Inf in BF16
    bits, ov, un, ot = conv.analyse(inf_bits)
    assert (ov, un, ot) == (0, 0, 0)


# -- planning --------------------------------------------------------------


def test_plan_picks_routed_scales_and_biases_only(tmp_path):
    shard = tmp_path / "model-00001.safetensors"
    write_shard(
        shard,
        {
            expert_key("scales"): ("BF16", [4], bf16_bits([1.0] * 4)),
            expert_key("biases"): ("BF16", [4], bf16_bits([0.5] * 4)),
            expert_key("weight"): ("U32", [2], b"\x00" * 8),
            shared_key("scales"): ("BF16", [4], bf16_bits([1.0] * 4)),
            "model.layers.0.ffn.gate.weight": ("F32", [2], b"\x00" * 8),
        },
    )
    plans = conv.plan_shard(shard, include_shared=False)
    keys = {p.key for p in plans}
    assert keys == {expert_key("scales"), expert_key("biases")}
    assert all(p.role in ("scales", "biases") for p in plans)


def test_plan_includes_shared_only_when_asked(tmp_path):
    shard = tmp_path / "model-00001.safetensors"
    write_shard(
        shard,
        {
            shared_key("scales"): ("BF16", [4], bf16_bits([1.0] * 4)),
            expert_key("scales"): ("BF16", [4], bf16_bits([1.0] * 4)),
        },
    )
    assert len(conv.plan_shard(shard, include_shared=False)) == 1
    assert len(conv.plan_shard(shard, include_shared=True)) == 2


def test_plan_ignores_already_f16(tmp_path):
    shard = tmp_path / "model-00001.safetensors"
    write_shard(
        shard,
        {expert_key("scales"): ("F16", [4], b"\x00\x3c" * 4)},
    )
    assert conv.plan_shard(shard, include_shared=False) == []


# -- end to end ------------------------------------------------------------


def _build(tmp_path, scales, weight_blob=None):
    shard = tmp_path / "model-00001.safetensors"
    weight_blob = weight_blob if weight_blob is not None else b"\x01" * 16
    write_shard(
        shard,
        {
            expert_key("scales"): ("BF16", [len(scales)], bf16_bits(scales)),
            expert_key("weight"): ("U32", [4], weight_blob),
        },
    )
    return shard


def test_convert_rewrites_dtype_and_preserves_size(tmp_path):
    scales = [1.0, 0.5, 0.25, 12.0]
    shard = _build(tmp_path, scales)
    before = shard.stat().st_size

    plans = conv.plan_shard(shard, include_shared=False)
    res = conv.convert_shard(shard, plans, tmp_path / "out.safetensors")

    assert res.n_converted == 1
    assert res.n_refused == 0
    out = tmp_path / "out.safetensors"
    # Payload is bit-identical in length (BF16 and F16 are both 2 bytes).
    # The *file* is one byte shorter per converted tensor because the header
    # string "BF16" shrinks to "F16" — compare payloads, not file sizes.
    assert _payload_len(out) == _payload_len(shard)
    assert out.stat().st_size == before - 1
    header = audit.read_header(out)
    assert header[expert_key("scales")]["dtype"] == "F16"
    assert header[expert_key("weight")]["dtype"] == "U32"  # untouched


def test_convert_preserves_values_and_other_tensors(tmp_path):
    scales = [1.0, 0.5, 0.25, 12.0]
    weight_blob = bytes(range(16))
    shard = _build(tmp_path, scales, weight_blob)

    plans = conv.plan_shard(shard, include_shared=False)
    conv.convert_shard(shard, plans, tmp_path / "out.safetensors")

    out = tmp_path / "out.safetensors"
    header = audit.read_header(out)
    base = conv._payload_base(out)
    b0, b1 = header[expert_key("scales")]["data_offsets"]
    with open(out, "rb") as f:
        f.seek(base + b0)
        got = np.frombuffer(f.read(b1 - b0), dtype="<f2").astype(np.float32)
    assert np.array_equal(got, f32_of_bf16(bf16_bits(scales)))

    w0, w1 = header[expert_key("weight")]["data_offsets"]
    with open(out, "rb") as f:
        f.seek(base + w0)
        assert f.read(w1 - w0) == weight_blob


def test_convert_refuses_overflow_and_leaves_bytes_alone(tmp_path):
    shard = _build(tmp_path, [1.0, 70000.0])
    original = shard.read_bytes()

    plans = conv.plan_shard(shard, include_shared=False)
    res = conv.convert_shard(shard, plans, tmp_path / "out.safetensors")

    assert res.n_converted == 0
    assert res.n_refused == 1
    assert "over F16 max" in res.results[0].reason
    header = audit.read_header(tmp_path / "out.safetensors")
    assert header[expert_key("scales")]["dtype"] == "BF16"
    assert (tmp_path / "out.safetensors").read_bytes() == original


def test_dry_run_writes_nothing(tmp_path):
    shard = _build(tmp_path, [1.0, 0.5])
    plans = conv.plan_shard(shard, include_shared=False)
    res = conv.convert_shard(shard, plans, None)
    assert res.n_converted == 1
    assert res.written == ""
    assert [p.name for p in tmp_path.iterdir()] == ["model-00001.safetensors"]


def test_in_place_leaves_no_temp_file(tmp_path):
    shard = _build(tmp_path, [1.0, 0.5, 4.0])
    plans = conv.plan_shard(shard, include_shared=False)
    conv.convert_shard(shard, plans, shard)
    assert audit.read_header(shard)[expert_key("scales")]["dtype"] == "F16"
    assert not list(tmp_path.glob("*.tmp"))


def test_verify_passes_after_in_place(tmp_path):
    shard = _build(tmp_path, [1.0, 0.5, 4.0])
    plans = conv.plan_shard(shard, include_shared=False)
    res = conv.convert_shard(shard, plans, shard)
    assert conv.verify_shard(shard, res.results) == []


def test_verify_catches_a_shard_that_was_never_converted(tmp_path):
    shard = _build(tmp_path, [1.0, 0.5])
    plans = conv.plan_shard(shard, include_shared=False)
    res = conv.convert_shard(shard, plans, None)  # dry run: dtype still BF16
    problems = conv.verify_shard(shard, res.results)
    assert problems and "dtype still BF16" in problems[0]


def test_multiple_tensors_across_slice_boundary(tmp_path):
    # More elements than one slice to exercise the chunked read path.
    conv.SLICE_ELEMS = 4
    try:
        scales = [1.0] * 100
        shard = _build(tmp_path, scales)
        plans = conv.plan_shard(shard, include_shared=False)
        res = conv.convert_shard(shard, plans, tmp_path / "out.safetensors")
        assert res.n_converted == 1
        assert conv.verify_shard(tmp_path / "out.safetensors", res.results) == []
    finally:
        conv.SLICE_ELEMS = 8 << 20


def test_convert_model_reports_dry_run(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    _build(model, [1.0, 0.5])  # writes model/model-00001.safetensors
    rep = conv.convert_model(model, None, False, False, True)
    assert rep["dry_run"] is True
    assert rep["tensors_converted"] == 1
    assert rep["tensors_refused"] == 0
