"""Trilha C: the direct-compatible layout audit.

Fixtures write **header-only** safetensors (8-byte length prefix + JSON, no
payload at all). That is deliberate: the audit must never read tensor bytes,
so a file with no tensor bytes is both the cheapest fixture and a live proof
of the read-only contract.
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

# Load by path, NOT by putting scripts/ on sys.path.
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
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_expert_streaming_layout")

_DT = audit.DTYPE_BYTES


def _write_shard(path: Path, tensors: dict) -> None:
    """tensors: {name: (dtype, shape)}. Writes header only, no payload."""
    header = {}
    off = 0
    for name, (dtype, shape) in tensors.items():
        n = _DT[dtype]
        for d in shape:
            n *= d
        header[name] = {
            "dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n],
        }
        off += n
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


@pytest.fixture()
def audit_mod():
    return audit


def _model(tmp_path, tensors_by_shard, index=True):
    """Build a checkpoint dir; tensors_by_shard: {filename: {name: (dt, shape)}}."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for fn, tensors in tensors_by_shard.items():
        _write_shard(tmp_path / fn, tensors)
    if index:
        wm = {}
        for fn, tensors in tensors_by_shard.items():
            for name in tensors:
                wm[name] = fn
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": wm})
        )
    return tmp_path


def _qwen_layers(dtype_weight="U32", dtype_side="F16", layers=(0, 1)):
    out = {}
    for layer in layers:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out[f"language_model.layers.{layer}.mlp.switch_mlp.{proj}.weight"] = (
                dtype_weight, [8, 64, 4])
            out[f"language_model.layers.{layer}.mlp.switch_mlp.{proj}.scales"] = (
                dtype_side, [8, 64, 1])
            out[f"language_model.layers.{layer}.mlp.switch_mlp.{proj}.biases"] = (
                dtype_side, [8, 64, 1])
    return out


def test_direct_compatible_checkpoint(tmp_path):
    """U32 weight + F16 scales/biases needs no conversion at all."""
    d = _model(tmp_path, {"model-00001-of-00001.safetensors": _qwen_layers()})
    rep = audit.audit_model(d)
    assert rep["verdict"] == "direct_compatible"
    assert rep["tensors"] == 18
    assert rep["needs_conversion_bytes"] == 0
    assert rep["incompatible_bytes"] == 0
    assert all(g["status"] == "ok" for g in rep["groups"])


def test_bf16_scales_are_flagged_convertible(tmp_path):
    """Trilha C's trigger case: BF16 side tensors -> needs_conversion."""
    d = _model(
        tmp_path,
        {"model-00001-of-00001.safetensors": _qwen_layers(dtype_side="BF16")},
    )
    rep = audit.audit_model(d)
    assert rep["verdict"] == "needs_conversion"
    assert rep["needs_conversion_bytes"] > 0
    assert rep["incompatible_bytes"] == 0
    side = [g for g in rep["groups"] if g["role"] in ("scales", "biases")]
    assert side and all(g["dtype"] == "BF16" for g in side)
    assert all(g["status"] == "convertible" for g in side)
    # Weight is still fine and must not be dragged into the verdict.
    assert [g for g in rep["groups"] if g["role"] == "weight"][0]["status"] == "ok"


def test_non_packed_weight_is_incompatible(tmp_path):
    """A weight that is not U32 has no path to direct publication."""
    d = _model(
        tmp_path,
        {"model-00001-of-00001.safetensors": _qwen_layers(dtype_weight="F32")},
    )
    rep = audit.audit_model(d)
    assert rep["verdict"] == "incompatible"
    assert rep["incompatible_bytes"] > 0
    assert [g for g in rep["groups"] if g["role"] == "weight"][0]["status"] == (
        "incompatible"
    )


def test_deepseek_ffn_experts_convention(tmp_path):
    """layers.N.ffn.experts.E.wK.ROLE is recognised (43 layers x 256 experts)."""
    tensors = {
        f"layers.{layer}.ffn.experts.{E}.{p}.{r}": (
            "U32" if r == "weight" else "F16", [2048, 16])
        for layer in (0, 1) for E in (0, 1) for p in ("w1", "w2", "w3")
        for r in ("weight", "scales", "biases")
    }
    d = _model(tmp_path, {"model-00001-of-00001.safetensors": tensors})
    rep = audit.audit_model(d)
    assert rep["verdict"] == "direct_compatible"
    assert rep["tensors"] == 36
    assert {g["arch"] for g in rep["groups"]} == {"ffn_experts"}
    assert {g["proj"] for g in rep["groups"]} == {"w1", "w2", "w3"}


def test_mtp_experts_count_as_routed(tmp_path):
    """The MTP draft module's experts stream too, so they gate the verdict."""
    tensors = {
        f"mtp.layers.0.mlp.experts.{p}.weight": ("U32", [8, 64, 4])
        for p in ("down_proj", "gate_up_proj")
    }
    tensors.update({
        f"mtp.layers.0.mlp.experts.{p}.{r}": ("BF16", [8, 64, 1])
        for p in ("down_proj", "gate_up_proj") for r in ("scales", "biases")
    })
    d = _model(tmp_path, {"model-00001-of-00001.safetensors": tensors})
    rep = audit.audit_model(d)
    assert {g["arch"] for g in rep["groups"]} == {"mtp_experts"}
    assert all(g["scope"] == "routed" for g in rep["groups"])
    assert rep["verdict"] == "needs_conversion"


def test_shared_experts_are_reported_but_excluded_from_verdict(tmp_path):
    """Dense shared experts are resident, so their dtype must not gate us."""
    tensors = _qwen_layers(layers=(0,))
    for proj in ("gate_proj", "up_proj", "down_proj"):
        tensors[f"model.layers.0.mlp.shared_experts.{proj}.weight"] = ("U32", [64, 4])
        tensors[f"model.layers.0.mlp.shared_experts.{proj}.scales"] = ("BF16", [64, 1])
    d = _model(tmp_path, {"model-00001-of-00001.safetensors": tensors})
    rep = audit.audit_model(d)
    shared = [g for g in rep["groups"] if g["scope"] == "shared"]
    assert shared and all(g["arch"] == "shared_experts" for g in shared)
    assert any(g["status"] == "convertible" for g in shared)
    # Routed side is clean, so the verdict stays clean.
    assert rep["verdict"] == "direct_compatible"


def test_router_gate_is_not_an_expert_tensor(tmp_path):
    """``ffn.gate.weight`` is F32 by design and must not be flagged."""
    tensors = dict(_qwen_layers(layers=(0,)))
    tensors["layers.0.ffn.gate.weight"] = ("F32", [256, 4096])
    d = _model(tmp_path, {"model-00001-of-00001.safetensors": tensors})
    rep = audit.audit_model(d)
    assert rep["verdict"] == "direct_compatible"
    assert not any(g["dtype"] == "F32" for g in rep["groups"])
    assert audit.classify("layers.0.ffn.gate.weight") is None


def test_generic_fallback_catches_unseen_convention(tmp_path):
    """An unknown naming still gets audited, and it counts as routed.

    Fail-closed on purpose: a convention we cannot name must not be waved
    through as "probably shared", otherwise the audit silently passes models
    it never actually classified.
    """
    tensors = {
        "weird.block.experts.7.weight": ("U32", [64, 4]),
        "weird.block.experts.7.scales": ("BF16", [64, 1]),
    }
    d = _model(tmp_path, {"model-00001-of-00001.safetensors": tensors})
    rep = audit.audit_model(d)
    assert {g["arch"] for g in rep["groups"]} == {"generic"}
    assert rep["verdict"] == "needs_conversion"


def test_index_selects_shards_and_ignores_unrelated_files(tmp_path):
    """Calibration blobs next to the model must not be audited."""
    good = _qwen_layers(layers=(0,))
    d = _model(tmp_path, {"model-00001-of-00001.safetensors": good})
    _write_shard(
        tmp_path / "awq-calibration.safetensors",
        {"language_model.layers.0.mlp.switch_mlp.gate_proj.weight": ("F32", [4, 4])},
    )
    files = audit.shard_files(d)
    assert [f.name for f in files] == ["model-00001-of-00001.safetensors"]
    rep = audit.audit_model(d)
    assert rep["verdict"] == "direct_compatible"
    assert not any(g["dtype"] == "F32" for g in rep["groups"])


def test_missing_index_falls_back_to_glob(tmp_path):
    d = _model(tmp_path, {"model-00001-of-00001.safetensors": _qwen_layers(layers=(0,))},
               index=False)
    assert [f.name for f in audit.shard_files(d)] == [
        "model-00001-of-00001.safetensors"]
    assert audit.audit_model(d)["verdict"] == "direct_compatible"


def test_audit_is_header_only(tmp_path):
    """A shard with zero payload bytes audits fine: we never read tensors."""
    tensors = _qwen_layers(layers=(0,))
    d = _model(tmp_path, {"model-00001-of-00001.safetensors": tensors})
    shard = d / "model-00001-of-00001.safetensors"
    (n,) = struct.unpack("<Q", shard.read_bytes()[:8])
    assert shard.stat().st_size == 8 + n  # header and nothing else
    rep = audit.audit_model(d)
    assert rep["verdict"] == "direct_compatible"
    assert rep["bytes"] > 0  # sizes come from shapes, not from disk


def test_empty_directory_reports_no_expert_tensors(tmp_path):
    d = _model(tmp_path, {"model-00001-of-00001.safetensors": {"embed.weight": ("F16", [4, 4])}})
    rep = audit.audit_model(d)
    assert rep["verdict"] == "no_expert_tensors"
    assert rep["tensors"] == 0


def test_strict_exit_code(tmp_path, monkeypatch, capsys):
    """--strict turns 'not direct_compatible' into a non-zero exit."""
    clean = _model(tmp_path / "clean",
                   {"model-00001-of-00001.safetensors": _qwen_layers(layers=(0,))})
    dirty = _model(tmp_path / "dirty",
                   {"model-00001-of-00001.safetensors": _qwen_layers(
                       layers=(0,), dtype_side="BF16")})

    monkeypatch.setattr(sys, "argv", ["x", str(clean), "--strict"])
    assert audit.main() == 0

    monkeypatch.setattr(sys, "argv", ["x", str(dirty), "--strict"])
    assert audit.main() == 1

    # Without --strict the same model still exits 0 (reporting, not failing).
    monkeypatch.setattr(sys, "argv", ["x", str(dirty)])
    assert audit.main() == 0


def test_json_report_is_written(tmp_path, monkeypatch):
    d = _model(tmp_path / "m",
               {"model-00001-of-00001.safetensors": _qwen_layers(layers=(0,))})
    out = tmp_path / "sub" / "report.json"
    monkeypatch.setattr(sys, "argv", ["x", str(d), "--json", str(out)])
    assert audit.main() == 0
    payload = json.loads(out.read_text())
    assert payload["models"][0]["verdict"] == "direct_compatible"


def test_classify_directly():
    assert audit.classify(
        "language_model.layers.28.mlp.switch_mlp.gate_proj.biases"
    ) == ("switch_mlp", "biases", "gate_proj")
    assert audit.classify("layers.21.ffn.experts.140.w3.weight") == (
        "ffn_experts", "weight", "w3")
    assert audit.classify("mtp.layers.0.mlp.experts.down_proj.scales") == (
        "mtp_experts", "scales", "down_proj")
    assert audit.classify("model.layers.10.mlp.shared_experts.up_proj.weight") == (
        "shared_experts", "weight", "up_proj")
    assert audit.classify("layers.0.mlp.down_proj.weight") is None
    assert audit.classify("layers.0.ffn.gate.weight") is None
