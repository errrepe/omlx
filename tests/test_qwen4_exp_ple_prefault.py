# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in SSD PLE prefault warm-up.

The vendored qwen4_exp module is ruff-excluded but still exercised here:
prefault is best-effort by construction, so the contract under test is
"same bytes, fewer serial faults" -- never a different result.
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    DiskBackedShardedEmbedding,
    _SafeTensorMMap,
    _align_ranges_to_page,
    _merge_byte_ranges,
    _ple_prefault_min_rows,
    _prefault_page_bytes,
    _ple_prefault_workers,
)

PREFAULT_ENV = "OMLX_QWEN4_PLE_PREFAULT"
WORKERS_ENV = "OMLX_QWEN4_PLE_PREFAULT_WORKERS"
MIN_ROWS_ENV = "OMLX_QWEN4_PLE_PREFAULT_MIN_ROWS"

PREFIX = "model.ple.ple_embedding"
DIMS = 64
SHARD_SIZE = 64
NUM_SHARDS = 4
BITS = 4
GROUP_SIZE = 64


def _build_ple_checkpoint(
    tmp_path,
    *,
    num_shards: int = NUM_SHARDS,
    shard_size: int = SHARD_SIZE,
    dims: int = DIMS,
    affine: bool = True,
    num_files: int = 1,
    seed: int = 0,
) -> int:
    """Write a synthetic sharded PLE checkpoint; returns num_embeddings."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    weight_map: dict[str, str] = {}
    per_file: dict[str, dict[str, np.ndarray]] = {}

    for shard in range(num_shards):
        filename = f"model-{shard % num_files}.safetensors"
        base = f"{PREFIX}.shard_{shard}"
        tensors = per_file.setdefault(filename, {})
        if affine:
            groups = dims // GROUP_SIZE
            tensors[f"{base}.weight"] = rng.integers(
                0, 2**32, size=(shard_size, dims * BITS // 32), dtype=np.uint32
            )
            tensors[f"{base}.scales"] = (
                rng.random((shard_size, groups)) * 0.1 + 0.01
            ).astype(np.float16)
            tensors[f"{base}.biases"] = (
                rng.random((shard_size, groups)) * 0.1 - 0.05
            ).astype(np.float16)
        else:
            tensors[f"{base}.weight"] = rng.random((shard_size, dims)).astype(
                np.float32
            )
        for key in (f"{base}.weight", f"{base}.scales", f"{base}.biases"):
            if key in tensors:
                weight_map[key] = filename

    for filename, tensors in per_file.items():
        save_file(tensors, str(tmp_path / filename), metadata={"format": "pt"})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return num_shards * shard_size


def _gather(model_path, indices, num_embeddings, dims=DIMS):
    table = DiskBackedShardedEmbedding(
        str(model_path), PREFIX, num_embeddings, dims, NUM_SHARDS
    )
    try:
        out = table(mx.array(indices, dtype=mx.int32))
        mx.eval(out)
        return out
    finally:
        table.close()


def _indices(count: int, num_embeddings: int, seed: int = 1) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(0, num_embeddings, size=count)]


class _PreadRecorder:
    """Stand-in for os.pread that records (fd, offset, length) per call."""

    def __init__(self):
        self.calls: list[tuple[int, int, int]] = []
        self._lock = threading.Lock()

    def __call__(self, fd, length, offset):
        with self._lock:
            self.calls.append((fd, offset, length))
        return b"\x00" * length


def test_prefault_is_bit_exact_on_and_off(tmp_path, monkeypatch):
    """Warming pages must not change a single gathered value."""
    num_embeddings = _build_ple_checkpoint(tmp_path)
    indices = _indices(128, num_embeddings)

    monkeypatch.setenv(PREFAULT_ENV, "0")
    baseline = _gather(tmp_path, indices, num_embeddings)

    monkeypatch.setenv(PREFAULT_ENV, "1")
    warmed = _gather(tmp_path, indices, num_embeddings)

    assert warmed.tolist() == baseline.tolist()


def test_prefault_is_bit_exact_for_dense_shards(tmp_path, monkeypatch):
    num_embeddings = _build_ple_checkpoint(tmp_path, affine=False, seed=7)
    indices = _indices(96, num_embeddings)

    monkeypatch.setenv(PREFAULT_ENV, "0")
    baseline = _gather(tmp_path, indices, num_embeddings)

    monkeypatch.setenv(PREFAULT_ENV, "1")
    warmed = _gather(tmp_path, indices, num_embeddings)

    assert warmed.tolist() == baseline.tolist()


def test_prefault_disabled_issues_no_pread(tmp_path, monkeypatch):
    num_embeddings = _build_ple_checkpoint(tmp_path)
    recorder = _PreadRecorder()
    monkeypatch.setattr(os, "pread", recorder)
    monkeypatch.setenv(PREFAULT_ENV, "0")

    _gather(tmp_path, _indices(128, num_embeddings), num_embeddings)

    assert recorder.calls == []


def test_decode_sized_gather_skips_prefault(tmp_path, monkeypatch):
    """The row floor keeps warm-up off the decode path."""
    num_embeddings = _build_ple_checkpoint(tmp_path)
    recorder = _PreadRecorder()
    monkeypatch.setattr(os, "pread", recorder)
    monkeypatch.setenv(PREFAULT_ENV, "1")

    _gather(tmp_path, _indices(8, num_embeddings), num_embeddings)

    assert recorder.calls == []


def test_prefault_reads_only_touched_ranges_without_overlap(tmp_path, monkeypatch):
    """Merged spans are coalesced per file, chunked, and never re-read."""
    num_embeddings = _build_ple_checkpoint(tmp_path, num_files=2)
    recorder = _PreadRecorder()
    monkeypatch.setattr(os, "pread", recorder)
    monkeypatch.setenv(PREFAULT_ENV, "1")

    _gather(tmp_path, _indices(160, num_embeddings), num_embeddings)

    assert recorder.calls
    by_fd: dict[int, list[tuple[int, int]]] = {}
    for fd, offset, length in recorder.calls:
        assert 0 < length <= 64 * 1024
        by_fd.setdefault(fd, []).append((offset, offset + length))

    for spans in by_fd.values():
        spans.sort()
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            assert next_start >= prev_end, "prefault re-read an already warm span"


def test_rows_sharing_a_page_collapse_into_one_read(tmp_path, monkeypatch):
    """Regression: N rows inside one page are one I/O, not N syscalls.

    The whole 4x64 fixture is ~9 KiB, i.e. a single 16 KiB page. Issuing
    one pread per row (768 of them) made the warm-up slower than the
    fault path it replaces at 8k-token prefills.
    """
    num_embeddings = _build_ple_checkpoint(tmp_path)
    recorder = _PreadRecorder()
    monkeypatch.setattr(os, "pread", recorder)
    monkeypatch.setenv(PREFAULT_ENV, "1")

    _gather(tmp_path, list(range(num_embeddings)), num_embeddings)

    assert len(recorder.calls) == 1
    fd, offset, length = recorder.calls[0]
    page = _prefault_page_bytes()
    assert offset % page == 0
    assert length == page


def test_align_ranges_to_page_quantizes_and_dedupes():
    page = 16 * 1024
    aligned = _align_ranges_to_page([(100, 136), (200, 236)], page)
    assert aligned == [(0, page), (0, page)]
    assert _merge_byte_ranges(aligned) == [(0, page)]
    assert _align_ranges_to_page([(0, 10), (page, page + 10)], page) == [
        (0, page),
        (page, 2 * page),
    ]
    # A 60 B row that straddles a page boundary must claim both pages.
    assert _align_ranges_to_page([(page - 10, page + 10)], page) == [(0, 2 * page)]
    # Degenerate page size is a pass-through, not a division by zero.
    assert _align_ranges_to_page([(5, 9)], 0) == [(5, 9)]


def test_prefault_page_bytes_is_a_power_of_two_page():
    page = _prefault_page_bytes()
    assert page >= 4096
    assert page & (page - 1) == 0


def test_prefault_survives_pread_failure(tmp_path, monkeypatch):
    """Best-effort: an I/O error must not corrupt or abort the gather."""
    num_embeddings = _build_ple_checkpoint(tmp_path)
    indices = _indices(128, num_embeddings)

    monkeypatch.setenv(PREFAULT_ENV, "0")
    baseline = _gather(tmp_path, indices, num_embeddings)

    def exploding_pread(fd, length, offset):
        raise OSError("boom")

    monkeypatch.setattr(os, "pread", exploding_pread)
    monkeypatch.setenv(PREFAULT_ENV, "1")
    # Scope to our prefault path: upstream's page-touch prefetch re-raises
    # I/O failures by contract (see test_qwen4_runtime_ple_fork_cpu.py), so
    # it is stubbed out here — the exploding pread must only reach
    # _prefault_fd_ranges, which is the best-effort path under test.
    monkeypatch.setattr(
        _SafeTensorMMap, "_prefetch_missing_pages", lambda *a, **k: True
    )

    assert _gather(tmp_path, indices, num_embeddings).tolist() == baseline.tolist()


def test_row_byte_ranges_are_row_aligned_and_disjoint(tmp_path):
    num_embeddings = _build_ple_checkpoint(tmp_path)
    table = DiskBackedShardedEmbedding(
        str(tmp_path), PREFIX, num_embeddings, DIMS, NUM_SHARDS
    )
    try:
        key = f"{PREFIX}.shard_0.weight"
        reader = table._tensor_readers[key]
        row_bytes = reader.tensor_shape(key)[1] * 4  # U32

        ranges = reader.row_byte_ranges(key, [1, 5, 63])
        for start, stop in ranges:
            assert stop - start == row_bytes
        ordered = sorted(ranges)
        for (_, prev_end), (next_start, _) in zip(ordered, ordered[1:]):
            assert next_start >= prev_end
        # Stride between row 1 and row 5 is exactly four rows.
        step = ordered[1][0] - ordered[0][0]
        assert step == 4 * row_bytes
        assert ordered[2][0] - ordered[0][0] == 62 * row_bytes

        # A repeated row maps to a repeated span; merging folds the copy.
        duplicated = reader.row_byte_ranges(key, [5, 5])
        assert duplicated[0] == duplicated[1]
        assert _merge_byte_ranges(duplicated) == [duplicated[0]]
    finally:
        table.close()


def test_row_byte_ranges_returns_empty_for_unsupported_layout():
    """Unknown dtypes/layouts must degrade to 'no warm-up', never raise."""
    from mlx_vlm.models.qwen4_exp.language import _SafeTensorMMap

    reader = _SafeTensorMMap.__new__(_SafeTensorMMap)
    reader._data_start = 8
    reader._header = {
        "w": {"dtype": "F64", "shape": [4, 8], "data_offsets": [0, 256]},
        "b": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
    }
    assert reader.row_byte_ranges("w", [0]) == []
    assert reader.row_byte_ranges("b", [0]) == []
    assert reader.row_byte_ranges("missing", [0]) == []


def test_merge_byte_ranges_coalesces_and_dedupes():
    assert _merge_byte_ranges([]) == []
    # Overlap, adjacency and sub-4 KiB gaps all fold into one span.
    assert _merge_byte_ranges([(100, 200), (150, 250), (260, 300), (7000, 7100)]) == [
        (100, 300),
        (7000, 7100),
    ]
    # A gap above the threshold stays split; output is always sorted.
    assert _merge_byte_ranges([(5000, 5100), (0, 100)]) == [(0, 100), (5000, 5100)]
    # Zero-length spans are dropped.
    assert _merge_byte_ranges([(10, 10), (10, 20)]) == [(10, 20)]
    merged = _merge_byte_ranges([(0, 10), (10, 20), (30, 40)], max_gap=0)
    assert merged == [(0, 20), (30, 40)]


def test_prefault_worker_env_is_clamped(monkeypatch):
    monkeypatch.setenv(WORKERS_ENV, "1000")
    assert _ple_prefault_workers() == 64
    monkeypatch.setenv(WORKERS_ENV, "0")
    assert _ple_prefault_workers() == 1
    monkeypatch.setenv(WORKERS_ENV, "-4")
    assert _ple_prefault_workers() == 1
    monkeypatch.setenv(WORKERS_ENV, "abc")
    assert _ple_prefault_workers() == 1
    monkeypatch.setenv(WORKERS_ENV, "8")
    assert _ple_prefault_workers() == 8
    monkeypatch.delenv(WORKERS_ENV, raising=False)
    # Serial wins on this hardware: GIL contention beats readahead gains.
    assert _ple_prefault_workers() == 1


def test_prefault_min_rows_env_is_parsed(monkeypatch):
    monkeypatch.setenv(MIN_ROWS_ENV, "256")
    assert _ple_prefault_min_rows() == 256
    monkeypatch.setenv(MIN_ROWS_ENV, "0")
    assert _ple_prefault_min_rows() == 0
    monkeypatch.setenv(MIN_ROWS_ENV, "nope")
    assert _ple_prefault_min_rows() == 64
    monkeypatch.delenv(MIN_ROWS_ENV, raising=False)
    assert _ple_prefault_min_rows() == 64
