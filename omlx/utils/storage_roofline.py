# SPDX-License-Identifier: Apache-2.0
"""Storage roofline for MoE expert streaming.

Decode under expert streaming is IO-bound by construction: every step pages
the active experts for every MoE layer off SSD. That makes the ceiling for
steady-state throughput a pure storage equation::

    ceiling_base_tok_s = random_read_bandwidth / bytes_per_decode_step

and the MTP profitability question a bytes inequality, not a FLOPs one::

    profitable  <=>  tok_per_cycle > bytes_verify / bytes_base_step

This module measures the left side (uncached storage bandwidth on the volume
that holds the checkpoints) and derives the right side (stored expert bytes
per decode step, straight from the safetensors headers — stored bytes are
what actually crosses the bus), so the WebUI/CLI can show a predicted
ceiling per model and a structural MTP verdict next to the measured bench.

Measurement integrity notes (learned the hard way):
- A 48 GiB unified-memory box swallows a 2 GiB scratch file whole, so a
  naive read benchmark reports RAM speed (~19 GiB/s) and lies. Every read
  here bypasses the page cache: ``F_NOCACHE`` on macOS, ``posix_fadvise``
  eviction on Linux. When neither is available the report is flagged
  ``cache_clean=False`` and the numbers are an upper bound, not a ceiling.
- Steady-state decode is mostly a cold miss (125B+51B of weights vs 48 GiB
  of RAM), so the *random* 2 MB number — one quantized expert — is the one
  that predicts decode, not the sequential number. Sequential predicts
  spill conversion / model load, which is why both are measured. Note the
  sequential read bypasses OS readahead (that is the price of a guaranteed
  cold read), so dd-style readahead-friendly sequential loads typically run
  ~1.3-1.6x higher; the random/decode number is unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import random
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_F_NOCACHE = 48  # fcntl(F_NOCACHE) on macOS — fd bypasses the page cache
_DEFAULT_READ_MB = 2  # ~one quantized expert (2560x640x3 @ ~4-bit)
_DEFAULT_SAMPLES = 256
_VERIFY_BYTE_MULT_DEFAULT = 2.3  # measured Gap-2: 2-position verify reads


# ===========================================================================
# Volume identity
# ===========================================================================


@dataclass
class VolumeInfo:
    path: str
    mount: str
    filesystem: str
    media_name: str
    protocol: str
    location: str
    solid_state: bool
    total_bytes: int
    free_bytes: int


def _diskutil_text(path: str) -> dict:
    """Parse `/usr/sbin/diskutil info <path>` (macOS only)."""
    out: dict = {}
    try:
        proc = subprocess.run(
            ["/usr/sbin/diskutil", "info", path],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        logger.warning("diskutil failed: %s", e)
        return out
    if proc.returncode != 0:
        return out
    want = {
        "Media Name": "media_name",
        "Protocol": "protocol",
        "Device Location": "location",
        "File System Personality": "filesystem",
        "Solid State": "solid_state",
        "Mount Point": "mount",
    }
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k in want:
            if want[k] == "solid_state":
                out[want[k]] = v.lower() == "yes"
            else:
                out[want[k]] = v
    return out


def _df_bytes(path: str) -> tuple[int, int]:
    """(total, free) bytes via statvfs — portable, no subprocess."""
    try:
        st = os.statvfs(path)
        return st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
    except Exception:
        return 0, 0


def volume_info_for(path: str | Path) -> VolumeInfo:
    """Identify the storage volume holding `path` (model dir or spill)."""
    p = str(Path(path).expanduser())
    total, free = _df_bytes(p)
    info = VolumeInfo(
        path=p, mount=p, filesystem="", media_name="", protocol="",
        location="", solid_state=False, total_bytes=total, free_bytes=free,
    )
    if sys.platform == "darwin":
        d = _diskutil_text(p)
        info.mount = str(d.get("mount") or p)
        info.filesystem = str(d.get("filesystem") or "")
        info.media_name = str(d.get("media_name") or "")
        info.protocol = str(d.get("protocol") or "")
        info.location = str(d.get("location") or "")
        info.solid_state = bool(d.get("solid_state", False))
    else:
        try:
            proc = subprocess.run(
                ["lsblk", "-J", "-o", "NAME,TYPE,TRAN,ROTA,MODEL,MOUNTPOINT"],
                capture_output=True, text=True, timeout=30,
            )
            info.media_name = (proc.stdout.strip().splitlines() or [""])[0][:160]
        except Exception:
            pass
    return info


# ===========================================================================
# Uncached bandwidth measurement
# ===========================================================================


@dataclass
class StorageMeasurement:
    volume_mount: str
    file_bytes: int
    # Sequential: predicts spill conversion / model load.
    seq_read_Bps: float
    # Random read_mb: predicts expert paging (the decode ceiling).
    rand_read_Bps: float
    rand_iops: float
    rand_lat_ms_p50: float
    rand_lat_ms_p90: float
    rand_lat_ms_p99: float
    rand_lat_ms_max: float
    write_Bps: float = 0.0
    samples: int = 0
    read_mb: int = _DEFAULT_READ_MB
    cache_clean: bool = False  # False => page cache may pollute numbers
    method: str = ""
    warnings: list = field(default_factory=list)


def _open_uncached(path: str, write: bool) -> int:
    """Open with page-cache bypass where the platform allows it."""
    if sys.platform == "darwin":
        import fcntl

        flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC) if write else os.O_RDONLY
        fd = os.open(path, flags, 0o644)
        try:
            fcntl.fcntl(fd, _F_NOCACHE, 1)
            return fd
        except OSError:
            os.close(fd)
            raise
    # Linux / other: plain open; eviction handled via posix_fadvise.
    flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC) if write else os.O_RDONLY
    return os.open(path, flags, 0o644)


def _evict(fd: int, offset: int = 0, length: int = 0) -> bool:
    """Drop a range from the page cache (Linux). No-op True on macOS."""
    if sys.platform == "darwin":
        return True  # F_NOCACHE already bypasses
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        return False
    try:
        fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        return True
    except Exception:
        return False


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def measure_storage(
    directory: str | Path,
    file_gb: float = 2.0,
    read_mb: int = _DEFAULT_READ_MB,
    samples: int = _DEFAULT_SAMPLES,
    seed: int = 7,
    include_write: bool = True,
    progress=None,
    scratch_name: str = ".omlx_storage_roofline.bin",
) -> StorageMeasurement:
    """Measure uncached bandwidth on the volume holding `directory`.

    Creates a scratch file (default 2 GiB, incompressible pattern),
    measures sequential read + random `read_mb` reads with per-read
    latency, then removes the scratch file. `progress` is an optional
    callable ``(phase: str, done: int, total: int)`` for UI polling.
    """
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    scratch = directory / scratch_name
    if scratch.exists():
        raise FileExistsError(f"scratch already present (stale run?): {scratch}")

    file_bytes = int(file_gb * 1024**3)
    chunk = 1024 * 1024
    if file_bytes < chunk:
        chunk = file_bytes
    pattern = os.urandom(chunk)
    meas = StorageMeasurement(
        volume_mount=str(directory), file_bytes=file_bytes,
        seq_read_Bps=0.0, rand_read_Bps=0.0, rand_iops=0.0,
        rand_lat_ms_p50=0.0, rand_lat_ms_p90=0.0,
        rand_lat_ms_p99=0.0, rand_lat_ms_max=0.0,
    )
    cache_clean = sys.platform == "darwin" or hasattr(os, "posix_fadvise")
    meas.cache_clean = cache_clean
    meas.method = (
        "F_NOCACHE" if sys.platform == "darwin"
        else ("posix_fadvise" if hasattr(os, "posix_fadvise") else "plain")
    )
    if not cache_clean:
        meas.warnings.append(
            "no page-cache bypass on this platform: numbers are an upper "
            "bound (RAM speed), not a storage ceiling"
        )

    def emit(phase: str, done: int, total: int) -> None:
        if progress is not None:
            try:
                progress(phase, done, total)
            except Exception:
                pass

    try:
        # -- write (also the only phase that mutates the volume) --
        if include_write:
            emit("write", 0, file_bytes)
            fd = _open_uncached(str(scratch), write=True)
            try:
                t0 = time.perf_counter()
                left, done = file_bytes, 0
                while left > 0:
                    n = os.write(fd, pattern[: min(chunk, left)])
                    left -= n
                    done += n
                    emit("write", done, file_bytes)
                os.fsync(fd)
            finally:
                os.close(fd)
            dt = time.perf_counter() - t0
            meas.write_Bps = file_bytes / dt if dt > 0 else 0.0
            emit("write", file_bytes, file_bytes)
        else:
            # Caller pre-created the file? No — without a write phase there
            # is nothing cold to read. Require the scratch to exist.
            if not scratch.exists():
                raise ValueError("include_write=False needs an existing scratch file")

        # -- sequential read (spill-convert / load predictor) --
        emit("seq_read", 0, file_bytes)
        fd = os.open(str(scratch), os.O_RDONLY)
        try:
            if sys.platform == "darwin":
                import fcntl

                fcntl.fcntl(fd, _F_NOCACHE, 1)
            else:
                _evict(fd)
            t0 = time.perf_counter()
            left, done = file_bytes, 0
            while left > 0:
                data = os.read(fd, min(chunk, left))
                if not data:
                    break
                left -= len(data)
                done += len(data)
                emit("seq_read", done, file_bytes)
            dt = time.perf_counter() - t0
            meas.seq_read_Bps = done / dt if dt > 0 else 0.0
        finally:
            os.close(fd)
        emit("seq_read", file_bytes, file_bytes)

        # -- random reads (expert-paging / decode predictor) --
        rsize = read_mb * 1024 * 1024
        if rsize > file_bytes:
            raise ValueError(f"read_mb={read_mb} exceeds scratch file_gb={file_gb}")
        rng = random.Random(seed)
        offs = [rng.randrange(0, file_bytes - rsize + 1) for _ in range(samples)]
        meas.samples = samples
        meas.read_mb = read_mb
        lats: list[float] = []
        fd = os.open(str(scratch), os.O_RDONLY)
        try:
            if sys.platform == "darwin":
                import fcntl

                fcntl.fcntl(fd, _F_NOCACHE, 1)
            t0 = time.perf_counter()
            total_rand = samples * rsize
            for i, off in enumerate(offs):
                if sys.platform != "darwin":
                    _evict(fd, off, rsize)
                os.lseek(fd, off, os.SEEK_SET)
                t1 = time.perf_counter()
                got = 0
                while got < rsize:
                    data = os.read(fd, rsize - got)
                    if not data:
                        break
                    got += len(data)
                lats.append((time.perf_counter() - t1) * 1000.0)
                emit("rand_read", (i + 1) * rsize, total_rand)
            dt = time.perf_counter() - t0
            total_read = samples * rsize
            meas.rand_read_Bps = total_read / dt if dt > 0 else 0.0
            meas.rand_iops = samples / dt if dt > 0 else 0.0
        finally:
            os.close(fd)
        lats.sort()
        meas.rand_lat_ms_p50 = _percentile(lats, 0.50)
        meas.rand_lat_ms_p90 = _percentile(lats, 0.90)
        meas.rand_lat_ms_p99 = _percentile(lats, 0.99)
        meas.rand_lat_ms_max = lats[-1] if lats else 0.0
        emit("done", samples, samples)
        return meas
    finally:
        try:
            if scratch.exists():
                scratch.unlink()
        except Exception as e:
            meas.warnings.append(f"scratch cleanup failed: {e}")


# ===========================================================================
# MoE step profile — stored bytes per decode step, from checkpoint headers
# ===========================================================================

_TOPK_KEYS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_top_k",
    "top_k",
    "num_active_experts",
    "n_experts_per_tok",
)


def _text_cfg(config: dict) -> dict:
    tc = config.get("text_config")
    return tc if isinstance(tc, dict) else {}


def _cfg_int(*sources: dict, keys: tuple[str, ...], default: int = 0) -> int:
    for src in sources:
        if not isinstance(src, dict):
            continue
        for k in keys:
            v = src.get(k)
            if v is None:
                continue
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return default


def _safetensors_index(model_dir: Path) -> dict[str, str]:
    """Global tensor-name -> file map (index or single-file scan)."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        try:
            return json.loads(index.read_text()).get("weight_map") or {}
        except Exception:
            pass
    wmap: dict[str, str] = {}
    for fp in sorted(model_dir.glob("*.safetensors")):
        try:
            with open(fp, "rb") as f:
                raw = f.read(8)
                if len(raw) != 8:
                    continue
                (hsize,) = struct.unpack("<Q", raw)
                hdr = json.loads(f.read(hsize))
            for k in hdr.keys():
                if k != "__metadata__":
                    wmap.setdefault(k, fp.name)
        except Exception:
            continue
    return wmap


def _header_sizes(model_dir: Path, weight_map: dict[str, str]) -> dict[str, int]:
    """Stored byte size per tensor (headers only — no tensor data read)."""
    sizes: dict[str, int] = {}
    files: dict[str, list[str]] = {}
    for k, fname in weight_map.items():
        files.setdefault(fname, []).append(k)
    for fname, keys in files.items():
        try:
            with open(model_dir / fname, "rb") as f:
                raw = f.read(8)
                if len(raw) != 8:
                    continue
                (hsize,) = struct.unpack("<Q", raw)
                hdr = json.loads(f.read(hsize))
        except Exception:
            continue
        want = set(keys)
        for k in want:
            entry = hdr.get(k)
            if not isinstance(entry, dict):
                continue
            try:
                s, e = entry["data_offsets"]
                sizes[k] = int(e) - int(s)
            except Exception:
                continue
    return sizes


@dataclass
class MoEStepProfile:
    model_dir: str
    model_type: str
    supported: bool
    reason: str = ""
    num_moe_layers: int = 0
    routed_total_per_layer: int = 0
    top_k: int = 0
    # Stored bytes actually paged per decode step (routed-active only;
    # shared experts are dense-resident, never streamed).
    routed_active_bytes_per_layer: int = 0
    shared_bytes_per_layer: int = 0  # resident, informational
    bytes_per_step: int = 0
    checkpoint_bytes: int = 0


def moe_step_profile(model_dir: str | Path) -> MoEStepProfile:
    """Stored expert bytes per decode step, from headers + config.

    Reuses :func:`expert_streaming_estimate` for the routed banks (same
    detector the load path gates on, so the prediction can never describe
    a model streaming refuses to run) and adds the top-k + shared-expert
    split the estimate does not track.
    """
    from omlx.patches.expert_streaming.residency import expert_streaming_estimate

    mdir = Path(model_dir).expanduser()
    try:
        config = json.loads((mdir / "config.json").read_text())
    except Exception:
        config = {}
    tc = _text_cfg(config)
    est = expert_streaming_estimate(mdir)
    prof = MoEStepProfile(
        model_dir=str(mdir),
        model_type=est.model_type or "unknown",
        supported=est.supported,
        reason=est.reason or "",
        num_moe_layers=est.num_moe_layers,
        routed_total_per_layer=est.experts_per_layer,
        checkpoint_bytes=est.checkpoint_bytes,
    )
    if not est.supported:
        return prof
    top_k = _cfg_int(config, tc, keys=_TOPK_KEYS, default=0)
    if top_k <= 0:
        prof.supported = False
        prof.reason = "active-expert count (num_experts_per_tok) missing in config"
        return prof
    prof.top_k = top_k
    # Uniform experts => active slice of the per-layer routed stored bytes.
    per_layer = est.per_layer_expert_bytes
    prof.routed_active_bytes_per_layer = per_layer * top_k // max(1, est.experts_per_layer)
    # Shared experts: resident, but report their per-layer stored bytes.
    try:
        wmap = _safetensors_index(mdir)
        shared_keys = [k for k in wmap if "shared_expert" in k]
        if shared_keys:
            sizes = _header_sizes(mdir, {k: wmap[k] for k in shared_keys})
            prof.shared_bytes_per_layer = sum(sizes.values()) // max(1, est.num_moe_layers)
    except Exception as e:
        logger.warning("shared-expert scan failed: %s", e)
    prof.bytes_per_step = prof.num_moe_layers * prof.routed_active_bytes_per_layer
    return prof


# ===========================================================================
# Prediction
# ===========================================================================


@dataclass
class RooflinePrediction:
    bytes_per_step: int
    bytes_verify: int
    verify_byte_mult: float
    tok_per_cycle: float
    ceiling_base_tok_s: float
    ceiling_mtp_tok_s: float
    mtp_profitable: bool
    margin_tok_per_cycle: float  # tok_per_cycle - verify_byte_mult
    explanation: str


def predict_roofline(
    profile: MoEStepProfile,
    measurement: StorageMeasurement,
    tok_per_cycle: float = 1.0,
    verify_byte_mult: float = _VERIFY_BYTE_MULT_DEFAULT,
) -> RooflinePrediction:
    """Ceilings + MTP verdict from profile x measurement.

    `tok_per_cycle` (1 + accept_rate for depth-1) and `verify_byte_mult`
    come from a Gap-2 style bench; defaults describe the base step only.
    The verdict is deliberately independent of absolute bandwidth: both
    arms share the same SSD, so only the byte ratio matters.
    """
    bw = measurement.rand_read_Bps
    bps = max(1, profile.bytes_per_step)
    bverify = int(bps * verify_byte_mult)
    ceiling_base = bw / bps
    ceiling_mtp = bw * tok_per_cycle / max(1, bverify)
    profitable = tok_per_cycle > verify_byte_mult
    margin = tok_per_cycle - verify_byte_mult
    # The ceilings assume every expert byte comes cold off SSD each step
    # (worst case). The real engine usually beats the base ceiling via
    # temporal locality (reselected experts hit the page cache) + prefetch
    # overlap — that shows up as calibration efficiency > 100% and does NOT
    # invalidate the verdict, which is a byte-*ratio* argument: both arms
    # share the same locality/prefetch regime.
    if profitable:
        expl = (
            f"MTP pays: {tok_per_cycle:.2f} tok/cycle > {verify_byte_mult:.2f}x "
            f"verify bytes (cold ceiling {ceiling_mtp:.2f} vs base "
            f"{ceiling_base:.2f} tok/s)"
        )
    else:
        expl = (
            f"MTP loses structurally: {tok_per_cycle:.2f} tok/cycle < "
            f"{verify_byte_mult:.2f}x verify bytes (cold ceiling {ceiling_mtp:.2f} "
            f"vs base {ceiling_base:.2f} tok/s). No faster SSD closes this "
            f"gap — only more stages or resident weights do."
        )
    return RooflinePrediction(
        bytes_per_step=profile.bytes_per_step,
        bytes_verify=bverify,
        verify_byte_mult=verify_byte_mult,
        tok_per_cycle=tok_per_cycle,
        ceiling_base_tok_s=ceiling_base,
        ceiling_mtp_tok_s=ceiling_mtp,
        mtp_profitable=profitable,
        margin_tok_per_cycle=margin,
        explanation=expl,
    )


# ===========================================================================
# Report persistence (bench/results convention — gitignored, like benches)
# ===========================================================================


def _results_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "bench" / "results" / "storage_roofline"


def save_report(report: dict, slug: str) -> Path:
    out_dir = _results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def load_report(slug: str) -> dict | None:
    path = _results_dir() / f"{slug}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def build_report(
    volume: VolumeInfo,
    measurement: StorageMeasurement,
    profile: MoEStepProfile | None,
    prediction: RooflinePrediction | None,
    measured_base_tok_s: float | None = None,
) -> dict:
    report: dict = {
        "version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "volume": asdict(volume),
        "measurement": asdict(measurement),
    }
    if profile is not None:
        report["profile"] = asdict(profile)
    if prediction is not None:
        report["prediction"] = asdict(prediction)
    if measured_base_tok_s:
        ceil = (prediction.ceiling_base_tok_s if prediction else 0.0) or 0.0
        report["calibration"] = {
            "measured_base_tok_s": measured_base_tok_s,
            "predicted_ceiling_base_tok_s": ceil,
            "efficiency": (measured_base_tok_s / ceil) if ceil > 0 else 0.0,
        }
    return report
