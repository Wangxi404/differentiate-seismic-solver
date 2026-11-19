"""Prepare the paper's 2D Marmousi-II inputs from UH AGL SEG-Y files."""

from http.client import HTTPException
from pathlib import Path
import gzip
import hashlib
import io
import os
import struct
import tempfile
from urllib.request import urlopen
import zipfile
import zlib

import numpy as np
from scipy.ndimage import gaussian_filter


ARCHIVE_DATES = {
    "vp_marmousi-ii.segy.gz": "20070821164514",
    "vs_marmousi-ii.segy.gz": "20070821164456",
    "density_marmousi-ii.segy.gz": "20070821164444",
}
PACKAGES = ("model_fields.npz", "training.npz", "truth_for_observations_and_offline_only.npz")


def default_cache():
    cache = os.environ.get("MARMOUSI2_AGL_CACHE")
    if cache:
        return Path(cache).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "marmousi2-agl"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_paper_npz(path, arrays):
    """Write the paper's NPY 1.0 members with reproducible ZIP headers."""
    with zipfile.ZipFile(path, "w", allowZip64=False) as archive:
        for key, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, version=(1, 0), allow_pickle=False)
            npy = buffer.getvalue()
            compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
            compressed_size = len(compressor.compress(npy)) + len(compressor.flush())
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o600 << 16
            # The reference ZIP reserves 64-bit sizes only in each local header.
            info.extra = struct.pack("<HHQQ", 1, 16, len(npy), compressed_size)
            archive.writestr(info, npy, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            info.extra = b""


def source_file(directory, name, record, download):
    path = directory / name
    if path.is_file() and digest(path) == record["sha256"]:
        return path
    if not download:
        raise ValueError(f"Missing or incorrect AGL source: {path}")
    https = record["url"]
    if name not in ARCHIVE_DATES or https != f"https://www.agl.uh.edu/downloads/{name}":
        raise ValueError(f"Unexpected AGL source: {name}")
    http = "http://www.agl.uh.edu/downloads/" + name
    snapshot = f"https://web.archive.org/web/{ARCHIVE_DATES[name]}id_/{http}"
    directory.mkdir(parents=True, exist_ok=True)
    guidance = f"Retry later, put the original .segy.gz files in {directory}, or supply --source-dir."
    for url, timeout in ((https, 30), (http, 30), (snapshot, 120)):
        try:
            with urlopen(url, timeout=timeout) as response:
                data = response.read(record["bytes"] + 1)
        except (OSError, HTTPException) as exc:
            if url == snapshot:
                raise RuntimeError(f"AGL download unavailable for {name}. {guidance}") from exc
            continue
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            if url == snapshot:
                raise ValueError(f"Downloaded archive differs from the UH AGL reference: {name}. {guidance}")
            continue
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            pending = Path(tmp) / name
            pending.write_bytes(data)
            pending.replace(path)
        return path


def sample_agl(path, x, depth):
    ix = (x / 1.25).astype(int)
    iz = (depth / 1.25).astype(int)
    if not (np.array_equal(ix * 1.25, x) and np.array_equal(iz * 1.25, depth)):
        raise ValueError("Coordinates must coincide with AGL's 1.25 m source grid")
    with gzip.open(path, "rb") as stream:
        header = stream.read(3600)
        nz = int.from_bytes(header[3220:3222], "big")
        if nz != 2801 or int.from_bytes(header[3224:3226], "big") != 1:
            raise ValueError(f"Unexpected SEG-Y sample count or encoding in {path.name}")
        columns = []
        for trace in ix:
            stream.seek(3600 + int(trace) * (240 + 4 * nz) + 240)
            words = np.frombuffer(stream.read(4 * nz), dtype=">u4")[iz]
            # SEG-Y format 1 uses a base-16 exponent and a 24-bit unsigned fraction.
            exponent = ((words >> 24) & 127).astype(int) - 64
            values = np.ldexp((words & 0xffffff).astype(float), 4 * exponent - 24)
            columns.append(np.where(words >> 31, -values, values))
    return np.stack(columns, axis=1) * 1000.0


def ensure_packages(output, profile, manifest, source_dir=None, cache_dir=None):
    """Create a complete directory atomically, or reuse an identical materialization."""
    output = Path(output)
    expected = {n: manifest["files"][n]["sha256"] for n in PACKAGES}

    def verify():
        for name, checksum in expected.items():
            path = output / name
            if not path.is_file() or digest(path) != checksum:
                raise ValueError(f"Existing model directory differs from the paper inputs: {path}")

    if output.exists():
        verify()
        return output
    cache = default_cache() if cache_dir is None else Path(cache_dir)
    source = Path(source_dir) if source_dir is not None else cache / "raw"
    geometry = profile["physical_input"]
    nz, nx = geometry["shape_depth_x"]
    step = geometry["spacing_m"]
    x = geometry["original_x_m"][0] + step * np.arange(nx)
    z = geometry["depth_m"][0] + step * np.arange(nz)
    sz = np.minimum(z, geometry["source_max_depth_m"])
    fields = []
    for component in ("vp", "vs", "density"):
        filename = f"{component}_marmousi-ii.segy.gz"
        path = source_file(source, filename, profile["sources"][filename], source_dir is None)
        fields.append(sample_agl(path, x, sz))
    vp, vs, rho = fields
    smoothed = gaussian_filter(vp, geometry["initial_smoothing_sigma_m"] / step, mode="reflect")
    initial = np.maximum(smoothed, np.sqrt(4 / 3 * vs**2 + 100**2) + 1)
    data = dict(vp_true=vp.astype("float32"), vp_initial=initial.astype("float32"),
                vs_fixed=vs.astype("float32"), rho_fixed=rho.astype("float32"),
                x_m=x, depth_m=z, sampling_depth_m=sz, water_mask=vs == 0)
    for key, array in data.items():
        if hashlib.sha256(array.tobytes(order="C")).hexdigest() != manifest["arrays"][key]["sha256"]:
            raise ValueError(f"Reconstructed array differs from the reference: {key}")
    subsets = (data, {k: v for k, v in data.items() if k != "vp_true"},
               {k: data[k] for k in ("vp_true", "x_m", "depth_m", "sampling_depth_m")})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staging = Path(tmp) / "prepared"
        staging.mkdir()
        for name, subset in zip(PACKAGES, subsets):
            path = staging / name
            write_paper_npz(path, subset)
            if digest(path) != expected[name]:
                raise ValueError(f"Reconstructed NPZ differs from the paper reference: {name}")
        try:
            staging.rename(output)
        except OSError:
            if not output.is_dir():
                raise
            verify()
    return output
