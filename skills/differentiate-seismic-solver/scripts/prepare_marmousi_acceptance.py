#!/usr/bin/env python3
"""Validate pinned supplied Marmousi-II arrays and split training from truth."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np

REFERENCE = Path(__file__).resolve().parents[1] / 'references/marmousi-2d-input.json'
FIELD_KEYS = ('vp_true', 'vp_initial', 'vs_true', 'rho_true')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<f8').tobytes()).hexdigest()


def validate(arrays, reference):
    """Reject changed input before exporting anything; never repair its fields."""
    shape = tuple(reference['shape_depth_x'])
    for key in FIELD_KEYS + ('x', 'z'):
        a = arrays[key]
        if a.dtype != np.dtype('float64') or not np.isfinite(a).all():
            raise ValueError(f'{key}: finite float64 samples required')
        expected = shape if key in FIELD_KEYS else (shape[1 if key == 'x' else 0],)
        if a.shape != expected or array_hash(a) != reference['array_sha256'][key]:
            raise ValueError(f'{key}: canonical shape/content mismatch; do not crop or reblur')
    for key, extent, n in [('x', reference['original_x_m'], shape[1]),
                           ('z', reference['depth_m'], shape[0])]:
        expected = extent[0] + np.arange(n, dtype=np.float64) * reference['spacing_m']
        if expected[-1] != extent[1] or not np.array_equal(arrays[key], expected):
            raise ValueError(f'{key}: canonical physical coordinates mismatch')
    water = arrays['vs_true'] == 0.
    expected_water = np.broadcast_to((arrays['z'] <= reference['water_depth_m'])[:, None], shape)
    if not np.array_equal(water, expected_water):
        raise ValueError('Acoustic-water mask changed')
    for key in ('vp_true', 'vp_initial'):
        vp = arrays[key]
        if not np.all(vp[water] == reference['water_vp_m_s']):
            raise ValueError('Fixed water Vp changed')
        if np.any(vp**2 - 4./3.*arrays['vs_true']**2 <= 0.):
            raise ValueError('Nonpositive isotropic bulk modulus')
    if np.any(arrays['vs_true'] < 0.) or not np.all(arrays['rho_true'] == reference['fixed_density_kg_m3']):
        raise ValueError('Fixed priors changed')
    return water


def prepare(source, output, reference_path=REFERENCE):
    source, output, reference_path = map(Path, (source, output, reference_path))
    if output.exists():
        raise FileExistsError(output)
    reference = json.loads(reference_path.read_text())
    with np.load(source, allow_pickle=False) as f:
        arrays = {key: f[key].copy() for key in FIELD_KEYS + ('x', 'z')}
    water = validate(arrays, reference)
    packages = {
        'training.npz': dict(vp_initial=arrays['vp_initial'], vs_fixed=arrays['vs_true'],
                             rho_fixed=arrays['rho_true'], x_m=arrays['x'], depth_m=arrays['z'],
                             water_mask=water),
        'truth_for_observations_and_offline_only.npz': dict(vp=arrays['vp_true'],
                                                          x_m=arrays['x'], depth_m=arrays['z'])}
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='inputs_verified_only_not_a_completed_inversion',
                    reference_name=reference['name'], source_sha256=file_hash(source),
                    reference_sha256=file_hash(reference_path), transforms=[], outputs={})
    with tempfile.TemporaryDirectory(prefix=output.name + '-preparing-', dir=output.parent) as tmp:
        staging = Path(tmp)
        for name, values in packages.items():
            path = staging / name
            np.savez_compressed(path, **values)
            with np.load(path, allow_pickle=False) as check:
                for key, value in values.items():
                    if not np.array_equal(check[key], value):
                        raise ValueError('Exported array changed')
            manifest['outputs'][name] = dict(sha256=file_hash(path), arrays={
                key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_hash(value))
                for key, value in values.items()})
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.output)
    print(json.dumps(dict(status=result['status'], output=str(args.output), reference=result['reference_name'])))


if __name__ == '__main__':
    main()
