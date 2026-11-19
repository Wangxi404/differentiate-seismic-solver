#!/usr/bin/env python3
"""Verify and copy the fixed 3-D salt acceptance inputs without transforming them.

This data-only helper does not run a solver or create observations, checkpoints,
or recovered models. The model and its Gaussian initial model are immutable;
solver-specific native realization belongs in the target project's adapter.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np


SKILL_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = SKILL_ROOT / 'assets' / 'salt3d-acceptance'
PROFILE_PATH = SKILL_ROOT / 'references' / 'salt3d-acceptance.json'
PROFILE_ID = 'seg-eage-salt3d-fixed33-v1'
MODEL_NAME = 'model_fields.npz'
PROFILE_NAME = 'salt3d-acceptance.json'
MODEL_SHA256 = 'fb0d4acc42525ed501e6fe1eb922a7445c7ff17ac9b7c1f0ab8a7e906da7c585'
MODEL_BYTES = 181788
PROFILE_SHA256 = 'bd0cbc03cae5dae7530370585f831c6f79d3237007ea4d24f12a751e94c50395'
MANIFEST_SHA256 = 'ce31bfe04cd68fdb2c5827b8bc94aa0a5dc6892d50f579382945af20dace72c0'
# Earlier exports contain the same model, with a different plotting policy.
PROFILE_BY_MANIFEST = {
    MANIFEST_SHA256: PROFILE_SHA256,
    '7c9e964fd4ad78eb0704894a9dd90245aa2d6f9e5633e5e8736cb7a463e46593':
        '197a769c73c56754b0a66e354e5b05a1dceb4d1436a03bfb308a73a5f72ace0d',
}
NOTICES = {
    'DATA_ATTRIBUTION.txt': {
        'sha256': '67fbc7ce9bbe81c0361978cfdd606f2c33f09af32c3dde063c28f04916d659a3', 'bytes': 3657},
    'DATA_LICENSE.txt': {
        'sha256': '1aaf0648e34e568b31c2f1511958c1915a164511989c693fc133e82a10e557cb', 'bytes': 19050},
}
AXIS_ORDER = ['x', 'y', 'depth']
UNITS = {'x': 'm', 'y': 'm', 'depth': 'm', 'true': 'm/s',
         'initial': 'm/s', 'rho': 'kg/m^3', 'vs': 'm/s'}
# rho and vs are scalar fixed priors in the supplied NPZ, not 3-D arrays.
ARRAYS = {
    'x': ((33,), '6aeb6349e94b72d8e8348956564f8076b5730e115bca2c287480dcf83c8d1a1e'),
    'y': ((33,), '6aeb6349e94b72d8e8348956564f8076b5730e115bca2c287480dcf83c8d1a1e'),
    'depth': ((17,), 'c129ead3149cc8102332960155130471ea03398ee392542a7668efbc9d9357f6'),
    'true': ((33, 33, 17), 'dbf91cd0cd4635ee4e1458b1b0416b056beba34e101350881154772c551fd275'),
    'initial': ((33, 33, 17), '6114c4edacb3149286659c9f151815a49cfea0729c6c5adb3aaf4e1898c6748d'),
    'rho': ((), 'a21cdf9eb908794c7fc839fc50781ee7233462021635fe9a242b4304b73103ed'),
    'vs': ((), '31d060b6110dde735a1c71c8ed6524c1d2755dc60fc34576029d2febb365d30e'),
}
EXPORT_NAMES = (MODEL_NAME, 'manifest.json', 'DATA_ATTRIBUTION.txt',
                'DATA_LICENSE.txt', PROFILE_NAME)


def byte_hash(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return byte_hash(Path(path).read_bytes())


def array_hash(array):
    """SHA-256 of little-endian float64 samples serialized in C index order."""
    return byte_hash(np.ascontiguousarray(array, dtype='<f8').tobytes(order='C'))


def _read_json(data, label):
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'{label}: invalid JSON') from exc
    if not isinstance(result, dict):
        raise ValueError(f'{label}: a JSON object is required')
    return result


def _validate_manifest(manifest):
    if manifest.get('schema_version') != 1 or manifest.get('profile_id') != PROFILE_ID:
        raise ValueError('Manifest schema/profile does not identify the fixed salt fixture')
    if manifest.get('axis_order') != AXIS_ORDER or manifest.get('units') != UNITS:
        raise ValueError('Manifest must preserve x/y/depth axis order and SI units')
    if manifest.get('profile', {}).get('name') != PROFILE_NAME or manifest['profile'].get('sha256') not in PROFILE_BY_MANIFEST.values():
        raise ValueError('Unrecognized fixed salt acceptance profile')
    if manifest.get('notices') != NOTICES:
        raise ValueError('Manifest must preserve canonical data provenance document hashes/sizes')
    model = manifest.get('file', {})
    if not isinstance(model, dict) or any(model.get(k) != v for k, v in {
            'name': MODEL_NAME, 'sha256': MODEL_SHA256, 'bytes': MODEL_BYTES}.items()):
        raise ValueError('Manifest cannot authorize a different NPZ file')
    records = manifest.get('arrays')
    if not isinstance(records, dict) or set(records) != set(ARRAYS):
        raise ValueError('Manifest must contain exactly the seven canonical arrays')
    for name, (shape, digest) in ARRAYS.items():
        record = records[name]
        expected = {'shape': list(shape), 'dtype': '<f8',
                    'sha256_le_f64_C_order': digest}
        if not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items()):
            raise ValueError(f'{name}: manifest canonical shape/dtype/hash mismatch')


def _read_manifest(data):
    expected_profile = PROFILE_BY_MANIFEST.get(byte_hash(data))
    if expected_profile is None:
        raise ValueError('Manifest byte identity mismatch; the bundled exact contract is required')
    manifest = _read_json(data, 'Manifest')
    if manifest.get('profile') != {'name': PROFILE_NAME, 'sha256': expected_profile}:
        raise ValueError('Manifest and acceptance profile hashes do not match')
    return manifest


def validate_arrays(arrays, manifest):
    """Check canonical arrays and physical coordinates; never repair or resample."""
    _validate_manifest(manifest)
    if set(arrays) != set(ARRAYS):
        raise ValueError('NPZ must contain exactly x, y, depth, true, initial, rho, vs')
    for name, (shape, _) in ARRAYS.items():
        array = arrays[name]
        if not isinstance(array, np.ndarray) or array.dtype.str != '<f8':
            raise ValueError(f'{name}: little-endian float64 samples required')
        if array.shape != shape:
            raise ValueError(f'{name}: canonical shape mismatch; do not crop or broadcast')
        if not np.isfinite(array).all():
            raise ValueError(f'{name}: finite samples required')
    coordinates = {'x': -10000. + np.arange(33, dtype=np.float64) * 625.,
                   'y': -10000. + np.arange(33, dtype=np.float64) * 625.,
                   'depth': np.arange(17, dtype=np.float64) * 261.25}
    for name, expected in coordinates.items():
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f'{name}: exact canonical SI coordinates required')
    if np.any(arrays['rho'] <= 0.) or np.any(arrays['vs'] < 0.):
        raise ValueError('Density must be positive and shear speed nonnegative')
    for name in ('true', 'initial'):
        if np.any(arrays[name] <= 0.):
            raise ValueError(f'{name}: positive compressional speed required')
        if np.any(arrays[name] ** 2 - (4. / 3.) * arrays['vs'] ** 2 <= 0.):
            raise ValueError(f'{name}: nonpositive isotropic bulk modulus')
    for name, (_, expected) in ARRAYS.items():
        if array_hash(arrays[name]) != expected:
            raise ValueError(f'{name}: canonical content mismatch; do not change, flip, or reblur')


def _validate_model_bytes(model_bytes, manifest):
    _validate_manifest(manifest)
    if len(model_bytes) != MODEL_BYTES or byte_hash(model_bytes) != MODEL_SHA256:
        raise ValueError('NPZ byte identity mismatch; the bundled exact salt model is required')
    try:
        with np.load(io.BytesIO(model_bytes), allow_pickle=False) as archive:
            if sorted(archive.files) != sorted(ARRAYS):
                raise ValueError('NPZ must contain exactly the seven canonical arrays')
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, EOFError, TypeError) as exc:
        raise ValueError('Cannot read the canonical NPZ arrays') from exc
    validate_arrays(arrays, manifest)


def _validate_profile(profile_bytes, manifest):
    if byte_hash(profile_bytes) != manifest['profile']['sha256']:
        raise ValueError('Acceptance profile hash mismatch')
    profile = _read_json(profile_bytes, 'Profile')
    if profile.get('profile_id') != PROFILE_ID:
        raise ValueError('Acceptance profile does not identify the fixed salt fixture')
    return profile


def validate_fixture(npz_path=BUNDLE_DIR / MODEL_NAME, manifest_path=None, profile_path=None):
    """Validate an NPZ and manifest, including a profile when its hash is declared."""
    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path) if manifest_path is not None else npz_path.parent / 'manifest.json'
    manifest = _read_manifest(manifest_path.read_bytes())
    _validate_model_bytes(npz_path.read_bytes(), manifest)
    if profile_path is None and 'profile' in manifest:
        profile_path = (PROFILE_PATH if npz_path.parent.resolve() == BUNDLE_DIR.resolve()
                        else npz_path.parent / PROFILE_NAME)
    if profile_path is not None:
        _validate_profile(Path(profile_path).read_bytes(), manifest)
    return {'status': 'inputs_verified_only_not_a_completed_inversion',
            'profile_id': PROFILE_ID, 'model_sha256': MODEL_SHA256,
            'transforms': []}


def _bundle_bytes(source_dir, profile_path):
    source_dir, profile_path = Path(source_dir), Path(profile_path)
    payload = {name: (source_dir / name).read_bytes() for name in EXPORT_NAMES if name != PROFILE_NAME}
    payload[PROFILE_NAME] = profile_path.read_bytes()
    manifest = _read_manifest(payload['manifest.json'])
    _validate_model_bytes(payload[MODEL_NAME], manifest)
    _validate_profile(payload[PROFILE_NAME], manifest)
    for name, record in NOTICES.items():
        if len(payload[name]) != record['bytes'] or byte_hash(payload[name]) != record['sha256']:
            raise ValueError(f'{name}: canonical provenance document hash/size mismatch')
    return payload


def export_fixture(output_dir, source_dir=None, profile_path=None):
    """Copy only the five input/provenance files byte for byte, or verify an identical export."""
    output = Path(output_dir)
    payload = _bundle_bytes(BUNDLE_DIR if source_dir is None else source_dir,
                            PROFILE_PATH if profile_path is None else profile_path)
    if output.exists():
        if not output.is_dir() or any(not (output / name).is_file()
                                     or (output / name).read_bytes() != data
                                     for name, data in payload.items()):
            raise FileExistsError(f'{output}: existing export is incomplete or conflicts; no files changed')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=output.name + '-preparing-', dir=output.parent) as temp:
            staging = Path(temp) / 'bundle'
            staging.mkdir()
            for name, data in payload.items():
                (staging / name).write_bytes(data)
            staging.rename(output)
    return {'status': 'inputs_verified_only_not_a_completed_inversion',
            'profile_id': PROFILE_ID, 'model_sha256': MODEL_SHA256,
            'transforms': [], 'outputs': {name: {'sha256': byte_hash(data), 'bytes': len(data)}
                                         for name, data in payload.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--output', type=Path, help='Export the five immutable acceptance input files')
    action.add_argument('--validate', type=Path, metavar='NPZ', help='Validate an existing exact model_fields.npz')
    parser.add_argument('--manifest', type=Path, help='Manifest for --validate (default: adjacent manifest.json)')
    parser.add_argument('--profile', type=Path, help='Profile for --validate (default: bundled or adjacent profile)')
    args = parser.parse_args()
    if args.output is not None:
        if args.manifest is not None or args.profile is not None:
            parser.error('--manifest/--profile apply only to validation')
        result = export_fixture(args.output)
    else:
        model = args.validate if args.validate is not None else BUNDLE_DIR / MODEL_NAME
        profile = args.profile
        if profile is None and args.validate is None:
            profile = PROFILE_PATH
        result = validate_fixture(model, args.manifest, profile)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
