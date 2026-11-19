#!/usr/bin/env python3
"""Prepare or verify the UH AGL Marmousi-II inputs for the full-shot inversion."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import sys

import numpy as np

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT/'scripts'))
from agl_marmousi import PACKAGES, default_cache, ensure_packages, write_paper_npz

REFERENCE = SKILL_ROOT/'references/marmousi-2d-fast40-softdtw.json'
BUNDLE_DIR = SKILL_ROOT/'assets/marmousi2d-acceptance'
PROFILE_ID = 'marmousi-ii-fast40-softdtw-v1'
MODEL_SHA256 = '17a59061b2c68095969fbc68ab2901ecc41ac8774cf1bfc70370bd1c8a424b93'
MANIFEST_SHA256 = '4af98c6f1a6b0b6a3356ad24896d35c73afe9e94b0746114acc46dc01c1c1564'
PROFILE_SHA256 = 'bdc40092e73856d9cf6f47c297fda60ce0e0b0442d8ad924127a9c4fd1898b9f'
BUNDLE_FILES = ('model_fields.npz', 'training.npz',
                'truth_for_observations_and_offline_only.npz',
                'DATA_ATTRIBUTION.txt', 'DATA_TERMS.txt')
MODEL_FIELDS = ('vp_true', 'vp_initial', 'vs_fixed', 'rho_fixed', 'x_m',
                'depth_m', 'sampling_depth_m', 'water_mask')
TRAINING_FIELDS = ('vp_initial', 'vs_fixed', 'rho_fixed', 'x', 'z', 'sampling_z_m')
TRUTH_FIELDS = ('vp_true', 'vs_true', 'rho_true', 'water_mask')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate(training, truth, reference):
    shape = tuple(reference['physical_input']['shape_depth_x'])
    fields = {**training, **truth}
    for key, item in reference['arrays'].items():
        require(key in fields, 'Missing pinned input array: '+key)
        array = fields[key]
        require(str(array.dtype) == item['dtype'] and tuple(array.shape) == tuple(item['shape']),
                'Pinned dtype/shape changed: '+key)
        require(np.isfinite(array).all() and array_hash(array) == item['sha256'],
                'Pinned array content changed; no crop, reblur or repair: '+key)
    for key in ('vp_initial', 'vs_fixed', 'rho_fixed'):
        require(training[key].shape == shape, 'Physical field shape differs')
    require(np.array_equal(training['vs_fixed'], truth['vs_true']) and
            np.array_equal(training['rho_fixed'], truth['rho_true']), 'Fixed priors differ from supplied true fields')
    for key, extent, length in [('x', 'original_x_m', shape[1]), ('z', 'depth_m', shape[0])]:
        expected = reference['physical_input'][extent][0]+np.arange(length, dtype=np.float64)*reference['physical_input']['spacing_m']
        require(np.array_equal(training[key], expected) and expected[-1] == reference['physical_input'][extent][1],
                'Physical coordinates differ: '+key)
    depth = np.minimum(training['z'], reference['physical_input']['source_max_depth_m'])
    require(np.array_equal(training['sampling_z_m'], depth), 'Source boundary sampling changed')
    water = truth['vs_true'] == 0
    require(np.array_equal(water, truth['water_mask']) and bool(water.any()) == reference['physical_input']['water_in_this_region'],
            'Declared physical water/solid mask changed')
    require((training['rho_fixed'] > 0).all() and (training['vs_fixed'] >= 0).all(), 'Nonphysical fixed field')
    for vp in (training['vp_initial'], truth['vp_true']):
        require((vp.astype(np.float64)**2-4./3*training['vs_fixed'].astype(np.float64)**2 > 0).all(),
                'Nonpositive isotropic bulk modulus')
    return water


def _npz_arrays(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _metadata(source_dir, reference_path):
    source_dir, reference_path = Path(source_dir), Path(reference_path)
    manifest_bytes = (source_dir/'manifest.json').read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256,
            'Reference manifest byte identity mismatch')
    manifest = json.loads(manifest_bytes)
    profile_bytes = reference_path.read_bytes()
    require(hashlib.sha256(profile_bytes).hexdigest() == PROFILE_SHA256,
            'Reference profile byte identity mismatch')
    profile = json.loads(profile_bytes)
    require(manifest['profile_id'] == profile['profile_id'] == PROFILE_ID,
            'Reference profile identity mismatch')
    require(manifest['profile'] == {'name': 'protocol.json', 'sha256': PROFILE_SHA256},
            'Reference profile binding mismatch')
    require(set(manifest['files']) == set(BUNDLE_FILES), 'Reference file list mismatch')
    return manifest, profile, manifest_bytes, profile_bytes


def _fixture_payload(source_dir, reference_path):
    source_dir = Path(source_dir)
    manifest, profile, manifest_bytes, profile_bytes = _metadata(source_dir, reference_path)
    payload = {}
    for name in BUNDLE_FILES:
        data = (source_dir/name).read_bytes()
        record = manifest['files'][name]
        require(len(data) == record['bytes'] and
                hashlib.sha256(data).hexdigest() == record['sha256'],
                'Reference file byte identity mismatch: '+name)
        payload[name] = data
    require(hashlib.sha256(payload['model_fields.npz']).hexdigest() == MODEL_SHA256,
            'Reference model byte identity mismatch')
    arrays = _npz_arrays(payload['model_fields.npz'])
    require(set(arrays) == set(MODEL_FIELDS), 'Reference model fields mismatch')
    for key, array in arrays.items():
        require(manifest['arrays'][key] == dict(dtype=str(array.dtype),
                shape=list(array.shape), sha256=array_hash(array)),
                'Reference array identity mismatch: '+key)
    training = dict(vp_initial=arrays['vp_initial'], vs_fixed=arrays['vs_fixed'],
                    rho_fixed=arrays['rho_fixed'], x=arrays['x_m'],
                    z=arrays['depth_m'], sampling_z_m=arrays['sampling_depth_m'])
    truth = dict(vp_true=arrays['vp_true'], vs_true=arrays['vs_fixed'],
                 rho_true=arrays['rho_fixed'], water_mask=arrays['water_mask'])
    validate(training, truth, profile)
    separated = {
        'training.npz': set(MODEL_FIELDS)-{'vp_true'},
        'truth_for_observations_and_offline_only.npz': {
            'vp_true', 'x_m', 'depth_m', 'sampling_depth_m'}}
    for name, fields in separated.items():
        values = _npz_arrays(payload[name])
        require(set(values) == fields, 'Truth-isolated package fields mismatch: '+name)
        for key, value in values.items():
            require(value.dtype == arrays[key].dtype and np.array_equal(value, arrays[key]),
                    'Truth-isolated package array mismatch: '+key)
    payload['manifest.json'] = manifest_bytes
    payload['protocol.json'] = profile_bytes
    return payload


def validate_fixture(npz_path=None, reference_path=None):
    """Verify the complete canonical bundle, not only a caller-supplied hash."""
    if npz_path is None:
        case = default_cache()/'skill'/PROFILE_ID
        export_fixture(case)
        npz_path = case/'model_fields.npz'
    npz_path = Path(npz_path)
    require(npz_path.name == 'model_fields.npz', 'Expected model_fields.npz')
    if reference_path is None:
        reference_path = npz_path.parent/'protocol.json'
    _fixture_payload(npz_path.parent, reference_path)
    return dict(status='inputs_verified_only_not_a_completed_inversion',
                profile_id=PROFILE_ID, model_sha256=MODEL_SHA256, transformations=[])


def export_fixture(output, source_dir=None, reference_path=None, *,
                   raw_dir=None, cache_dir=None, prepared_pair=None):
    """Prepare and atomically export the paper inputs, or reuse a verified export."""
    output = Path(output)
    reference = REFERENCE if reference_path is None else Path(reference_path)
    metadata_dir = BUNDLE_DIR if source_dir is None else Path(source_dir)
    manifest, profile, manifest_bytes, profile_bytes = _metadata(metadata_dir, reference)
    if output.exists():
        try:
            payload = _fixture_payload(output, output/'protocol.json')
        except (OSError, ValueError) as exc:
            raise FileExistsError(f'{output}: conflicting/incomplete export; no files changed') from exc
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=output.name+'-preparing-', dir=output.parent) as temporary:
            staging = Path(temporary)/'case'
            if prepared_pair is not None:
                training_path, truth_path = prepared_pair
                with np.load(training_path, allow_pickle=False) as train, np.load(truth_path, allow_pickle=False) as truth:
                    fields = {name: truth[name] if name == 'vp_true' else train[name] for name in MODEL_FIELDS}
                staging.mkdir()
                write_paper_npz(staging/'model_fields.npz', fields)
                (staging/'training.npz').write_bytes(training_path.read_bytes())
                (staging/'truth_for_observations_and_offline_only.npz').write_bytes(truth_path.read_bytes())
            elif source_dir is not None:
                staging.mkdir()
                for name in PACKAGES:
                    (staging/name).write_bytes((metadata_dir/name).read_bytes())
            else:
                ensure_packages(staging, profile, manifest, source_dir=raw_dir, cache_dir=cache_dir)
            for name in ('DATA_ATTRIBUTION.txt', 'DATA_TERMS.txt'):
                (staging/name).write_bytes((metadata_dir/name).read_bytes())
            (staging/'manifest.json').write_bytes(manifest_bytes)
            (staging/'protocol.json').write_bytes(profile_bytes)
            payload = _fixture_payload(staging, staging/'protocol.json')
            try:
                staging.rename(output)
            except OSError:
                if not output.is_dir():
                    raise
                _fixture_payload(output, output/'protocol.json')
    return dict(status='inputs_verified_only_not_a_completed_inversion',
                profile_id=PROFILE_ID, model_sha256=MODEL_SHA256, transformations=[],
                native_execution_not_prescribed=True,
                outputs={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
                         for name, data in payload.items()})


def prepare(training_path, truth_path, output, reference_path=REFERENCE):
    """Verify supplied training and truth packages for the selected edition."""
    training_path, truth_path, output, reference_path = map(Path, (training_path, truth_path, output, reference_path))
    if output.exists():
        raise FileExistsError(output)
    reference = json.loads(reference_path.read_text())
    if reference['profile_id'] == PROFILE_ID and 'prepared_packages' in reference:
        require(file_hash(reference_path) == PROFILE_SHA256, 'AGL profile byte identity mismatch')
        for name, path in [('training.npz', training_path),
                           ('truth_for_observations_and_offline_only.npz', truth_path)]:
            require(file_hash(path) == reference['prepared_packages'][name]['sha256'],
                    'Supplied file differs from the AGL paper inputs: '+name)
        return export_fixture(output, reference_path=reference_path, prepared_pair=(training_path, truth_path))
    expected_packages = reference['source_packages']
    for name, path in [('training.npz', training_path), ('truth.npz', truth_path)]:
        require(file_hash(path) == expected_packages[name]['sha256'], 'Supplied package bytes changed: '+name)
    with np.load(training_path, allow_pickle=False) as package:
        require(not {'vp_true', 'target_vp', 'vp'} & set(package.files), 'Unknown target Vp present in training input')
        training = {key: package[key].copy() for key in TRAINING_FIELDS}
    with np.load(truth_path, allow_pickle=False) as package:
        truth = {key: package[key].copy() for key in TRUTH_FIELDS}
    water = validate(training, truth, reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    contents = {
        'training.npz': dict(vp_initial=training['vp_initial'], vs_fixed=training['vs_fixed'],
            rho_fixed=training['rho_fixed'], x_m=training['x'], depth_m=training['z'],
            sampling_depth_m=training['sampling_z_m'], water_mask=water),
        'truth_for_observations_and_offline_only.npz': dict(vp_true=truth['vp_true'],
            x_m=training['x'], depth_m=training['z'], sampling_depth_m=training['sampling_z_m'])}
    manifest = dict(schema='fast40_marmousi_prepared_v1', profile_id=reference['profile_id'],
        status='inputs_verified_only_not_a_completed_inversion', profile_sha256=file_hash(reference_path),
        source_sha256={'training.npz':file_hash(training_path), 'truth.npz':file_hash(truth_path)},
        transformations=[], native_execution_not_prescribed=True, outputs={})
    with tempfile.TemporaryDirectory(prefix=output.name+'-preparing-', dir=output.parent) as temporary:
        staging = Path(temporary)
        for name, arrays in contents.items():
            target = staging/name
            write_paper_npz(target, arrays)
            with np.load(target, allow_pickle=False) as check:
                for key, array in arrays.items():
                    require(check[key].dtype == array.dtype and np.array_equal(check[key], array), 'Exported samples changed')
            manifest['outputs'][name] = dict(sha256=file_hash(target), arrays={
                key:dict(dtype=str(value.dtype), shape=list(value.shape), sha256=array_hash(value)) for key,value in arrays.items()})
        (staging/'protocol.json').write_bytes(reference_path.read_bytes())
        manifest['outputs']['protocol.json'] = dict(sha256=file_hash(staging/'protocol.json'))
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--output', type=Path, help='Prepare and export the UH AGL paper inputs')
    action.add_argument('--validate', type=Path, metavar='MODEL_NPZ', help='Verify an existing model_fields.npz bundle')
    parser.add_argument('--training', type=Path, help='Verify a prepared AGL training.npz together with --truth')
    parser.add_argument('--truth', type=Path, help='Verify the corresponding AGL truth NPZ together with --training')
    parser.add_argument('--source-dir', type=Path, help='Use three local AGL .segy.gz archives offline')
    parser.add_argument('--cache-dir', type=Path, help='Cache for original archives downloaded from AGL')
    args = parser.parse_args()
    if (args.source_dir is not None or args.cache_dir is not None) and (args.validate or args.training or args.truth):
        parser.error('--source-dir and --cache-dir apply only to preparation from original archives')
    if (args.training is None) != (args.truth is None):
        parser.error('--training and --truth must be supplied together')
    if args.validate:
        if args.training is not None:
            parser.error('--validate cannot be combined with external input packages')
        result = validate_fixture(args.validate)
    elif args.training is not None:
        result = prepare(args.training, args.truth, args.output)
    else:
        result = export_fixture(args.output, raw_dir=args.source_dir, cache_dir=args.cache_dir)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
