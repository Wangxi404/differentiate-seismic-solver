# Fixed3D salt inversion acceptance

For a3D seismic source-solver application, use **this exact bundled model**.
The task does not select or generate the test geology. This is the same
reduced/refined SEG/EAGE salt Vp and Gaussian initial model used by the recorded
3D inversion. It is a solver-independent input fixture, not an AxiSEM3D adapter.

## Export the supplied inputs

From the installed skill directory, with Python and NumPy available:

```sh
python scripts/prepare_salt3d_acceptance.py --output /path/to/application/case-inputs
python scripts/prepare_salt3d_acceptance.py --validate /path/to/application/case-inputs/model_fields.npz
```

The helper resolves its bundled assets relative to its own location. It exports
five byte-identical files: `model_fields.npz`, `manifest.json`,
`salt3d-acceptance.json`, `DATA_ATTRIBUTION.txt` and `DATA_LICENSE.txt`. It never
downloads/rebuilds geology or runs a simulator. Export checks model, manifest,
profile and original data notices; model validation checks the NPZ/manifest/profile.
Missing/corrupt inputs or conflicting output files fail explicitly. Never fill
the gap with another model or a freshly generated Gaussian blur.

## Mandatory model identity

| Input | Fixed value |
|---|---|
| Profile | `seg-eage-salt3d-fixed33-v1` |
| Vp arrays | `true` and `initial`, float64,33×33×17 |
| Array axis order | x, y, positive-down depth |
| x / y | −10000 to10000m,625m spacing,33 nodes each |
| Depth | 0 to4180m,261.25m spacing,17 nodes |
| Active field | Vp |
| Supplied fixed priors | `vs` scalar850m/s; `rho` scalar2200kg/m³ |
| Initial model | Bundled `initial` exactly, with empty optimizer state for a fresh test |

Package SHA256:
`fb0d4acc42525ed501e6fe1eb922a7445c7ff17ac9b7c1f0ab8a7e906da7c585`.
Canonical little-endian float64/C-order array hashes:

- `true`: `dbf91cd0cd4635ee4e1458b1b0416b056beba34e101350881154772c551fd275`
- `initial`: `6114c4edacb3149286659c9f151815a49cfea0729c6c5adb3aaf4e1898c6748d`

All seven keys, dtypes, scalar/array shapes, values and coordinate vectors are
also independently pinned by the helper and [asset manifest](../assets/salt3d-acceptance/manifest.json).
The [machine-readable profile](salt3d-acceptance.json) carries the full physical
case, preparation provenance and reference configuration.

Do not crop, transpose/flip silently, rescale, regenerate/reblur, refine again,
downsample further or replace the volume by an analytic anomaly, radial average
or extruded section. Same shape/range is insufficient: exact fingerprints must
match. A resource limit is a capacity outcome, not permission to change the
fixture. Only a later explicit user override creates a separately labelled case;
it does not count as completion of this fixed acceptance profile.

The33×33×17 fields are exact nested trilinear prolongations of a previously
prepared17×17×9 case. They preserve those coarse nodes and add no new raw
geological detail. Raw antialiasing, nearest outer extension and Gaussian
initial preparation have already happened; their resulting arrays are
authoritative. The original raw footprint is x/y±6750m, embedded in the supplied
±10000m controls. Keep this provenance and the CC BY4.0 attribution/license
with any redistributed input. Never claim a new raw-volume preparation.

## Run the inversion test

The reference acquisition uses nine independent shots on a3×3 square grid
(x/y−4500,0,4500m; depth522.5m) and169 receivers on a13×13 grid
(x/y−6000 to6000m by1000m; depth261.25m). Exact positions, source waveform
normalization and reference displacement components are in the profile.
Generate observations with the target's own qualified native forward using
the supplied true model and fixed fields. Synthetic truth is for generating
observations and posthoc recovery diagnostics, never optimizer/step selection.

Use **debiased soft-DTW divergence**, gamma0.1, float64 and no band, clamp or
prediction-dependent trace mask. The reference output grid is−2.5..8s at0.01s
(1051samples); zero-padded antialias8±2Hz plus decimation5 gives211samples.
Different native time steps require an explicit qualified observation map;
all prediction-side processing remains in AD. Preserve declared whole-shot
normalization, complete trace/component weights and one mean over all nine
shots. Pass the actual objective VJP and full native model derivative gates.

Start from `initial` and complete **125 accepted full-shot updates** by default:
all shots at one current model, complete gradient reduction, then one update.
Every line-search candidate uses all shots. An explicit continuation restores
the full actual model/optimizer checkpoint; it does not restart or count old
partial-shot work as an update. Keep numerical validity, iteration completion
and geological recovery as separate outcomes.

## Preserve the native solver

The **input fields are fixed**. Native mesh/order/modes, precision, integrator,
coordinate/basis conventions, history strategy, MPI allocation and optimizer
implementation remain target-specific. Derive and test the literal native
model map/VJP while preserving native kernels/execution. Native interpolation
or precision conversion must be disclosed and qualified; neither changes the
canonical package. This fixture supplies no previous adapter, derivative
binary, seismograms, inverted model or optimizer checkpoint.

Do not interpret AxiSEM3D's cylindrical geometry, SPZ/ZRT labels, GaussianSTF
class, NEWMARK implementation, modal layout or host CPU IDs as universal APIs.
The profile labels these as reference provenance. Bind equivalent physical
source/observation coordinates, units, components and timing explicitly.
Unsupported physics/coverage must be reported honestly; it never authorizes
choosing a different/easier model or claiming the full named protocol passed.

Finish with the concise English model-recovery HTML: True / Initial / Inverted
columns and XZ / YZ / XY rows at y = 0, x = 0 and depth = 2090 m. Use one
Blues scale spanning the three complete Vp fields within native coverage,
with high velocities in dark blue. Show loss in a collapsed panel. No kernel/gradient
gallery or duplicate volume view. Bind the last accepted checkpoint and actual
stopping reason; interruption or an iteration cap is not convergence.
