# Model-centred visual delivery

Use concise English labels and captions by default. Lead with the actual model
recovery, not a diagnostics gallery. Default to one main model plate and a short
English title/status; a small results table is optional. Link detailed provenance
and numerical evidence instead of repeating it in the page text.

## Default model plate

| Actual model | Main comparison |
|---|---|
| Planar 2-D | One row: **True / Initial / Inverted** |
| Volumetric Cartesian 3-D | One 3×3 plate: columns **True / Initial / Inverted**; rows **XZ / YZ / XY** |
| Other geometry or parameterization | Matching, physically meaningful views of the actual fields; do not invent Cartesian volumes or dense fields |

For a 3-D plate declare one y-coordinate for XZ, one x-coordinate for YZ and
one depth for XY. Use those same physical sections in all three model columns.
Choose these coordinates from the case geometry before inspecting recovery.
Use the geometric midplanes by default and record the actual grid indices. Label physical axes,
section coordinates and units. Preserve the declared extent. Do not add a second
central-section triptych below the 3×3 plate.

Use a shared **Blues** scale for comparable positive velocities: low values
light, high values dark. Compute a single minimum and maximum from all three
complete models within the same native coverage, before extracting sections.
Use this full range for every panel and the initial view of an interactive volume.
Where a coverage mask is needed, derive it from the native mesh or declared
simulation domain before evaluating recovery and identify the excluded cells.
Apply the same mask to every model. Convert units only in
an explicit offline export. Respect an explicit user colourmap choice. Other
physical quantities need an appropriate scale rather than inheriting a velocity
palette. Fixed true priors are inputs, not reconstructed fields.

If truth is unavailable, show Initial / Inverted and label that fact; never
substitute a prior or fabricate truth. If no inversion has run, label an input
view as input-only rather than calling the initial array an inverted model.
A single field's matching sections suffice. This rendering layout is generic;
the named [3D acceptance](salt3d-acceptance.md) separately mandates its bundled
fixed model, section coordinates and baseline acquisition. Do not interpret
layout flexibility as permission to substitute that model. Solver-native mesh
and execution remain target-specific.

## Minimal report and archived evidence

Show the actual accepted update count, running/completed/interrupted status and
stopping reason in a short English line. A small table may include the objective
and independently computed recovery metrics. Distinguish data fit from model
recovery; more epochs or a small loss do not guarantee geological resolution.
The manifest reports initial and final RMSE on the full control grid and, when a
native-coverage mask is supplied, also over the covered cells, with sample counts
and physical units for each. Keep true-model diagnostics outside optimizer selection.

An interactive **3D View** and **Loss** plot may be collapsed supplements when
useful. Use actual accepted iteration numbers and frequency-stage boundaries;
do not join differently normalized losses as one monotone curve. The 3-D volume
supplements the orthogonal sections and does not replace them.

Do not include sensitivity, kernel or gradient plots in the default report,
even as automatically added hidden galleries. Do not require wavefield, data,
residual, cotangent, optimizer-direction or update-map galleries. Generate such
diagnostic plots only for an explicit user request;
keep them separate from the default model-recovery presentation.

All numerical requirements remain in force. Archive the following as raw arrays,
JSON records, configurations and manifests, with a short evidence link:

- full native-forward, data-VJP, sampling/propagation/model adjoint and Taylor
  results, including failed steps and roundoff regions;
- raw coefficient/model gradients with their real coordinates and units,
  optimizer directions and committed updates as distinct quantities;
- full-shot model fingerprints, complete reductions, accepted-update ledger,
  objective/stage history, constraints and actual stopping reason;
- native state/data/cotangent artifacts required by the derivative, history or
  reproduction contract, with source/component/time and resource provenance.

Evidence lists in other references specify what must be retained and verified;
they do not require a plot of every artifact in the public report. Hiding a
figure never permits discarding a failed check, smoothing a gradient to claim
correctness, or replacing quantitative acceptance with a model picture.

## End-of-inversion model comparison

Wire an offline plotting action into the runner's terminal path for normal
completion and budget exhaustion. Render the main model plate from the actual
truth, exact initial input and **last accepted checkpoint**, not a trial, stale
preview or pre-update gradient model. Save a readable PNG plus editable PDF/SVG
and link the main plate in the final report/README. A periodic preview alone
does not close the deliverable.

Bind the figure manifest to the checkpoint/model hash, accepted update count,
actual stopping reason and plotting command. For interruptions, label the figure
as the last accepted model at interruption; an iteration cap or terminal
evaluation is not proof of convergence. Keep re-rendering independent of
propagation and updates: presentation changes must use saved arrays and must
not trigger additional solver runs. Unavailable optional state views can be
omitted rather than simulated just to fill the report.

Use matching physical coordinates and units. Preserve the original arrays and
record the coverage and full range used for the shared scale in the manifest.

## Provenance and rendered review

Provide plotting source, an exact offline command and a figure manifest with
input hashes/array keys, active field, physical coordinates/sections, units,
transforms, shared physical range and output files. Archive complete numerical
settings and evidence JSON separately from the concise report. Publish only
permitted data and keep workstation paths out of captions.

Open the main rendered plate and verify its model identities, row/column order,
matching sections, colourbar direction, limits, labels and accepted-update
count. Inspect optional views when included; do not claim browser-interaction
QA from HTML serialization alone. This presentation review requires no solver
calls and makes no new numerical or performance claim.

When diagnostic state plots are explicitly requested, use saved native states,
physical times/components and correct topology. Label modal, meridional or
interpolated sections honestly. Merge MPI state replicas only after checking
physical identity and agreement; element-local coefficient derivatives must
follow the actual parameter-map reduction. Never fabricate wavefields or
broadcast a few regional derivatives into a dense-looking kernel. Signed
residuals/gradients need zero-centred diverging scales, not the velocity scale.

## Optional portable renderers

For saved Cartesian model volumes, the included report helper implements the
final presentation directly from an accepted checkpoint:

```bash
python scripts/model_recovery_report.py models.npz metadata.json report
```

Use NPZ keys `true`, `initial`, `inverted` and strictly increasing physical
`x`, `depth` axes (shape nx,nz), or `x`, `y`, `depth` (shape nx,ny,nz).
Transpose explicitly upstream; no implicit axis guessing is performed. Optional
`loss` and `loss_iterations` describe one consistent objective/stage. The JSON
requires `coordinate_unit`, `value_unit`, `accepted_updates`, `status` and
`model_is_last_accepted: true`; completed runs also require `stopping_reason`
and `source_checkpoint_sha256`. Supply requested physical `sections`
(`x`, `y`, `depth`), conversion scales/units if needed and a `provenance` object.
The helper records the nearest saved section coordinates, shared limits,
coverage counts, input hashes and output hashes in its manifest. Old metadata
with explicit limits can be used to regenerate the report with the full range. Inspect
the command's docstring for optional masks, acquisition markers and controls.

Enable `include_3d` or `include_loss` for collapsed supplements. Offline 3D
shows the saved model cells in a standalone browser canvas. Rotate the three
models together or raise the displayed-value threshold to inspect the interior.
The viewer uses the physical grid spacing, native coverage and the same Blues
scale as the section plate. Source and receiver markers can be toggled.
The report helper is optional: adapters with other physical geometry can render
equivalent native-aware comparisons and retain the same evidence bindings.

`python scripts/visualize.py --arrays case.npz --spec figures.json --output figures`
uses saved arrays and does not run or differentiate a solver. A 2-D model plate
uses three map panels with `columns: 3`, labels `True`, `Initial`, `Inverted`,
physical extents/axis units, `cmap: "Blues"` and one shared
`scale_group: "velocity"`. Use the volume report helper above for 3-D models so that values outside the
plotted sections contribute to the shared colour scale.

The helper also supports points, curves, bars and animations for requested
diagnostics. Other plotting libraries and native-aware exports are valid when
the same model identity, coordinate, scale and evidence requirements are met.
