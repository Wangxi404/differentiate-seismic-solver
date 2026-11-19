# End-to-end acceptance for a planar 2-D solver

This case tests whether a skill-guided extension of the supplied forward source
repository can carry a structured model through inversion. The final inverted
model is the outcome to assess. Numerical checks are controls on its validity;
the experiment is not a reconstruction ranking or gradient comparison against
HybridADFWI, which is already differentiable and is not a skill target.

## Select the versioned physical input and protocol

The default profile `marmousi-ii-fast40-softdtw-v1` uses the University of
Houston AGL download edition of Marmousi-II. The model, starting field,
acquisition and inversion settings match the paper’s 2D case.
[Protocol and checksums](marmousi-2d-fast40-softdtw.json)

| Quantity | Current default |
|---|---|
| Physical samples | 68 depth rows × 200 columns; 45 m spacing |
| Original coordinates | x = 5000–13955 m; depth = 500–3515 m |
| Unknown / initial | Vp; exact supplied Gaussian sigma180 m starting array |
| Fixed priors | Supplied true Vs and supplied heterogeneous true density |
| Input precision | Original float32 fields and float64 coordinates; native arithmetic is qualified separately |
| Source-boundary sampling | Nominal final depth3515 m uses the supplied SEG-Y depth3500 m boundary value |
| Acquisition | 40 independent source x-indices2,7,…197; 200 receiver x-indices0,…199; both at depth index2; 5 Hz integrated Ricker |
| Misfit | Debiased soft-DTW; gamma0.1; squared normalized-amplitude local cost; no alignment band |
| Observation axis | 2500 samples at0.003 s, with explicit native/reference time origin and transpose-consistent resampling |
| Misfit sampling | Fixed10 Hz anti-alias filter with2 Hz transition, decimation8 →313 samples |
| Updates | 40 at3 Hz,60 at4.5 Hz,200 full-band;300 maximum |
| Optimizer | One globally averaged full-shot gradient, max-absolute normalization, Adam; physical rate10 m/s for updates1–200 and7.5 m/s thereafter; float64 state |

This region lies below the original water layer; it does not test an
acoustic/elastic interface. Do not relabel it as the former marine case.
The [legacy marine profile](marmousi-2d-legacy-acceptance.md) retains its exact
555×401 /6.25 m arrays, x8–10.5 km, acoustic water, sigma60 m starting model and
constant2450 kg/m³ density. Keep its manifest/helper and all earlier frozen
versions intact. An explicit request for that case or another physical input is
a separately named application, not a silent replacement of this profile.

## Prepare the physical model

Run the helper before generating observations. It uses a verified cached copy
of each Vp, Vs and density **Model Definition** archive when available. For a
missing file it tries the AGL site first, then a Wayback Machine snapshot of
the original AGL URL if the site is unavailable. Each download must match the
paper input's SHA-256 before it is cached and processed locally.
The data directory and the skill installation can be on different file systems.

| NPZ field | Meaning | Units |
|---|---|---|
| `vp_true` | True Vp, for native observations and offline evaluation | m/s |
| `vp_initial` | Exact Gaussian initial Vp | m/s |
| `vs_fixed` | Heterogeneous true Vs held fixed | m/s |
| `rho_fixed` | Heterogeneous true density held fixed | kg/m³ |
| `x_m` | Original horizontal coordinates | m |
| `depth_m` | Nominal depth coordinates | m |
| `sampling_depth_m` | SEG-Y sampling depths, including the final boundary row | m |
| `water_mask` | Boolean water mask; false throughout this region | dimensionless |

The arrays use depth × distance order. Run the commands from the solver
workspace. Set `skill_dir` to the directory containing this skill's `SKILL.md`.

```bash
python -m pip install numpy scipy
skill_dir=/path/to/differentiate-seismic-solver
python "$skill_dir/scripts/prepare_marmousi_fast40.py" --output prepared_fast40
python "$skill_dir/scripts/prepare_marmousi_fast40.py" --validate prepared_fast40/model_fields.npz
```

The helper samples the 1.25 m input grid at the paper’s 45 m control points,
applies a 180 m reflected Gaussian to Vp and enforces the fixed-Vs bulk-modulus
floor. It verifies all eight arrays and the complete NPZ file hashes. A
validated output directory can be reused offline. For an offline first run,
pass `--source-dir /path/to/agl` pointing to the three original `.segy.gz`
files. The default download cache is `~/.cache/marmousi2-agl/raw/`; set
`MARMOUSI2_AGL_CACHE` or pass `--cache-dir` to choose another cache root.

Use `training.npz` with the observed traces in the optimizer. Keep
`truth_for_observations_and_offline_only.npz` for native observation generation
and offline evaluation. The combined `model_fields.npz` is for plotting and
input verification. Prepared AGL training and truth packages can also be
verified and exported with paired `--training` and `--truth` arguments.

Attribute Marmousi-II to Gary S. Martin, Robert Wiley and Kurt J. Marfurt,
Allied Geophysical Laboratory, University of Houston. The helper exports the
AGL data notice and processing description with the local use case. Pass the
profile’s `figure_credit` as `provenance.model_credit` in the report metadata.
Generate the 2-D model plate locally after the run. The release repository
contains the download and plotting code. The AGL arrays, figures and HTML
that embeds them stay in the user's run directory.
[AGL downloads](https://www.agl.uh.edu/downloads/downloads.htm) ·
[AGL data terms](https://www.agl.uh.edu/resources-data.php) ·
[Model paper](https://doi.org/10.1190/1.2172306)

The optional [shared soft-DTW fixture](soft-dtw-fixture.md) supplies a portable
data-only module and small CPU VJP checks. Its fast backend avoids constructing
a large scalar Torch DP graph for production records. It contains no target
solver or native source/gradient implementation; all actual application checks
below still apply.

## Preserve the native solver, not the reference implementation

Keep the full physical region and starting values. Use an appropriate native
mesh, timestep, parallel execution, supported observable and precision. **FD4,
float32 propagation, seven checkpoints, pressure recordings, GPU execution and
PyTorch propagation are reference-framework details, not requirements of this
solver-independent acceptance test.** Pin and qualify the target's own choices.

Record the physical-grid-to-native material map and its transpose, interpolation,
native resolution, fixed masks/bounds, source components and units, source-time
function, recording duration and receiver convention. Promoting supplied
float32 material values to native float64 preserves their input identity; a
new smoothing or different physical region does not. A coarser propagation mesh
needs its own resolution qualification and cannot be labelled an exact copy of
all supplied detail. Native pressure, displacement, velocity or another
supported component set may define the data; declare components, weights and
shot/trace reduction before observations are generated.

Bind the native sample times explicitly. When the native trace timestep, time
origin or length differs, construct and verify the processing/sampling operator
to the declared observation axis, including anti-alias filters and its transpose.
Do not silently truncate traces, shift them by one step, decimate without a
filter, or change the313-sample alignment problem for convenience. The entire
prediction-side conversion remains in the data-domain VJP.

Use a fixed per-shot/per-stage observed RMS across all declared receiver,
component and time samples after stage filtering. Divide prediction and
observation by that same finite positive scalar; the prediction cannot choose
its own scale. The reference retains all samples with a unity window. Its stage
and anti-alias filters use raised-cosine frequency responses with one original
record of zero padding at each end; cut back to the original record before
decimation. The JSON profile fixes this order. Any required native observable
conversion and component weighting must be stated and differentiated, not
introduced as an undocumented mask.

Use native-solver observations generated from the supplied target in a separate
action. Traces from a different discretization are not the target's native
observations. The optimizer receives the prepared training package, known Vs
and density, and observed data; it must not receive unknown true Vp or truth-based
tuning/stopping feedback.

## Verify, run and judge the outcome

1. Establish the small native-forward/L2 derivative and matched native-cost
   baseline, respecting [numerical acceptance](numerical-acceptance.md) and
   [history/performance accounting](performance-history.md).
2. Implement and check the actual soft-DTW data VJP, including fixed observation
   normalization, filtering, resampling and both predicted self arguments.
   Verify informative data directions and the intended long-record resources;
   an L2 source or a short-trace check cannot certify this replacement.
3. On the actual pinned starting model, verify a non-vacuous full40-shot model
   directional derivative over multiple physical step sizes. Verify the actual
   source/sampling transpose and native parameter map; retain raw gradients and
   perturbed traces. Where an independent native gradient exists, compare it
   in the same coordinates. **Do not require another solver's gradient to
   match a different discretization.** Failed gates prevent a production pass.
4. At every iteration hold one current model fixed for all40 independent shots.
   Reduce their sum and divide by40 once, then normalize/precondition that mean
   and commit one update. Computational batches, MPI groups and optimizer
   candidates obey [the full-shot contract](full-shot-inversion.md). Preserve
   full shot IDs, actual model fingerprints, reduction and update proofs.
5. Use the pinned optimizer/stages unless the user explicitly selects a new
   protocol. Carry Adam moments and its global StepLR count through frequency
   stages; only a separately selected optimizer with curvature memory needs its
   own reset policy. Keep native stability/bounds fixed and qualified, rather
   than choosing them from target-model recovery. Report every actual change.
6. Complete the declared schedule or its actual data-based stopping rule. Save
   each accepted model, raw full-shot gradient, optimizer direction, actual
   update, optimizer state and separate stage losses. A first gradient, first
   update or detached process is not a completed inversion. Budget300 is not
   convergence: report budget exhaustion, convergence, failure, interruption or
   resource limitation distinctly.
7. Judge the final model and numerical validity separately. Show recovered and
   missing structures with shared scales and offline true-Vp errors/masks. A
   valid optimizer path may still recover little geology or stagnate; report
   that outcome. More iterations do not guarantee detail or convergence.

At termination automatically save and inspect the required **True / Initial /
Inverted Vp** comparison from the final accepted checkpoint, then link it in the
report/README. Follow the [terminal visualization contract](visualization-deliverables.md#end-of-inversion-model-comparison): retain the full pinned region,
shared physical units/colour scale, source hashes, actual update number and stop
reason. A stale periodic preview or the model before the last update is not the
final figure, and a budget stop must not be captioned as convergence.

Lead the concise English report with one True / Initial / Inverted Vp row,
shared Blues limits (high velocity dark) and a short title/status; a small
results table is optional. Follow the [presentation contract](visualization-deliverables.md);
Loss may be a collapsed supplement. Archive error/update arrays, native states,
observed/initial/final traces, the actual AD cotangent, first/latest raw
full-shot gradients, separate optimizer directions and complete stage-labelled
loss/model-error records. Diagnostic plots are only for an explicit user request,
not an automatic gallery. The gradient used for updateK belongs to modelK−1
unless recomputed at the final model. Keep
full-gradient, complete accepted-iteration and failed-attempt costs separate;
report host RSS, device/process totals and allocated/reserved GPU memory in
their actual scopes. Five repeated gradient benchmarks are not five independent
inversions; a single final model per route has no repeat-based standard deviation.

For a primary skill-transfer validation use
[the clean-source application protocol](clean-source-application.md). Pin its
packet version before the application and verify integrity afterwards. An
existing adapter may support a labelled follow-up test; it cannot be presented
as a new clean-source adaptation or proof of lower implementation effort.
