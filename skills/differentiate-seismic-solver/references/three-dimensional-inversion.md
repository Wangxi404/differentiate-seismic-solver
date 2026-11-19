# Realize the fixed3D salt inversion case affordably

For a3D source-solver acceptance test, first read and export the
[fixed salt benchmark](salt3d-acceptance.md). The skill supplies the actual
33 × 33 × 17 fields and Gaussian initial model. This document governs the target's
native realization, qualification, resources and result; it supplements the
derivative/full-shot/visual contracts without prescribing one solver's internals.

## Keep the fixed input; declare the native realization

Verify the bundled NPZ and all seven array fingerprints before native mapping.
Before generating observations or committing a production schedule, record:

| Layer | Required declaration |
|---|---|
| Original input | Bundled salt profile, package/array hashes and original notices; exact axes/order, SI units,33×33×17 dimensions and physical extent |
| Simulated material | Unchanged canonical arrays and qualified target-native interpolation/coordinate map; boundaries and actual coverage/masks |
| Starting material | Bundled `initial` verbatim, never reblurred; scalar Vs850m/s and density2200kg/m³ remain supplied fixed priors |
| Model coordinates | Active parameters, units, bounds, control coordinates and control-to-native map; distinguish volumetric controls from a reduced regional parameterization |
| Native resolution | Actual spatial/modal/mesh representation, numerical settings, time grid, source bandwidth, internal/export precision and output/history policy |
| Acquisition/objective | Independent source IDs and weights, actual observations and coordinates, processing/VJP, global normalization and stages |
| Budget | Resources, measured pilot costs, accepted-update/walltime caps and retained-artifact/storage policy |

The benchmark reduction and nested refinement are already applied. Do not
reduce it again, crop it, change amplitudes/fixed fields or recompute its
Gaussian initial model. Native material interpolation is permitted and requires
a consistent parameter transpose; it does not authorize changing the fixture.
Disclose native precision conversion and actual simulation coverage separately
from canonical-input identity. Preserve full non-axisymmetric variation; a
section, radial average or extruded2D model is not this test. If the native
representation cannot realize it, report unsupported scope.

Check that the actual simulated native fields retain informative variation
in the advertised directions. Display matching sections through that volume,
not unrelated favourable crops. A small number of regional coefficients may
be a useful preliminary derivative test but is not a dense3D model inversion.

## Default production objective: debiased soft-DTW

For synthetic3D acceptance, use **debiased soft-DTW divergence** unless the
user explicitly selects a different objective. L2 remains a useful native
qualification/performance baseline; passing it is not permission to run the
production inversion with L2. Preserve the requested objective when fixing
resource or implementation problems, and make runner, ledger, configuration
hashes, report labels and plotted loss agree on the actual objective.

For each declared scalar receiver/component trace, compute
`D(p,o) = softDTW(p,o) - 0.5*softDTW(p,p) - 0.5*softDTW(o,o)`.
Differentiate both arguments of the predicted self term. Use float64, exact
squared normalized-amplitude cost, gamma0.1 and no alignment band by default;
record the fixed profile and any later explicit user override separately.
With a fixed positive observed RMS
per shot, apply identical declared preprocessing to observation and prediction,
divide each divergence by the processed sample count, take one mean over all
declared traces/components, then one mean over independent shots. Do not add a
positivity clamp, prediction-dependent mask or rank-local normalization.

The [shared data-only fixture](soft-dtw-fixture.md) can supply this objective.
Bind the real native time/component axes; include every prediction-dependent
filter, sampling and permutation in AD. Bind the [fixed profile's](salt3d-acceptance.json)
declared reference sampling/preprocessing to actual native output, not the
planar2D numbers.
The filter stopband must reach zero by the processed Nyquist frequency. Record
the unchanged native record and the actual processed record separately.

Dense alignment costs grow quadratically with processed record length. Stream
**complete traces** in bounded chunks; weight chunk/rank sums by the complete
declared trace count, release each chunk graph after its VJP, and retain all
shots at one model before updating. Cache observed self terms only under
identical inputs and objective settings. Declare AD worker affinity, thread
counts, dependencies and measured time/memory alongside native resources;
installed packages do not prove an import-compatible environment.

Before production, check a small independent value/VJP reference; check the
actual full-native-record data VJP at multiple informative finite-difference
steps; verify chunk/rank loss and cotangent equivalence; then pass the complete
full-shot native model directional/Taylor test at the actual starting array.
Keep these new-objective gates distinct from unchanged native L2 qualification.
Measure a complete new-objective gradient and full-shot line-search trial for
the runtime forecast. Never substitute a short-trace fixture for these gates.

When the objective changes, start a separate, accurately labelled experiment
with fresh optimizer state and a separate loss history. A fresh benchmark run
starts from the prescribed Gaussian initial model. Reusing a previously qualified native extension is an objective
replacement follow-up, not a fresh-source application of the new skill packet.

## Make the wave simulation affordable

Reducing control or input-array dimensions can lower interpolation and optimizer
cost while leaving the native propagation cost unchanged. Inspect the actual
native degrees of freedom, time steps and source/history costs. Choose a
compatible native spatial/modal realization and integration step while retaining
the fixed input/source/acquisition and declared output window. Retain meaningful3D variation
and an observation window that samples the requested anomaly. Check numerical
stability at the declared material bounds. Disclose spatial/modal resolution
limits separately from correctness of the native discrete gradient.

Use a small native forward early to rule out a plainly unaffordable case. After
derivative qualification, measure a complete-gradient pilot at the intended
production realization, including required history/replay and the actual data
VJP. If pilots use a smaller realization, do not label their timings as full-case
measurements. Measure a full-shot point when feasible; source concurrency,
setup and shared I/O can invalidate a simple per-source multiplier.

Before a long inversion, forecast its cost from measured calls and the declared
optimizer schedule. Include all shots, adjoints, replays, candidate/closure
evaluations, stage changes and terminal evaluation/plotting. Record the assumed
trial count and whether the estimate is typical or a bounded worst case. Keep
measured costs and projections distinct. Choose a feasible resource schedule
without altering the fixed arrays; report capacity limits if it cannot fit.
A small preliminary case or explicit user-selected variant is not completed
fixed salt acceptance. Preserve earlier timing/capacity failures.
Never obtain an apparent speedup by dropping shots from an evaluation.

Continue to use [history/performance accounting](performance-history.md) for
matched native/record-enabled baselines and resource evidence. A reduced
discretization has its own baseline; it is not a controlled wrapper-overhead
comparison against an older, larger experiment.

## Larger physical steps with safeguards

Specify what the learning scale means in the advertised parameter coordinates.
A scalar learning rate depends on gradient normalization and parameterization;
it is not a comparable physical model change on its own. Record proposed and
accepted model-change norms in the actual physical field, including any outer
transform, bounds or projection. Preserve raw gradients separately from any
normalized/preconditioned direction and the committed update.

When larger steps are requested, test them using the declared objective-based
acceptance and physical/stability constraints. A bounded line/trust search or
another justified safeguard is acceptable. Previous successful steps may seed
a larger trial where appropriate; bound and record the trial budget. Reject an
invalid or unsuitable candidate without forcing acceptance or counting a new
model. Every candidate follows the complete [full-shot contract](full-shot-inversion.md).
Do not use synthetic-truth error, anomaly visibility or a desired-looking
gradient to choose steps, regularization or stopping.

Record the initial trial scale, maximum trial scale, growth rule and backtrack
budget in the external case profile. For a requested more aggressive rerun,
increase the initial physical trial as well as the cap, then let the full-shot
acceptance test determine the usable step. For normalized directions, state whether
the scale means maximum or RMS physical change, and log both proposed and actual
accepted maximum/RMS changes after projection. For quasi-Newton directions,
retain curvature safeguards and resets; increasing a nominal learning rate must
not bypass a non-descent or invalid direction. When continuation is requested
from a fresh blurred truth, use the bundled `initial` exactly rather than
regenerating the blur or importing an old inverted checkpoint.

Inspect accepted-step and backtrack history before attributing slow recovery to
a step cap. A larger cap is irrelevant when accepted steps already lie below
it. Larger steps, more cores and a fixed update count are testable proposals,
not guarantees of convergence or detail. Report data misfit, model recovery and
actual stopping status separately; hold case, acquisition, source bandwidth and
native realization fixed when isolating optimizer or resource changes.

## Accepted-update budget and continuation

The default 3D acceptance budget is **125 accepted full-shot updates**. A user
budget takes precedence; resource limits require an explicit prospective decision,
not a hidden smaller cap. Keep the runner's termination condition, checkpoint
counts, ledger, report target and export validator consistent with the declared
budget. An iteration cap is not a convergence claim.

To extend a run, resume its last accepted model, optimizer state, next trial
scale, objective stage and full-shot ledger. Keep the previous report and record
the starting checkpoint, cumulative update budget and resource forecast.

## Close with the actual model

Wire the [terminal visual comparison](visualization-deliverables.md#end-of-inversion-model-comparison)
into the runner. At completion or budget exhaustion, use the last accepted
checkpoint for one English 3×3 model-recovery plate: True / Initial / Inverted
columns and matching XZ / YZ / XY rows. Declare the section coordinates once,
use shared Blues limits for positive velocity (high values dark), and omit a
redundant central-section triptych. Adapt the views honestly for other native
geometries. This is the required3D model; if a target cannot represent it,
report that limitation instead of silently picking other geology. The named
3D fixture is not applied to a2D input repository.

Keep front-facing text to a short English title/status; a small results table is
optional. A3D HTML
view and Loss curve may be collapsed supplements; an input-volume viewer alone
cannot close the inversion test. Do not add sensitivity/kernel/gradient or
other diagnostic galleries by default. Preserve the complete objective history,
raw full-shot gradients, directions, actual updates, fits and all numerical
checks in the linked evidence archive. Record checkpoint identity, accepted
count and stop reason in the manifest. Report remaining structure/amplitude
errors honestly and keep truth-based diagnostics outside optimizer selection.
Re-render saved arrays for styling; do not run additional propagation for it.
