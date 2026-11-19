# A verified data-domain objective demonstration

Establish **L2 derivative correctness and matched native-performance baselines**
before attributing behavior to a complex loss. Repair an incorrect or inefficient
native path on that baseline; keep performance, active-parameter/history,
full-shot and visualization contracts. Record the application's actual objective,
inversion scope, priorities and completed evidence outside the generic packet.

Then use the requested or named-profile nontrivial data-domain objective to
demonstrate interface flexibility. The current [planar2D acceptance case](marmousi-2d-acceptance.md) selects the versioned fast40 soft-DTW protocol, and
[synthetic3D acceptance](three-dimensional-inversion.md#default-production-objective-debiased-soft-dtw)
uses the bundled [fixed salt model/profile](salt3d-acceptance.md) and debiased
soft-DTW, with an explicitly qualified map from actual native output to the
declared sampling/preprocessing.
Other tasks may select their own objective; neither profile prescribes a
mandatory schedule for every solver. A task limited to an L2
repair does not imply a complex-loss production run; a requested composability
demonstration does need its own actual VJP and, where requested, inversion checks.
Passing L2 does not verify a replacement objective or its inversion behavior.

## Select and freeze the actual objective

Declare preprocessing, samples/components, shot/trace weights and reduction,
units, fixed versus prediction-dependent quantities, stabilization and stage
schedule. AD must include every prediction-dependent operation. Fixed
observations may supply normalization and regularization scales; the unknown
true model may generate synthetic data and independent diagnostics, but may not
choose loss/optimizer hyperparameters, line-search acceptance or stopping.

## Candidate: AWI-based reverse matching filter

One data-AD candidate is an **AWI-based reverse matching-filter objective**.
Record any existing implementation and its local checks in the application;
those are distinct from production recovery evidence. Choose implementation and
inversion work according to the actual task scope. This candidate is neither a
new AWI method nor a required objective for every seismic solver.

For predicted trace p and fixed observed trace o, define the declared convolution
operator D_o and solve

```text
w_lambda(p,o) = argmin_w 0.5 ||D_o w - p||^2 + 0.5 lambda ||w||^2
J(p,o)       = ||T w_lambda||^2 / (2 ||w_lambda||^2)
```

Here T weights filter lags in declared time units. This formula assumes a
nonzero filter. Record the actual regularization and denominator stabilization,
including its derivative. Differentiate through the regularized matching solve
and normalized lag penalty to p; do not detach the filter or replace the
resulting VJP with a residual. The reverse matching direction above must be
named explicitly when comparing to other AWI formulations.

FFT implementations must specify linear versus circular convolution, zero
padding/FFT length, crop/window, filter support, time origin/lag ordering,
normalization, real/complex conventions, spectral regularization and trace
weighting. A stable FFT ratio can make the data operation practical for long
records, but circular/padded and regularized choices define a particular
discrete objective. Do not call an illustrative implementation equivalent to an
industrial AWI package without evidence.

AWI is motivated by [Warner and Guasch (2016), Theory](https://doi.org/10.1190/GEO2015-0387.1)
and [Guasch, Warner and Ravaut (2019), Practice](https://doi.org/10.1190/GEO2018-0377.1).
These are prior methods, not results of this project. [Symes (2024)](https://arxiv.org/abs/2402.17696)
analyzes conditions and limitations: multiple arrivals can invalidate a simple
traveltime interpretation and optimization can stagnate. Do not promise that
AWI, another complex loss, or additional epochs will remove cycle skipping or
recover specific geological details.

## Candidate: soft-DTW divergence

[Soft-DTW](https://proceedings.mlr.press/v70/cuturi17a.html) smoothly aggregates
alignment paths. Its entropic bias means the raw loss need not be minimized at
identical series. [Soft-DTW divergence](https://proceedings.mlr.press/v130/blondel21a.html)
corrects self-similarity terms; for smoothing gamma > 0, write

```text
Div_gamma(p,o) = softDTW_gamma(p,o)
                - 0.5 softDTW_gamma(p,p) - 0.5 softDTW_gamma(o,o)
```

For the named fast40 profile an optional [data-only fixture](soft-dtw-fixture.md)
provides the shared implementation and a small independent reference check.
Native propagation, source conversion and model derivatives are not bundled.

The predicted self term depends on p **in both argument positions**. Do not
detach either copy or treat that term as a constant. The observed self term may
be cached only when its data, preprocessing, ground cost and gamma stay fixed.
Differentiate the entire actual divergence before native source conversion.
Document the ground cost, gamma and scaling, amplitude/time normalization,
alignment constraints, masks and any approximations. Nonnegativity and identity
properties have conditions on the ground cost. The original 2021 paper proves
stationarity at equality for the ordinary squared-Euclidean cost, while validity
as a divergence for that cost is a conjecture there. Do not present this case as
its proved general theorem or extend it to arbitrary modified alignment costs.

The definition contains three soft-DTW evaluations; fixed observed self costs
can be reused. Unrestricted dense dynamic programming has quadratic time and
space in equal sequence length, with additional AD workspace. A 16,000-sample
record has 256 million alignment cells per dense table, before batch/trace axes
and gradient storage. Treat the actual long-record objective/AD time and peak
memory as a resource gate before selecting this loss. Do not infer feasibility
from a short-trace gradient check. Windows, banding, downsampling, parallel
kernels or specialized backward algorithms may change cost or the objective;
declare them, validate the resulting VJP and compare resources at the intended
length. Fix choices from observations/numerics, not true-model reconstruction.

There is already seismic precedent: [Kalita, Purcell and Casasanta (2023)](https://www.shearwatergeo.com/technical-publications/soft-dynamic-time-warping-divergence-as-a-misfit-measure-in-full-waveform-inversion)
report soft-DTW divergence FWI with Marmousi and offshore-Australia marine data.
Using this loss in FWI is therefore not itself a new contribution. Alignment
may match unintended arrivals or alter amplitude sensitivity. Published cases
do not establish recovery in a new solver/configuration, universal removal of
cycle skipping, or superiority to AWI. Those require the actual experiment.

## Verify and demonstrate

1. Check the exact data VJP at informative nonzero traces over several step
   sizes, including the chosen regularization, padding and normalization.
   Include a long-record case and report objective/AD time and memory.
2. Check the complete model directional derivative through the native source
   conversion and adjoint. Preserve raw gradients. Small-case AD comparisons
   are independent checks, not a production temporal-AD fallback.
3. Pin the objective implementation/configuration. For multiple applications,
   use the same declared loss definition and configuration; document any
   resampling/time-coordinate conversion and validate it. Different native
   forward operators may generate their own observations, so this is not a
   cross-solver reconstruction ranking.
4. Run every shot at the same current model, reduce once, then update once.
   Full-shot closures/trials obey the same rule. If a separately selected
   protocol changes objective or frequency, clear L-BFGS/optimizer curvature
   history at that boundary; no AWI-to-L2 schedule is prescribed here.
5. Retain separate objective/stage histories and common evaluation diagnostics.
   Never concatenate unlike raw losses into one apparently continuous curve.
   Lead the concise English report with model recovery under the
   [visual contract](visualization-deliverables.md); Loss may be collapsed.
   Archive raw full-shot gradients, actual updates, native states and per-shot
   fits with their provenance. Additional diagnostic plots require an explicit
   user request; no gradient gallery is part of the default report.
6. State actual accepted iterations and stopping reason. A data-gradient check
   establishes local implementation correctness, not Marmousi recovery. Report
   failed/stagnant reconstructions and costs as well as successful cases.
