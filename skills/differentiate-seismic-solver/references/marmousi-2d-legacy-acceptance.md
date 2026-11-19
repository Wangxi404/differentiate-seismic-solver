# Legacy marine 2-D acceptance profile (v0.4.0–v0.5.2)

This historical profile remains available for explicitly selected marine/full-depth tests. It is not the current fast40 soft-DTW default. Earlier frozen packets and their input identities are unchanged.

# End-to-end acceptance for a planar 2-D solver

The question is whether the supplied forward source repository can acquire a
verified inversion interface and actually invert a structured model. It is
not whether its reconstruction beats a different solver. HybridADFWI is a
separate already-differentiable-framework comparison, not an adaptation target.

## Physical input

For a historical marine run, use the UH AGL horizontal patch below. It retains the
benchmark depth but is not the original 17 km wide model.

| Quantity | Canonical input |
|---|---|
| Original x | 8000–10500 m, inclusive |
| Depth | 0–3462.5 m, inclusive |
| Supplied samples | 555 depth rows × 401 columns; 6.25 m spacing; float64 |
| Unknown | Vp; existing Gaussian sigma = 60 m starting array |
| Fixed priors | Supplied true Vs and designated constant 2450 kg/m³ density |
| Acoustic water | Depth 0–450 m; Vp = 1500 m/s, Vs = 0; first positive-Vs row at 456.25 m |

Use the **actual supplied initial array**, not a new smoothing operation after
cropping or a blur chosen for easier recovery. Canonical array hashes and the
original SEG-Y sources/checksums are in [marmousi-2d-input.json](marmousi-2d-input.json).
The source dataset has its own redistribution/citation terms; this legacy dataset is
not bundled with the skill. Obtain permitted inputs from the recorded origin or
the experiment's separately distributed prepared-input package. The helper
checks the pinned fields and exports a portable training/truth split:

```bash
python scripts/prepare_marmousi_acceptance.py \
  --source supplied_marmousi_vponly.npz --output prepared_marmousi
```

It does not download data, run a solver, crop, interpolate or smooth fields.
A reference preparation recipe is bilinear SI resampling from the original
1.25 m SEG-Y samples to this x/depth grid, constant designated density, then
solid-normalized Gaussian sigma 60 m with `mode=nearest` and exact restoration
of water. Fixed Vs is the target Vs, not blurred Vs. Use the pinned arrays
when rebuilding changes floating-point values; do not silently relax their
hashes. A different explicitly requested model is a separately named protocol.

## Native realization

Keep the complete physical model region. The propagation mesh can be native
FD, SEM or another discretization; it need not equal the supplied material
sample grid. Record the model-to-native map and its transpose, interpolation,
interface treatment, effective native resolution, numerical source revision
and a forward-only material-map inspection. Do not discard model rows or
thin layers to save memory. Coarser propagation is allowed only with its
declared numerical resolution qualification, never relabelled as an exact
copy of all input detail.

Preserve Vs=0 water and the acoustic/elastic interface. Do not turn water into
solid with a Vs floor. If the target cannot support the required physics,
report this acceptance case as unsupported; a solid-only or smaller fallback
may be useful, but cannot pass this case. Do not impose this planar marine
case on spherical, axisymmetric, 3-D-only or incompatible-physics solvers.

Choose and disclose native sources, receivers, observations, boundary
conditions, bandwidth, recording duration and active parameter map. They
need not match HybridADFWI: the experiment tests inversion in the adapted
repository. Qualify frequency/time-step and material admissibility on the
actual initial, target-generation and accepted native models. Resource
pressure should change checkpoint/replay, scheduling or the declared budget;
it must not silently change the physical input.

## Run and judge the inversion

1. Pass the small native forward/data/model derivative checks before the long
   run. On the actual canonical starting point, check a non-vacuous full-shot
   objective directional derivative at multiple physical step sizes. Retain
   raw gradients and perturbed receiver data. A failed qualification prevents
   starting the production inversion under a claimed passed status.
2. Generate observations in a separate action. The optimizer receives no
   unknown true Vp array or truth-based tuning/stopping feedback. Fixed true
   Vs is a strong supplied prior and must remain labelled.
3. Declare all independent shots, weights, objective/processing, frequency
   stages, constraints, optimizer, maximum accepted epochs and data-based
   stopping rules. At **each iteration evaluate all shots at one current
   model, accumulate all gradients, then make one accepted update**. Follow
   [the full-shot contract](full-shot-inversion.md) for batches, trials and
   global reductions. Save accepted model states, the raw full-shot gradient,
   search direction, actual update and replay ledger.
4. Complete the declared inversion protocol or report the actual incomplete,
   failed or resource-limited state. A first gradient, one trial update,
   detached process or iteration cap does not establish convergence. Do not
   stop after producing an appealing image.
5. Judge numerical validity and inversion outcome separately. Verified
   gradients, genuine accepted full-shot updates and reduced data loss
   establish an operational inversion workflow. Report final offline Vp error
   and recovered/missing structures independently; an operational workflow
   need not be a successful high-resolution reconstruction. If geological
   recovery is weak or worse, say so rather than calling this a recovered
   Marmousi result. No cross-solver quality score is required.

For the concise English visual report lead with one True / Initial / Inverted
Vp row using shared Blues limits (high velocity dark) and a short title/status.
A small results table is optional. Follow the [presentation contract](visualization-deliverables.md);
Loss may be collapsed. Archive error/update arrays, native states, observed/
initial/final data, actual residual and AD cotangent, first/latest raw full-shot
gradients and complete stage-labelled loss/model-error records. Save the
physical evaluation mask with those metrics. Additional diagnostic plots
require an explicit user request, not an automatic gallery. A gradient used
for update K belongs to model K−1 unless recomputed at the final model; raw
gradients remain distinct from optimizer directions. Use the shared full range
of the three models on their physical domain.

Record this as a new frozen-version application, with source/build/input
hashes, pre/post packet integrity, repairs, resource use, all accepted updates,
actual stop reason and exact reproduction/plotting commands. Reusing an
existing adapter may be a follow-up capability test; it is not a new blind
adaptation or proof that the skill reduced implementation effort.
