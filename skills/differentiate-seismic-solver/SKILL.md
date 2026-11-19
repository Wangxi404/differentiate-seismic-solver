---
name: differentiate-seismic-solver
description: Add and verify a differentiable inversion interface to an existing high-performance seismic forward source solver, using data-domain AD and a native adjoint. Preserve native kernels and execution, archive numerical evidence, and deliver a concise English model-recovery report. Use for a missing differentiable interface, not an already differentiable framework benchmark.
---

# Differentiate an existing seismic solver

Deliver a runnable objective-to-model-gradient extension around the supplied source solver. The reusable part is the adaptation and verification procedure; choose interfaces appropriate to the actual solver, not a mandatory common architecture or language.

## Establish the numerical contract

- Pin the input source revision and a small, working native forward example. Record build/runtime settings, precision inside the solver and in exported files, model coordinates, source and receiver conventions, and the original output. Keep the upstream baseline recoverable.
- Identify the state updates, coefficient map, observation operator, fixed numerical settings, native adjoint and kernel facilities, and available history/storage mechanisms. An existing native adjoint is useful but is not a prerequisite; deriving missing transpose/sensitivity operations from the source is within scope.
- Prefer an independent wrapper/extension and optional history hooks. If source instrumentation is necessary, keep it outside the vendor tree where possible, or produce an explicit patch. Preserving numerical updates is not the same as claiming zero source changes or zero overhead.
- Read [the derivative contract](references/derivative-contract.md) before implementing source conversion, the adjoint, or parameter gradients. Trace the complete native parameter initialization, including resampling and nonlinear coefficient order, rather than differentiating an assumed outer map. State the supported physics, active parameters, and quantities deliberately held fixed.

## Implement the differentiation boundary

1. Execute native propagation without a general-purpose temporal AD tape. Preserve its numerical kernels, build, mesh, and parallel execution. Use its existing adjoint/sensitivities when suitable; otherwise implement the required transpose updates and sensitivity accumulation consistently with the actual discretization.
2. Expose receiver data as a new differentiable leaf. Evaluate the real data objective, including all differentiable processing, and compute its VJP with AD. Convert this data cotangent to the native adjoint source using the sampling transpose and the solver's units, quadrature, signs, components, and time convention.
3. Run the native adjoint and return coefficient/model-coordinate gradients. An optional static coefficient or outer model-parameterization VJP may use AD; keep that graph outside the no-grad temporal calculation.
4. Derive history dependencies for the advertised active parameters, then choose and disclose full history, checkpoint replay, native reconstruction, or another consistent policy. An all-parameter backward API may require the union of those histories; make that contract explicit. No temporal AD tape does not mean no history cost. Read [history and performance accounting](references/performance-history.md) before choosing fields or optimizing storage.

Do not replace the production gradient with a parameter finite-difference sweep or an AD reimplementation of the wave solver. Such computations may be independent checks on small cases. A custom backward wrapper may itself have a `grad_fn`; verify the internal propagation route rather than demanding that every external output have none.

## Full-shot inversion: one current model, one accepted update

When running inversion, **every iteration must evaluate all declared shots at the same current model, accumulate their objective and model gradients, and only then make one accepted model update**. Never update between shots or computational batches and label that a full-shot iteration. This is a correctness requirement, not an optional optimization.

- Declare the complete shot IDs, objective weights and sum/mean normalization before running. Independent shots may be processed serially, in parallel or in memory-sized batches; retain separate wavefields/data and the same model fingerprint throughout each full evaluation. Simultaneous encoded sources are a different experiment.
- Complete the global reduction, including all MPI workers, before applying regularization once, preconditioning the total gradient or invoking the optimizer. A batch mean must be weighted by its actual shot count; do not average unequal batch means.
- Keep the accepted model unchanged during accumulation. Line-search candidates are provisional models, each evaluated over all shots at one fixed candidate fingerprint. Rejected candidates do not count as accepted updates.
- Preserve an iteration ledger: declared shot IDs; the actual model fingerprint for each shot/batch; full reduction completion; and the single accepted update. Check it with [the optional ledger auditor](scripts/check_iterations.py) or equivalent adapter-native assertions. A log cannot replace independent numerical checks of the summed gradient.

Read [the full-shot inversion contract](references/full-shot-inversion.md) before implementing an optimizer loop. Verify batch/serial gradient equivalence and a full-objective directional derivative on a small case. If the user explicitly requests stochastic/minibatch optimization, label that separate protocol; do not silently substitute it for full-shot inversion. More epochs alone do not guarantee geological resolution or convergence.

Establish a standard L2 derivative and matched native-performance baseline first. Then verify the actual task-scoped objective with [objective demonstration](references/objective-demonstration.md). The planar2D profile and default synthetic3D acceptance below select **debiased soft-DTW divergence**; L2 qualification does not authorize an L2 production substitute. Verify each actual VJP, choose settings without the unknown true model, and retain separate stage/loss curves. Do not silently replace the requested objective while debugging native correctness or performance.

## Two-dimensional inversion acceptance

For an input **planar 2-D seismic forward repository**, use [the University of Houston AGL download edition of Marmousi-II](references/marmousi-2d-acceptance.md). Run the installed `scripts/prepare_marmousi_fast40.py --output prepared_fast40` with the solver workspace as the working directory. The helper retrieves the three original SEG-Y archives from AGL, checks their SHA-256 hashes and samples the paper’s 68 × 200 grid. It constructs the initial Vp with 180 m Gaussian smoothing and the fixed-Vs stability floor. Run the helper as part of case setup and give the optimizer only the locally generated training inputs and native observations. Distribute the preparation code, AGL source links and checksums with the adapted repository; generate the True / Initial / Inverted plate on the user's machine. Keep the AGL source files, derived model arrays and 2-D figures out of the release repository.

Use 45 m spacing, 40 independent shots, debiased soft-DTW and up to 300 all-shot updates. Qualify the solver's native mesh, timestep, observables, arithmetic and execution. Verify the prepared models against the paper's hashes, and retain the AGL attribution and source terms in reports. An explicit request for the [legacy marine case](references/marmousi-2d-legacy-acceptance.md) selects a separate profile.

## Three-dimensional inversion cases

For an input **3-D seismic forward repository**, use the bundled [fixed SEG/EAGE salt acceptance case](references/salt3d-acceptance.md). Its **exact33×33×17 true Vp, Gaussian initial Vp, coordinate vectors and fixed Vs/density** are mandatory inputs, supplied inside the skill. Export and verify them with [prepare_salt3d_acceptance.py](scripts/prepare_salt3d_acceptance.py); do not ask the task to supply/select a model, regenerate the blur, crop/downsample it, or substitute an easier anomaly. Only a later explicit user override may define a separately labelled case. The input model is solver-independent; derive and qualify the target's native coordinate/material map without replacing its numerical kernels or imposing AxiSEM3D's mesh, API or execution.

Read [the3D realization and affordability contract](references/three-dimensional-inversion.md). Use **debiased soft-DTW divergence** and **125 accepted full-shot updates** by default, retaining the fixed nine-shot/receiver-grid baseline and concise English True / Initial / Inverted XZ / YZ / XY plate. Qualify the actual data VJP and complete native model derivative, measure native/full-gradient costs, and safeguard physical trial steps. A small native example may establish numerical validity first; it does not replace the fixed-model inversion test. If the target cannot realize the full3D input or the run is unaffordable, report that limitation rather than silently changing the canonical model or claiming the required test completed.

## Validate the skill from clean source

For a primary skill-transfer test, use [the clean-source application protocol](references/clean-source-application.md): an independent application receives the frozen skill, fresh pinned upstream source/native example and supplied case inputs, without an inherited differentiable adapter or hand-carried target repair. Target-specific implementation and ordinary debugging are expected application outputs. If failure exposes a generic guidance gap, preserve the failed run, improve and freeze a new skill version, then reapply from fresh original source. Do not silently repair an old adapter and credit the unchanged packet. Existing-adapter continuations remain labelled follow-up tests. Numerical gates validate the final-model outcome; they do not require another solver's gradients or prove lower adaptation effort without a matched baseline.

## Verify before making resource claims

Read [numerical acceptance](references/numerical-acceptance.md). At minimum provide:

- native-versus-extension forward agreement on the same discrete example;
- data-objective VJP checks for the actual requested objective; validate each nontrivial processing/target separately before advertising that capability, rather than assuming an L2 check covers replacements;
- sampling and, where feasible, propagation/model adjoint inner-product checks in declared coordinates;
- non-vacuous, multi-step-size model directional/Taylor checks for the advertised parameter families;
- evidence that the end-to-end production path uses data AD, native adjoint and the declared history policy, including absence of a hidden temporal-AD or parameter-FD fallback.

Use the reusable functions or CLI in [scripts/numerics.py](scripts/numerics.py) for comparisons, dot tests, and Taylor sweeps. They support checks; they do not choose physical coordinates or excuse a mismatched test. Do not relax thresholds after a failed run without documenting a numerical justification and retaining the failure.

## Package the result

Deliver a concise English report centred on model recovery. Read [visualization deliverables](references/visualization-deliverables.md): default to a short title/status and one **True / Initial / Inverted** comparison. A small results table is optional. For 2-D use one three-panel row; for 3-D use the matching-section 3×3 plate above, without a redundant central-section triptych. Use shared `Blues` scales for positive velocity: low values light, high values dark. Keep an optional 3D view and Loss section collapsed. Do not add sensitivity, kernel or gradient plots to the default report, or require wavefield, data or cotangent galleries.

After inversion ends, automatically render and save that comparison from the actual final accepted checkpoint, including budget-limited runs. Record its accepted update number and actual stopping reason; an iteration cap is not convergence. If truth is unavailable, show Initial / Inverted and say so. Keep all required raw gradients, VJPs, adjoint/Taylor checks, full-shot ledgers, failures, configuration and source/figure manifests in the linked evidence archive. Simplifying the report changes no numerical acceptance requirement. Re-render saved arrays for presentation changes; do not run additional propagation just for styling. Follow the [end-of-inversion comparison contract](references/visualization-deliverables.md#end-of-inversion-model-comparison). Use the joint minimum and maximum of all model values on the declared physical domain, before extracting sections.

The optional [model-recovery report helper](scripts/model_recovery_report.py) produces the final English HTML, shared-scale PNG/PDF/SVG and source manifests directly from saved Cartesian models, with optional offline 3D and Loss views. Its small input contract is documented in [visualization deliverables](references/visualization-deliverables.md#optional-portable-renderers). The more general [visualization helper](scripts/visualize.py) supports additional requested plots. Neither helper runs a solver. Use custom native-aware plotting when it better represents the discretization; no common solver architecture or export schema is mandatory. Produce additional diagnostic plots only when the user requests them; the numerical checks and raw records remain required. Never invent wavefields or turn a few regional derivatives into an apparent spatial kernel.

Keep the adapter, runnable native input, objective example, tests, visual report, source/build provenance and exact reproduction commands together. Follow [the application record](references/application-record.md): retain failed attempts and target-specific repairs, not only the final success. Record supported and unsupported configurations explicitly.

After correctness, use [history and performance accounting](references/performance-history.md) and its [report template](references/performance-report-template.md). Compare a matched native forward baseline with record-enabled execution, then account for data AD, adjoint, replay, kernels, transfer and I/O. Respect existing host jobs; verify multicore/MPI correctness before changing production execution. Separate complete gradient time from failed-attempt time, GPU allocation from device/process totals, and host RAM from GPU RAM. A reduced fallback is a separate matched case; it does not replace the original capacity outcome.

If a frozen packet was supplied, verify its manifest before and after application. Keep target code and application records outside the packet. Target-specific code repairs are expected and must be logged; do not silently edit the frozen skill to make a target pass. A generic skill correction requires a new version and an explicit account of which applications used it.

Return the actual runnable extension and validation report. Do not declare success from a build, a waveform plot, an all-zero gradient, a single favorable epsilon, or the presence of an upstream file called an adjoint.
