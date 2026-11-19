# Full-shot inversion contract

This contract applies when an inversion loop is requested. It does not require
a common simulator architecture, optimizer, field format or array framework.

## Objective and update

For the usual separable objective with declared independent shots S and fixed weights w_s, evaluate
`J(m) = sum_s w_s J_s(F_s(m), d_s) / Z + R(m)` and
`g(m) = sum_s w_s D F_s(m)^T v_s / Z + grad R(m)`.
Here `v_s` is the AD derivative of that shot's actual data objective. Use Z=1
for a sum or the declared total weight for a weighted mean. The weights,
normalization, data processing and active parameterization must agree between
the reported objective, VJP, summed gradient and line search.

This sum is an example, not a restriction on differentiable objectives. If
`Phi(d_1,...,d_S)` couples shots, evaluate its true joint data-domain VJP over
the complete required data. Then accumulate
`sum_s D F_s(m)^T (partial Phi / partial d_s)` at the same model. Do not replace
that joint objective with independent batch losses. Record any extra native
forward replay needed to obtain each adjoint history after the joint cotangent
is known; preserving native kernels does not imply zero replay cost.

One iteration has this invariant:

```text
anchor = immutable current model
total_loss = 0; total_gradient = 0
for every declared shot or independent-shot batch:
    assert actual evaluated model fingerprint == fingerprint(anchor)
    compute fresh forward data, objective VJP and native adjoint
    add the shot-summed loss and gradient, without changing anchor
assert every declared shot contributed exactly once
apply the single declared global normalization and regularizer
choose a physically admissible update using this total gradient
commit at most one new accepted model
```

An accepted iteration commits one update. A converged, rejected or interrupted
iteration may commit none, with its status recorded. Never call optimizer.step
inside the shot/batch loop. All distributed workers must finish their global
loss/gradient reduction before the update. Computational batching is allowed;
each shot retains its own state/data axis. A simultaneous-source calculation
with a loss on superposed records is not the same full-shot objective.

If a batch returns a mean over B shots, convert it to a sum using B before
the global reduction, or use the equivalent B/total_shots weight. Incomplete
last batches need their actual size. Do not add the regularizer once per shot.
Apply smoothing/preconditioning only to the accumulated gradient, and save
the raw total separately from the update direction.

## Candidates, constraints and resumption

Evaluate each line-search candidate consistently over all shots at one fixed
candidate model. Temporary candidate assignments are not accepted updates.
An invalid or interrupted candidate can be aborted and rejected; its partial
loss must not be accepted as a full objective. L-BFGS closures must also obey
the all-shot invariant for every gradient evaluation they request. Reset
curvature history when the objective/frequency stage changes.

Use physical bounds, known fixed fields and solver stability guidance. A
synthetic true model may generate observations and supply explicitly declared
known priors. Never use the unknown true target field to choose optimizer
steps, regularization strength, line-search acceptance or stopping. Save any
true-model error only as an independent diagnostic.

When learning steps are a tuning target, declare their scale in physical/model
coordinates and log proposed versus accepted changes. The [3D step guidance](three-dimensional-inversion.md#larger-physical-steps-with-safeguards)
explains why a larger bare learning-rate number alone is insufficient. Keep
the same full-shot objective and bounded acceptance policy for every trial.

Checkpoint the accepted model, optimizer state, objective stage and iteration
ledger. If resuming a partial accumulation, reuse a contribution only when
its model/objective fingerprints and shot IDs match exactly; otherwise abort
that evaluation and restart it under a fresh attempt ID. Do not mix stale
gradients from different models or frequency stages. Foreign GPU jobs must
not be killed or displaced to finish an iteration.

Choose a stated iteration cap and data-based convergence/stagnation rule,
then report the actual stopping reason. A single gradient or an unfinished
background process is not a completed inversion. Extra epochs cannot supply
missing bandwidth, aperture or information.

## Required evidence

- A small independent comparison of serial shot sums, computational batches
  and the derivative of the complete sum/mean objective; include an uneven
  final batch when batching is supported.
- A ledger showing all declared shot IDs at the same evaluated model before
  each accepted update. Hash actual realized fields or bind an immutable
  model version to their content in the advertised inversion coordinates;
  include objective/frequency/discretization identifiers separately. A change
  of numerical mesh is not an accepted change of the underlying model vector.
- Initial and final accepted models, accumulated raw gradients, optimizer
  directions and actual updates, loss by iteration/stage and per-shot fits.
  Fixed true priors must be labelled as inputs, not reconstructed outputs.
- Automatically save the final [True / Initial / Inverted comparison](visualization-deliverables.md#end-of-inversion-model-comparison)
  from the last accepted checkpoint after termination; label missing truth or
  interrupted output honestly, with the actual update number and stop reason.
- Actual completed iterations and epochs, source coverage, normalization,
  constraints, frequency schedule, step acceptance and stopping reason.

The optional `scripts/check_iterations.py` audits orchestration metadata.
It cannot prove that a producer's claimed hashes are truthful or that the
numerical gradient is correct. Keep the native/data/model VJP checks.

## Portable optional ledger

Use JSONL events with `event` and a unique `iteration_id`. Every evaluation
also has an `evaluation_id`; all fingerprints below are `model_sha256` values.

| Event | Additional fields |
|---|---|
| iteration_start | shot_ids (complete dataset), model_sha256, reduction (sum/mean), objective_id |
| evaluation_start | evaluation_id, phase (gradient/closure/trial), model_sha256, objective_id |
| shot_batch | evaluation_id, shot_ids, model_sha256; contributions have shot-sum units |
| evaluation_complete | evaluation_id, model_sha256, normalizer (1 for sum, number of shots for an unweighted mean) |
| evaluation_abort | evaluation_id, reason; cannot be used to authorize an update |
| accepted_update | gradient_evaluation_id, model_before_sha256, model_after_sha256; optional candidate_evaluation_id |
| iteration_end | status (accepted/converged/rejected/failed/interrupted) |

The auditor supports this unweighted sum/mean ledger. Weighted objectives may
use adapter-native equivalent assertions with the declared total weight.
The evaluation authorizing an update must be a completed gradient/closure at
the iteration's anchor model. A supplied candidate evaluation must be complete
and match the accepted new model. Record each trial/closure separately. Use
fresh iteration/attempt IDs after aborted attempts; leave an ongoing prefix
explicitly incomplete rather than reporting it as a completed audit.
