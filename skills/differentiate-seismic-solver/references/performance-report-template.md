# Application performance report template

Copy into the application repository and replace placeholders with actual
artifacts. Mark unavailable measurements `not measured`; do not use a planned
row as a result. Read [the accounting contract](performance-history.md).

## Identity and derivative scope

| Field | Actual value / artifact |
| --- | --- |
| Skill version and packet hash, pre/post verification | |
| Native revision, patch/build/config hashes | |
| Physics, physical model extent, grid/mesh, dt and sample count | |
| Active/fixed parameters and requested gradient API | masked / all supported |
| Complete shot IDs, reduction, objective/config hash | |
| History policy, precision, reconstruction/replay count | |
| Phase boundaries, synchronization, exclusions | |
| Repeats, warmup, filesystem/cache conditions | |

## Required history dependencies

| Active parameter | Required native sensitivities | Fields/substeps | Save/reconstruct | Predicted bytes | Measured bytes |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

Attach specialized/full-API gradient equivalence checks if pruning is claimed.
Record both history file size and peak resident memory; they are different.

## Matched time and memory

| Configuration / scope | Native forward | Record-enabled forward | Matched wrapper control | Complete gradient | Accepted iteration |
| --- | --- | --- | --- | --- | --- |
| Allocation, active API and policy ID | | | | | |
| Wall time: each repeat and summary (s) | | | | | |
| Peak process / rank-aggregate host RAM (GiB) | | | | | |
| Peak GPU allocator allocated / reserved (GiB) | | | | | |
| Peak device/process GPU memory, method (GiB) | | | | | |
| History / read / written bytes | | | | | |
| Completed scope / failure or OOM stage | | | | | |

## Phase detail

| Run ID | Phase | Inclusive / disjoint / overlap | Wall time (s) | Calls | I/O or transfer bytes | Boundaries / evidence |
| --- | --- | --- | --- | --- | --- | --- |
| | Forward numerical propagation + recording | | | | | |
| | Data processing and AD | | | | | |
| | Source conversion / data transfer | | | | | |
| | Native adjoint + fused kernels, if inseparable | | | | | |
| | Separate kernels / static parameter VJP | | | | | |
| | Forward replay / reconstruction | | | | | |
| | History read/write / reductions | | | | | |
| | Total complete gradient | enclosing timer | | | | |

Do not sum nested/fused/overlapping timers. Record trial/closure evaluations and
failed attempts separately from a successful full-gradient measurement.

## CPU topology, MPI/thread scaling and correctness

Record CPU model, sockets/NUMA, physical/logical/core types, affinity, rank/thread
placement, BLAS/OpenMP limits, GPU IDs, shared-host load and resource budget.

| Allocation ID | MPI ranks × threads / physical CPUs | Native / record forward (s) | Adjoint + kernel (s) | Replay / I/O (s) | Complete gradient (s) | Speedup / efficiency | Peak RAM | Forward / gradient check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | | | | | | 1 / 1 | | |
| tested allocation | | | | | | | | |

Hold the problem and declared output/history policy fixed for strong scaling.
Keep multicore correctness artifacts from before the production switch. Label
fused phases and changed allocations explicitly rather than filling invented
separate timings.

## Inversion and visual evidence

- Link complete all-shot/model-fingerprint/accepted-update ledgers.
- List objective-stage configuration and optimizer resets, stopping reason and
  actual accepted iterations; retain every objective's separate curve.
- Lead the concise English report with a short title/status and the True /
  Initial / Inverted model comparison. A small results table is optional;
  optional 3D View and Loss stay collapsed.
- Link raw full-shot gradients, directions/updates, native state/data/cotangent
  artifacts, complete dot/Taylor records and source/configuration manifests as
  evidence. These remain required records, not a mandatory diagnostics gallery;
  additional diagnostic plots require an explicit user request.
- State remaining failures and which performance conclusions the matching
  supports. Keep geological recovery distinct from operational correctness.
