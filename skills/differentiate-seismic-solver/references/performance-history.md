# History and native execution costs

Use this contract after the numerical checks, and when designing history hooks.
It applies to source solvers with different languages, meshes and parallel
architectures. Retain raw measurements in the application repository; the
[report template](performance-report-template.md) is optional scaffolding.

## Derive the history before storing it

Removing the temporal AD tape removes one differentiation implementation. It
does not remove the forward states needed by the native parameter sensitivities,
adjoint live states, PML/memory variables, checkpoints, data or objective graph.
For the chosen discrete operator, tabulate:

| Active physical parameter | Required coefficient cotangents | Forward dependency, time/substep | Saved / reconstructed | Shape, dtype, bytes |
| --- | --- | --- | --- | --- |
| actual parameter family | derive through the material-map VJP | derive from each sensitivity term | actual policy | actual allocation/file size |

Include fixed-field inputs that an active sensitivity still requires. Document
history precision, temporal stride/interpolation, boundaries and auxiliary
states. Approximate or decimated history changes the gradient unless its
consistency is established. Derive rather than guess whether velocity, strain,
stress, displacement or auxiliary fields are required.

An API that always computes all supported parameter gradients needs the union
of their dependencies. Selecting one returned array in Python does not prune
native histories. A parameter-specialized API must accept and honor an explicit
active set, reject unsupported requests, and pass raw-gradient comparisons and
directional checks against the full API. Record both compute and storage scope;
do not advertise the specialized cost as an all-parameter result.

For dense uncompressed storage, reconcile field dimensions, saved time levels,
partitions and dtype with expected bytes. Report the measured peak RAM, bytes
written/read and file sizes separately. Include ghost/duplicate points, buffers,
temporary copies and compression if present. A file-backed history can reduce
RSS while increasing storage traffic and wall time. Checkpoint reconstruction
must state replay counts and retain its correctness evidence.

## Paired baselines and phase accounting

Use the same source revision, numerical updates, physical configuration, model,
observations, precision, source count, CPU/GPU allocation and output policy:

1. **Native forward:** the original runnable forward, with its declared normal
   output. This measures the cost before differentiability instrumentation.
2. **Record-enabled forward:** the same propagation with the history needed by
   the stated gradient API. The difference includes recording and any associated
   synchronization/transfer; it is not automatically wrapper overhead.
3. **Matched interface control, when feasible:** the native and wrapped routes
   with identical history/output. Use this to isolate interface overhead.
4. **Complete gradient and iteration:** include data processing/AD, source
   conversion, native adjoint, sensitivity accumulation, static VJP, all history
   replay/I/O and required reductions. Count all shots and line-search/closure
   evaluations; one accepted update can require several full evaluations.

Measure phases with explicit boundaries and synchronization. Distinguish
inclusive timers from disjoint timers: kernel accumulation fused into an
adjoint must not be added twice. For overlapping asynchronous I/O/compute,
report wall time and overlap without pretending summed timers equal wall time.
Separate setup/mesh/compilation, allocation, warmup, forward recording,
objective/AD, adjoint, kernels, replay, file read/write, transfers, reductions
and total time where observable. Keep an aggregate phase when the implementation
cannot reliably separate fused operations. State such limits rather than infer
compute time by subtracting unrelated runs.

Report repeat points and the chosen summary, cache/storage conditions and
contention. Distinguish per-shot, full-gradient and accepted-iteration scopes.
Keep failed and OOM attempts with their achieved scope; they have no completed
runtime or gradient error. A smaller fallback is a separate matched case.

## CPU topology and parallel execution

Inventory existing jobs before selecting an allocation. Respect their CPU/RAM,
affinity and priority; do not kill, suspend or oversubscribe them to obtain a
benchmark. Use a conservative declared resource budget on a shared host, with
bounded inner BLAS/OpenMP threads so MPI ranks do not multiply hidden threads.
Record physical/logical cores, heterogeneous core types if present, sockets,
NUMA placement, affinity, MPI ranks, threads per rank, GPU device and other load.

Interpret a requested core fraction against distinct **physical** cores unless
the task says otherwise; do not count SMT siblings as extra physical cores.
Plan shot groups, native ranks per group and threads per rank together within
that budget. The number of independent shots limits useful shot concurrency.
A larger budget may instead benefit native domain decomposition, but allocation
is not utilization and more ranks do not guarantee faster iterations. On mixed
core types, measure the slowest group rather than assuming equal rank capacity.
Keep the task's core fraction and machine-specific affinity outside this packet.

First run small multicore/MPI correctness checks: native-versus-extension
forward agreement at that configuration, model/propagation checks and any
partition/history mapping. Compare gradients across feasible partition counts
in consistent global coordinates. Reduction order can change roundoff; choose
and justify tolerances before the run. Only then switch production execution.

Native forward MPI support does not establish derivative MPI correctness.
Account for communication transposes, interface/replicated degrees of freedom,
partition-local histories, stable global identities and coefficient-gradient
reduction. Check native data parity and raw model gradients across rank counts,
then qualify the intended full-shot configuration at one unchanged model before
starting production. Keep serial and parallel orchestration checks distinct
from validation of a newly distributed native adjoint.

Use feasible rank/thread counts within the approved budget, not a fixed list
required of every machine. Pair native and extended runs at each allocation;
hold the problem and output/history policy fixed for strong scaling. Report
speedup against the declared reference, parallel efficiency, peak memory and
I/O per configuration. Separate rank scaling from per-rank thread scaling and
from changed problem sizes. Repartitioning, memory-bandwidth or filesystem
contention can dominate scaling and must remain visible.

Use bounded prospective pilots to compare feasible shot/rank allocations at the
same full case. Include controller work, history I/O, global reduction and
candidate evaluations in the end-to-end scope. Measure aggregate process-tree
PSS where available (RSS sums can double-count shared pages), peak temporary
storage and host memory reserve. Sample after several accepted updates as well
as at initialization: a first-call memory measurement is not a long-run bound.
Value-only observations or line-search trials may omit unneeded histories only
after parity with the record-enabled observation path is established.

## Inversion scope

Before a long requested3D inversion, use [the prospective case/cost gate](three-dimensional-inversion.md#make-the-wave-simulation-affordable)
to choose a measured, feasible native realization and declared schedule.

All shots in an evaluation use one immutable current model, including parallel
workers and memory batches. Complete the global loss/gradient reduction before
one accepted update; each line-search candidate has its own full evaluation.
Use the [full-shot contract](full-shot-inversion.md). Record all objective-stage
and optimizer resets. Faster individual shots do not establish a faster
accepted iteration if replay, closures or I/O increased.
