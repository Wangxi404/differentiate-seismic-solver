# Application record

Keep a machine-readable record and the raw run artifacts in the adapter repository, outside any frozen skill packet. Measure hands-on development time directly if reporting it.

Record these fields where applicable:

- skill version, complete packet/manifest hash, pre/post integrity check;
- primary clean-source application versus inherited-adapter follow-up; initial upstream inventory, native examples and supplied implementation hints; external interventions; generic guidance gaps, failed versions and fresh-source replay after each skill correction;
- target source URL/revision, source hashes and any pre-existing local changes;
- exact build/compiler/library/runtime settings and a runnable native baseline;
- supported physics, model/coefficients, active/fixed quantities, units and precision;
- active-parameter-to-coefficient/history dependencies, masked versus all-parameter gradient API, retained fields/time levels, history bytes and replay policy;
- data and adjoint-source layout/conversion, raw kernel and model-gradient coordinates;
- numerical source changes or instrumentation and their rationale;
- complete commands, failures, retries, adapter patches, human interventions and their provenance;
- AD data-objective checks, native-forward discrepancy, transpose tests, complete Taylor sweeps and raw gradients;
- temporal-AD/replay route, stored history and complete resource measurements;
- matched native/record-enabled baselines, exclusive/inclusive phase timing, I/O/transfer bytes and time, CPU topology/affinity/threads/MPI partitions, contention and multicore correctness/scaling evidence;
- objective definition/version and configuration hash, data-VJP/model checks, stabilization and FFT/filter conventions, parameter-selection provenance and objective-stage transitions;
- for inversion: the complete shot set and weights; objective reduction; shot/batch model fingerprints; full reduction and accepted-update ledger; optimizer/constraints/frequency stages; accepted iteration count, resume state and actual stopping reason;
- for planar 2-D Marmousi acceptance: named profile/version and input/reference hashes, complete physical extent and native realization, actual initial Vp identity, fixed true priors, model-to-native map, completed inversion status and independent recovery diagnostics; distinguish operational inversion from geological recovery;
- for3D acceptance: bundled `seg-eage-salt3d-fixed33-v1` package and all seven array fingerprints, unchanged true/Gaussian/fixed-prior inputs and full extent, original preparation provenance, native map/coverage/precision and actual3D variation, prospective measured-cost budget and proposed/accepted physical update norms; distinguish the fixed3D profile from planar2D Marmousi and any explicit user-selected variant;
- exact reproduction commands, package dependencies, scope and remaining failures.
- main model-comparison figure and offline plotting command, input/checkpoint hashes, active fields, physical sections and units, shared full colour range, coverage and unit conversions, report status and rendered QA outcome; inventory any explicitly requested diagnostic figures separately;
- linked raw numerical evidence, configurations and manifests, including gradient/state/data coordinates and units, selected native times and aggregation rules when applicable; preserve these records without requiring a front-facing diagnostics gallery.

Use statuses such as `planned`, `attempted`, `failed`, and `passed` for individual gates. A build or a zero exit code is not a numerical pass. Unsupported parameters should be rejected or clearly excluded, not returned as zero gradients.

Keep the default visual report concise and in English: lead with model recovery,
not this full record. Archive the detailed fields above as machine-readable
evidence and link them. Numerical/raw-artifact retention is unchanged by the
smaller presentation; restyling saved outputs requires no new solver run.

For a frozen application, target-specific repairs do not modify the packet. Record each repair and rerun the affected checks. A generic packet correction creates a new version; preserve the failed version and keep versions separate in reported results. An independent agent context and a frozen packet do not by themselves make the experiment blinded or prove that the skill reduces effort; disclose prior familiarity and use an appropriate no-skill baseline before making a causal cost claim.
