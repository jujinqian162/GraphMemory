## Context

The current experiment surface is a single argparse program at `scripts/experiment.py`. It delegates configuration layering and run initialization to `scripts/workflow/manifest.py`, workflow selection and ordering to `planner.py`, `registry.py`, and `workflows.py`, stage configuration projection to `stage_configs.py`, and cache inspection/resume to `status.py` and `resume.py`. Low-level scripts then parse a generated JSON config plus additional scientific flags and independently repeat run-summary lifecycle code.

The resulting system already has important behavior that must survive the migration: three dataset adapters, eight public retrieval methods, hidden train/tune dependencies, direct stage execution, process isolation, ablation invalidation and aliasing, artifact-aware status, and completed-prefix resume. The refactor changes the configuration and orchestration boundaries without changing those scientific semantics.

The implementation must run on Python 3.10 even though the development checkout can use a newer interpreter. Hydra, Pydantic, and MLflow are new dependencies. The migration intentionally removes the old JSON/argparse/config compatibility surface, so parity must be established before the cutover and the removal must happen only after all replacement paths are verified.

## Goals / Non-Goals

**Goals:**

- Make Hydra the only experiment configuration composer and public experiment CLI parser.
- Validate every resolved experiment and stage input through closed Pydantic V2 contracts before planning or execution.
- Give repository-owned code sole responsibility for run layout, method dependencies, stage DAGs, artifact binding, ablation behavior, and cache resume.
- Preserve low-level subprocess isolation and direct script debugging through one resolved YAML input per stage process.
- Preserve artifact-backed `missing`/`complete`/`stale`/`alias` status and completed-prefix resume without broadening cache semantics.
- Add strict MLflow tracking as a queryable mirror of parameters, metrics, stage attempts, and curated small artifacts.
- Preserve current prediction, checkpoint, failure-case, metric, and aggregate table contracts across HotpotQA, 2WikiMultiHopQA, and MuSiQue.

**Non-Goals:**

- Using Hydra as a workflow engine or using `hydra.utils.instantiate()` for runtime construction.
- Adding content hashes, source fingerprints, artifact digests, cross-run cache reuse, or independent per-DAG-node caching.
- Adding a parallel Hydra launcher or a non-SQLite MLflow backend.
- Uploading datasets, graphs, train pairs, full predictions, or model/checkpoint directories to MLflow.
- Enabling strict deterministic algorithms or promising bitwise equality for GPU-trained methods.
- Preserving the legacy JSON, positional CLI, argparse stage flags, recipe concept, custom run roots, `--force`, or other compatibility adapters.

## Decisions

### 1. Hydra composes only at experiment entrypoints

Five thin modules under `experiment/` own public Hydra entrypoints: `plan.py`, `run.py`, `status.py`, `inspect.py`, and `reset.py`. `plan` and `run` use the same composition and planning service; `status`, `inspect`, and `reset` use narrow command-specific root configs. Low-level scripts remain ordinary Python programs that accept only `--config <absolute-stage-yaml>`.

Hydra uses BasicLauncher and `hydra.job.chdir=false`. Single runs use `runs/${name}`. Multirun jobs use Hydra's job number and override dirname below the requested name, and `RunState.mode` prevents a single-run directory from being reused as a multirun root or vice versa.

Alternative considered: make every stage script a Hydra app. Rejected because nested Hydra jobs create competing output-directory and override semantics and make direct subprocess plans harder to inspect.

### 2. YAML groups own defaults; Pydantic owns the external boundary

`configs/experiment/config.yaml` composes dataset, profile, all eight method config groups, search spaces, and tracking defaults. Dataset groups define sources and validated split capacities; profile groups define `fixed` or `all_available` count policies and trainable-method settings. Root presets exist only for exceptional combinations such as `2wiki_tiny`; there is no recipe abstraction.

Hydra/OmegaConf resolves interpolation into primitive containers. `ExperimentConfig.model_validate()` then applies closed models (`extra='forbid'`), discriminated method/stage unions, scientific scalar validators, capacity checks, and absence-of-`???` checks. Models are not globally strict because OmegaConf emits ordinary containers, but bool/string coercion into scientific numeric fields is explicitly rejected. Application code supplies no business defaults.

Alternative considered: Pydantic settings or hand-written dataclass conversion. Rejected because Hydra already owns composition and both alternatives would recreate a second layering/default system.

### 3. The experiment core is repository-owned and typed

New code under `graph_memory/experiment/` owns:

- `config.py`: root and nested Pydantic contracts plus config resolution;
- `layout.py`: `RunLayout`, the only component allowed to derive run-local paths;
- `registry.py`: typed method lifecycle/dependency metadata projected from the existing runtime registry;
- `planning.py`: typed `StageInvocation` DAG construction, stage range selection, prerequisite checks, and plan formatting data;
- `state.py`: `RunState`, `StageRunSummary`, `ArtifactRef`, and atomic YAML persistence;
- `status.py`: method/stage-specific expected-value inspection and live status derivation;
- `resume.py`: ordered-prefix pruning;
- `execution.py`: stage YAML persistence and subprocess execution;
- `tracking.py`: MLflow adapter and artifact allowlist;
- `service.py`: shared plan/run initialization flow used by entrypoints.

`StageInvocation` contains the stage identifier, method/split/variant identity, absolute script/config paths, typed stage config, declared inputs/outputs, and dependency edges. Method dependencies live only in the typed registry. YAML contains scientific configuration, not `_target_` runtime construction metadata.

Alternative considered: move the existing `scripts/workflow/` package intact. Rejected because it would retain dict-heavy manifests and generated-config ownership rather than establish the intended contracts.

### 4. Run layout and run identity are singular

`RunLayout` derives every path under an absolute `runs/${name}` directory, including resolved config, override list, run state, stage configs, artifacts, ablation namespaces, and aliases. Consumers receive paths from the layout or typed artifact bindings and must not reconstruct them.

On first `plan` or `run`, the service atomically writes resolved YAML, Hydra overrides, stage YAMLs, and `RunState`. Reopening an existing name compares the normalized resolved Pydantic config and mode. A mismatch fails and directs the user to choose another name or invoke `experiment/reset.py`. Ordinary plan/run paths never delete content.

Alternative considered: include date/time in the run path or permit configurable roots. Rejected because stable named identity is required for resume and the design deliberately narrows the public surface.

### 5. Cache truth remains local artifact plus typed stage summary

Each primary stage artifact keeps an adjacent `StageRunSummary`. The summary records stage identity, method/split/variant, status, start/end timestamps, effective config, expected inputs/outputs, selected-config/checkpoint bindings where applicable, counts/timing, error data, and optional MLflow child run id. Writes are atomic.

Status is recomputed from the expected typed invocation, artifact kind (`file` or `directory`), artifact existence/validity, and matching successful summary. An output without a summary, a failed summary, or a mismatched input/output/config is `stale`. Ablation aliases remain `alias`. Resume removes only the longest continuous `complete`/`alias` prefix from the ordered selected plan. `cache.enabled=false` disables pruning but still writes ordinary summaries.

Alternative considered: use MLflow FINISHED status or add hashes. Rejected because MLflow does not prove local artifact compatibility and the approved scope explicitly preserves the current cache boundary.

### 6. Stage execution uses one resolved YAML contract and a shared lifecycle

Every low-level script accepts exactly `--config <path>`, loads a discriminated Pydantic stage model, and delegates scientific work to `graph_memory/stages/`. It supplies no scientific defaults or compatibility flags. Direct invocation writes the same local artifacts and summary as orchestrated execution.

A shared context manager/helper writes running, success, and failure `StageRunSummary` states, records counts/timing, mirrors allowlisted observations to an optional active MLflow child run, and preserves exception propagation. Direct scripts without parent tracking context do not create orphan MLflow runs.

Migration proceeds stage by stage with failing direct-script tests before removing the corresponding old parser/config branch.

Alternative considered: in-process stage calls. Rejected because subprocess boundaries preserve independent debugging, failure isolation, and release of GPU resources.

### 7. MLflow is strict observability, not storage or control flow

The repository uses one SQLite backend store and one artifact root. A Hydra job maps to one parent MLflow run, and every actually executed stage attempt maps to a nested child run. Resume reuses the parent id for an identical run config but creates a fresh child for a new attempt. Cache hits create no child runs. A tuning stage logs one candidate-table artifact and best candidate parameters/metrics rather than a child per candidate.

The adapter logs flattened scientific parameters, dataset/profile/method/seed/device/top-k, source revision and dirty state, environment versions, timings, train/tune/retrieval/evaluation metrics, status, and exceptions. Only resolved configs, override lists, stage summaries, selected tuning configs, candidate tables, aggregate CSVs, report images, and small failure summaries are uploadable. Large artifacts are represented by path, size, kind, and role metadata. Tracking initialization or logging failure aborts orchestrated runs; there is no silent disable/backend fallback.

Alternative considered: one MLflow store per named run or treating MLflow artifacts as canonical. Rejected because cross-run comparison requires a shared store and direct-script/offline delivery requires local files to remain authoritative.

### 8. Cutover is parity-first and compatibility-free

Before modifying execution, tests capture current plans, resolved configs, artifact schemas, and smoke outputs for the supported dataset/method combinations. New config/contracts and planner are then built alongside the current executor long enough to prove semantic parity. Status/resume and stages migrate incrementally. After MLflow and aggregate delivery pass, the legacy JSON files, `scripts/experiment.py`, `scripts/workflow/`, hand-written config codec/converter/patch/loader, obsolete observability forwarding, old CLI tests, and compatibility adapters are deleted in one explicit cleanup phase.

Alternative considered: retain a translation layer indefinitely. Rejected because the approved change explicitly removes compatibility baggage and duplicate sources of truth.

## Risks / Trade-offs

- [The migration can silently omit an existing method or special dataset path] → Freeze plan/config snapshots first and require HotpotQA, 2Wiki, MuSiQue, Memory Stream, hidden Dense-FT seed dependency, and ablation parity tests before cutover.
- [A Pydantic model can accidentally absorb domain or leakage validation] → Keep artifact/data/checkpoint validators in their existing domain packages and test that config validation only handles input shape and cross-field configuration invariants.
- [GPU training is not bitwise deterministic] → Compare exact plans/configs/artifact schemas for all methods, exact results for deterministic methods, and predeclared metric tolerances for trained methods.
- [SQLite writes can contend] → Support only Hydra BasicLauncher sequential execution in this change and reject parallel launcher configurations.
- [MLflow can duplicate large data] → Centralize a deny-by-default artifact allowlist and test that prohibited artifact kinds never call MLflow upload APIs.
- [A failed tracking call can leave local output behind] → Mark the stage summary failed/stale and propagate the error so output existence alone cannot become a cache hit.
- [Removing legacy entrypoints is disruptive] → Update all active docs/tests/commands in the same change and provide explicit old-to-new command mappings; do not keep runtime shims.
- [Python 3.10 compatibility can drift from the local interpreter] → Lock dependencies and run composition, Pydantic, MLflow SQLite, tests, and workflow smoke in a real Python 3.10 environment before completion.

## Migration Plan

1. Record legacy plan snapshots, resolved configs, schema fixtures, and same-split smoke baselines for all supported datasets and method families.
2. Add Python 3.10-compatible Hydra, Pydantic V2, and MLflow constraints; lock them and prove composition, sequential multirun paths, and SQLite parent/child tracking in an isolated spike test.
3. Add Pydantic experiment/stage contracts and Hydra YAML groups without changing the active executor; prove all legacy configs are expressible and root overrides reach every intended consumer.
4. Add `RunLayout`, typed registry, stage DAG, public `experiment/` plan path, and parity tests while still targeting existing stage scripts.
5. Add typed run state, stage summaries, status derivation, and ordered-prefix resume; migrate existing cache fixtures.
6. Migrate prepare, graph, pairs, tune, train, retrieve, evaluate, and aggregate scripts one at a time to resolved YAML and shared lifecycle handling.
7. Add MLflow parent/child tracking and curated artifact mirroring, including failure and resume behavior.
8. Verify aggregate/delivery artifacts and update active documentation and command examples.
9. Delete legacy JSON/config/argparse/workflow code and run residual scans for forbidden compatibility surfaces.
10. Run the full engineering gates and real workflow matrix, including a real Python 3.10 lock install.

Rollback before step 9 is performed by switching callers back to the untouched legacy entrypoint. After step 9, rollback is repository-level reversion of this change; runtime dual paths are intentionally not provided.

## Open Questions

None. `docs/10-plans/refactor-7-9-analysis.md` locks the architecture and public behavior; discoveries that require a different decision must update that document and this OpenSpec change before implementation continues.
