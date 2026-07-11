## Why

The experiment runner currently spreads configuration merging, workflow planning, artifact layout, cache status, and stage CLI behavior across JSON files, argparse entrypoints, and `scripts/workflow/` helpers. This makes experiment inputs weakly typed and forces the repository to maintain duplicate validation and observability plumbing while still lacking a unified, queryable experiment view.

## What Changes

- **BREAKING** Replace the positional/subcommand experiment CLI and JSON configuration entrypoints with five Hydra entrypoints using `key=value` overrides: `plan`, `run`, `status`, `inspect`, and `reset`.
- **BREAKING** Replace JSON experiment/method/profile/stage configs and hand-written structuring/patching with Hydra YAML groups validated by Pydantic V2 models that reject unknown keys and invalid scientific scalar coercions.
- Introduce a repository-owned experiment core containing the single run layout owner, typed method registry, typed stage-invocation DAG, dependency validation, ablation invalidation, and ordered-prefix resume planning.
- Preserve the current artifact-backed cache semantics by replacing loose manifests and run summaries with typed `RunState` and `StageRunSummary` contracts; do not add hashes, cross-run reuse, or per-node content-addressed caching.
- **BREAKING** Make each low-level stage script accept exactly one resolved stage YAML config and remove scientific argparse defaults and legacy config compatibility paths.
- Add MLflow tracking backed by a repository-local SQLite store, with one parent run per Hydra job and nested child runs per executed stage attempt; keep local artifacts as the scientific source of truth and upload only an explicit allowlist of small artifacts.
- Preserve current dataset, method, tuning, ablation, direct-script, aggregate-delivery, and failure-validation behavior, and verify plan/workflow parity across HotpotQA, 2WikiMultiHopQA, and MuSiQue.

## Capabilities

### New Capabilities

- `experiment-config-composition`: Hydra configuration groups, root presets, overrides, Pydantic validation, and resolved YAML contracts.
- `experiment-workflow-planning`: Typed run layout, method dependencies, stage DAG construction, plan parity, ablation invalidation, and public experiment entrypoints.
- `experiment-cache-resume`: Typed run state and stage summaries, artifact status derivation, and completed-prefix resume behavior.
- `experiment-stage-execution`: Resolved-YAML stage adapters, shared stage execution/summary lifecycle, direct-script behavior, and final artifact delivery.
- `experiment-tracking`: MLflow parent/child run lifecycle, metadata and metric mirroring, artifact allowlisting, and strict tracking failure behavior.

### Modified Capabilities

None. The repository has no baseline specs under `openspec/specs/`; this change establishes the experiment workflow contracts as new capabilities.

## Impact

- Affects `experiment/`, `scripts/experiment.py`, every stage script under `scripts/`, `scripts/workflow/`, and the experiment/config/stage packages under `graph_memory/`.
- Replaces committed JSON configuration trees with Hydra YAML groups and updates tests, operational documentation, command examples, and workflow snapshots.
- Adds and locks stable Python 3.10-compatible releases of Hydra, Pydantic V2, and MLflow, including a shared SQLite tracking store and MLflow artifact root.
- Removes legacy CLI/config compatibility rather than maintaining parallel entrypoints or fallback loaders.
