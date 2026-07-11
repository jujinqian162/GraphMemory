## Why

The first `refactor-experiment-config-workflow` implementation preserves most scientific workflows, but it does not satisfy the repository's minimality and readability requirements. It adds an unnecessary config namespace and root-preset system, changes locked public and MLflow contracts, and introduces many dead fields, unreachable branches, test-only optional parameters, duplicate registries, and repeated stage-binding logic that make the code harder to read than the system it replaced.

This correction is needed before the refactor can be treated as complete or committed as the new foundation. The goal is not to add capability; it is to remove non-required structure while preserving the verified workflow, artifact, cache-resume, ablation, dataset, method, and scientific-result behavior.

## What Changes

- **BREAKING** Restore the five required public file entrypoints at `experiment/{plan,run,status,inspect,reset}.py` and remove the undocumented module-only public command contract.
- **BREAKING** Flatten the Hydra authoring root from `configs/experiment/` to `configs/`, replace `_base.yaml` plus forwarding `config.yaml` with one canonical root `config.yaml`, and remove root presets that only save ordinary CLI overrides.
- Keep only config groups that represent real user choices; remove one-option groups, duplicate runtime dependency metadata, dead config fields, and the rejected per-method package-override defaults syntax.
- Move the shared MLflow SQLite store and managed artifacts to the locked `runs/.mlflow/` paths and ensure aggregate results are represented as queryable metrics as well as curated table artifacts.
- Remove unused optional parameters, unused default values, impossible type branches, no-op exception handling, silent fallbacks, one-line wrappers, repeated state writes, dead functions, and test-only dependency-injection seams from production APIs.
- Replace wide optional config models with discriminated contracts where the alternatives have different required fields, including raw versus importance preparation, BM25 versus Dense tuning, ordinary versus Dense-FT-seeded R-GCN training, ordinary versus ablation aggregation, and normal versus alias artifacts.
- Eliminate duplicate method, ablation, status, summary-path, stage-binding, and artifact-binding authorities. Each fact shall have one owner, and direct stage execution shall consume the same persisted contract rather than reconstruct a second incomplete `StageInvocation`.
- Remove legacy `Prepare*Args` config carriers and unreachable `None` branches left behind after the Pydantic stage-YAML migration.
- Preserve the existing artifact-backed `missing`/`complete`/`stale`/`alias` states, completed-prefix resume semantics, all eight methods, three datasets, Memory Stream, hidden Dense-FT dependencies, ablations, direct stage execution, local scientific artifacts, and successful workflow outputs.
- Replace tests that freeze accidental implementation shapes with boundary tests that prove the smaller public and ownership contracts.

## Capabilities

### New Capabilities

- `minimal-experiment-configuration`: Defines the single-root config tree, legal config groups, root override behavior, command configuration, and absence of presets or duplicate runtime metadata.
- `minimal-experiment-core`: Defines required ownership boundaries and forbids dead optional/default branches, duplicate registries and bindings, legacy config carriers, silent fallbacks, and test-only production interfaces.
- `experiment-public-interface`: Defines the five file entrypoints, their Hydra/key-value command behavior, and the separation between public scripts and reusable experiment-core code.
- `experiment-observability-repair`: Defines the fixed MLflow store, parent/child run behavior, local artifact authority, curated uploads, and queryable evaluation/aggregate metrics.

### Modified Capabilities

None. The previous refactor capabilities have not been archived into `openspec/specs/`; this change introduces explicit corrective contracts without claiming a base-spec delta.

## Impact

- Configuration files under `configs/experiment/**` will be moved, merged, or deleted; active documentation and tests must use the flattened paths and the canonical root config.
- Public experiment commands and imports under `graph_memory/experiment/{plan,run,status,inspect,reset}.py` will be replaced by top-level `experiment/*.py` adapters plus narrower reusable core modules.
- `graph_memory/experiment/{config,service,planning,registry,layout,stage_cli,state,status,resume,execution,tracking,persistence}.py` will be simplified and, where ownership is currently duplicated, split or removed.
- Migrated stage scripts, especially prepare, retrieval, training, and aggregation, will lose legacy argument carriers and dynamic `getattr`-based field discovery.
- OpenSpec status for the earlier refactor must no longer be treated as implementation-complete until this corrective change is implemented and the real workflow matrix is rerun.
- No new third-party dependency, cache feature, compatibility adapter, recipe system, fallback backend, or alternate workflow engine is introduced.
