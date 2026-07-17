## Why

The current experiment runner implements its own typed planner, subprocess protocol, run-local cache state, multi-method expansion, and MLflow parent/child projection. These layers duplicate workflow responsibilities that Prefect can own directly and prevent expensive prepared data, graphs, training pairs, and trained models from being reused naturally across independent experiments.

Prefect 3.7.8 is already installed and its basic runtime has been spiked successfully in the project environment, so the repository can now replace the custom runner instead of adding another orchestration layer around it.

## What Changes

- **BREAKING** Replace the planner/invocation/subprocess runner with one readable Prefect Flow whose directly called Tasks represent expensive reusable stages.
- **BREAKING** Replace root `methods: [...]` selection with one final `method` and one optional method variant per Hydra job. Multi-baseline and ablation studies become independently launched Hydra jobs whose compatible stages reuse shared Task caches.
- **BREAKING** Do not accept an in-Flow `ablation.variants` list or fan variants out with `task.submit()` in the first implementation. Users launch one command/process per variant, directly or through a Hydra/external launcher, so variants can be assigned to separate GPUs without sharing one Python/CUDA runtime.
- **BREAKING** Remove the public plan command, stage-range execution, generated stage YAML, completed-prefix resume, run-local computation cache, and workflow-owned aggregate stage.
- Add persistent cross-experiment Task caching with cache keys derived only from effective stage inputs, relevant implementation identity, and upstream artifact identity.
- Keep cache refresh and Task state inside Prefect instead of mirroring Task-run records, retry counters, or cache status through every Flow call. The first implementation uses no automatic Task retries or repository-owned cache lock layer.
- Store reusable large computation assets under `data/processed/`; Prefect persists and passes small typed artifact references rather than serializing datasets, graphs, predictions, checkpoints, or model directories as Task results.
- Preserve `runs/<run-name>/` as an output-only experiment surface. Files below `runs/` may be inspected, logged, delivered, or collected, but no experiment Task may consume them as computation input.
- Replace MLflow parent/child tracking with one comparable top-level MLflow run per final method/variant. Active experiment tracking uses only MLflow's fluent active-run API; `MlflowClient`, child-run lookup/reuse, and writes to runs other than the current active run are removed while resolved configuration, parameters, final metrics, small reports, training summaries, and reusable asset references are preserved.
- Preserve `scripts/deliver/collect_run_artifacts.py` by making each output-only run directory contain the complete small-file delivery surface and references to omitted reusable assets.
- Separate cached scientific outputs from fresh runtime benchmarking so a reused ranking, graph, index, or model does not report historical production latency as current execution time.

## Capabilities

### New Capabilities

- `single-method-prefect-workflow`: Defines one Hydra job as one final method/variant, one Prefect Flow, and one direct readable stage chain without planner or subprocess execution.
- `cross-experiment-task-cache`: Defines Task boundaries, effective cache inputs, implementation invalidation, dependency reuse, cache refresh, and interrupted-run recovery behavior.
- `processed-asset-and-run-output-boundaries`: Defines reusable assets under `data/processed/`, typed artifact references, temporary publication, output-only `runs/<run-name>/`, and collectable delivery manifests.
- `single-run-experiment-tracking`: Defines one MLflow run per final method/variant, cache-aware logging, comparable final metrics, asset references, and honest runtime/benchmark measurements.

### Modified Capabilities

None. The repository has no canonical specs under `openspec/specs/`; this change introduces replacement capabilities and supersedes the completed experiment-runner, cache-resume, ablation-orchestration, config-workflow, and MLflow-tracking change contracts.

## Impact

- Replaces most of `graph_memory/experiment/`, including planning, invocation, execution, resume, state, stage CLI, run-local layout, `MlflowClient`, and parent/child tracking responsibilities.
- Simplifies `experiment/` to the Hydra run entrypoint plus narrow discovery/export utilities; removes `experiment/plan.py` and obsolete status/reset semantics tied to run-local computation state.
- Moves stage execution bodies from `scripts/*.py` CLI wrappers into importable `graph_memory/stages/` functions used by Prefect Tasks; obsolete generated-stage CLI entrypoints are removed after parity verification.
- Changes `configs/config.yaml` and method configuration ownership from multi-select `method_configs` to a single final-method contract.
- Adds Prefect as a runtime dependency; the dependency and lockfile update are already present in the worktree.
- Changes new experiment output layout and delivery metadata without migrating or rewriting historical `runs/`, `results/`, Prefect, or MLflow records.
