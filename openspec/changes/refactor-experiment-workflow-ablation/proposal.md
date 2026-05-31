## Why

The experiment runner currently derives lifecycle behavior from scattered method flags such as `requires_checkpoint` and central method-specific branches. Adding automatic R-GCN ablation runs directly to that structure would make `scripts/experiment.py` and its orchestration helpers increasingly difficult to extend for Dense-FT, Memory Stream, GraphRAG, or future graph retrievers.

This change introduces a small workflow-planning layer and uses it to run R-GCN ablations as explicit experiment variants. Each variant declares changed dimensions, each workflow step declares its dimension dependencies, and the planner derives the earliest invalidated step without hard-coding variant names.

## What Changes

- Add a typed workflow-planning package under `scripts/workflow/` while keeping `scripts/experiment.py` as the user-facing entry point and low-level `scripts/*` commands as explicit IO adapters.
- Replace central lifecycle inference from broad retriever flags with a lightweight static method-to-workflow registry.
- Represent lifecycle stages, workflow identifiers, artifact roles, changed dimensions, variant identifiers, and status values with explicit typed values instead of unconstrained `str` values where the value set is closed.
- Let each workflow declare its full ordered steps, semantic artifact dependencies, produced artifacts, and command adapters.
- Let the planner expand run units by method and variant, assign deterministic artifact namespaces, alias reusable artifacts, validate dependencies, and produce concrete low-level commands.
- Add config-controlled R-GCN ablation orchestration with a simple `enable_ablation` switch, optional variant selection, and minimal per-variant overrides layered over the main training config.
- Derive reuse boundaries declaratively: variants declare changed dimensions and workflow steps declare which dimensions invalidate their outputs.
- Reuse the main `full_rgcn` run instead of retraining an identical ablation baseline.
- Generate run-local effective training configs, per-variant artifacts, status rows, and `tables/ablation_results.csv`.
- Preserve strict downstream resume behavior: an explicit downstream stage range fails fast when required upstream artifacts are absent instead of silently scheduling hidden training work.
- Keep Dense-FT, Memory Stream, and GraphRAG implementations out of scope, but make their later workflow adapters local additions rather than new planner branches.

## Capabilities

### New Capabilities

- `workflow-driven-experiment-planning`: Provides typed method workflows, declarative step dependencies, deterministic plan expansion, artifact reuse, and strict downstream resume behavior.
- `retriever-ablation-orchestration`: Provides config-controlled retriever ablation expansion, R-GCN variant execution, per-variant artifacts, status inspection, and ablation result aggregation.

### Modified Capabilities

- None.

## Impact

- Affected entry point:
  - `scripts/experiment.py`
- New orchestration package:
  - `scripts/workflow/`
- Existing orchestration compatibility surface:
  - `graph_memory/experiment.py`
- Affected configs and documentation:
  - `configs/experiments/`
  - `configs/training/dense_rgcn_graph_retriever/`
  - `docs/configs/`
  - `docs/40-operations/`
- Affected low-level adapters:
  - `scripts/aggregate_tables.py`
  - existing retrieval, training, and evaluation scripts only where an explicit variant artifact or model-visible graph-view argument is required
- Affected tests:
  - runner planning, config resolution, resume behavior, artifact paths, aggregation, and R-GCN ablation integration tests
- No new runtime dependency and no dynamic plugin discovery, general DAG engine, or implementation of Phase 2 baseline methods.
