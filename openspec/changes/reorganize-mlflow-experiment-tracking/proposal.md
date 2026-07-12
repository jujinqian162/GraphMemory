## Why

The current MLflow projection mirrors internal stage attempts instead of presenting experiments at the level researchers compare them: Hydra jobs and retrieval baselines. Multirun jobs have unreadable filesystem identities, parent runs contain an uncurated expansion of every method configuration but no useful result summary, and one baseline is fragmented across many stage children with duplicated single-value metrics.

## What Changes

- **BREAKING** Replace Hydra's full override-derived multirun directory leaf with a concise job identity such as `0_num_layers=2`, while retaining the complete resolved configuration and override list as local files and MLflow artifacts.
- **BREAKING** Replace one MLflow child per executed stage attempt with one child per user-visible baseline, without changing local stage execution, summaries, cache, resume, or artifact authority.
- Limit parent parameters to concise experiment identity and execution selections; do not recursively publish `method_configs` or search-space internals on the parent.
- Make the parent run an experiment-summary container: render the all-baseline result table in Overview and publish the resolved config, overrides, aggregate result tables, and workflow summaries as organized artifacts.
- Explicitly keep comparison metrics and comparison plots out of the parent. The repository will not generate parent comparison images or register parent result metrics.
- Give every baseline child the same concise final evaluation metric keys so MLflow Compare Runs can compare baselines across single runs and multirun jobs.
- For trainable baselines, additionally log epoch-indexed training and development series on the same baseline child.
- Publish only the selected method's flattened parameters on its child, together with its tuning, training, retrieval, evaluation, and stage-summary artifacts.
- Stop creating MLflow children for `prepare`, `graphs`, `pairs`, `tune`, `train`, `retrieve`, `evaluate`, and `aggregate` individually. Preserve those identities in local summaries and organized parent/baseline artifacts.
- Stop logging duplicated aggregate/evaluate copies, counts, timings, artifact sizes, and every numeric CSV cell as model metrics. Final baseline metrics remain queryable because Compare Runs requires native MLflow metrics; MLflow's own run-detail UI may still render those scalar metrics as single-point charts.
- Do not add compatibility readers, legacy tracking branches, schema/version tags, migration code, or old-run rewriting. New executions use the new organization directly; existing MLflow records remain untouched historical data.

## Capabilities

### New Capabilities

- `mlflow-experiment-organization`: Defines concise multirun identities, parent experiment summaries, one child per baseline, consistent final metric keys, trainable epoch series, organized artifacts, Compare Runs behavior, and the no-compatibility boundary.

### Modified Capabilities

None. The repository has no archived base capability under `openspec/specs/`; this change introduces the replacement tracking contract as a new capability and explicitly supersedes the stage-attempt projection implemented by the completed but unarchived experiment-workflow changes.

## Impact

- Primary implementation surfaces: `configs/config.yaml`, `graph_memory/experiment/layout.py`, `service.py`, `execution.py`, `tracking.py`, and tracking-related state/tests.
- Existing training, retrieval, evaluation, aggregation, local artifacts, typed stage summaries, direct stage scripts, cache/resume decisions, SQLite backend path, and artifact root remain authoritative and unchanged in scientific meaning.
- Existing tests that require one child per stage attempt, cache-hit child suppression, fully flattened parent parameters, or exhaustive aggregate metric mirroring must be replaced with researcher-facing parent/baseline organization and Compare Runs assertions.
- Active operations and reproducibility documentation must describe the new directory names, parent/child responsibilities, metric placement, artifact layout, and the standard MLflow Compare Runs workflow.
