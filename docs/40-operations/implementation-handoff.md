# Implementation handoff

The experiment workflow has four explicit ownership layers:

- `graph_memory/experiment/config.py` defines the closed singular Hydra/Pydantic job contract.
- `graph_memory/experiment/artifacts.py` owns external identities, processed references, manifests, validation, staging, and atomic publication below `data/processed/`.
- `graph_memory/experiment/tasks.py` owns thin cached Prefect Tasks and shared Prefect result storage. It has no repository lock manager, broad retry policy, or cache-state projection.
- `graph_memory/experiment/workflow.py` owns one synchronous Flow with direct method branches; `output.py` and `tracking.py` project the final result to output-only `runs/` files and one active MLflow run.

Importable scientific bodies live in `graph_memory/stages/{prepare,graphs,pairs,models,retrieve,evaluate}.py`. They consume typed external or processed references and return typed stage results. Large datasets, graphs, pairs, predictions, checkpoints, and model directories are processed assets; a later stage never reads them from `runs/`.

The public commands are:

- `experiment/run.py` for one singular method/variant job, including Hydra multiruns of independent jobs;
- `experiment/inspect.py` for output discovery and summaries;
- `scripts/deliver/collect_run_artifacts.py` for copying complete output-only trees into `results/`.

When extending the workflow:

1. Add or change the singular Hydra method config and its closed Pydantic discriminator.
2. Put scientific behavior in an importable stage service and keep the Prefect Task wrapper thin.
3. Include every behavior-bearing input, external digest/revision, runtime identity, and implementation version in the Task signature.
4. Publish every reusable file below `data/processed/` through `ArtifactPublisher`; declare every workspace output.
5. Add the direct branch to the one Flow, then project only small current-run reports and asset references below `runs/`.
6. Log through fluent active-run MLflow APIs only. One Hydra job is one Flow run and one MLflow run.
7. Verify focused cache invalidation, fresh-name workflow execution, output collection, static checks, and the full test suite.

Do not add a planner, arbitrary stage range, generated stage command, subprocess execution, completed-prefix resume, run-local cache truth, multi-method job, Flow-internal variant fan-out, MLflow parent/child lifecycle, run-ID reuse, or scientific reads from `runs/`. To compare methods or variants, launch independent Hydra jobs and group their peer MLflow runs with the shared study name.
