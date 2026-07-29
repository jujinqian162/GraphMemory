# Implementation handoff

## Layers

| Module | Owns |
|---|---|
| `experiment/config.py` | Closed singular Hydra/Pydantic job contract |
| `experiment/artifacts.py` | Content-addressed publish under `data/processed/` |
| `experiment/tasks.py` | Thin cached Prefect Tasks |
| `experiment/workflow.py` | One synchronous Flow; direct method branches |
| `experiment/output.py`, `tracking.py` | `runs/` projection and one MLflow run |
| `stages/*` | Importable scientific bodies |

## Public commands

- `experiment/run.py` — one method/variant job (or Hydra multirun of independent jobs)
- `experiment/inspect.py` — discovery/summaries
- `scripts/deliver/collect_run_artifacts.py` — copy output-only trees to `results/`

## Extending

1. Add singular Hydra method config + Pydantic discriminator.
2. Put science in a stage service; keep the Prefect wrapper thin.
3. Put every behavior-bearing input and implementation version in the Task signature.
4. Publish reusable files only via `ArtifactPublisher` under `data/processed/`.
5. Branch the Flow; project small reports under `runs/`.
6. One job = one Flow = one MLflow run. No planner, stage ranges, multi-method jobs, in-Flow variant fan-out, or scientific reads from `runs/`.
