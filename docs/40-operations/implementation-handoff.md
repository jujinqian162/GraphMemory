# Implementation handoff

The experiment workflow is owned by `graph_memory/experiment/`:

- `config.py` defines closed Hydra/Pydantic contracts and resolution.
- `layout.py` is the only run-local path constructor.
- `graph_memory/registry/methods.py` is the single lifecycle, dependency, and train-artifact authority for all eight methods.
- `graph_memory/registry/ablations.py` owns typed ablation patches and invalidation stages.
- `stage_models.py` defines discriminated stage payloads; `invocation.py` defines the one complete persisted execution contract.
- `planning.py` owns workflow selection, ordering, dependencies, and stage bounds.
- `state.py`, `status.py`, and `resume.py` own typed local truth and prefix resume.
- `execution.py` owns sequential subprocess fail-fast execution.
- `tracking.py` owns strict MLflow parent/child mirroring and artifact allowlisting.
- `service.py` is shared by the five public entrypoints.

Public commands are the five files `experiment/{plan,run,status,inspect,reset}.py`. The package-level command facades, positional runner, workflow helper package, JSON config codec, generic stage wrappers, and parameter-style stage parsers are removed without adapters.

All stage scripts under `scripts/` accept exactly `--config <absolute-yaml>`, validate a closed stage model, call reusable code under `graph_memory/stages/`, and use the shared stage lifecycle. Direct scripts never create MLflow runs.

When extending the workflow:

1. Add scientific defaults to the appropriate Hydra group.
2. Extend the closed Pydantic contract and discriminated stage model.
3. Project paths only through `RunLayout` and dependencies only through the runtime method registry and planner.
4. Keep local artifacts and stage summaries authoritative.
5. Add plan, direct-stage, status/resume, tracking, cross-dataset, and delivery tests as applicable.
6. Update the command runbook and run the full verification gates.

Do not introduce Hydra object-instantiation runtime construction, a second default source, alternate workflow dispatch, custom run roots, compatibility readers, MLflow-based cache decisions, or large artifact upload paths.
