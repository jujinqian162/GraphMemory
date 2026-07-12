# Reproducibility

The reproducible unit is a named Hydra job. Its local source of truth is:

```text
runs/<name>/
  run_state.yaml
  config/resolved.yaml
  config/overrides.yaml
  config/stages/**/*.yaml
  **/*.run_summary.yaml
  predictions/
  metrics/
  tables/
  learned/
```

`run_state.yaml` fixes the resolved typed config, mode, selected methods/stages, artifact bindings, and MLflow parent id. Each stage summary fixes identity, effective config, input/output bindings, attempt, timestamps, counts, timings, error, and optional owning baseline-child id. A cache hit requires valid outputs and an exactly matching successful summary.

Hydra groups under `configs/` own scientific defaults. `configs/config.yaml` is the only root. Dataset groups own sources and capacities, profile groups own count policies and trainable scale, method groups own method-specific settings, and search-space groups own tuning candidates. Fixed tracking paths resolve to `runs/.mlflow/tracking.db` and `runs/.mlflow/artifacts/`. Pydantic validation rejects unknown or unresolved values before planning.

Use a fresh name for changed configuration:

```powershell
uv run python experiment/plan.py name=hotpot_smoke_13 profile=smoke seed=13 device=cpu
uv run python experiment/run.py name=hotpot_smoke_13 profile=smoke seed=13 device=cpu
```

Reusing the name with different overrides fails. `cache.enabled=false` reruns the selected ordered plan without weakening summary validation. Interrupted runs resume only after the longest continuous complete/alias prefix.

MLflow is a strict presentation mirror, not cache or artifact authority. One Hydra job maps to one concise parent and each selected baseline or executable ablation variant maps to one stable child, including fully cached baselines. The parent carries concise selections, the all-baseline Overview table, and `config/`, `results/`, and shared `workflow/` artifacts, but no native model metrics. Baseline children carry method-relative parameters, common `final.*` metrics, genuine `train.*` epoch series, and organized small artifacts. Counts, timings, statuses, errors, paths, kinds, and sizes stay outside the model-metric namespace. Datasets, graphs, predictions, pairs, checkpoints, and model directories are metadata-only and are never uploaded.

Multirun directory names are concise identifiers such as `0_num_layers=2`; they are not a serialization of the full job. Reproduction still uses each job's complete `config/resolved.yaml`, `config/overrides.yaml`, typed summaries, and local artifacts. Existing MLflow rows from the former projection are historical and are not migrated. Use a fresh name or explicitly reset an old local name before executing the replacement projection.

For offline transfer:

```powershell
uv run python scripts/deliver/collect_run_artifacts.py --name hotpot_smoke_13
```

The delivery can be inspected without MLflow and includes typed state, resolved configuration, YAML summaries, selected tuning files, compact metrics/failure artifacts, and final tables.
