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

`run_state.yaml` fixes the resolved typed config, mode, selected methods/stages, artifact bindings, and MLflow parent id. Each stage summary fixes identity, effective config, input/output bindings, attempt, timestamps, counts, timings, error, and child run id. A cache hit requires valid outputs and an exactly matching successful summary.

Hydra groups under `configs/experiment/` own scientific defaults. Dataset groups own sources and capacities; profile groups own count policies and trainable scale; method groups own method-specific settings; search-space groups own tuning candidates; tracking defaults own the shared SQLite and artifact locations. Pydantic validation rejects unknown or unresolved values before planning.

Use a fresh name for changed configuration:

```powershell
uv run python -m graph_memory.experiment.plan name=hotpot_smoke_13 profile=smoke seed=13 device=cpu
uv run python -m graph_memory.experiment.run name=hotpot_smoke_13 profile=smoke seed=13 device=cpu
```

Reusing the name with different overrides fails. `cache.enabled=false` reruns the selected ordered plan without weakening summary validation. Interrupted runs resume only after the longest continuous complete/alias prefix.

MLflow is a strict queryable mirror, not cache or artifact authority. One Hydra job maps to one parent and each executed attempt to one child. The adapter logs flattened configuration, environment/provenance tags, counts, timings, train/tune/evaluation metrics, and an allowlist of small artifacts. Datasets, graphs, predictions, pairs, checkpoints, and model directories are recorded as prohibited path metadata and are never uploaded.

For offline transfer:

```powershell
uv run python scripts/deliver/collect_run_artifacts.py --name hotpot_smoke_13
```

The delivery can be inspected without MLflow and includes typed state, resolved configuration, YAML summaries, selected tuning files, compact metrics/failure artifacts, and final tables.
