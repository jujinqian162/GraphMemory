# Reproducibility

The reproducible computation unit is the typed input set of one Prefect Task. Prepared splits, evidence graphs, training pairs, models, predictions, evaluations, and Prefect results live below `data/processed/`. The repository does not maintain separate cache-key files or filesystem locks. External files and local model directories are content-addressed; immutable remote models must be named with an explicit revision.

The inspectable experiment unit is one Hydra job:

```text
runs/<name>/[<hydra-job-selector>/]
  config/resolved.yaml
  config/overrides.yaml
  workflow/summary.yaml
  workflow/ranking_origin.yaml
  assets/manifest.yaml
  metrics/final.metrics.csv
  tables/*.csv
  training/*                 # trainable methods only
  debug/failure_cases.jsonl
```

The run directory is output-only and is not consumed by later computation. Re-running the complete command is the recovery operation: equal Tasks use Prefect cache results, failed Tasks run again, and changed inputs invalidate only their consumers. A missing or corrupt processed asset is a strict error; rerun with `cache.refresh=true` after correcting storage.

Run identity, Hydra job number, output path, Prefect IDs, MLflow IDs, and tracking configuration do not enter scientific Task signatures. Therefore an equal experiment under a new name can reuse all compatible assets while creating a new output directory and one new MLflow run.

One singular method/variant per job also applies to sweeps. Hydra multirun or independent commands create peer jobs that share cache storage. Assign separate commands to `cuda:0`, `cuda:1`, and other devices for multi-GPU ablations; there is no in-Flow variant fan-out.

For offline transfer:

```powershell
uv run python scripts/deliver/collect_run_artifacts.py --name hotpot_smoke_13
```

The delivery preserves all output-only children under `results/<name>/` and the reusable asset references, but does not copy `data/processed/`. Historical planner-era runs and MLflow rows remain untouched and are not migration inputs for the new cache.
