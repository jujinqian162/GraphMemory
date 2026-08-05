# Reproducibility

Scientific unit: typed inputs of one Prefect Task. Assets live under `data/processed/`. External files and local models are content-addressed; remote models need an explicit revision.

Inspectable unit: one Hydra job under `runs/<name>/[<hydra-job-selector>/]`:

```text
config/resolved.yaml
config/overrides.yaml
workflow/summary.yaml
assets/manifest.yaml
metrics/final.metrics.csv
metrics/per_task.jsonl
training/*          # trainable only
debug/failure_cases.jsonl
```

`runs/` is output-only. Re-run the same command to recover: equal Tasks hit cache; failed Tasks retry; changed inputs invalidate consumers. Corrupt processed assets are hard errors — fix storage, then `cache.refresh=true`.

Run name, Hydra job number, output path, Prefect/MLflow IDs do not enter scientific Task signatures. Runtime placement fields (`device`, `workers`, `encoding.enable_gpupool`, and encoding `chunk_size`) are also excluded. Processed inputs contribute artifact kind + content digest rather than their materialization URI, so byte-identical assets remain reusable after rematerialization. Equal work under a new name reuses assets and creates a new run directory + MLflow run.

Offline transfer:

```powershell
uv run python scripts/deliver/collect_run_artifacts.py --name hotpot_smoke_13
```

Delivery keeps output-only trees and asset references; it does not copy `data/processed/`.
