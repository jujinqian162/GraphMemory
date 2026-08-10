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
predictions/ranked_prefix.jsonl.gz  # per-query fixed-k/token-budget reproduction prefix
training/*          # trainable only
debug/failure_cases.jsonl
```

`runs/` is output-only. Re-run the same command to recover: equal Tasks hit cache; failed Tasks retry; changed inputs invalidate consumers. Corrupt processed assets are hard errors — fix storage, then `cache.refresh=true`.

Run name, Hydra job number, output path, Prefect/MLflow IDs do not enter scientific Task signatures. Runtime placement fields (`device`, `workers`, `encoding.enable_gpupool`, and encoding `chunk_size`) are also excluded. Processed inputs contribute artifact kind + content digest rather than their materialization URI, so byte-identical assets remain reusable after rematerialization. Equal work under a new name reuses assets and creates a new run directory + MLflow run.

## Complete-run validation and offline transfer

The collector validates the complete output contract by default. It rejects missing/empty `metrics/per_task.jsonl`, duplicate task IDs, malformed final metrics, missing asset references, and required files that would be skipped by the size limit. Output schema v2 additionally requires a deterministic gzip JSONL prediction prefix containing enough ranked-node records for fixed `top_k` and the maximum reported token budget; the full processed ranking remains content-addressed through `assets/manifest.yaml`. The delivery manifest records SHA-256 for every copied file plus a sorted task-ID fingerprint for each job. Legacy schema-v1 deliveries remain readable but do not contain the compact prediction prefix.

```powershell
uv run python scripts/deliver/collect_run_artifacts.py --name hotpot_smoke_13

# Formal ISETrace main-test delivery: enforce the frozen 2,000-query population.
uv run python scripts/deliver/collect_run_artifacts.py `
  --name isetrace_rgcn_natural_s13 `
  --expected-per-task-count 2000
```

Use `--allow-incomplete` only for explicitly historical/debug trees; such a delivery is not a complete scientific run. Delivery keeps output-only trees and asset references; it does not copy `data/processed/`.

## Main-results aggregation

The aggregator consumes complete run outputs, checks an identical test task set and prepared-test artifact digest, reports trainable mean/sample standard deviation, includes token-budget metrics, and computes `method - baseline` paired cluster-bootstrap intervals. New ISETrace evaluations persist `graph_id` and `memory_mode` in every per-task row. For legacy outputs, join the frozen query-authoring metadata sidecar:

```powershell
uv run python scripts/aggregate_main_results.py `
  --run dense_ft_natural=results/isetrace/isetrace_dense_ft_natural_refactor_s13 `
  --run rgcn_natural=results/isetrace/isetrace_rgcn_natural_only_refactor_s13 `
  --baseline dense_ft_natural `
  --trainable dense_ft_natural `
  --trainable rgcn_natural `
  --query-metadata data/isetrace/query-authoring/isetrace-v7-raw.jsonl.metadata.jsonl `
  --expected-task-count 2000 `
  --output results/isetrace/main_results_natural.json `
  --output-csv results/isetrace/main_results_natural.csv
```

For final multi-seed results, repeat each trainable label with seeds 13/17/29. Explicit `LABEL=PATH` syntax is required when the same raw method ID has natural, mixed, or template-only supervision variants.
