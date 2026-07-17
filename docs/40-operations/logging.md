# Logging and run records

Every Hydra job owns exactly one top-level MLflow run for its final method and optional variant. The active run receives the resolved scientific parameters, stable identity/study tags, common `final.*` metrics, Task cache metadata, reusable asset references, and small output artifacts. The tracking layer uses fluent active-run APIs only; there are no parent, child, dependency, or stage runs and no run-ID lookup/reuse.

Prefect owns Task state and cache reuse. Tasks do not retry automatically in the initial workflow; rerunning the Flow reuses compatible completed results. Cached Task results still return complete typed result objects, so a new MLflow run and a new output-only run directory are complete even when no scientific stage executes freshly.

Reusable assets and Prefect persistence live below `data/processed/`. A run output below `runs/<name>/` is a presentation and delivery projection only. It is never cache truth and is never read by a scientific Task. Prefect remains the only authority for Task execution and cache state; `assets/manifest.yaml` records every reusable asset URI and digest without copying the large asset into the run directory.

Production-time values stored with a cached ranking are provenance metadata, not a current runtime measurement. Enable `benchmark.enabled=true` to run the uncached warmup/repetition benchmark Task and log fresh `benchmark.*` metrics.
