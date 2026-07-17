# Direct cutover and semantic parity matrix

This change is a direct cutover. Historical `runs/` trees are immutable evidence and are not migrated into the processed cache. All new scientific inputs and reusable outputs live below `data/processed/`; new `runs/<run-name>/` trees contain projection-only artifacts.

## Accepted execution matrix

| Scenario | New command/behavior | Acceptance rule |
|---|---|---|
| Single method | `uv run python experiment/run.py name=<name> dataset=<dataset> method=<method>` | One Prefect Flow, one final method/variant result, one top-level MLflow run. |
| Baseline comparison | Hydra `-m method=bm25,dense,graphrag` | Independent peer jobs under one named multirun directory; no parent run or Flow fan-out. |
| Ablation comparison | Launch one command per singular `method.variant`; assign `CUDA_VISIBLE_DEVICES` outside the Flow | Independent jobs may run concurrently and reuse compatible Prefect results. Concurrent first computations may duplicate work, but opaque atomic asset publication prevents overwrite. |
| Cache hit | Repeat equal scientific inputs under another run name | Compatible Tasks return `Cached`; the job still emits a new complete run projection and MLflow run. |
| Cache refresh | Add `cache.refresh=true` | Every selected scientific Task executes and atomically republishes its result; refresh is not a cache-key input. |
| Failure | Task or tracking exception escapes | MLflow run is marked failed, no incomplete processed artifact is published, and the output tree is not computation truth. |
| Recovery | Correct the input or use `cache.refresh=true`, then relaunch the job | Valid processed assets remain reusable; missing/corrupt assets fail closed instead of falling back to `runs/`. |
| Delivery | `uv run python scripts/deliver/collect_run_artifacts.py --name <name>` | Copies the complete output-only run tree and records omitted reusable asset URI/digest pairs without traversing `data/processed/`. |
| Historical run | Inspect old `runs/<name>` in place | No automatic migration, cache import, or use as a scientific input is allowed. |

## Frozen parity evidence

`tests/fixtures/prefect_semantic_parity.json` records behavior-bearing prepared, ranking, and evaluation summaries from the retired runner alongside fresh Prefect smoke runs for all seven cutover branches. `tests/test_prefect_semantic_parity.py` keeps that matrix free of paths, timestamps, latency, memory, and transient timing bytes.

The stable behavior contract is task identity, ranked-result structure, ordered-node availability, and named evaluation metrics. Exact task IDs, sample counts, learned scores, and tie order are not compared when the legacy and Prefect smoke inputs or freshly trained weights differ. The preparation rename from legacy `memory_items/query` to the current `candidate_sentences/question` is an intentional internal contract normalization; provenance preparation retains its dataset-specific shape.

The frozen sources are evidence only. Tests never read them from `runs/`; the extracted fixture is the durable compatibility boundary.
