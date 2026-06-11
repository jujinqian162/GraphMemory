# Commands

Date: 2026-05-20

Status: Experiment runner runbook with low-level Phase 1 command reference.

## Purpose

This document shows the recommended experiment runner path and the low-level Phase 1 HotpotQA evidence-tracing commands. Commands should be run from the repository root.

The safe path uses separate input and label artifacts:

- Retrieval and graph construction read `*_memory_tasks.input.json`.
- Evaluation and dev tuning read `*_memory_tasks.labels.json`.
- Combined `*_memory_tasks.json` files are compatibility artifacts for humans and original-plan readers only.

## Recommended Experiment Runner

Use `scripts/experiment.py` for normal runs. It creates an isolated directory under `runs/<experiment_name>/`, writes `manifest.json`, writes `config/effective_config.json`, and generates all low-level script paths.

By default, `plan` and `run` inspect live artifact status and prune the completed prefix of the selected command plan. This lets repeated full-stage invocations resume from the first missing or stale command. Pass `--no-cache` when you want to see or execute the full selected plan even if earlier commands are already complete.

Initialize a run:

```powershell
python scripts/experiment.py init quick_valid_100 `
  --config configs/experiments/hotpotqa_evidence_retrieval.json `
  --profile quick `
  --methods bm25,dense,bm25_graph_rerank,dense_graph_rerank,dense_rgcn_graph_retriever
```

Plan without executing:

```powershell
python scripts/experiment.py plan quick_valid_100 `
  --run-root runs `
  --stages prepare,graphs,retrieve,evaluate,aggregate `
  --methods bm25
```

Render the full selected plan without cache-aware pruning:

```powershell
python scripts/experiment.py plan quick_valid_100 `
  --run-root runs `
  --methods bm25 `
  --no-cache
```

Run a BM25-only quick path:

```powershell
python scripts/experiment.py run quick_valid_100 `
  --config configs/experiments/hotpotqa_evidence_retrieval.json `
  --profile quick `
  --methods dense_rgcn_graph_retriever `
  --stages prepare,graphs,pairs,train,retrieve,evaluate,aggregate
```

Force every selected command to execute instead of resuming from cached completed work:

```powershell
python scripts/experiment.py run quick_valid_100 `
  --methods dense_rgcn_graph_retriever `
  --stages prepare,graphs,pairs,train,retrieve,evaluate,aggregate `
  --no-cache
```

Resume from retrieval for selected methods:

```powershell
python scripts/experiment.py run quick_valid_100 `
  --from retrieve `
  --methods dense,dense_graph_rerank
```

Inspect artifact status:

```powershell
python scripts/experiment.py status quick_valid_100
```

Status uses expected output files plus matching run summaries where the low-level script writes a summary. Existing outputs with mismatched inputs, outputs, method, or effective config are reported as `stale` and stop cache-aware prefix pruning.

## Run R-GCN Ablations

List registered variants:

```powershell
python scripts/experiment.py ablations list --method dense_rgcn_graph_retriever
```

Initialize an ablation-enabled run:

```powershell
python scripts/experiment.py init rgcn_ablation_cloud `
  --config configs/experiments/hotpotqa_rgcn_ablation_selected.json `
  --profile cloud-full
```

Plan or run only selected variants:

```powershell
python scripts/experiment.py plan rgcn_ablation_cloud `
  --ablations-only `
  --variant wo_bridge `
  --variant wo_hard_negatives
```

`full_rgcn` aliases the ordinary main R-GCN run. Model-only variants reuse main train pairs and start local work from `train`; `wo_hard_negatives` owns local pairs and starts from `pairs`.

Before running `--ablations-only`, complete the ordinary `dense_rgcn_graph_retriever` run under the same experiment name. The ablation table includes the aliased `full_rgcn` metrics row. If that baseline metrics file is absent, the runner fails before execution and prints the missing path. When `--variant` narrows the work, aggregation reads only `full_rgcn` plus the selected variants.

Resume downstream work only after checkpoints exist:

```powershell
python scripts/experiment.py run rgcn_ablation_cloud `
  --ablations-only `
  --variant wo_bridge `
  --from retrieve
```

If the variant checkpoint is absent, the runner fails before execution and prints the missing path. It does not silently insert `train`.

The low-level commands below remain useful for debugging, contract review, and manually reproducing one stage. They deliberately keep explicit input and output paths.

## Verify The Environment

```powershell
uv run pytest tests -q
```

If the local sandbox blocks `uv` cache access, an already prepared local virtual environment can run the same tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Download Raw Dataset

Download HotpotQA v1 raw files into `data/hotpotqa/raw/`:

```powershell
python scripts/prepare_dataset.py `
  --dataset hotpotqa-v1 `
  --name hotpotqa
```

`--dataset` selects the registered source bundle. `--name` controls the local directory under `data/`, so `--name hotpotqa` writes to `data/hotpotqa/raw/`.

## Prepare HotpotQA Splits

`prepare_hotpotqa.py` drops invalid raw HotpotQA examples by default before split sampling; pass
`--strict_invalid_examples` to fail on the first invalid raw record when auditing data quality.

Train from labeled HotpotQA train:

```powershell
python scripts/prepare_hotpotqa.py `
  --input data/hotpotqa/raw/train.json `
  --output_input data/hotpotqa/processed/train_memory_tasks.input.json `
  --output_labels data/hotpotqa/processed/train_memory_tasks.labels.json `
  --output_combined data/hotpotqa/processed/train_memory_tasks.json `
  --max_examples 5000 `
  --seed 13 `
  --offset 0
```

Dev from labeled HotpotQA dev:

```powershell
python scripts/prepare_hotpotqa.py `
  --input data/hotpotqa/raw/dev.json `
  --output_input data/hotpotqa/processed/dev_memory_tasks.input.json `
  --output_labels data/hotpotqa/processed/dev_memory_tasks.labels.json `
  --output_combined data/hotpotqa/processed/dev_memory_tasks.json `
  --max_examples 500 `
  --seed 13 `
  --offset 0
```

Test from labeled HotpotQA dev with a disjoint offset:

```powershell
python scripts/prepare_hotpotqa.py `
  --input data/hotpotqa/raw/dev.json `
  --output_input data/hotpotqa/processed/test_memory_tasks.input.json `
  --output_labels data/hotpotqa/processed/test_memory_tasks.labels.json `
  --output_combined data/hotpotqa/processed/test_memory_tasks.json `
  --max_examples 1000 `
  --seed 13 `
  --offset 500
```

## Build Graphs

```powershell
python scripts/build_graphs.py `
  --input data/hotpotqa/processed/train_memory_tasks.input.json `
  --output data/hotpotqa/processed/train_graphs.json `
  --max_query_overlap 20 `
  --max_entity_neighbors 10 `
  --max_bridge_edges 50
```

```powershell
python scripts/build_graphs.py `
  --input data/hotpotqa/processed/dev_memory_tasks.input.json `
  --output data/hotpotqa/processed/dev_graphs.json `
  --max_query_overlap 20 `
  --max_entity_neighbors 10 `
  --max_bridge_edges 50
```

```powershell
python scripts/build_graphs.py `
  --input data/hotpotqa/processed/test_memory_tasks.input.json `
  --output data/hotpotqa/processed/test_graphs.json `
  --max_query_overlap 20 `
  --max_entity_neighbors 10 `
  --max_bridge_edges 50
```

## Build Train Pairs

Phase 2 trainable retrieval uses `*_pairs.json` as the supervised query-node training artifact. The script reads already prepared input, label, and graph artifacts; it writes the pair artifact, `*.summary.json`, and `*.run_summary.json`.

```powershell
python scripts/build_train_pairs.py `
  --tasks data/hotpotqa/processed/train_memory_tasks.input.json `
  --labels data/hotpotqa/processed/train_memory_tasks.labels.json `
  --graphs data/hotpotqa/processed/train_graphs.json `
  --output data/hotpotqa/processed/train_pairs.json `
  --config configs/training/dense_rgcn_graph_retriever/base.json
```

For normal experiment runs, edit `configs/training/dense_rgcn_graph_retriever/base.json` instead of passing negative-sampling flags directly. For example, change `profiles.quick.pair_sampling.hard_dense_per_positive` only when the configured dense encoder is available in the environment. The field-by-field Chinese reference lives in `docs/configs/training/dense_rgcn_graph_retriever/base.md`.

## Train Phase 2 R-GCN Retriever

`train_graph_retriever.py` reads train input/label/graph/pair artifacts and dev input/label/graph artifacts. It writes `train_metrics.jsonl`, `train_run_summary.json`, `checkpoints/checkpoint_epoch_<n>.pt`, and `checkpoints/best.pt` under `--output_dir`.

```powershell
python scripts/train_graph_retriever.py `
  --train_tasks data/hotpotqa/processed/train_memory_tasks.input.json `
  --train_labels data/hotpotqa/processed/train_memory_tasks.labels.json `
  --train_graphs data/hotpotqa/processed/train_graphs.json `
  --train_pairs data/hotpotqa/processed/train_pairs.json `
  --dev_tasks data/hotpotqa/processed/dev_memory_tasks.input.json `
  --dev_labels data/hotpotqa/processed/dev_memory_tasks.labels.json `
  --dev_graphs data/hotpotqa/processed/dev_graphs.json `
  --output_dir runs/manual_rgcn `
  --config configs/training/dense_rgcn_graph_retriever/base.json
```

Training hyperparameters live in `configs/training/dense_rgcn_graph_retriever/base.json`. To change batch size for the quick profile, edit `profiles.quick.optimization.batch_size`; to change the default for all profiles, edit `defaults.optimization.batch_size`. Automatic variant selection is configured at the experiment level; `base.json` remains the only hand-maintained complete R-GCN training config. The field-by-field Chinese reference lives in `docs/configs/training/dense_rgcn_graph_retriever/base.md`.

### Dense encoder batching behavior

`encoder.batch_size` controls Sentence-Transformers text mini-batches. `optimization.batch_size` controls how many task graphs are tensorized and trained together. They are independent settings.

Dense retrieval, graph feature construction, graph-rerank initial-score precomputation, and hard-dense negative preparation flatten bounded groups of tasks before encoding. The default graph path also derives dense seed signals from the same normalized embeddings used as node embeddings, and training builds frozen dev graph batches once per invocation rather than once per epoch.

These changes reduce logical encoder calls and partially filled GPU mini-batches, but they do not guarantee wall-clock speedup equal to either batch size. Actual throughput depends on text lengths, model/device transfer, GPU saturation, host memory, and kernel behavior. Increasing `encoder.batch_size` can improve utilization but can also cause GPU OOM.

No persistent embedding-cache artifact or process-global embedding cache is created. Normal `scripts/run_retrieval.py` remains task-oriented so `latency_ms` continues to represent one task's `rank_task()` boundary; cross-task public retrieval throughput and amortized latency require a separate artifact contract.

## Run Flat Retrieval On Test

BM25:

```powershell
python scripts/run_retrieval.py `
  --method bm25 `
  --tasks data/hotpotqa/processed/test_memory_tasks.input.json `
  --output results/ranked_results_bm25.json `
  --top_k 10
```

Frozen dense retrieval:

```powershell
python scripts/run_retrieval.py `
  --method dense `
  --tasks data/hotpotqa/processed/test_memory_tasks.input.json `
  --output results/ranked_results_dense.json `
  --top_k 10 `
  --encoder_model intfloat/e5-base-v2 `
  --query_prefix "query: " `
  --passage_prefix "passage: "
```

Dense retrieval requires the Sentence-Transformers model to be available locally or downloadable in the active environment.

## Tune Graph Rerank On Dev

Graph-rerank tuning computes the seed retriever's complete initial scores once per task in memory, then reuses those scores across the graph-rerank grid. This keeps `dense_graph_rerank` tuning compatible with the existing CLI while avoiding one dense encoding pass per candidate. No persistent score-cache artifact is written.

Search-space and selected-config artifacts use `neighbor_type_weights` for memory-to-memory graph edge calibration. Deprecated `type_weights` configs are rejected; convert them before rerunning graph-rerank commands.

BM25-seeded graph rerank:

```powershell
python scripts/tune_graph_rerank.py `
  --method bm25_graph_rerank `
  --tasks data/hotpotqa/processed/dev_memory_tasks.input.json `
  --labels data/hotpotqa/processed/dev_memory_tasks.labels.json `
  --graphs data/hotpotqa/processed/dev_graphs.json `
  --output_config runs/manual_hotpotqa/tuned/bm25_graph_rerank.dev_selected.json `
  --top_k 10 `
  --grid_config configs/search_spaces/graph_rerank.json
```

Dense-seeded graph rerank:

```powershell
python scripts/tune_graph_rerank.py `
  --method dense_graph_rerank `
  --tasks data/hotpotqa/processed/dev_memory_tasks.input.json `
  --labels data/hotpotqa/processed/dev_memory_tasks.labels.json `
  --graphs data/hotpotqa/processed/dev_graphs.json `
  --output_config runs/manual_hotpotqa/tuned/dense_graph_rerank.dev_selected.json `
  --encoder_model intfloat/e5-base-v2 `
  --query_prefix "query: " `
  --passage_prefix "passage: " `
  --top_k 10 `
  --grid_config configs/search_spaces/graph_rerank.json
```

## Run Fixed Graph Rerank On Test

BM25-seeded graph rerank:

```powershell
python scripts/run_retrieval.py `
  --method bm25_graph_rerank `
  --tasks data/hotpotqa/processed/test_memory_tasks.input.json `
  --graphs data/hotpotqa/processed/test_graphs.json `
  --graph_config runs/manual_hotpotqa/tuned/bm25_graph_rerank.dev_selected.json `
  --output results/ranked_results_bm25_graph_rerank.json `
  --top_k 10
```

Dense-seeded graph rerank:

```powershell
python scripts/run_retrieval.py `
  --method dense_graph_rerank `
  --tasks data/hotpotqa/processed/test_memory_tasks.input.json `
  --graphs data/hotpotqa/processed/test_graphs.json `
  --graph_config runs/manual_hotpotqa/tuned/dense_graph_rerank.dev_selected.json `
  --output results/ranked_results_dense_graph_rerank.json `
  --encoder_model intfloat/e5-base-v2 `
  --query_prefix "query: " `
  --passage_prefix "passage: " `
  --top_k 10
```

## Run Trainable R-GCN Retrieval On Test

The retrieval-only path reads only task inputs, graphs, and checkpoint. It must not read labels or train pairs.

Use the shared retrieval script with the registered trainable method:

```powershell
python scripts/run_retrieval.py `
  --method dense_rgcn_graph_retriever `
  --tasks data/hotpotqa/processed/test_memory_tasks.input.json `
  --graphs data/hotpotqa/processed/test_graphs.json `
  --checkpoint runs/manual_rgcn/checkpoints/best.pt `
  --output results/ranked_results_dense_rgcn_graph_retriever.json `
  --top_k 10
```

## Evaluate Methods

```powershell
python scripts/evaluate_retrieval.py `
  --pred results/ranked_results_bm25.json `
  --labels data/hotpotqa/processed/test_memory_tasks.labels.json `
  --graphs data/hotpotqa/processed/test_graphs.json `
  --output results/main_results_bm25.csv `
  --failure_cases_output results/debug/failure_cases_bm25_test.jsonl `
  --failure_case_limit 50
```

Repeat the same command shape for:

- `results/ranked_results_dense.json` -> `results/main_results_dense.csv`
- `results/ranked_results_bm25_graph_rerank.json` -> `results/main_results_bm25_graph_rerank.csv`
- `results/ranked_results_dense_graph_rerank.json` -> `results/main_results_dense_graph_rerank.csv`
- `results/ranked_results_dense_rgcn_graph_retriever.json` -> `results/main_results_dense_rgcn_graph_retriever.csv`

The compatibility alias `--gold` is accepted, but `--labels` is preferred.

## Aggregate Tables

```powershell
python scripts/aggregate_tables.py `
  --input_dir results `
  --output_main results/main_results.csv `
  --output_path results/path_results.csv `
  --output_efficiency results/efficiency_results.csv
```

## Leakage Check

Input-visible artifacts should not contain label-only fields:

```powershell
rg "gold_answer|gold_evidence_nodes|supporting_facts|is_gold" data/hotpotqa/processed -g "*input*.json" -g "*graphs*.json"
```

Expected: no matches.

## Review The Code

After running or modifying the pipeline, use `docs/40-operations/implementation-handoff.md` to review the code entry points, control flow, abstractions, tests, and current package ownership. Public commands above are unchanged by the core package refactor; complete retrieval run orchestration lives under `graph_memory/stages/`, while retrieval implementation lives under domain packages such as `graph_memory/retrieval/`, `graph_memory/graphs/`, `graph_memory/training_pairs/`, and `graph_memory/models/graph_retriever/`.
