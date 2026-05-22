# Commands

Date: 2026-05-20

Status: Phase 1 implementation runbook.

## Purpose

This is the canonical command sequence for the Phase 1 HotpotQA evidence-tracing pipeline. Commands should be run from the repository root.

The safe path uses separate input and label artifacts:

- Retrieval and graph construction read `*_memory_tasks.input.json`.
- Evaluation and dev tuning read `*_memory_tasks.labels.json`.
- Combined `*_memory_tasks.json` files are compatibility artifacts for humans and original-plan readers only.

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

BM25-seeded graph rerank:

```powershell
python scripts/tune_graph_rerank.py `
  --method bm25_graph_rerank `
  --tasks data/hotpotqa/processed/dev_memory_tasks.input.json `
  --labels data/hotpotqa/processed/dev_memory_tasks.labels.json `
  --graphs data/hotpotqa/processed/dev_graphs.json `
  --output_config configs/phase1_bm25_graph_rerank_dev_selected.json `
  --top_k 10
```

Dense-seeded graph rerank:

```powershell
python scripts/tune_graph_rerank.py `
  --method dense_graph_rerank `
  --tasks data/hotpotqa/processed/dev_memory_tasks.input.json `
  --labels data/hotpotqa/processed/dev_memory_tasks.labels.json `
  --graphs data/hotpotqa/processed/dev_graphs.json `
  --output_config configs/phase1_dense_graph_rerank_dev_selected.json `
  --encoder_model intfloat/e5-base-v2 `
  --query_prefix "query: " `
  --passage_prefix "passage: " `
  --top_k 10
```

## Run Fixed Graph Rerank On Test

BM25-seeded graph rerank:

```powershell
python scripts/run_retrieval.py `
  --method bm25_graph_rerank `
  --tasks data/hotpotqa/processed/test_memory_tasks.input.json `
  --graphs data/hotpotqa/processed/test_graphs.json `
  --graph_config configs/phase1_bm25_graph_rerank_dev_selected.json `
  --output results/ranked_results_bm25_graph_rerank.json `
  --top_k 10
```

Dense-seeded graph rerank:

```powershell
python scripts/run_retrieval.py `
  --method dense_graph_rerank `
  --tasks data/hotpotqa/processed/test_memory_tasks.input.json `
  --graphs data/hotpotqa/processed/test_graphs.json `
  --graph_config configs/phase1_dense_graph_rerank_dev_selected.json `
  --output results/ranked_results_dense_graph_rerank.json `
  --encoder_model intfloat/e5-base-v2 `
  --query_prefix "query: " `
  --passage_prefix "passage: " `
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

After running or modifying the pipeline, use `docs/40-operations/implementation-handoff.md` to review the code entry points, control flow, abstractions, tests, and known Phase 1 limitations.
