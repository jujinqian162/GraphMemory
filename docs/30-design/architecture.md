# Architecture

Date: 2026-06-03

Status: Maintained project-level reference.

## Core Decision

Use a library-core architecture with thin CLI scripts.

```text
Artifacts are the external contract.
Domain packages are the internal ownership map.
CLI scripts are adapters, not the system.
```

Public script names, CLI arguments, retrieval method names, JSON/JSONL/CSV schemas, and checkpoint metadata are the compatibility boundary. Internal `graph_memory.*` imports are allowed to move when ownership becomes clearer.

## External Structure

The public experiment structure remains stable:

```text
data/
  hotpotqa/
    raw/
    processed/
results/
runs/
scripts/
```

The workflow runner and low-level scripts remain the user-facing entry points:

- `scripts/experiment.py`
- `scripts/prepare_hotpotqa.py`
- `scripts/build_graphs.py`
- `scripts/run_retrieval.py`
- `scripts/tune_graph_rerank.py`
- `scripts/build_train_pairs.py`
- `scripts/train_graph_retriever.py`
- `scripts/run_trainable_retrieval.py`
- `scripts/evaluate_retrieval.py`
- `scripts/aggregate_tables.py`

## Package Shape

The current core package is organized by domain ownership:

```text
graph_memory/
  contracts/
  datasets/
  evaluation/
  graphs/
  infrastructure/
  models/
    graph_retriever/
  retrieval/
  text/
  training_pairs/
  validation/
  experiment.py
  io.py
  observability.py
  retrieval_registry.py
  training_config.py
```

Only these root modules are retained as workflow integration ports:

```text
graph_memory/io.py
graph_memory/observability.py
graph_memory/retrieval_registry.py
graph_memory/training_config.py
graph_memory/experiment.py
```

They must stay thin. New core logic belongs in the domain package that owns the behavior.

## Domain Responsibilities

| Package | Responsibility |
|---|---|
| `contracts/` | Artifact-shaped aliases, `TypedDict`s, and stable data language. |
| `validation/` | Fail-fast validators for task, graph, ranking, training-pair, metric, and model contracts. |
| `infrastructure/` | JSON/CSV IO, run summaries, and runtime environment capture. |
| `datasets/` | Dataset-specific parsing, conversion, compatibility records, and split helpers. |
| `text/` | Tokenization, lexical scoring, and entity extraction helpers. |
| `graphs/` | Graph build config, construction rules, graph index, statistics, and graph views. |
| `retrieval/` | Retrieval contracts, request resolution, method factory, execution service, flat methods, graph rerank, trainable adapter, and tuning. |
| `training_pairs/` | Deterministic positive/negative train-pair construction and sampling config. |
| `models/graph_retriever/` | Trainable graph retriever config, tensor batches, neural model, checkpointing, training, dev evaluation, and inference. |
| `evaluation/` | Metric primitives, connectivity, aggregate evaluation service, table splitting, and failure cases. |

## Dependency Direction

Allowed high-level flow:

```text
scripts/*.py
  -> graph_memory domain packages
  -> graph_memory infrastructure / validation

scripts/workflow/*
  -> graph_memory workflow integration ports

retrieval execution
  -> retrieval methods
  -> graphs views when a method needs graph structure

trainable graph retrieval
  -> models.graph_retriever
  -> retrieval.signals
```

Important forbidden directions:

- `contracts/` must not import algorithm packages.
- `graphs/` must not import retrieval, training pairs, models, evaluation, application code, or scripts.
- `retrieval/` must not import scripts or workflow orchestration.
- `models/graph_retriever/` must not import scripts or workflow orchestration.
- `infrastructure/` must not import research-domain packages.
- Core algorithms must not read/write JSON, CSV, or JSONL artifacts directly.
- Core algorithms must not parse CLI arguments.

These rules are enforced by `tests/test_core_refactor_final_boundaries.py`.

## Retrieval Boundary

Public method metadata lives in `graph_memory/retrieval_registry.py`. Runtime construction lives under `graph_memory/retrieval/`:

```text
retrieval.requests / retrieval.resolver
  -> exact method-family build requests
retrieval.factory
  -> method object construction
retrieval.execution.service
  -> per-task ranking and artifact assembly
retrieval.methods.flat
  -> BM25 and dense flat seed methods
retrieval.methods.graph_rerank
  -> graph-rerank engine, components, config, and method adapter
retrieval.methods.trainable_graph
  -> checkpoint-backed trainable retrieval adapter
retrieval.tuning
  -> graph-rerank grid and selected-config service
```

`RetrievalBuildContext` is removed. Dense prefixes, graph configs, checkpoints, and seed providers belong to typed request/runtime objects for the method family that actually needs them.

## Trainable Retriever Boundary

Train-pair generation and trainable model runtime are separate domains:

```text
training_pairs/
  config.py
  samplers.py
  builder.py

models/graph_retriever/
  config/
  internals/
  batching.py
  checkpoint.py
  factory.py
  inference.py
  training.py
  dev_evaluation.py
  text_embeddings.py
```

`training_pairs` may consume retrieval seed signals for hard negatives, but it does not depend on trainable model internals. `models/graph_retriever` owns tensorization, graph-scoring model construction, checkpoint parsing, training, and inference, but it does not parse CLI args or read experiment workflow state.

## Script Boundary

Scripts own:

- CLI/config parsing.
- file paths and artifact IO.
- top-level logging.
- run summary writing.
- invoking validators at artifact boundaries.

Scripts do not own:

- core algorithms.
- metric definitions.
- graph scoring formulas.
- retrieval implementation details.
- trainable model internals.

## Future Extraction Rule

Extract a package or submodule only when ownership is clear and the behavior has multiple independent implementations or has become hard to navigate. Do not introduce dynamic plugin discovery, a dependency-injection container, or a generic pipeline engine while the local static registry and explicit workflow recipes remain sufficient.
