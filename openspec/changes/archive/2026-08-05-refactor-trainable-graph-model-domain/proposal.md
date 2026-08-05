## Why

The accepted core package refactor plan assigns train pair generation and trainable graph retriever internals to Change C after retrieval boundaries have been split. The current `graph_memory.learned` package still mixes training-pair artifact generation, tensor feature construction, batching, neural model components, checkpoint IO, training, development evaluation, and checkpoint-backed inference, which keeps model internals coupled to data-building and forces inference to import training code.

## What Changes

- Move train pair generation into a `graph_memory/training_pairs/` domain with an explicit builder and parallel negative samplers.
- Move trainable graph retriever config, checkpoint, tensorization, batching, neural components, factory, training, dev evaluation, text embedding, and inference into `graph_memory/models/graph_retriever/`.
- Move the retrieval adapter for `dense_rgcn_graph_retriever` into `graph_memory/retrieval/methods/trainable_graph.py`.
- Keep public CLI flags, workflow command contracts, artifact schemas, checkpoint schema, ablation mappings, relation vocab ordering, tensor ordering, ranking semantics, and training semantics unchanged.
- Remove production/script/test dependencies on `graph_memory.learned.data`.
- Remove the inference-to-training dependency by making inference and training share a model factory.
- Retain `graph_memory/training_config.py` as the narrow workflow integration port for trainable config loading; do not modify `scripts/workflow/`.
- Do not perform Batch 9 final old-module deletion, durable docs promotion, or broad facade cleanup in this change.

## Capabilities

### New Capabilities
- `training-pair-boundaries`: Behavior-preserving training-pair artifact construction through an owned training-pairs domain and sampler boundary.
- `graph-retriever-model-boundaries`: Behavior-preserving trainable graph retriever model, tensorization, training, checkpoint, inference, and retrieval-adapter boundaries.

### Modified Capabilities

## Impact

- Affected production areas: `graph_memory/learned/*`, `graph_memory/types.py`, `graph_memory/training_config.py`, `graph_memory/retrieval/factory.py`, `graph_memory/retrieval/requests.py`, `scripts/build_train_pairs.py`, `scripts/train_graph_retriever.py`, `scripts/run_trainable_retrieval.py`, and related tests.
- Public CLI and workflow behavior are not intended to change.
- No new production dependency is introduced.
