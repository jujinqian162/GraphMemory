## Why

Evidence R-GCN already interprets `batch_size` as multiple task graphs merged into one disconnected-union forward pass, while provenance R-GCN previously used it as a single-graph optimizer interval and incorrectly scaled an incomplete final group. This semantic split made experiment configuration and reported batch sizes misleading, prevented provenance training from using real graph parallelism, and left both R-GCN paths outside a reproducible PyTorch `DataLoader` boundary.

## What Changes

- Establish one physical batch contract for both R-GCN families: `per_device_graph_batch_size` is the number of task graphs in one device-local forward, backward pass, and optimizer step.
- Move the generic `GraphBatch` disconnected-union contract out of the evidence-specific internals and provide shared graph collation/device-transfer primitives with explicit node, edge, query, task, and offset invariants.
- Replace prebuilt epoch batch lists with map-style task datasets and seeded PyTorch `DataLoader` instances for evidence and provenance train/dev execution. Frozen encoder features are materialized once as CPU task tensors; DataLoader workers do not own or invoke the encoder.
- Add a provenance-specific multi-graph tensor contract that partitions candidates and logical transitions by task, maps every candidate/transition to the correct query, and supports one vectorized model forward over multiple disconnected graphs.
- Preserve provenance candidate-loss protocol `provenance-candidate-loss-v2`: pairwise comparisons and class-balanced logical-edge BCE remain task-local, and a graph batch is reduced as an equal-weight mean over task losses. The batching refactor does not create a new candidate-loss protocol version.
- Preserve the current evidence R-GCN sample-level BCE semantics while changing only its physical data-loading and collation path.
- Make partial DataLoader batches normalize by their actual task or supervised-sample count, eliminating the provenance tail-scaling bug.
- Add deterministic shuffle/sampler behavior and batch observability for per-device graph count, actual tasks per optimizer step, task count, and optimizer step.
- **BREAKING**: replace ambiguous trainer `batch_size` configuration/checkpoint fields with the explicit graph-batch field, advance the evidence R-GCN checkpoint schema from v2 to v3, and advance the provenance R-GCN checkpoint schema from v3 to v4. Legacy checkpoint batch semantics must not be silently reinterpreted.
- Keep public single-task provenance inference behavior by executing the same batched tensor/model contract with graph batch size one; batch-aware dev evaluation splits outputs back into task-local ranking and structured-edge decisions.

## Capabilities

### New Capabilities

- `rgcn-graph-batch-runtime`: shared disconnected-union graph batching, precomputed task datasets, deterministic DataLoader behavior, explicit batch configuration, and runtime/observability invariants for both R-GCN families.
- `provenance-rgcn-multigraph-learning`: provenance multi-query/multi-transition model execution, task-local v2 candidate and edge supervision, batch-aware dev evaluation, and self-describing checkpoint semantics.

### Modified Capabilities

None. The repository currently has no promoted baseline specs under `openspec/specs/`; this change introduces current batching contracts as new capabilities.

## Impact

- Shared/evidence graph contracts, tensorization, batching, training, dev evaluation, and tests under `graph_memory/models/graph_retriever/` and a new shared model-batching module.
- Provenance tensorization, contracts, model forward, loss, training, dev evaluation, inference adapter, checkpoint schema, and tests under `graph_memory/models/provenance_rgcn/`.
- Trainer Pydantic models, stage adapters, Hydra method/profile configuration, checkpoint validation, MLflow training metadata, and documentation under `graph_memory/experiment/`, `graph_memory/stages/`, `configs/`, and `docs/`.
- PyTorch remains the only required training runtime; this change does not introduce Accelerate, PyTorch Lightning, a new graph library, online encoder execution in DataLoader workers, persistent embedding artifacts, or a new candidate-loss protocol.
