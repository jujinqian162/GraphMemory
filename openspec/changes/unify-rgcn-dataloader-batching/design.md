## Context

The repository has three trainable paths with incompatible data-loading boundaries. Dense fine-tuning already passes a PyTorch `DataLoader` into SentenceTransformers 2.7, evidence R-GCN preconstructs a fixed `list[TrainingBatch]`, and provenance R-GCN preconstructs one `ProvenanceGraphTensor` per task and performs one forward per task. Evidence `batch_size` means task graphs per disconnected-union forward; provenance previously used it as an optimizer interval and incorrectly scaled an incomplete final group.

The two R-GCN implementations already share the same `RGCNGraphEncoder`, and provenance imports `GraphBatch` from evidence-owned internals. The shared graph layer aggregates only along `edge_index`, uses per-target relation-degree normalization, and applies per-node LayerNorm; it contains no cross-node batch statistic. A correctly offset disconnected union is therefore mathematically separable by task when dropout is disabled.

Provenance adds constraints that evidence batching does not have. Its current tensor contract contains one scalar query index, one flat candidate list, and logical transitions with single-graph node indices. Its candidate loss is `provenance-candidate-loss-v2`: task-local pairwise logistic ranking over materialized positive/negative pairs. Its logical-edge head uses task-local class-balanced BCE. Neither loss may compare or rebalance examples across tasks after graphs are merged.

This change is a training-runtime and tensor-contract refactor. It must be completed before the formal multi-seed v2 provenance experiment so that reported batch sizes are truthful and the current tail-normalization bug cannot affect the result.

## Goals / Non-Goals

**Goals:**

- Give evidence and provenance R-GCN one physical batch definition: task graphs per device-local forward.
- Execute multiple provenance graphs as one disconnected-union graph without cross-task query, candidate, transition, message, or loss leakage.
- Put evidence and provenance train/dev iteration behind seeded map-style PyTorch DataLoaders.
- Materialize frozen encoder features once into CPU task tensors and dynamically collate only the current graph batch.
- Preserve evidence sample-level BCE and provenance task-balanced v2 pairwise plus logical-edge BCE semantics.
- Make partial graph batches mathematically exact and apply one optimizer step per DataLoader batch.
- Record graph-batch and optimizer-step semantics in configs, checkpoints, metrics, and experiment tracking.
- Keep public single-task inference behavior and checkpointed model parameters compatible with the batched forward contract.

**Non-Goals:**

- Changing provenance candidate loss away from `provenance-candidate-loss-v2` or changing logical-edge BCE.
- Making evidence loss task-balanced or otherwise changing its scientific loss reduction.
- Jointly training the frozen text encoder, invoking an encoder from DataLoader workers, or adding persistent embedding artifacts.
- Introducing Accelerate, PyTorch Lightning, PyG, DGL, distributed training, online hard-negative mining, or a generic training-framework abstraction.
- Guaranteeing bitwise equality with historical dropout-enabled training runs.
- Loading legacy checkpoint training semantics by silently treating an old `batch_size` as a new graph count.

## Decisions

### 1. Define batch size as device-local task-graph count

Both R-GCN trainer configs use:

```text
per_device_graph_batch_size
    number of task graphs collated into one forward, backward pass,
    and optimizer step on one device
```

The final DataLoader batch may be smaller, so metrics separately record its actual task count. Gradients are never retained across DataLoader batches.

The existing ambiguous `batch_size` field is removed from evidence/provenance trainer configs. Profile settings are separated by method family so graph-size and memory differences do not force identical numeric values merely because the semantics are identical. Initial migration defaults are:

- smoke: graph batch `1` for both families;
- quick: graph batch `8` for both families;
- full evidence: graph batch `128`, preserving current true-batch behavior;
- full provenance: graph batch `8`, introducing true graph parallelism conservatively.

The full provenance graph batch may be raised only after measured node/edge/candidate counts, CPU memory, peak VRAM, and throughput are recorded. Changing this physical batching implementation does not change the candidate-loss protocol tag.

Alternative: keep `batch_size` and change only its provenance interpretation. Rejected because old checkpoints, configs, paper notes, and metrics would silently assign two meanings to the same field.

Alternative: require the same numeric graph batch for both families. Rejected because identical semantics do not imply identical variable-graph memory costs.

### 2. Promote graph batching to shared model infrastructure

A shared model-batching package owns two levels of graph data:

```text
TaskGraphTensor
    one task's CPU node/edge/query tensors and identity metadata

GraphBatch
    one or more TaskGraphTensor values collated as a disconnected union
```

The shared collator concatenates node tensors, offsets every edge and query index, and builds:

```text
query_node_indices: [B]
task_node_offsets: [0, n_0, n_0+n_1, ..., total_nodes]
task_ids: B entries
node_ids_by_task: B entries
```

The shared contract requires `len(task_node_offsets) == B + 1`, a zero first offset, a final offset equal to total nodes, one query index per task, and edge endpoints contained in the corresponding task interval. Shared device transfer moves tensors without mutating metadata.

Evidence-specific and provenance-specific collators wrap this primitive to offset their own supervision indices. Training loops and losses remain model-owned; this change does not create one generic R-GCN trainer.

Alternative: duplicate provenance graph concatenation beside evidence `_build_batch`. Rejected because edge/query offset invariants are identical and duplicated implementations would make cross-graph leakage fixes diverge.

Alternative: move all model-specific supervision into a universal graph batch. Rejected because evidence samples and provenance candidate/transition supervision have different reduction and inference contracts.

### 3. Separate frozen feature materialization from DataLoader collation

Each family builds a map-style dataset whose item is one validated, precomputed CPU task tensor. Frozen encoder calls happen once in the parent training process before epoch iteration. Feature materialization may internally chunk encoder requests, but DataLoader `__getitem__` and `collate_fn` do not call the encoder.

Train loaders use `shuffle=True`, a `torch.Generator` seeded from trainer `random_seed`, `drop_last=False`, and a family-specific `collate_fn`. Dev loaders use `shuffle=False` and `drop_last=False`. The first implementation fixes `num_workers=0`; it does not expose worker tuning until profiling proves that copying already-materialized custom tensor records through workers is beneficial. CUDA tensors are created only by the explicit batch-to-device step.

Evidence currently prebuilds already-collated epoch batches. It is refactored to materialize per-task graph/features/supervision once and collate current tasks after DataLoader sampling. Provenance similarly retains one precomputed task tensor per request instead of one GPU forward per item.

Alternative: store raw requests in Dataset and encode in `__getitem__`. Rejected because every epoch would repeat frozen encoding and multi-worker execution could duplicate or fail to pickle/CUDA-initialize the encoder.

Alternative: keep prebuilt batch lists and use DataLoader only as a wrapper. Rejected because the batch boundaries would remain fixed across epochs and `shuffle=True` would shuffle batches rather than task graphs.

### 4. Represent provenance candidates and transitions as task-partitioned flat tensors

A single task is tensorized into a provenance task item containing local node indices, node type IDs, candidate IDs/indices, logical transition metadata/indices, and optional training targets. Collation produces a `ProvenanceGraphBatch` with:

```text
graph_batch
node_type_ids

candidate_node_indices
candidate_query_indices
candidate_offsets
candidate_ids_by_task

transition_source_indices
transition_target_indices
transition_query_indices
transition_offsets
logical_transitions_by_task
```

All model-facing indices are global after adding the task node offset. Candidate and transition offsets have length `B + 1`, start at zero, and end at their corresponding flattened logit count. Metadata stays partitioned by task so batched outputs can be split without reconstructing ownership from IDs.

Training supervision is a separate `ProvenanceTrainingBatch` layered on the graph batch. Candidate targets align with all flattened candidates and use `1` for a materialized positive, `0` for a materialized negative, and `-1` for an unpaired candidate excluded from candidate loss. Logical-edge targets align with flattened logical transitions. Pair sample-type metadata remains available for diagnostics but does not control loss grouping.

Alternative: retain `LogicalProvenanceTransition.source_node_index/target_node_index` as the only model indices and mutate every object during collate. Rejected because immutable domain/trace metadata should not double as offset-sensitive model tensors.

Alternative: identify task ownership by splitting candidate IDs such as `task::node`. Rejected because node identity formatting is not a safe tensor partition contract.

### 5. Vectorize the provenance model with explicit query mappings

The provenance model consumes the flattened batch once. It computes node states over the disconnected union, then gathers:

```text
candidate state = states[candidate_node_indices]
candidate query = states[candidate_query_indices]

transition source = states[transition_source_indices]
transition target = states[transition_target_indices]
transition query = states[transition_query_indices]
```

Candidate and edge logits remain flat tensors aligned with their offsets. There is no scalar query broadcast. A one-task batch uses exactly the same forward path as a multi-task batch. Empty transition batches produce an empty edge-logit tensor without changing candidate scoring.

The disconnected-union equivalence gate runs with `dropout=0` or model evaluation mode. Dropout-enabled grouped and sequential runs are distributionally comparable but not bitwise identical because random-mask draw order changes.

Alternative: loop through graph slices inside `forward`. Rejected because it preserves single-graph execution overhead and defeats true graph parallelism.

### 6. Preserve model-specific loss reductions

Evidence retains its current scientific objective: global training-set `pos_weight` when enabled and BCE reduced over supervised evidence samples. DataLoader/graph collation changes must not make tasks equally weighted if their supervised sample counts differ.

Provenance retains candidate-loss protocol `provenance-candidate-loss-v2`. For each task `t`:

```text
L_candidate(t)
    mean pairwise logistic loss over materialized P_t x N_t

L_edge(t)
    class-balanced BCE over that task's logical transitions,
    using that task's positive/negative transition counts

L_task(t)
    candidate_loss_weight * L_candidate(t)
    + edge_loss_weight * L_edge(t)

L_batch
    mean_t L_task(t)
```

No candidate margin, edge `pos_weight`, or mean reduction crosses a task boundary. A task with positives but no selected negatives contributes a zero candidate term as today; its edge term still contributes. Unpaired candidates participate in message passing and inference but not candidate loss.

Each DataLoader batch is normalized and applied independently before the next batch begins:

- evidence denominator: supervised sample count in that batch;
- provenance denominator: task count in that batch.

The normalized loss is backpropagated, clipped, and applied in exactly one optimizer step. This handles a short final DataLoader batch without retaining gradients across batches. Epoch metrics use the corresponding numerators and observed counts.

Alternative: take one global mean across all provenance comparisons or edges. Rejected because tasks with more negatives/transitions would dominate and `provenance-candidate-loss-v2` would no longer be task-balanced.

Alternative: use a global provenance edge `pos_weight`. Rejected because it changes the declared task-local logical-edge supervision.

### 7. Batch dev model execution but keep task-local decoding

Evidence dev evaluation uses an ordered DataLoader and retains its existing task reconstruction and sample-weighted dev BCE.

Provenance dev evaluation tensorizes/collates multiple requests, performs one batched model forward, splits candidate/edge logits by offsets, and invokes structured candidate promotion, edge abstention, and metric construction independently for each task. Candidate ordering, transition conflict resolution, protected-prefix behavior, and trace metadata remain single-task operations.

Public `ExecutionProvenanceRgcnRetriever.rank_task` constructs a one-task batch and uses the same model output splitting path. This avoids separate single-graph and multi-graph model implementations while preserving its public request/result API.

Alternative: batch structured reranking itself into one global operation. Rejected because ranking pools, target conflicts, top-k, and abstention are task-local domain behavior.

### 8. Make the new training semantics self-describing and breaking

Provenance checkpoint schema advances from v3 to v4. The payload records at least:

```text
candidate_loss_protocol = provenance-candidate-loss-v2
candidate_loss_type = task_balanced_pairwise_logistic
batch_semantics = disconnected_union_task_graphs
per_device_graph_batch_size
```

Model parameter shapes do not change because batching adds no learned layer, but the training configuration and scientific identity do. A provenance v3 checkpoint is not loaded through the v4 strict loader by silently translating `batch_size`; previous artifacts remain diagnostic-only. The evidence checkpoint schema likewise advances from v2 to v3 so the removed `batch_size` field and new graph-batch field cannot share a schema identity with historical artifacts.

MLflow/training metrics report optimizer/global steps, actual task counts, and node, edge, candidate, transition, and supervised-item counts per batch or epoch summary. Large per-task tensors remain local artifacts and are not uploaded.

Alternative: keep either checkpoint schema unchanged because weights are shape-compatible. Rejected because schema identity covers training semantics, not only tensor shapes, and old `batch_size` has a materially different meaning.

## Risks / Trade-offs

- **[A fixed graph count produces variable memory use]** → Record node/edge/candidate/transition distributions and benchmark conservative provenance graph batches before changing full-profile values.
- **[The full provenance graph batch may not fit at evidence scale]** → Keep its true graph batch conservative and report the actual configured value rather than simulating a larger batch.
- **[Precomputed per-task tensors still consume CPU memory]** → Remove duplicate prebuilt batch lists, retain only one task-tensor copy, collate transiently, and defer persistent/on-demand caches to another change.
- **[DataLoader workers can copy large custom objects]** → Start with `num_workers=0`; do not expose worker tuning until platform-specific profiling and deterministic worker seeding tests exist.
- **[Dropout changes exact RNG draw order]** → Require deterministic structural/loss equivalence with dropout disabled, document that historical training is not bitwise reproducible, and compare formal results only within the new runtime.
- **[A wrong offset can leak messages or queries across tasks]** → Centralize graph collation and add index-range, no-cross-edge, single-vs-multi output, and deliberately corrupted-edge sensitivity tests.
- **[A generic shared trainer would erase loss differences]** → Share only data/graph runtime primitives; retain evidence and provenance training/loss/evaluation owners.
- **[Checkpoint v4 invalidates current v3 experiment artifacts]** → Complete this change before formal v2 runs, keep candidate-loss tag unchanged, and document old artifacts as diagnostic-only.
- **[DataLoader shuffle can complicate exact resume]** → Seed the loader generator and record iteration/optimizer counters; exact mid-epoch resume remains out of scope until the experiment runtime supports trainer resume.

## Migration Plan

1. Freeze single-task tensor/model/loss behavior and add failing multi-task isolation, task-local loss, partial-batch, deterministic-loader, and config/checkpoint tests.
2. Introduce shared `TaskGraphTensor`/`GraphBatch`, disconnected-union collation, and device transfer; migrate evidence contracts without changing its training loop yet.
3. Replace ambiguous trainer/profile fields with the explicit graph-batch setting, advance checkpoint validation, and update tracking parameter projection.
4. Split evidence feature materialization from collation, create evidence train/dev datasets and DataLoaders, and prove loss/prediction parity at graph batch size one and structural parity at larger sizes.
5. Introduce provenance task/batch/training tensor contracts and collation, including candidate/transition offsets, target alignment, and graph/query isolation validation.
6. Vectorize provenance model candidate/edge scoring and keep single-task inference on the batch-size-one path.
7. Replace provenance single-graph interval training with true multi-graph DataLoader execution, one optimizer step per batch, and exact task-denominator normalization while preserving `provenance-candidate-loss-v2`.
8. Batch provenance dev forwards, split outputs, and retain task-local structured inference and metric computation.
9. Run focused tests, ruff, basedpyright, broad pytest, and graph-batch microbenchmarks; freeze full-profile physical batch values before formal seed `13/17/29` training.
10. Roll back by reverting the code/config change and using v3 checkpoints with the old runtime. No v3-to-v4 semantic translation shim is added.

## Open Questions

None blocking. The initial full provenance graph batch `8` is a conservative migration value and may be increased using predeclared train/dev-only memory and throughput measurements before the formal v2 multi-seed matrix; any change must update the resolved config and checkpoint identity.
