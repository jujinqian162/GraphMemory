## 1. Freeze Batch and Loss Contracts

- [x] 1.1 Add failing shared graph-collation tests for one-graph offsets, multi-graph cumulative offsets, query ownership, edge-range isolation, metadata preservation, and device transfer.
- [x] 1.2 Add failing evidence tests proving new task-level collation matches the current reference tensors/logits/loss at dropout zero and preserves sample-weighted BCE when tasks have unequal supervision counts.
- [x] 1.3 Add failing provenance tests proving multi-graph node states, candidate logits, and edge logits equal concatenated single-graph results at dropout zero.
- [x] 1.4 Add failing provenance locality tests proving candidate comparisons equal `sum_t |P_t||N_t|`, edge class weights remain task-local, unpaired candidates are excluded, and no candidate/query/transition index crosses a task interval.
- [x] 1.5 Add failing optimizer tests for short final DataLoader batches; prove each batch produces one optimizer step, evidence normalizes by actual supervised-sample count, provenance normalizes by actual task count, and both normalize before gradient clipping.
- [x] 1.6 Add failing DataLoader tests for seeded reproducible task order, deterministic epoch-to-epoch reshuffling, stable dev order, `drop_last=False`, and no encoder calls during iteration.
- [x] 1.7 Add failing config/checkpoint tests for the explicit graph-batch field, removal of ambiguous `batch_size`, evidence checkpoint schema v3, provenance checkpoint schema v4, legacy semantic rejection, and retained `provenance-candidate-loss-v2` identity.

## 2. Introduce Shared Disconnected-Union Infrastructure

- [x] 2.1 Create shared `TaskGraphTensor` and `GraphBatch` contracts outside evidence-specific internals with explicit task/query/node/offset invariants.
- [x] 2.2 Implement the shared disconnected-union collator that concatenates node fields, offsets edge/query indices, preserves task order, and rejects malformed or cross-task structures.
- [x] 2.3 Implement shared batch-to-device transfer without changing CPU metadata or task partitioning.
- [x] 2.4 Move evidence and provenance imports from `graph_retriever.internals.contracts.GraphBatch` to the shared owner and remove the obsolete duplicate contract.
- [x] 2.5 Add reusable validation/debug helpers for task ranges and graph-batch counts without putting model-specific supervision into the shared batch.

## 3. Make Training Configuration and Checkpoints Truthful

- [x] 3.1 Replace evidence and provenance trainer `batch_size` fields with `per_device_graph_batch_size` in Pydantic and model training configs.
- [x] 3.2 Split profile runtime values by evidence/provenance family and apply smoke `1`, quick `8`, full evidence `128`, and initial full provenance `8` defaults.
- [x] 3.3 Update stage trainer adapters, resolved configs, validation, planner/cache identity, and inspection output to consume the explicit fields.
- [x] 3.4 Advance the provenance checkpoint schema to v4 and record candidate-loss protocol v2, loss type, disconnected-union semantics, and graph batch.
- [x] 3.5 Advance the evidence checkpoint schema from v2 to v3 and update its config validation and experiment tracking projections with the explicit graph-batch field while preserving the existing model/loss identity.
- [x] 3.6 Make legacy evidence/provenance checkpoint and config errors name the old batch semantics explicitly rather than translating old `batch_size` values.

## 4. Migrate Evidence R-GCN to Task Dataset and DataLoader

- [x] 4.1 Split evidence tensorization into one-time CPU task-item materialization and dynamic evidence supervision collation, reusing existing frozen embedding and seed-signal providers.
- [x] 4.2 Implement a map-style evidence train dataset whose item contains one task graph and its validated training samples.
- [x] 4.3 Implement an ordered evidence dev dataset whose item contains one full-ranking task graph and labels needed for reconstruction.
- [x] 4.4 Build seeded training and ordered dev DataLoaders with custom evidence collators, `drop_last=False`, `num_workers=0`, and no encoder ownership in Dataset/worker code.
- [x] 4.5 Replace fixed prebuilt training-batch iteration with DataLoader graph batches while preserving global `pos_weight`, sample-level BCE, scheduler stepping, gradient clipping, global-step meaning, and checkpoint selection.
- [x] 4.6 Replace prebuilt evidence dev batches with ordered DataLoader execution and preserve prediction ordering, task reconstruction, sample-weighted dev loss, and evaluation metrics.
- [x] 4.7 Record evidence optimizer/global steps, configured and actual tasks per graph batch, supervised-sample denominator, and node/edge/task batch summaries.
- [x] 4.8 Remove obsolete list-slicing batch builders or retain only clearly named compatibility-free task materializers/collators with direct tests.

## 5. Add Provenance Multi-Graph Tensor and Model Contracts

- [x] 5.1 Refactor single-request provenance tensorization to produce a validated CPU task item with local graph, node-type, candidate, transition, ID, and trace metadata.
- [x] 5.2 Materialize aligned provenance training targets using `1/0/-1` candidate supervision and logical-transition edge targets while retaining pair-source diagnostics.
- [x] 5.3 Implement provenance graph/training collators that offset graph/candidate/transition indices and build candidate/transition/task offsets plus per-task metadata.
- [x] 5.4 Implement provenance batch-to-device transfer and output-splitting helpers for candidate logits, edge logits, IDs, and logical transitions.
- [x] 5.5 Vectorize `ExecutionProvenanceRGCN.forward` to gather per-candidate and per-transition queries explicitly and execute one disconnected-union graph forward.
- [x] 5.6 Make the batch-size-one path use the same provenance batch/model contract and remove the scalar-query-only model assumption.

## 6. Preserve V2 Loss While Enabling True Provenance Batches

- [x] 6.1 Refactor provenance loss to compute one task-local pairwise logistic candidate scalar per task from aligned targets, with no cross-task comparison.
- [x] 6.2 Compute logical-edge class-balanced BCE independently for each task, including differentiable-zero behavior for tasks without transitions.
- [x] 6.3 Reduce combined provenance losses as an equal-weight task sum/mean and preserve candidate/edge weights, comparison counts, and negative-category diagnostics.
- [x] 6.4 Implement a map-style provenance training dataset and seeded DataLoader over precomputed CPU task tensors.
- [x] 6.5 Replace sequential one-graph forward plus the overloaded optimizer interval with one true multi-graph forward and one optimizer step per DataLoader batch.
- [x] 6.6 Reduce each graph batch by its actual task count before clipping/step and eliminate the incomplete-tail scaling bug without retaining gradients across batches.
- [x] 6.7 Record optimizer/global step, task denominator, candidate comparison, transition, node, edge, and actual batch-size metrics without changing `provenance-candidate-loss-v2`.

## 7. Batch Provenance Dev Execution and Preserve Public Inference

- [x] 7.1 Implement an ordered provenance dev dataset/DataLoader that reuses materialized CPU task tensors and performs batched model forwards.
- [x] 7.2 Split batched provenance outputs by task before candidate sorting, structured promotion, target-conflict resolution, edge abstention, trace construction, and metrics.
- [x] 7.3 Refactor `ExecutionProvenanceRgcnRetriever.rank_task` to use the one-task batch and shared output-splitting path without changing its public API or result schema.
- [x] 7.4 Prove batched and single-task dev rankings, logical edges, traces, joint checkpoint metrics, and tie-breaking are equal at dropout zero.
- [x] 7.5 Preserve every provenance model ablation and `wo_edge_rerank` inference boundary under the new batch contract, including shuffled-feed isolation.

## 8. Document, Validate, and Benchmark the Migration

- [x] 8.1 Update `docs/10-plans/provenance-rgcn-loss-protocol.md` to link this OpenSpec change, retain candidate-loss protocol v2, and distinguish checkpoint schema v4 from loss/dataset versions.
- [x] 8.2 Update training/operations documentation and paper-facing setup notes to report the true per-device graph batch and short-tail count; defer scientific result claims until formal runs exist.
- [x] 8.3 Run focused graph-batching, evidence R-GCN, provenance R-GCN, checkpoint/config, Registry/planner, workflow, and experiment-tracking tests.
- [x] 8.4 Run `uv run ruff check`, `uv run basedpyright`, and the repository's broader required pytest gate in the effective host environment.
- [ ] 8.5 Benchmark evidence and provenance task/node/edge/candidate/transition distributions plus CPU tensor memory, peak VRAM, examples/tasks per second, and optimizer steps at graph batches `1/2/4/8/16` (and larger only when safe).
- [ ] 8.6 Freeze final full-profile physical batch values from train/dev-only runtime evidence, update resolved config/checkpoint identities, and run one seed-13 quick smoke proving no dropped tasks and successful save/load/retrieval.
- [x] 8.7 Confirm the pending formal provenance seeds `13/17/29` consume checkpoint schema v4 and `provenance-candidate-loss-v2`; do not compare old single-graph runs as a batching-only ablation.
- [x] 8.8 Run strict OpenSpec validation and `git diff --check`, then review that only intended source/docs/spec files are included and no Accelerate/Lightning/new graph dependency was introduced.

> Benchmark deferral: tasks 8.5-8.6 are intentionally reserved for the formal multi-GPU environment. The current host has one CUDA device, no transformed provenance dataset/checkpoint, and the formal transform is configured for eight workers/devices. Synthetic one-GPU CUDA smoke passed for both R-GCN families, but provenance graph batch `8` remains provisional and MUST NOT be presented as a benchmark-frozen value.
