## 1. Baseline and Contract Tests

- [x] 1.1 Add fake-encoder tests that freeze current dense text prefixes, normalization flag, scores, complete-node coverage, ranking order, and node-ID tie-breaks.
- [x] 1.2 Add variable-length bulk encoding tests that verify deterministic flatten order, task/node reconstruction, configured encoder mini-batch forwarding, and invalid output-shape failures.
- [x] 1.3 Add bulk-capability fallback tests for single-task text providers, seed rankers, and seed signal providers.
- [x] 1.4 Add graph batch equivalence tests covering embeddings, numeric features, edge tensors, offsets, query indices, sample metadata, dtypes, and task/node ordering.
- [x] 1.5 Add call-count tests proving the compatible default graph feature path performs one logical encoder call per task group rather than separate embedding and seed calls.
- [x] 1.6 Add multi-epoch training tests proving dev feature/tensor construction occurs once while model evaluation still occurs every epoch.
- [x] 1.7 Add initial-score and hard-dense negative tests that freeze score maps, latency-map shape, hard-pool behavior, de-duplication, selected node order, and pair artifact order.

## 2. Shared Dense Encoding Foundation

- [x] 2.1 Create `graph_memory.embeddings` with the shared sentence-encoder protocol and typed ordered task encoding request/result records.
- [x] 2.2 Implement `DenseEncodingService` with query/passage formatting, bounded-group flattening, one logical normalized encode call, matrix validation, embedding-dimension detection, and deterministic result slicing.
- [x] 2.3 Replace the duplicated retrieval/model sentence-encoder protocols with imports from the shared embeddings contract without adding broad compatibility facades.
- [x] 2.4 Add architecture tests proving the embeddings package does not import retrieval methods, graph-model internals, registry modules, stages, scripts, or workflow state.

## 3. Dense Retrieval Bulk Path

- [x] 3.1 Adapt `DenseTaskRetriever` to use `DenseEncodingService` for both single-task and bulk ranking.
- [x] 3.2 Add typed optional bulk ranker/provider capability protocols and centralized deterministic fallback helpers.
- [x] 3.3 Verify bulk dense ranking is behavior-equivalent to single-task ranking for fake encoders and preserves existing `rank(task_input)` callers.
- [x] 3.4 Keep normal `retrieval.execution.service.run_retrieval()` task-oriented and add a regression test that preserves its per-task latency measurement boundary.

## 4. Trainable Graph Feature Integration

- [x] 4.1 Replace the default graph text provider with a joint dense graph feature provider backed by `DenseEncodingService`.
- [x] 4.2 Implement joint bulk feature construction that returns ordered node embeddings and derives dense seed signals from the same normalized embeddings.
- [x] 4.3 Refactor graph batch construction to request embeddings/features once per task-graph batch while preserving separate-provider fallback behavior.
- [x] 4.4 Update training-script and checkpoint registry construction so the same compatible joint provider occupies the existing embedding and seed dependency slots.
- [x] 4.5 Verify independently injected embedding and seed providers retain their own semantics and are not replaced by dense-derived scores.

## 5. Frozen Training Lifecycle Reuse

- [x] 5.1 Refactor dev evaluation so it can consume prebuilt immutable full-ranking batches.
- [x] 5.2 Build dev batches once before the epoch loop and reuse them for every epoch without mutating the retained CPU batches.
- [x] 5.3 Verify train batches remain constructed once, model forward/backward behavior remains unchanged, and checkpoint selection metrics remain equivalent.
- [x] 5.4 Add invocation-lifecycle tests proving no process-global cache or new persistent embedding artifact is created.

## 6. Collection Consumer Integration

- [x] 6.1 Refactor dense initial-score precomputation to use bounded bulk ranking when supported and deterministic single-task fallback otherwise.
- [x] 6.2 Refactor hard-dense negative preparation to precompute bounded bulk rankings before per-task sampler selection.
- [x] 6.3 Preserve existing tuning cache schemas, train-pair schemas, summary statistics, and public builder/script contracts.
- [x] 6.4 Add call-count assertions demonstrating that bulk-capable collection consumers no longer invoke dense encoding separately for every task.

## 7. Configuration and Documentation

- [x] 7.1 Keep encoder text mini-batch size in dense-owned settings and verify it remains distinct from trainable task-graph batch size through config conversion and registry construction.
- [x] 7.2 Update architecture and model/retrieval contract documentation with the shared embeddings dependency direction, bulk capability fallback, and joint graph feature path.
- [x] 7.3 Update training and operations documentation to explain expected batching gains, GPU memory trade-offs, and the absence of a hardware-independent speedup guarantee.
- [x] 7.4 Document that persistent embedding caches and batched public retrieval latency attribution remain deferred.

## 8. Verification

- [x] 8.1 Run focused dense retrieval, graph batching/training, tuning, training-pair, registry, and architecture tests outside the Windows sandbox.
- [x] 8.2 Run the full pytest suite outside the Windows sandbox.
- [x] 8.3 Run `uv run ruff check .` and `uv run basedpyright --level error` outside the Windows sandbox.
- [x] 8.4 Run a local real-encoder smoke command when the configured model/runtime is available and verify finite embeddings, legal output schemas, and ranking invariants.
- [x] 8.5 Record before/after logical encoder call counts and optional wall-clock timings on the same task fixture without making timing a correctness gate.
- [x] 8.6 Run strict OpenSpec validation and confirm all change tasks and capability requirements are represented.
