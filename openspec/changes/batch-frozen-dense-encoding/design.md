## Context

The repository has two sentence-transformer adapters:

- `DenseTaskRetriever` encodes one query and one passage list per task.
- `DenseTextEmbeddingProvider` encodes one graph task at a time.

`build_training_batches()` and `build_full_ranking_batches()` group task graphs, but `_build_batch()` still invokes the text provider and seed provider separately for every task. The default dense seed provider shares the same encoder object with the text provider, but sharing a model instance avoids only model reload; it does not avoid repeated forward passes.

The existing 500-task sample has a mean of about 41 passages per task while the encoder mini-batch default is 64 texts. Task-local calls therefore leave many mini-batches partially filled. For that sample, flattening graph task texts across tasks reduces the theoretical mini-batch count from 506 to 326 at encoder batch size 64. Dense ranking, which currently separates every single query from its passages, has a larger theoretical reduction from 1004 to 326.

The trainable encoder is frozen. Training batches are already constructed once before the epoch loop, but dev batches and their dense features are rebuilt every epoch. Dense seed signals for graph features are also computed from the same normalized query and passage embeddings that the graph model consumes.

The core refactor design already identifies a shared `DenseEncodingService` as a later low-level capability. This change implements that extraction while preserving the current registry, retrieval, model, script, artifact, and checkpoint boundaries.

## Goals / Non-Goals

**Goals:**

- Fill sentence-transformer mini-batches across multiple tasks instead of invoking one logical encode operation per task.
- Give flat dense retrieval and trainable graph feature construction one shared implementation of text formatting, encoder invocation, normalization, shape validation, and deterministic result slicing.
- Preserve single-task protocols and support injected test/custom providers that do not implement bulk capabilities.
- Let the default graph feature path derive dense seed scores from the same normalized embeddings used as graph node embeddings.
- Construct frozen dev graph batches once per training invocation and reuse them across epochs.
- Use bounded bulk ranking for initial-score precomputation and hard-dense negative generation.
- Preserve task order, node order, ranking order, scores, tie-breaks, tensor dtypes, graph offsets, sample order, CLI contracts, artifact schemas, and checkpoint schemas.
- Keep encoder text mini-batch size separate from task-graph training batch size.

**Non-Goals:**

- Add a persistent on-disk embedding or score artifact.
- Add an unbounded process-global text embedding cache.
- Change BM25, CrossEncoder, graph neural network, or graph-rerank scoring behavior.
- Batch normal `run_retrieval()` execution while it owns per-task wall-clock latency measurement.
- Add distributed encoding, multi-GPU scheduling, asynchronous request queues, or dynamic batching services.
- Guarantee a speedup equal to encoder batch size or define a hardware-independent timing threshold.

## Decisions

### Decision: Extract a low-level `graph_memory.embeddings` package

Create a public low-level package containing:

- The single sentence-encoder protocol currently duplicated by retrieval and model contracts.
- Typed task encoding requests/results that preserve task and node identity.
- `DenseEncodingService`, which owns model invocation, query/passage formatting, encoder mini-batch size, normalization, shape checks, flattening, and deterministic slicing.

Both `graph_memory.retrieval` and `graph_memory.models.graph_retriever` may depend on this package. The package must not import retrieval methods, graph-model internals, registry objects, scripts, or workflow state.

Alternative considered: place the service under `retrieval`. Rejected because trainable graph text features are an independent consumer and should not depend on a retrieval implementation package.

Alternative considered: place the service under `models.graph_retriever`. Rejected because flat dense retrieval must not depend on a specific trainable model domain.

### Decision: Bulk encoding accepts bounded task groups and returns aligned results

The bulk API accepts an ordered sequence of typed requests. Each request carries one `MemoryTaskInput` and the ordered node IDs required by the caller. The service:

1. Resolves each node ID to its fully prefixed query or passage text.
2. Flattens texts across the complete request group.
3. Calls `SentenceEncoder.encode()` once for the logical group with the configured text mini-batch size and normalized embeddings.
4. Validates the returned matrix shape.
5. Slices rows back into results with the original request, task, and node ordering.

Callers remain responsible for bounded task grouping. R-GCN uses its existing task-graph batch groups. Collection-oriented dense consumers use a fixed internal task chunk convention in this change rather than exposing another public config field.

Alternative considered: pass the entire dataset to one encode call. Rejected because the returned embedding matrix and flattened text list can become too large on full datasets.

Alternative considered: expose only `encode_texts()`. Rejected because every consumer would duplicate task/node flattening and reconstruction logic, which is the error-prone part of this change.

### Decision: Add optional bulk protocols with explicit single-task fallbacks

Keep existing `rank(task)`, `score_task(task)`, and `encode_task_nodes(task, node_ids)` contracts. Add separate bulk capability protocols rather than making every fake and custom provider implement new required methods.

Domain helpers dispatch to the bulk capability when available and otherwise call the existing single-task method in deterministic input order. This keeps custom injections behavior-compatible and makes bulk support observable in focused tests.

Alternative considered: add required bulk methods to all existing protocols. Rejected because it would create a breaking contract change for tests and external injected implementations.

Alternative considered: detect methods with untyped `getattr` calls throughout consumers. Rejected because capability detection and fallback behavior should be centralized and type-checkable.

### Decision: Use a joint default graph feature provider for one-pass embedding and seed scoring

Replace the default pair of separately constructed dense graph providers with one provider object that implements:

- The text-embedding provider capabilities.
- The dense seed-signal provider capabilities.
- A joint bulk capability that returns ordered node embeddings and seed signals derived from those embeddings.

The training and checkpoint registry paths pass this same object through both existing dependency slots. Graph batching uses the joint capability only when both dependencies identify the same compatible provider. Separately injected providers continue through the bulk-or-single fallback paths and retain their current semantics.

Dense scores are computed as normalized passage matrix times normalized query vector. Ranking, rank percentile calculation, and node-ID tie-breaks remain identical to the existing dense retriever and `RetrieverSeedSignalProvider`.

Alternative considered: add a service-wide embedding cache so separate providers happen to reuse calls. Rejected because cache lifetime, memory bounds, and invalidation would be implicit and difficult to test.

Alternative considered: add a third optional provider field to every training runtime request. Rejected because it expands the dependency surface and recreates optional-bag pressure.

### Decision: Reuse immutable dev batches across epochs

Training constructs train batches once as today. It also constructs full-ranking dev batches once before entering the epoch loop. Dev prediction consumes the prebuilt batches on every epoch while model logits are recomputed normally.

The reusable batches contain frozen CPU tensors and immutable task metadata. Device movement continues to create moved batch values per evaluation pass, so cached batches are not mutated.

Alternative considered: cache only raw text embeddings and rebuild graph batches every epoch. Rejected because graph tensorization and seed features are also frozen for one invocation and the complete batches already form the natural immutable boundary.

### Decision: Limit bulk ranking to collection-oriented consumers in this change

`DenseTaskRetriever` gains bulk ranking, and single-task `rank()` delegates through the same scoring implementation. Initial-score precomputation and hard-dense negative preparation use bounded bulk dispatch.

Normal `retrieval.execution.service.run_retrieval()` remains task-oriented. It currently measures wall-clock latency around each `rank_task()` call. A cross-task GPU operation has no exact per-task time attribution, so changing this path would silently change latency semantics even if ranked artifacts remained identical.

A later change may add explicit batched-run throughput artifacts or define an amortized latency contract. That decision is not required to obtain the training, tuning, and pair-generation improvements in this change.

### Decision: Validate behavior through equivalence and call-shape tests

Fake encoder tests require exact text order, prefixes, configured encoder mini-batch size, normalized flag, embedding rows, dense scores, ranking, and tie-break equality. Variable-length tasks verify flatten/slice reconstruction.

Real local encoder validation is a smoke check only: command completion, output schema, finite normalized embeddings, and ranking invariants. Hardware timing is reported when available but is not a pass/fail gate because the local environment and accelerator availability vary.

Architecture tests ensure the embeddings package remains low-level and that registry/workflow layers do not acquire encoding mechanics.

## Risks / Trade-offs

- [Batch-dependent floating-point differences on real GPU kernels] -> Require exact fake-encoder equivalence and ranking invariants for real encoder smoke tests; do not require bitwise identity across different physical mini-batch shapes.
- [Flattened task groups increase peak host memory] -> Keep grouping bounded by caller-owned task batches and avoid whole-dataset encode calls.
- [Joint-provider capability could accidentally bypass custom seed semantics] -> Activate it only when the embedding and seed dependencies are the same compatible object; otherwise use existing providers independently.
- [Reusing dev batches can retain substantial CPU memory] -> Reuse only within one training invocation and release with the invocation; do not introduce global or persistent caches.
- [A larger encoder mini-batch can cause GPU OOM] -> Preserve the existing encoder batch-size setting and surface the original encoder failure; task grouping does not override the encoder's internal mini-batch bound.
- [Performance work can expand into orchestration refactoring] -> Keep scripts and registry limited to constructing and injecting the shared service/provider; batching mechanics remain in embeddings and domain consumers.

## Migration Plan

1. Add failing equivalence, variable-length reconstruction, bulk fallback, call-count, joint reuse, and dev lifecycle tests.
2. Add the low-level embeddings contracts and `DenseEncodingService`; update duplicate encoder protocols to import the shared contract.
3. Adapt `DenseTaskRetriever` and graph text features to the shared service while preserving single-task behavior.
4. Add bulk provider/ranker dispatch and convert graph batching to one bulk operation per task-graph batch.
5. Construct the joint default graph feature provider in training and checkpoint registry paths, retaining separate-injection fallback behavior.
6. Prebuild immutable dev batches once per training invocation.
7. Batch initial-score and hard-dense negative collection paths in bounded task groups.
8. Update durable architecture/contracts/config docs and run focused tests, full tests, lint, type checks, and strict OpenSpec validation.

Rollback is a normal code revert. No data migration, artifact conversion, or checkpoint conversion is required.

## Open Questions

None. Persistent caches and batched public retrieval latency semantics are intentionally deferred to separate changes.
