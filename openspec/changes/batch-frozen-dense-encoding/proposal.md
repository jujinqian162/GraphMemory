## Why

Frozen sentence-transformer consumers currently invoke the encoder separately for each task, leaving many GPU mini-batches partially filled and repeating forward passes for the same query and passage texts. This makes dense retrieval, trainable graph batch construction, dev evaluation, initial-score generation, and hard-dense sampling substantially slower than the available accelerator can support.

## What Changes

- Add a shared frozen dense-encoding capability that can flatten texts across multiple tasks, encode them in configured mini-batches, and restore task/node alignment deterministically.
- Add optional bulk ranking and seed-signal paths while preserving the existing single-task retrieval contracts as compatibility fallbacks.
- Make trainable graph batch construction encode all texts for one task-graph batch through the bulk provider path instead of invoking the provider once per task.
- Reuse already-computed normalized embeddings to derive dense seed scores when encoder identity, prefixes, and normalization semantics match.
- Reuse frozen train/dev dense features within one training invocation so dev evaluation does not re-encode unchanged texts every epoch.
- Batch dense initial-score generation and hard-dense negative ranking across task collections.
- Preserve ranking order, score semantics, tie-breaks, CLI behavior, artifact schemas, checkpoint schemas, and registry ownership boundaries.
- Keep persistent on-disk embedding caches, distributed encoding, BM25, and CrossEncoder batching out of scope.

## Capabilities

### New Capabilities

- `batched-dense-encoding`: Cross-task frozen sentence encoding, bulk dense ranking, deterministic result reconstruction, and single-task fallback behavior.
- `frozen-dense-feature-reuse`: In-memory reuse of frozen embeddings and compatible dense seed scores during graph-model training, evaluation, and batch construction.

### Modified Capabilities

None.

## Impact

- Affected code: shared embedding services and contracts, dense retrieval methods, trainable graph text-embedding contracts and batching, seed-signal providers, initial-score tuning, training-pair hard-negative generation, registry builders, and focused configuration conversion.
- Affected tests: dense ranking equivalence, encoder call grouping, variable-length task reconstruction, graph batch equivalence, seed-score reuse, dev lifecycle reuse, initial-score generation, hard-negative ordering, and architecture boundaries.
- Public single-task method names, CLI arguments, output artifacts, ranking formulas, and checkpoint formats remain compatible.
- No new runtime dependency is required.
