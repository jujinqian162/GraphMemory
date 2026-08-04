## Why

The revised RQ2 ISETrace benchmark now has deterministic trajectory-count-based train/dev/test selection, shared lossless flat chunks, and exact-SourceSpan evaluation, but the existing `dense_ft` lifecycle cannot consume those contracts. A supervised flat-text baseline is required now so the graph-based `provenance_rgcn` result can be compared against encoder fine-tuning under the same trajectory split, candidate representation, supervision policy, and natural-only test set.

## What Changes

- Extend the existing public `dense_ft` method to the current ISETrace execution-provenance family without introducing an ISETrace-specific method ID.
- Project ISETrace natural and template exact spans onto the shared flat trajectory chunks and materialize every overlapping chunk as a positive candidate.
- Make training-pair construction select a method-native ISETrace view: flat chunks for `dense_ft` and provenance content units for `provenance_rgcn`.
- Keep ISETrace Dense-FT graph-free. It receives no `ProvenanceGraph`, EvidenceGraph, motif metadata, dependency edge, graph ID, node ID, query origin, or graph-neighbor negative.
- Use one family-aware effective Dense-FT pair configuration that forces `hard_graph_neighbor_per_positive=0` for ISETrace while preserving existing evidence-dataset behavior.
- Replace global-corpus Dense-FT dev selection on ISETrace with task-local candidate ranking that matches final retrieval semantics. Mixed dev selects by natural Recall@5; template-only dev selects by template Recall@5.
- Prevent in-batch false negatives by keeping examples from the same trajectory out of the same Dense-FT batch while retaining the current SentenceTransformers 2.7 fit and MultipleNegativesRankingLoss path.
- Reuse the existing SentenceTransformer model-directory artifact, metadata, Dense retriever, flat ranking output, and exact-span evaluation suite.
- Add ISETrace Dense-FT workflow, configuration, tracking, synthetic CPU smoke, and operations documentation for natural-only and size-matched mixed-supervision experiments.

## Capabilities

### New Capabilities

- `isetrace-dense-ft-retrieval`: Defines exact-span-to-flat supervision, task-local Dense-FT training/selection, trajectory-safe batching, checkpointed flat retrieval, and natural-only exact-span evaluation on ISETrace.

### Modified Capabilities

- `retrieval-workflow-matrix`: Adds `dense_ft` to the current ISETrace trainable lifecycle while preserving flat/provenance graph-access boundaries and evidence-dataset Dense-FT behavior.

## Impact

Affected surfaces include ISETrace training adapters and neutral span-overlap utilities; pair-stage method dispatch and effective sampling configuration; Dense-FT data, batching, dev evaluation, metric records, and model metadata; Registry task-family compatibility; Prefect workflow planning and artifact identity; Hydra profiles and method composition; focused config/pair/trainer/retrieval/evaluation tests; and the ISETrace operations runbook. The change adds no dependency, graph schema, model ID, alternate dense scorer, second trainer backend, or checkpoint compatibility layer.
