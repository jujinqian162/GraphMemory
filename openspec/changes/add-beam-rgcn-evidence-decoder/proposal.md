## Why

The current R-GCN retrievers encode graph context but still score every evidence node independently and sort the logits once. MuSiQue exposes a large gap between Full Support@5 and Full Support@10, so the existing model needs selection-conditioned decoding that can recover a complete two-to-four-hop evidence set within the top-five budget.

## What Changes

- Replace the existing R-GCN one-shot ranking objective with a shared beam evidence decoder for both `dense_rgcn_graph_retriever` and `dense_ft_rgcn_graph_retriever`; no new public method id is introduced.
- Encode each task graph once, then score each next evidence candidate from the question state, R-GCN node states, the already selected evidence set, the last selected node, and graph-frontier features.
- Train the decoder with dependency-aware dynamic-oracle supervision, beam states produced by the model, a learned `STOP` action, and an auxiliary node-ranking loss.
- Decode up to five unique evidence nodes, deduplicate hypotheses that represent the same selected set, and use the base R-GCN ranking to fill the remaining output positions after `STOP`.
- Preserve the existing request-authoritative graph input, `RankedResult` output, retrieved-subgraph construction, Dense-FT seed dependency, metrics, and public R-GCN method ids.
- Add beam configuration and diagnostics to the existing R-GCN configs, checkpoints, training metrics, prediction metadata, and run summaries.
- **BREAKING**: R-GCN checkpoints written before this change do not contain decoder parameters or beam configuration and must be retrained; the implementation will not add a legacy checkpoint fallback.

## Capabilities

### New Capabilities

- `beam-rgcn-evidence-decoding`: Selection-conditioned R-GCN training and inference with dependency-aware dynamic-oracle supervision, beam search, set deduplication, `STOP`, and complete ranked output.

### Modified Capabilities

- `graph-retriever-model-boundaries`: The existing R-GCN model boundary must expose graph-encoded node states to its decoder while keeping dataset projection, graph construction, training, and inference responsibilities separate.
- `current-trainable-method-config`: Existing R-GCN method configs gain strict typed beam-decoder and beam-training settings without adding a new method id.
- `current-trainable-artifacts`: Existing R-GCN checkpoints and runtime provenance gain required decoder state and beam configuration, making pre-change checkpoints incompatible.

## Impact

- Affected model code: R-GCN internal contracts, node encoding/scoring, beam decoder, training loop, dev evaluation, checkpoint loading, and inference.
- Affected configuration: the canonical configs for `dense_rgcn_graph_retriever` and `dense_ft_rgcn_graph_retriever`, plus typed experiment/stage config models.
- Affected workflow/runtime: the existing R-GCN train and retrieve stages continue to use their current method ids and artifact locations but write the new checkpoint contents and beam diagnostics.
- Affected dataset supervision: `EvidenceLabel.gold_evidence_item_ids` and `gold_dependency_edges` drive training only; no label-only data enters graph construction or inference.
- Affected verification: focused decoder/oracle/checkpoint tests, R-GCN training and retrieval tests, MuSiQue workflow coverage, type checking, and a fresh consuming workflow run.
