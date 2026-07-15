## 1. Typed Beam Configuration and Contracts

- [x] 1.1 Add failing tests for required R-GCN decoder, beam-search, loss-weight, and optimizer-phase config fields in both existing method configs.
- [x] 1.2 Add failing tests for invalid beam size, invalid maximum steps, unknown beam fields, and mismatched training/inference beam sizes.
- [x] 1.3 Add typed beam decoder, beam search, beam loss, and optimizer-phase records to the model and composed experiment config contracts.
- [x] 1.4 Update `dense_rgcn_graph_retriever` and `dense_ft_rgcn_graph_retriever` canonical YAML with default beam size `2`, maximum steps `5`, set deduplication, length penalty, and loss weights.
- [x] 1.5 Verify registry and experiment method-selection tests still expose only the two existing R-GCN public method ids.

## 2. Encoded Graph State and Decoder Inputs

- [x] 2.1 Add failing model tests that require one graph encoding to be reused across multiple hypothesis expansions.
- [x] 2.2 Add an internal encoded-graph result containing R-GCN node states, question state/index, task offsets, node ids, scorer features, and base node logits.
- [x] 2.3 Refactor the existing evidence model so its current forward path delegates to the shared encode/base-score operations without changing graph tensorization order or dtypes.
- [x] 2.4 Add deterministic selected-set attention pooling and last-selected-node extraction for empty and non-empty hypotheses.
- [x] 2.5 Add deterministic model-visible graph-frontier feature construction with relation-wise selected-to-candidate summaries and no label inputs.
- [x] 2.6 Add import-boundary tests proving encoded graph tensors remain internal and inference does not import training modules.

## 3. Beam Decoder Core

- [x] 3.1 Add failing tests for first-hop scoring, subsequent-hop selected-set conditioning, stop scoring, and question-node exclusion.
- [x] 3.2 Implement first-hop, subsequent-hop, and stop heads over cached R-GCN states and configured decoder dimensions.
- [x] 3.3 Add failing tests for unique-node expansion, maximum five-step decoding, stopped-hypothesis retention, and deterministic tie-breaking.
- [x] 3.4 Implement immutable beam hypothesis state, candidate expansion, cumulative length-normalized scoring, and the learned `STOP` action.
- [x] 3.5 Add failing tests showing equivalent selected sets in different orders consume one beam slot and retain the higher-scoring sequence.
- [x] 3.6 Implement selected-set deduplication and bounded beam pruning for beam sizes `1`, `2`, and `4`.

## 4. Dependency-Aware Dynamic Oracle

- [x] 4.1 Add failing tests for chain dependencies, branching dependencies, multiple valid roots, and completion-triggered `STOP`.
- [x] 4.2 Add failing tests that future gold nodes are masked rather than negative and that labels without dependency edges fall back to all remaining gold nodes.
- [x] 4.3 Implement a dataset-neutral dynamic oracle over `EvidenceLabel.gold_evidence_item_ids` and `gold_dependency_edges`.
- [x] 4.4 Add validation for missing graph nodes, cyclic gold dependencies, duplicate dependency edges, and unreachable completion states with actionable task ids.
- [x] 4.5 Add leakage-boundary tests proving decomposition text, answers, supporting flags, and gold edges are absent from decoder inference inputs.

## 5. Beam-Aware R-GCN Training

- [x] 5.1 Add failing loss tests for multi-valid-action probability mass, stop BCE, auxiliary sampled node BCE, and configured weighted totals.
- [x] 5.2 Add failing training tests that retain model-produced distractor hypotheses and preserve one oracle-reachable hypothesis while recovery is possible.
- [x] 5.3 Extend R-GCN training batches with per-task evidence labels needed by the decoder while preserving existing sampled train pairs for auxiliary node loss.
- [x] 5.4 Implement vectorized beam expansion and dynamic-oracle loss accumulation over cached task graph states.
- [x] 5.5 Implement decoder-only warmup and joint R-GCN/decoder optimizer phases with separate typed learning rates while keeping text embeddings frozen.
- [x] 5.6 Switch both existing R-GCN methods to beam-aware training and remove the independent one-shot objective as the primary training path.
- [x] 5.7 Update dev prediction and checkpoint selection to use beam-decoded rankings with Full Support@5-centered selection.

## 6. Strict Checkpoint and Inference Migration

- [x] 6.1 Add failing checkpoint round-trip tests for decoder config, decoder weights, beam loss settings, optimizer phases, optimizer state, and scheduler state.
- [x] 6.2 Add a failing test that a pre-change R-GCN checkpoint is rejected with an explicit retraining error and no compatibility fallback.
- [x] 6.3 Extend strict R-GCN checkpoint serialization and the shared model factory for the required beam decoder state.
- [x] 6.4 Add failing inference tests for selected-prefix ordering, early-STOP base-score filling, complete unique rankings, and deterministic residual ordering.
- [x] 6.5 Replace existing R-GCN one-shot inference with beam decoding while preserving request-authoritative graph use and induced retrieved-subgraph construction.
- [x] 6.6 Verify both base-dense and Dense-FT-seeded checkpoint loaders restore their recorded encoder providers and identical beam semantics.

## 7. Observability, Provenance, and Documentation

- [x] 7.1 Add failing metric-record tests for next-action loss, stop loss, auxiliary node loss, retained hypotheses, oracle-reachable rate, premature-stop rate, average selected length, beam size, and maximum steps.
- [x] 7.2 Record effective decoder, beam, loss, and optimizer-phase settings in R-GCN train checkpoints and train/retrieve run summaries.
- [x] 7.3 Add winning selected sequence, selected count, stop state, stop score, beam score, beam size, and maximum steps to prediction metadata.
- [x] 7.4 Add dev/debug diagnostics for any-beam gold coverage, final-beam gold coverage, premature stopping, and failure breakdown by required evidence count.
- [x] 7.5 Update R-GCN method configuration and operations documentation to explain beam semantics, maximum-five recovery budget, checkpoint incompatibility, and beam-size comparison commands.

## 8. Verification and Consuming Workflows

- [x] 8.1 Run focused dynamic-oracle, decoder, training-loss, checkpoint, and inference tests.
- [x] 8.2 Run the complete existing R-GCN tensorization, model, pairs, training, retrieval, configuration, planning, and tracking test set.
- [x] 8.3 Run formatting/lint checks, `basedpyright --level error`, and strict OpenSpec validation for this change.
- [x] 8.4 Run fresh smoke workflows for both existing R-GCN methods and confirm predictions, metrics, metadata, checkpoints, and run summaries are produced.
- [x] 8.5 Run fresh quick consuming workflows on MuSiQue and at least one non-MuSiQue evidence dataset to verify dependency-aware and unordered-label training paths.
- [x] 8.6 Document the server-side formal same-split MuSiQue comparison for beam sizes `1`, `2`, and `4`, using Full Support@5 as primary and Full Support@10, path metrics, latency, and multi-seed stability as guardrails; per delivery direction, do not block implementation handoff on this long-running experiment.
