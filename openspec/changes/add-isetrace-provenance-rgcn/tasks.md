## 1. Freeze the concise configuration and split contracts

- [x] 1.1 Add failing strict-config tests for `trajectory_source`, `natural_query_source`, normalized `queries.split_ratio`, and train/dev `queries.mix_ratio`; reject the removed test-only split shape, `allow_unreviewed`/`accepted_only`, `answer_only`/`support`/`intent_aware`, and other retired policy/kind/version fields for ISETrace.
- [x] 1.2 Remove user-facing ISETrace `source_revision` and infer the pinned revision from dataset registration, validating authoring run metadata/source identity when available.
- [x] 1.3 Add failing preparation tests that resolve valid natural queries before splitting, exclude/report malformed or unresolvable records, and require no hand-authored filtered corpus.
- [x] 1.4 Add failing deterministic split tests proving all queries for one trajectory stay together, split assignments depend on `split_seed` rather than training seed, and configured weights produce the expected target counts.
- [x] 1.5 Add failing mixture tests for all-natural retention, ratio-derived template counts, insufficient-template failure, train/dev trajectory ownership, and natural-only test.
- [x] 1.6 Add prepared query-origin metadata and summaries without adding origin to model-facing requests, candidates, graphs, or features.

## 2. Add minimal template supervision over existing motifs

- [x] 2.1 Add a small training-only template query/label contract referencing graph ID, query text, focused output IDs, and audit metadata outside `ProvenanceGraph`.
- [x] 2.2 Implement a deterministic renderer over existing dependency `MotifSpec`/`MotifAuthoringTarget` records using only safe tool/artifact/path descriptions; do not add another motif extractor.
- [x] 2.3 Map focused ToolOutputs through existing `execution.has_content` edges to positive output-content candidates; keep nonfocused participants nonpositive.
- [x] 2.4 Deterministically enumerate/select template records only from trajectories assigned to train or dev and satisfy configured mix counts without generating test templates.
- [x] 2.5 Add leakage, focus-versus-participant, graph-fingerprint, duplicate, deterministic-rendering, and insufficient-pool tests.

## 3. Adapt the existing R-GCN runtime to ProvenanceGraph

- [x] 3.1 Add `provenance_rgcn` to method/config/Registry contracts for the execution-provenance family without restoring the deleted provenance method ID or compatibility aliases.
- [x] 3.2 Add a provenance tensorizer that consumes `ProvenanceGraph` directly, appends one ephemeral disconnected query node, encodes provenance/query text with the existing frozen embedding provider, and emits shared task-graph tensors.
- [x] 3.3 Map the fixed enabled physical relations to forward/reverse relation IDs with uniform weights; exclude `temporal.precedes` and forbidden metadata features.
- [x] 3.4 Reuse the maintained `RGCNGraphEncoder`, node scorer, disconnected-union collator/DataLoader, batch-to-device path, and output splitting; do not add a second graph-convolution stack.
- [x] 3.5 Add single-task and multi-task tests for edge ranges, query ownership, candidate ownership, no cross-task messages, dropout-zero batch equivalence, and unchanged persisted graph fingerprints.

## 4. Reuse training pairs, training, and checkpoints

- [x] 4.1 Map natural exact spans and template focused-output content into candidate positives for train/dev provenance tasks.
- [x] 4.2 Adapt the existing easy-random, BM25-hard, Dense-hard, and graph-neighbor pair sampling to provenance candidates while keeping nonfocused motif participants eligible as negatives.
- [x] 4.3 Reuse the existing node-ranking loss, optimizer loop, graph batching, gradient clipping, metrics, and selection machinery; add no edge head or edge loss.
- [x] 4.4 Add a strict current `provenance_rgcn` checkpoint round trip using the maintained checkpoint owner and record the effective method, encoder, relation vocabulary, model/training config, and selected epoch.
- [x] 4.5 Add focused CPU smoke tests that train a tiny mixed natural/template fixture, select on natural dev, save/load the checkpoint, and reproduce retrieval output.

## 5. Integrate the trainable ISETrace workflow

- [x] 5.1 Extend prepared ISETrace train/dev/test artifacts to carry the mixed query pools, provenance graphs, candidate spans, labels, origins, and deterministic count summaries.
- [x] 5.2 Add the existing trainable lifecycle dependencies for `provenance_rgcn`: prepare, pair construction, frozen encoding, training/selection, checkpointed retrieval, and evaluation.
- [x] 5.3 Add Registry retrieval settings/payload/builder wiring that validates execution-provenance requests and loads the strict current checkpoint.
- [x] 5.4 Keep BM25, Dense, GraphRAG, and `provenance_path` on their current nontraining lifecycle; prove they evaluate only the derived natural test split, preserve their graph-access boundaries, and schedule no pair-building or training stage. Keep evidence R-GCN/Dense-FT behavior unchanged.
- [x] 5.5 Add smoke/quick/full Hydra coverage; ensure the full ISETrace trainable profile consumes the complete configured mixed splits rather than silently applying the evidence benchmark's fixed dev cap.
- [x] 5.6 Add planner, cache-identity, resume, artifact, Registry-family, unsupported-method, and end-to-end synthetic workflow tests.

## 6. Evaluate natural-query behavior without unsupported edge claims

- [x] 6.1 Compute natural and template dev metrics separately and use natural dev metrics for primary checkpoint selection.
- [x] 6.2 Run test retrieval only over natural queries and preserve exact candidate source spans through ranking and evaluation.
- [x] 6.3 Reuse existing ISETrace span Recall/Coverage, Full Support, span F1, MRR, evidence-density, and token-budget evaluation.
- [x] 6.4 Keep path/edge accuracy unavailable without independent labels; retrieved provenance paths remain diagnostic native traces only.
- [x] 6.5 Add a no-graph model variant through the existing zero-layer R-GCN mechanism for a direct template-training control, without adding a separate model implementation.

## 7. Documentation and verification

- [x] 7.1 Update maintained architecture, data/retrieval contracts, configuration examples, and RQ2 operations docs with the concise query config, trajectory-grouped split, mixed supervision, natural-only test boundary, and explicit retirement of the old ISETrace test-only review/target-policy contract.
- [x] 7.2 Document that generated natural queries remain unreviewed until separately reviewed and that formal paper claims require a frozen reviewed test corpus.
- [x] 7.3 Run focused ISETrace config/preparation/template, provenance graph, graph batching, pair sampling, model, checkpoint, Registry, workflow, and evaluation tests.
- [x] 7.4 Run `uv run ruff check`, `uv run basedpyright`, and the repository's broad pytest gate in the effective host environment.
- [x] 7.5 Run strict OpenSpec validation when the CLI is available, `git diff --check`, and a final review proving no new dependency, graph schema, duplicate R-GCN stack, query-conditioned persisted graph, or compatibility layer was introduced.
- [x] 7.6 Stop after verified engineering smoke runs; do not add paper result tables until reviewed natural data and formal seeds 13/17/29 are complete.
