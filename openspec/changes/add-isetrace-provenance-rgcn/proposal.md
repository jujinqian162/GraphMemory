## Why

RQ2 now has a revision-pinned ISETrace trajectory corpus, a query-independent provenance graph, a reusable natural-query authoring corpus, and a training-free `provenance_path` method. The remaining question is whether the existing relation-aware R-GCN machinery can learn from inexpensive template supervision and improve retrieval on natural queries.

The implementation should reuse the current provenance graph and the existing RQ1 R-GCN encoder, batching, negative sampling, training, and checkpoint infrastructure. It must not restore the deleted label-conditioned provenance stack or introduce a second graph-learning framework.

Natural LLM-authored queries are no longer a test-only file. Configuration must support deterministic train/dev/test allocation of that corpus and simple natural/template mixing in train and dev, while formal test remains natural-only.

## What Changes

- Replace the ISETrace test-only query configuration with a small corpus configuration containing only the trajectory source, natural-query source, natural split ratio, and train/dev natural-template mix ratios.
- Infer and validate the pinned ISETrace revision from repository data registration and authoring metadata instead of exposing it as a routine experiment field.
- Resolve valid natural queries to trajectories before splitting, keep every trajectory in exactly one split, and use the existing `split_seed` independently of model training seeds.
- Generate train/dev template queries from the existing query-independent provenance motifs. Template labels stay outside the graph and identify only focused output content; participant nodes are not automatically positive.
- Add a new `provenance_rgcn` method for execution-provenance retrieval. It reuses the existing frozen text encoder, R-GCN layers, graph batching, node scorer, pair sampling, training loop, checkpoint handling, and evaluation pipeline.
- Tensorize the existing `ProvenanceGraph` directly. Query text is represented as an ephemeral disconnected query node for scoring and is never persisted into or used to rebuild the provenance graph.
- Train on the configured natural/template mixture, select checkpoints primarily on natural dev queries, and evaluate the formal test split only on natural queries with the existing exact-span metrics.
- Keep the first implementation node-ranking only. It does not add a dependency-edge prediction head or claim path/edge accuracy without independent edge labels.

## Capabilities

### New Capabilities

- `isetrace-query-mixture`: deterministic trajectory-grouped natural-query splitting and simple natural/template composition for train, dev, and natural-only test.
- `provenance-rgcn-retrieval`: frozen-encoder R-GCN node ranking over the existing ISETrace provenance graph, integrated with the current train/retrieve/evaluate workflow.

### Modified Capabilities

- `isetrace-nontrain-benchmark`: replace the prior test-only split, config-level review/target policies, and ToolOutput-level candidate contract with one revision-pinned natural-query corpus, trajectory-grouped derived splits, exact-span provenance content candidates, and an operationally frozen reviewed natural test source. Existing non-training methods continue to execute only the derived natural test split and schedule no training lifecycle.
- `retrieval-workflow-matrix`: add `provenance_rgcn` as a trainable execution-provenance method that consumes the query-independent physical provenance graph, while BM25, Dense, GraphRAG, and `provenance_path` retain their non-training execution paths and graph-access boundaries.

Existing evidence R-GCN methods retain their current behavior. The old ISETrace benchmark contracts are replaced explicitly rather than kept as compatibility inputs.

## Impact

- ISETrace dataset configuration and strict Pydantic experiment configuration.
- ISETrace preparation artifacts, query-origin metadata, template-query materialization, and split summaries.
- A provenance-specific tensorization adapter around the existing graph-retriever runtime, plus Registry, workflow, training, checkpoint, and retrieval wiring for `provenance_rgcn`.
- Focused tests for configuration readability, deterministic grouped splitting, mixture counts, graph/query independence, natural-span labels, template labels, R-GCN training, and natural-only test evaluation.
- Maintained architecture, contracts, and RQ2 operations documentation.
- No new dependency, graph schema, generic training framework, Dense-FT stage, edge decoder, compatibility alias, or paper result is included.
