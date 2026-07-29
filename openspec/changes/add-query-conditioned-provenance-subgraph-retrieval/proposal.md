## Why

The current training-free EPGM uses dense-seeded local beam paths and independently reranks candidate nodes. Server-side seed-13 RQ2 analysis shows that globally multiplying recorded `feeds` weights lowers Path Recall@10 from 18.48% to 11.15% and gold-edge true positives from 505 to 310, while RQ3 structural questions remain limited by Dense candidate generation. EPGM needs a single non-trained architecture that interprets source-local dependency confidence correctly, conditions graph traversal on the query, and selects a coherent evidence subgraph rather than unrelated node-wise paths.

## What Changes

- Replace training-free EPGM's default typed-beam retrieval with a deterministic query-conditioned typed transition model.
- Use the frozen retriever encoder to score query-to-relation compatibility from schema-owned relation descriptions; no task labels or dataset-specific rules are used.
- Consume calibrated recorded edge confidence only through source-local outgoing transition normalization; constant weights are an identity signal.
- Run typed Personalized PageRank over the native provenance graph to make graph structure a candidate source rather than only a Dense top-seed reranker.
- Jointly select at most `top_k` retrievable evidence nodes with a deterministic budgeted Steiner-style connected-subgraph extractor; non-retrievable execution nodes may serve as connectors without consuming the evidence budget.
- Emit only selected, oriented provenance dependencies and retain a complete typed trace of transition inputs, diffusion convergence, selected connectors, objective contributions, and fallback state.
- Preserve the public method id `execution_provenance_retriever`, frozen E5 usage, full ranked-list result contract, registry/task-family boundaries, and Dense-identity fallback.
- Retire global min-max `w_eff` multiplication and keep legacy presets available only as explicit diagnostic ablations during migration.

## Capabilities

### New Capabilities
- `query-conditioned-provenance-subgraph-retrieval`: Deterministic typed transitions, PPR candidate expansion, and budgeted connected-subgraph extraction for one non-trained EPGM method across synthetic dependency graphs and real audit traces.

### Modified Capabilities
- `execution-provenance-retrieval`: Changes the default training-free provenance retrieval behavior from independent bounded paths to joint query-conditioned subgraph retrieval while preserving request and result boundaries.
- `method-native-retrieval-traces`: Extends the closed native trace contract with typed diffusion and connected-subgraph selection evidence.

## Impact

- Primary code: `graph_memory/retrieval/methods/epgm/`, retrieval contracts/result serialization/validation, experiment config and cache identity.
- Tests: EPGM domain behavior, source-local transition invariants, deterministic PPR convergence, connected selection, edge orientation, fallback identity, and native-trace round trips.
- Documentation/paper: replace the two-preset main-method description with one query-conditioned subgraph retriever and report legacy path modes only as diagnostics.
- No new trainable parameters, dataset-specific switches, label access, data migration, graph regeneration, or mandatory third-party solver dependency.
