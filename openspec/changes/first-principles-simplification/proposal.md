## Why

GraphMemory's scientific behavior is implemented behind repeated workflow, task, stage, payload, registry, factory, request, and result layers. These layers copy configuration and context without adding a second implementation, making experiments harder to trace and causing one scientific decision to have several authorities.

This refactor removes transport and extension machinery while preserving the current eight retrieval methods, both graph domains, experiment outputs, and scientific behavior. It establishes a smaller base before any later decision about removing research methods.

## What Changes

- **BREAKING** Remove internal retrieval registries, builder specifications, execution-task wrappers, result envelopes, and result-batch validation wrappers; retrieval uses one explicit static dispatch and concrete method requests.
- **BREAKING** Remove training payloads, trainer protocols/wrappers, single-implementation factories, sampler protocols, and other internal dependency carriers; stage materializers call concrete training functions directly.
- Collapse repeated method/stage/ranking/resolved configuration projections where they only copy fields, with one parsed configuration owner at the experiment boundary.
- Flatten experiment orchestration so scientific control flow is expressed once rather than repeated in workflow, Prefect task, and stage layers.
- Reduce prepared and result artifacts to authoritative scientific outputs while retaining source identity, digests, manifests, models, rankings, metrics, and per-task evaluation needed for reproducibility.
- Preserve all eight current methods: BM25, Dense, Dense-FT, GraphRAG, provenance path, evidence R-GCN, provenance R-GCN, and Dense-FT-seeded evidence R-GCN.
- Preserve `EvidenceGraph` and `ProvenanceGraph` as distinct domain contracts while retaining their shared tensor, batching, R-GCN, and optimization core.
- Replace tests that assert removed wrapper shapes with behavior tests at preparation, training, ranking, evaluation, and persisted-output boundaries.
- Introduce no replacement plan, representation, adapter, validator, registry, policy, factory, or compatibility layer.

## Capabilities

This is an internal structural refactor. It intentionally changes no product or scientific capability and therefore opts out of capability deltas in `.openspec.yaml`.

### New Capabilities

None.

### Modified Capabilities

None.

## Impact

- Primary production impact: `graph_memory/experiment`, `graph_memory/stages`, `graph_memory/registry`, `graph_memory/retrieval`, `graph_memory/models`, `graph_memory/training_pairs`, and `graph_memory/evaluation`.
- Tests that directly instantiate internal wrappers or freeze duplicate artifact projections will be removed or rewritten around observable behavior.
- Active documentation and configuration will describe the retained direct flow and its single parameter owners.
- Existing derived run artifacts remain historical read-only data; no compatibility adapters will be added for internal Python types removed by this change.
- No dependency, dataset algorithm, retrieval algorithm, graph construction algorithm, metric primitive, or model architecture is added or replaced.
