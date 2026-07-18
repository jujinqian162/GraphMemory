## Why

The current non-training graphrag and execution_provenance_retriever methods can damage Dense head ranking because they apply global graph-score fusion or broad path bonuses. The requested change is limited to improving those two methods. Dataset cleanup, cache migration, training methods, and general evaluation infrastructure are separate concerns and must not be bundled into this implementation.

## What Changes

- Replace GraphRAG PPR/global score fusion with deterministic typed entity evidence, bounded sentence resolution, pair-local bridge proposals, protected-prefix stable insertion, and exact Dense fallback.
- Replace execution-provenance global path bonuses with bounded validity-gated path proposals, one best local partner per Dense seed, deterministic conflict resolution, protected-prefix stable insertion, and exact Dense fallback.
- Keep existing public method IDs and the existing non-training workflow.
- Add method-native traces and focused behavior tests sufficient to audit accepted/rejected proposals and final movement.
- Preserve the existing twowiki_provenance schema, converter, prepared artifacts, cache identity, fixtures, and all training-method contracts.
- Preserve shared evaluation metric/output schemas; broader counterfactual and statistical evaluation is out of scope.

## Capabilities

### New Capabilities

- local-execution-provenance-retrieval: bounded validity-gated path proposals, local promotion, deterministic conflicts, and exact Dense fallback over the existing request graph.
- typed-local-graphrag-retrieval: typed entity evidence, bounded sentence resolution, local bridge promotion, and exact Dense fallback.
- stateless-method-intervention-traces: method-local trace evidence for proposal gates, rejection reasons, conflicts, displacement, and exact fallback.

### Modified Capabilities

None.

## Impact

- GraphRAG request/config/index/method/builder code under graph_memory/retrieval/requests/, graph_memory/retrieval/methods/graphrag/, and graph_memory/registry/retrieval_builders.py.
- Execution-provenance config/search/method code under graph_memory/retrieval/methods/execution_provenance/.
- Method-specific retrieval trace contracts, strict config wiring, focused tests, and method documentation.
- No dataset schema, converter, parser, projector, manifest, prepared artifact, training pair, model, shared metric, or evaluation artifact changes.
