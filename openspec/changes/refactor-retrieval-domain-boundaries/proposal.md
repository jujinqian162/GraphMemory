## Why

The accepted core package refactor plan identifies retrieval as the next boundary problem after the foundation, graph, and evaluation domains have been split. `graph_memory/retrieval.py` still mixes method-family resolution, dense runtime details, graph rerank orchestration, result assembly, and trainable checkpoint wiring, which makes new retrieval families likely to expand a universal context instead of adding focused composition objects.

## What Changes

- Split the retrieval domain into explicit contracts, method catalog access, typed build requests, resolver/factory, execution service, flat BM25/dense methods, seed signals, graph-rerank methods, and tuning modules.
- Delete `RetrievalBuildContext` and replace it with method-family build requests plus runtime objects.
- Keep public retrieval method names, script arguments, workflow commands, artifact schemas, ranking semantics, tuning objective, candidate ordering, and validation behavior unchanged.
- Keep graph-rerank scoring formulas, normalization, candidate expansion, debug records, and tuning search-space parsing behavior unchanged.
- Keep trainable graph retrieval behavior as a temporary factory branch until Change C reorganizes the trainable model domain.
- Retain `graph_memory/retrieval_registry.py` as the documented workflow integration port; do not change `scripts/workflow/`.
- Avoid adding a wide `graph_memory/retrieval.py` facade or any new catch-all context.

## Capabilities

### New Capabilities
- `retrieval-domain-boundaries`: Behavior-preserving retrieval construction, execution, flat method, and runtime-composition boundaries.
- `graph-rerank-tuning-boundaries`: Behavior-preserving graph-rerank method, scoring, debug, configuration, and tuning boundaries.

### Modified Capabilities

## Impact

- Affected production areas: `graph_memory/retrieval.py`, `graph_memory/rerank.py`, `graph_memory/rerank_config.py`, `graph_memory/tuning.py`, `graph_memory/indexes/`, retrieval-related imports in scripts, tests, and any moved retrieval type definitions still temporarily stored in `graph_memory/types.py`.
- Public CLI and workflow behavior are not intended to change.
- No new production dependency is introduced.
