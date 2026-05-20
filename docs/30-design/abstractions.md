# Abstractions

Date: 2026-05-20

Status: Working reference.

## Design Rule

Use abstraction only where behavior varies or where a boundary improves testing.

```text
Stable domain data     -> named aliases, TypedDicts, dataclasses
Replaceable behavior   -> small Protocol or class
Deterministic utility  -> plain function
```

## Data Representation

Use two layers:

```text
Disk artifacts:
  JSON/CSV-shaped records, validated strictly.

Internal algorithm state:
  small named dataclasses for configs, ranked nodes, and rerank results.
```

Recommended forms:

| Concept | Form |
|---|---|
| `TaskId`, `NodeId`, `MethodName`, `Score` | simple aliases |
| node types, edge types, method names | `Literal` first; `Enum` only if needed |
| parsed raw dataset examples | small frozen dataclasses |
| `MemoryTaskInput`, `MemoryTaskLabels`, `MemoryGraph` | `TypedDict` |
| `RankedNode`, `RerankResult`, score components | frozen dataclasses |
| configs | frozen dataclasses |
| metric rows and run summaries | JSON/CSV-shaped dicts with validation |

Do not mirror every JSON field with a class. Keep artifact records close to their serialized shape.

Raw JSON parsing should be explicit and dataset-specific. Prefer named functions such as `parse_hotpotqa_examples`
that turn untrusted JSON objects into small dataclasses before conversion. Do not introduce a generic
`JsonToDataClass` base class in Phase 1; it would add framework surface without replacing semantic artifact
validators.

Public core signatures should use project domain types. Avoid `list[dict]`, `tuple[list[dict], list[dict]]`,
or other unstructured containers in conversion, retrieval, graph, tuning, and evaluation interfaces when a
`TypedDict`, dataclass, alias, or protocol exists.

## Retriever

Purpose:

```text
MemoryTaskInput -> complete ranking over memory node IDs
```

Contract:

```python
class Retriever(Protocol):
    method_name: str

    def rank(self, task: MemoryTaskInput) -> list[RankedNode]:
        ...
```

Rules:

- Handles one task at a time.
- Returns every memory node exactly once.
- Does not read labels.
- Does not compute metrics.
- Does not write files.
- May keep explicit model/index state, such as a dense encoder.

## Reranker

Graph rerank is a separate reusable module.

```text
BM25 Retriever  -> initial ranking -> Graph Reranker -> final ranking
Dense Retriever -> initial ranking -> Graph Reranker -> final ranking
```

Rules:

- Does not run BM25 or dense retrieval itself.
- Does not read labels.
- Does not compute metrics.
- Uses only initial scores and graph structure.
- May return optional score components for debug artifacts.
- Does not own persistent score caching in Phase 1.

Core implementation can be a function:

```text
graph_rerank(initial_scores, graph, config) -> list[RankedNode]
```

A thin `GraphReranker` wrapper is acceptable if it makes config/debug handling clearer.

The explicit `initial_scores` argument is the only cache-friendly boundary needed for Phase 1. The first implementation may recompute initial rankings during dev tuning; a persisted score artifact can be introduced later if runtime becomes a blocker.

## Graph Construction

Keep graph construction function-based in Phase 1:

```text
build_graph(task_input, config) -> MemoryGraph
build_graphs(task_inputs, config) -> list[MemoryGraph]
graph_statistics(graphs) -> dict
```

Rules:

- Reads only input-visible task records.
- Does not read labels.
- Does not run retrieval.
- Does not compute evaluation metrics.

Introduce a `GraphBuilder` protocol only when Phase 2 adds genuinely different graph builders.

## Evaluation

Metric primitives should be pure functions:

```text
recall_at(...)
evidence_f1_at(...)
full_support_at(...)
mrr(...)
connected_evidence_at(...)
query_evidence_connectivity_at(...)
```

The aggregate evaluator owns task joins:

```text
evaluate_results(predictions, labels, graphs) -> list[EvaluationRow]
```

Rules:

- Reads labels and graphs.
- Never re-runs retrieval.
- Never reads gold fields from input task artifacts.
- Raises if predictions, labels, and graphs do not align.

## Validation

Validators are fail-fast functions:

```text
validate_memory_task_inputs(records) -> None
validate_memory_task_labels(records, inputs_by_task_id) -> None
validate_graphs(graphs, inputs_by_task_id) -> None
validate_ranked_results(predictions, inputs_by_task_id) -> None
```

Rules:

- Raise `ContractValidationError` for artifact contract violations.
- Do not clean, repair, drop, sort, or infer data.
- Transformation must be a separate named step.

## Experiment Services

Batch orchestration belongs in service functions beneath scripts:

```text
run_retrieval(...) -> list[RankedResult]
tune_graph_rerank(...) -> GraphRerankConfig
aggregate_tables(...) -> table artifacts
```

Service functions may loop over tasks, measure latency, and assemble artifacts. They should not read or write files directly.

Tuning services may call retrieval repeatedly in the first implementation. Prefer correctness and clear run summaries over introducing cache invalidation logic before the pipeline is stable.

## Anti-Abstractions

Avoid in Phase 1:

- dataset plugin registry
- method plugin discovery
- generic pipeline engine
- repository objects for local JSON files
- global config singleton
- large class hierarchies with lifecycle hooks
- object graphs that duplicate the JSON schema
