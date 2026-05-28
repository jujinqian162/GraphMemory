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

Concrete types that define stable contracts must follow the bilingual triple-quoted docstring rule in
`docs/20-contracts/README.md`. Field meaning belongs in the type docstring and contract document, not only in
inline comments.

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

## RetrievalMethod

Purpose:

```text
MemoryTaskInput + optional graph context -> final ranked nodes and retrieved subgraph edges
```

Contract:

```python
class RetrievalMethod(Protocol):
    name: str

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        ...
```

Rules:

- Is the top-level internal boundary for public baseline names such as `bm25`, `dense`, `bm25_graph_rerank`, and future methods.
- Owns method-specific requirements such as whether graphs and graph configs are required.
- Returns every memory node exactly once in the ranked node list.
- Does not read labels, compute metrics, or write files.
- May be implemented by a score pipeline, graph traversal method, hierarchical memory method, or trainable graph retriever.

This is the stable abstraction for future baseline growth. Do not make weighted scoring the only top-level model; some later baselines may retrieve communities, paths, buffers, or learned graph neighborhoods before producing compatible ranked nodes.

## ScorePipelineMethod

Purpose:

```text
Retriever.rank(task_input) -> complete flat ranking
```

Use this implementation for public flat baselines whose final ranking is exactly the seed retriever output:

```text
bm25 = BM25Score
dense = DenseScore
```

Rules:

- Baseline BM25/dense scores remain raw for flat methods.
- Does not own graph candidate expansion or graph component combination.
- Returns no retrieved edges for flat methods.

## RetrievalMethodSpec Registry

Public method dispatch uses the static registry in `graph_memory/retrieval_registry.py` because methods have different required inputs.

Purpose:

```text
method name -> method metadata + runtime builder id
```

Rules:

- Registry keys define supported public method names and are the single source for validator and CLI method choices.
- Registry metadata declares whether graphs, graph rerank config, dense encoder args, or checkpoint are required.
- `experiment.py`, tuning, and scripts use registry capability queries instead of copied method tuples or string matching such as `"dense" in method`.
- Runtime builders live in `graph_memory/retrieval.py` and are selected by the registry entry's local `builder_id`.
- Builders receive explicit build context objects, not raw CLI args.
- Heavy learned imports may be lazy inside the trainable method builder.
- This is a local dispatch table, not dynamic plugin discovery.

The field contract lives in `docs/20-contracts/retrieval-contracts.md`.

## NodeScoreComponent

Purpose:

```text
ScoreContext -> {node_id: component_score}
```

Rules:

- Lives in `graph_memory/rerank.py` with the graph scoring helpers.
- Computes one interpretable signal such as initial retrieval score, query-overlap score, neighbor propagation, or bridge score.
- Does not sort final rankings.
- Does not validate artifacts or read labels.
- Should be small enough to test with tiny task and graph fixtures.

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
rank_graph_from_initial_scores(initial_scores, graph, config, top_k) -> RerankResult
```

Graph-rerank methods in `retrieval.py` select the BM25/Dense seed retriever, compute explicit initial scores, and delegate candidate expansion, component normalization, weighted combination, and top-k induced subgraph extraction to `graph_memory/rerank.py`.

The explicit `initial_scores` argument is the only cache-friendly boundary needed for Phase 1. The first implementation may recompute initial rankings during dev tuning; a persisted score artifact can be introduced later if runtime becomes a blocker.

`graph_rerank(...)` and `graph_rerank_with_breakdown(...)` remain compatibility helpers for direct tests, debug analysis, and callers that already have initial scores.

## SeedSignalProvider

Trainable graph retrieval needs a shared abstraction for frozen baseline retrieval signals.

Purpose:

```text
MemoryTaskInput -> one seed score/rank signal per memory node
```

Rules:

- Used by hard negative sampling, node feature construction, and trainable retrieval inference.
- Returns every memory node exactly once and never returns `q`.
- Does not read labels.
- Uses deterministic tie-breaking.
- Keeps score, rank, and rank-percentile semantics explicit.

The field contract lives in `docs/20-contracts/retrieval-contracts.md`.

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
- Validators accept `object` at public boundaries, check the JSON/list/map shape at runtime, and only then narrow to
  internal validation record types. Call sites should pass loaded JSON artifacts or domain-typed artifacts directly;
  do not copy records through `dict(...)` or `dataclasses.asdict(...)` just to satisfy a type checker.

## Experiment Services

Batch orchestration belongs in service functions beneath scripts:

```text
run_retrieval(...) -> list[RankedResult]
tune_graph_rerank(...) -> GraphRerankConfig
aggregate_tables(...) -> table artifacts
```

Service functions may loop over tasks, measure latency, and assemble artifacts. They should not read or write files directly.

Tuning services may call retrieval repeatedly in the first implementation. Prefer correctness and clear run summaries over introducing cache invalidation logic before the pipeline is stable.

## Trainable Graph Components

Trainable graph retrieval should keep replaceable behavior small:

| Concept | Form | Reason |
|---|---|---|
| `GraphBatch`, `TrainingBatch` | frozen dataclasses | Prevent raw dicts from leaking into model code. |
| `GraphEncoder` | small protocol or `nn.Module` boundary | Allows identity, R-GCN, and future GAT encoders. |
| `MessageTransform` | concrete strategy class | Keeps edge-type ablation out of the training loop. |
| `EdgeWeightPolicy` | small function or class | Keeps edge-weight ablation in tensorization. |
| `NodeFeatureBuilder` | concrete class with config | Keeps feature-order changes explicit and checkpointable. |

Rules:

- `EvidenceScoringModel.forward` should only express tensor data flow.
- Ablations should be handled by construction-time component replacement or tensorization filtering.
- Training loops should not know R-GCN relation math.
- Model internals should not receive raw JSON artifact records.

Tensor and checkpoint contracts live in `docs/20-contracts/model-contracts.md`.

## Anti-Abstractions

Avoid in Phase 1:

- dataset plugin registry
- method plugin discovery
- generic pipeline engine
- repository objects for local JSON files
- global config singleton
- large class hierarchies with lifecycle hooks
- object graphs that duplicate the JSON schema
