# Retrieval Contracts

Status: Maintained project-level reference.

This document defines public retrieval method names, registry metadata, rank behavior, and seed retrieval signal contracts. Disk artifact shapes live in `data-contracts.md`; model tensors and checkpoints live in `model-contracts.md`.

## Scope

Retrieval code must provide:

- a complete ranking over memory nodes for each task.
- a stable public `method` name written into ranked result artifacts.
- explicit declaration of required inputs such as graphs, graph rerank configs, or trainable checkpoints.
- no direct file IO inside core retrieval implementations.

Retrieval code must not:

- read labels or train pairs during inference.
- compute evaluation metrics.
- silently fall back when required graphs, configs, or checkpoints are missing.

## Public Method Names

Current registry methods:

| Method | Kind | Required graph | Required config | Required checkpoint | Dense encoder args | Seed method |
|---|---|---:|---:|---:|---:|---|
| `bm25` | flat seed retrieval | no | no | no | no | none |
| `dense` | flat seed retrieval | no | no | no | yes | none |
| `bm25_graph_rerank` | graph rerank | yes | yes | no | no | `bm25` |
| `dense_graph_rerank` | graph rerank | yes | yes | no | yes | `dense` |
| `dense_rgcn_graph_retriever` | trainable graph retrieval | yes | no | yes | yes | `dense` |

Method names are artifact-level contract values. Renaming a method creates a new result namespace and must be treated as a compatibility change.

## Retrieval Method Registry

Scattered `method in {...}` checks are not allowed for public method dispatch. Use the static lightweight catalog in `graph_memory/retrieval/catalog.py`. `graph_memory/retrieval_registry.py` remains a thin workflow integration port that re-exports the catalog for scripts and workflow code. This is not dynamic plugin discovery.

```python
@dataclass(frozen=True)
class RetrievalMethodSpec:
    """
    Static metadata for one public retrieval method.
    一个公开检索方法的静态元数据。

    Fields / 字段:
    - name: Public method name written into ranked result artifacts.
      name：写入 ranked result artifact 的公开方法名。
    - requires_graphs: Whether this method requires `*_graphs.json`.
      requires_graphs：该方法是否需要 `*_graphs.json`。
    - requires_graph_config: Whether this method requires graph rerank config.
      requires_graph_config：该方法是否需要 graph rerank config。
    - requires_checkpoint: Whether this method requires a trainable model checkpoint.
      requires_checkpoint：该方法是否需要可训练模型 checkpoint。
    - requires_dense_encoder: Whether this method needs dense encoder runtime args.
      requires_dense_encoder：该方法是否需要 dense encoder 运行参数。
    - seed_method: Optional flat seed method used by this method, such as `dense`.
      seed_method：该方法使用的可选 flat seed method，例如 `dense`。
    - builder_id: Local runtime builder selected by `graph_memory.retrieval.factory`.
      builder_id：由 `graph_memory.retrieval.factory` 选择的本地运行时 builder。
    """

    name: MethodName
    requires_graphs: bool
    requires_graph_config: bool
    requires_checkpoint: bool
    requires_dense_encoder: bool
    seed_method: MethodName | None
    builder_id: str
```

Registry rules:

- Supported methods, validation method checks, experiment method filters, and CLI `choices` are derived from `METHOD_REGISTRY.keys()` or registry capability queries.
- Method capability queries such as graph-rerank methods and dense-encoder methods are derived from `RetrievalMethodSpec` fields, not string matching.
- All public methods, including trainable methods, are registered through the same registry.
- Runtime builders live in `graph_memory/retrieval/factory.py` and method-family packages under `graph_memory/retrieval/methods/`.
- Registry entries declare requirements; scripts validate missing inputs before invoking core retrieval.
- Adding a method requires adding one registry entry, tests for requirement validation, and an example command in operations docs when it becomes user-facing.

## Build Requests

Retrieval construction uses application run requests plus method-family build requests rather than one wide context object. CLI parsing may accept optional inputs, but those values must be grouped into typed runtime/config objects before application orchestration and resolver output must be a precise request for the selected method family.

Current request layers:

| Request | Meaning |
|---|---|
| `RunRetrievalRequest` | Application-level request for one complete retrieval run. |
| `RetrievalMethodResolveRequest` | Typed resolver input that turns a use-case request into one method-family build request. |

Current request families:

| Request | Meaning |
|---|---|
| `FlatMethodBuildRequest` | Flat BM25 or dense runtime state. |
| `GraphRerankMethodBuildRequest` | Seed retriever, `GraphIndex`, and `GraphRerankConfig`. |
| `TrainableGraphMethodBuildRequest` | `GraphIndex` plus checkpoint/text/seed runtime state. |

Scripts own paths and file IO. Builders receive already-loaded objects or runtime paths only when the object is intrinsically runtime state, such as a PyTorch checkpoint load target.

`graph_memory/retrieval/execution/service.py` is not a build boundary. It receives a built `RetrievalMethod`, task inputs, and `top_k`; it does not accept loose dense prefix fields, graph config, checkpoint path, providers, or device values.

## Retriever Protocol

Purpose:

```text
MemoryTaskInput -> complete ranking over memory node IDs
```

Contract:

```python
class Retriever(Protocol):
    """
    Flat retrieval behavior over one memory task.
    针对单个 memory task 的扁平检索行为。

    Methods / 方法:
    - rank: Return every memory node exactly once, sorted by descending score.
      rank：返回每个 memory node 且只返回一次，按 score 降序排列。
    """

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
- May keep explicit model or index state.

## RetrievalMethod Protocol

Purpose:

```text
MemoryTaskInput + optional graph context -> final ranked nodes and retrieved subgraph edges
```

Contract:

```python
class RetrievalMethod(Protocol):
    """
    Public retrieval method used by retrieval execution services.
    retrieval execution 服务使用的公开检索方法。

    Methods / 方法:
    - rank_task: Return final ranking and optional retrieved subgraph edges for one task.
      rank_task：为一个 task 返回最终 ranking 和可选 retrieved subgraph edges。
    """

    name: str

    def rank_task(
        self,
        task_input: MemoryTaskInput,
        *,
        top_k: int,
    ) -> tuple[list[RankedNode], list[GraphEdge]]:
        ...
```

Rules:

- Owns method-specific requirements through registry metadata.
- Returns every memory node exactly once in the ranked node list.
- Does not read labels, compute metrics, or write files.
- May be implemented by a score pipeline, graph traversal method, hierarchical method, or trainable graph retriever.

## Ranking Invariants

- Higher score means better rank.
- Ties must be deterministic; use ascending `node_id` unless a method documents a stronger tie-breaker.
- Complete rankings include every memory node exactly once.
- The question node `q` is never ranked as a candidate memory result.
- Scores must be finite numbers.

## Graph Rerank Boundary

Graph rerank consumes explicit initial scores and graph structure:

```text
graph_rerank(initial_scores, graph, config) -> list[RankedNode]
rank_graph_from_initial_scores(initial_scores, graph, config, top_k) -> RerankResult
```

Rules:

- Graph rerank does not run BM25 or dense retrieval itself.
- Graph rerank does not read labels.
- Graph rerank does not own persistent score caching.
- Seed retriever methods compute initial scores, then delegate graph score composition to rerank helpers.
- Score breakdowns may be returned for debug artifacts, but ranking behavior must not depend on debug mode.

## Seed Signal Contract

Seed signal is the frozen baseline retriever signal used before learned graph scoring. The first trainable graph retriever uses dense retrieval as the default seed.

The same seed signal provider must be used for:

- hard dense negative sampling.
- node numeric feature construction.
- trainable retrieval inference.

```python
@dataclass(frozen=True)
class SeedSignal:
    """
    Frozen seed retrieval signal for one memory node.
    一个 memory node 的冻结初始检索信号。

    Fields / 字段:
    - node_id: Memory node id receiving this seed signal.
      node_id：该 seed signal 对应的 memory node id。
    - score: Raw seed retriever score, dense cosine similarity for the default dense provider.
      score：seed retriever 原始分数；默认 dense provider 中为 dense cosine similarity。
    - rank: One-based rank after sorting by descending score and ascending node id tie-break.
      rank：从 1 开始的排名；按 score 降序、node id 升序打破平局。
    - rank_percentile: Rank percentile in [0, 1], where 0 means best and 1 means worst.
      rank_percentile：范围 [0, 1] 的排名百分位，0 表示最好，1 表示最差。
    """

    node_id: NodeId
    score: float
    rank: int
    rank_percentile: float
```

Rank percentile rule:

```text
rank_percentile = 0.0 if num_memory_nodes == 1
rank_percentile = (rank - 1) / (num_memory_nodes - 1) otherwise
```

```python
class SeedSignalProvider(Protocol):
    """
    Replaceable provider for frozen seed retrieval signals.
    可替换的冻结初始检索信号提供器。

    Methods / 方法:
    - score_task: Return one SeedSignal for every memory node in the task.
      score_task：为 task 中每个 memory node 返回一个 SeedSignal。
    """

    def score_task(self, task_input: MemoryTaskInput) -> list[SeedSignal]:
        ...
```

Provider rules:

- Returns one signal for every memory node.
- Does not include `q`.
- Does not read labels.
- Uses deterministic tie-breaking.
- May share implementation with a public flat retrieval method, but the signal object must make score and rank semantics explicit.

## Extension Rules

When adding a retrieval method:

1. Add or reuse a concrete `RetrievalMethod` implementation.
2. Add one `RetrievalMethodSpec` registry entry.
3. Add requirement validation tests for missing graphs, configs, or checkpoints.
4. Add ranked-result validation tests if the method returns subgraph edges.
5. Add command documentation only when the method is intended for normal user runs.

Avoid dynamic plugin discovery until there are external method providers that cannot reasonably be maintained in the local registry.
