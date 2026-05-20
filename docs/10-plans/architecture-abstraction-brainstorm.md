# Architecture And Abstraction Brainstorm

Date: 2026-05-20

Status: Exploration record. Stable decisions from this discussion have been promoted into `docs/30-design/architecture.md` and `docs/30-design/abstractions.md`.

## Context

Phase 1 builds an evidence-tracing retrieval pipeline:

```text
raw HotpotQA
  -> memory task inputs + labels
  -> typed graphs
  -> flat retrieval scores
  -> graph reranking
  -> ranked results
  -> evaluation tables
```

The architecture must support correctness, readability, testability, observability, and later extension to Phase 2/3 methods without making Phase 1 unnecessarily abstract.

## Design Pressure

The central tension:

```text
Too concrete:
  easy to build Phase 1, but later baselines and diagnostics become messy.

Too abstract:
  elegant-looking interfaces, but Phase 1 becomes slow to implement and hard to read.

Target:
  small domain abstractions around stable concepts, concrete implementations around experimental methods.
```

## Candidate Architecture Styles

### Option A: Script-First Pipeline

Each script owns most logic directly. Shared helpers exist only when duplication appears.

Pros:

- Fastest to write.
- Very direct for one-off experiments.
- Easy for beginners to follow command by command.

Cons:

- Harder to unit test core behavior without invoking scripts.
- Harder to add GraphRAG, Dense-FT, or new datasets cleanly.
- Observability becomes scattered.
- Contracts become comments rather than enforceable boundaries.

### Option B: Library-Core With Thin CLI

Core package owns domain logic. Scripts only parse arguments, load config, call library functions, validate input/output, and write artifacts.

Pros:

- Best balance for this project.
- Easy to test converters, graph builders, retrievers, rerankers, and evaluators directly.
- Keeps CLI stable while implementations evolve.
- Makes observability and validation reusable.
- Supports later baselines without turning scripts into large files.

Cons:

- Requires discipline to keep interfaces small.
- Slightly more upfront design work.

### Option C: Framework/Plugin Architecture

Everything is registered through generic interfaces, factories, registries, and plugin discovery.

Pros:

- Maximum extensibility.
- Clean if many teams add methods and datasets independently.

Cons:

- Too much machinery for Phase 1.
- Risk of abstracting before the real variation is known.
- Makes debugging and onboarding harder.

## Current Recommendation

Use Option B: library-core with thin CLI.

Architecture slogan:

```text
Artifacts are the external contract.
Domain objects are the internal language.
CLIs are adapters, not the system.
```

## External Structure Constraint

The original student experiment plan fixes the external experiment structure more strongly than it fixes internal Python package layout:

- `data/{dataset}/raw/`
- `data/{dataset}/processed/`
- `results/`
- script entry points such as `scripts/prepare_hotpotqa.py`, `scripts/build_graphs.py`, `scripts/run_retrieval.py`, `scripts/evaluate_retrieval.py`, and `scripts/aggregate_tables.py`
- artifact names such as `*_memory_tasks.json`, `*_graphs.json`, `ranked_results_{method}.json`, and final result CSVs

The Phase 1 plan also proposes concrete root-level modules such as `graph_memory/hotpotqa.py`, `graph_memory/graphs.py`, `graph_memory/retrieval.py`, `graph_memory/rerank.py`, and `graph_memory/evaluation.py`.

Decision direction:

- Respect the required external structure.
- Do not introduce a broad subpackage hierarchy at the start.
- Keep abstractions in the code through interfaces, dataclasses, protocols, validators, and small services.
- Extract subpackages later only when a module accumulates multiple independent implementations.

## Proposed Layers

Conceptual layers still matter, but they do not need to become directories immediately.

```text
scripts/
  thin command adapters
  parse CLI/config
  call graph_memory services
  write artifacts and run summaries

graph_memory/
  contracts/
    validation and typed records near JSON boundaries
  datasets/
    raw dataset conversion and split sampling
  text/
    tokenization, lexical scoring, entity extraction
  graphs/
    graph construction and graph statistics
  retrieval/
    retriever implementations and dispatch
  rerank/
    graph-aware reranking and score components
  evaluation/
    metrics, aggregation, failure cases
  observability/
    run summaries, debug dumps, score breakdowns
  io/
    JSON/CSV read-write helpers and config loading
```

The Phase 1 plan currently lists flatter modules such as `graph_memory/hotpotqa.py`, `graph_memory/graphs.py`, and `graph_memory/retrieval.py`. The layered idea should be expressed through dependency direction and responsibilities, not directory depth.

Recommended Phase 1 shape:

```text
graph_memory/
  types.py
  validation.py
  io.py
  hotpotqa.py
  splits.py
  text.py
  entities.py
  graphs.py
  indexes/
    bm25.py
    dense.py
  retrieval.py
  rerank.py
  tuning.py
  evaluation.py
  observability.py
```

`indexes/` may remain as a small exception because the Phase 1 plan already names it and BM25/Dense are naturally parallel implementations. Other subpackages can wait.

Dependency direction:

```text
scripts
  -> io / validation / observability
  -> domain modules

retrieval
  -> indexes
  -> rerank when graph rerank is requested

evaluation
  -> validation
  -> graph connectivity helpers

graphs
  -> text / entities

hotpotqa
  -> no retrieval, no graph, no evaluation
```

Avoid reverse dependencies:

- `hotpotqa.py` should not import `graphs.py`, `retrieval.py`, or `evaluation.py`.
- `graphs.py` should not import labels, retrieval, tuning, or evaluation logic.
- retrievers should not import evaluation metrics.
- evaluation should not import raw dataset conversion.

## Proposed Core Abstractions

### Artifact Boundary

Artifacts remain JSON/CSV dictionaries on disk. They are the stable cross-module and cross-run contract.

Core question:

```text
Should internal code convert JSON dictionaries into dataclasses immediately, or operate on validated dicts?
```

Initial leaning:

- Use validated dictionaries at I/O boundaries.
- Use small dataclasses for internal algorithm inputs, configs, and results when they improve readability.
- Do not create large object graphs that mirror every JSON field.

### Dataset Converter

Responsibility:

- Convert raw dataset examples into input and label records.
- Preserve stable IDs.
- Validate mapping from labels to memory nodes.

Should not:

- Build graph edges.
- Run retrieval.
- Compute metrics.

### Graph Builder

Responsibility:

- Convert task input records into typed graph records.
- Use only input-visible fields.
- Optionally emit graph statistics/debug information.

Should not:

- Use gold labels.
- Know about evaluation metrics.
- Know which retrieval method will consume the graph.

### Retriever

Responsibility:

- Given a task input, produce a complete ranking over memory node IDs.

Suggested interface shape:

```python
class Retriever(Protocol):
    method_name: str

    def rank(self, task: MemoryTaskInput) -> RankedNodes:
        ...
```

Important rule:

- A flat retriever should not know about labels or metrics.

### Graph Reranker

Responsibility:

- Given initial node scores and a graph, produce a new complete ranking.
- Optionally expose score components for observability.

Key boundary:

- Reranking is not a retriever by itself; it is a scoring transform over an initial retriever.

### Evaluator

Responsibility:

- Join predictions, labels, and graphs by `task_id`.
- Compute metrics.
- Export aggregate rows and failure cases.

Should not:

- Re-run retrieval.
- Interpret raw dataset examples.
- Read gold fields from input artifacts.

### Validator

Responsibility:

- Enforce contracts at script boundaries.
- Fail fast on malformed artifacts.

Open question:

- Should validation functions be standalone functions, or should every artifact type have a small validator object?

Initial leaning:

- Standalone validation functions are enough for Phase 1.

## Core Abstraction Proposal

The project should distinguish between three kinds of abstraction:

```text
Stable domain data:
  Name it clearly with dataclasses, type aliases, enums, or TypedDicts.

Replaceable behavior:
  Give it a small Protocol or class interface.

One-off deterministic transformation:
  Keep it as a function unless state or multiple implementations appear.
```

### Must-Have Named Data Concepts

These concepts appear across modules and should have clear names even if their serialized form remains JSON:

| Concept | Why it needs a name |
|---|---|
| `TaskId` | Join key across all artifacts. |
| `NodeId` | Used by memory items, graph edges, labels, predictions. |
| `MethodName` | Controls retrieval dispatch and result rows. |
| `MemoryTaskInput` | Main input-visible task contract. |
| `MemoryTaskLabels` | Gold-only evaluation contract. |
| `MemoryGraph` | Graph contract consumed by rerank and evaluation. |
| `RankedNode` | Atomic score result. |
| `RankedResult` | Per-task retrieval output. |
| `GraphBuildConfig` | Stable graph construction parameters. |
| `GraphRerankConfig` | Tuned graph rerank parameters. |
| `EvaluationRow` | Stable metric output row. |

Open representation choice:

- Use `TypedDict` for JSON-like records if it keeps boundaries readable.
- Use `dataclass` for configs and internal result objects.
- Avoid converting every nested JSON object into a class unless it improves clarity.

### Data Representation Strategy

The code should use a two-layer representation model:

```text
Disk artifacts:
  JSON/CSV-shaped records, validated strictly at boundaries.

Internal algorithm state:
  Small named objects for configs, scores, and reusable results.
```

This avoids two bad extremes:

- passing anonymous dictionaries everywhere
- building a heavy class hierarchy that mirrors every JSON field

#### Recommended Representations

| Data concept | Recommended form | Reason |
|---|---|---|
| `TaskId`, `NodeId` | type aliases | Makes signatures readable without runtime overhead. |
| method names, edge types, node types | `Literal` or `Enum` | Prevents typo-prone strings in important branches. |
| `MemoryTaskInput` | `TypedDict` | It mirrors a JSON artifact and is mostly passed across boundaries. |
| `MemoryItem` | `TypedDict` | Nested JSON record, simple and stable. |
| `MemoryTaskLabels` | `TypedDict` | Gold-label artifact, should stay close to JSON contract. |
| `GraphNode`, `GraphEdge`, `MemoryGraph` | `TypedDict` | Graph artifact is persisted and exchanged as JSON. |
| `RankedNode` | frozen `dataclass` | Algorithms repeatedly create, sort, and inspect these. |
| `RankedResult` | `TypedDict` or dataclass at assembly boundary | Persisted as JSON, but can be assembled from dataclass pieces. |
| `GraphBuildConfig` | frozen `dataclass` | Internal config with defaults and validation. |
| `GraphRerankConfig` | frozen `dataclass` | Tuned config with defaults and easy construction in tests. |
| `DenseConfig` | frozen `dataclass` | Keeps encoder model, prefixes, and batch size explicit. |
| `RerankResult` | frozen `dataclass` | Contains ranked nodes plus optional score components. |
| score components | frozen `dataclass` or dict keyed by `NodeId` | Should be inspectable without polluting ranked-result schema. |
| metric rows | plain dict or `TypedDict` | CSV-shaped output, not core algorithm state. |
| run summary | `TypedDict` or plain validated dict | Observability artifact, JSON-shaped. |

#### Suggested Alias Vocabulary

```python
TaskId = str
NodeId = str
MethodName = str
Score = float
```

Keep aliases boring. They are for readability, not for building a new type system.

#### Suggested Enum/Literal Vocabulary

Edge types:

```text
sequential
query_overlap
entity_overlap
bridge
```

Node types:

```text
question
document_sentence
```

Retrieval methods:

```text
bm25
dense
bm25_graph_rerank
dense_graph_rerank
```

Use `Literal` if the list is tiny and stays local. Use `Enum` if values are shared across modules, serialized often, or need methods.

#### Dataclass Rules

Use dataclasses when:

- the value is created and consumed by algorithms
- defaults matter
- tests need easy construction
- the object has a stable meaning independent of JSON field order

Prefer:

```text
frozen=True
```

for config and result objects so accidental mutation does not hide bugs.

Avoid dataclasses when:

- the object only mirrors a persisted JSON record
- the class would contain many optional fields
- conversion from dict to object adds no clarity

#### TypedDict Rules

Use `TypedDict` when:

- the object is a JSON artifact record
- field names are part of the public contract
- validators enforce correctness at script boundaries

Avoid deeply nesting `TypedDict` signatures inline. Name every important record.

#### What Not To Do

Avoid:

- signatures with deeply nested `tuple[list[dict[...]]]` forms
- returning multiple unrelated values as raw tuples
- using `Any` to avoid thinking about contracts
- using large dataclasses as a second copy of the JSON schema
- mutating task, graph, or label records inside algorithms

#### Boundary Flow

Recommended flow:

```text
read JSON
  -> validate TypedDict-shaped records
  -> algorithms consume records and configs
  -> algorithms emit dataclass result pieces where useful
  -> assemble JSON-shaped output
  -> validate output
  -> write JSON/CSV
```

This keeps persisted artifacts transparent while giving the algorithm code enough named structure to stay readable.

### Core Data Structure Sketch

This section sketches the intended data shapes. It is not implementation code yet; it is a readability target for `graph_memory/types.py` and related validators.

#### Primitive Aliases

Purpose: make signatures read in domain language without creating a heavy type system.

```python
TaskId = str
NodeId = str
MethodName = str
Score = float
```

Guideline:

- Keep these aliases simple.
- Do not introduce wrapper classes for IDs in Phase 1.

#### Node And Edge Vocabulary

Purpose: keep important string values consistent across graph construction, rerank, evaluation, and debug output.

```python
NodeType = Literal["question", "document_sentence"]

EdgeType = Literal[
    "sequential",
    "query_overlap",
    "entity_overlap",
    "bridge",
]

RetrievalMethod = Literal[
    "bm25",
    "dense",
    "bm25_graph_rerank",
    "dense_graph_rerank",
]
```

Recommendation:

- Start with `Literal` because the value sets are small.
- Move to `Enum` only if methods/edge types start needing behavior or frequent conversion helpers.

#### Memory Item

Purpose: one candidate evidence sentence.

```python
class MemoryItem(TypedDict):
    id: NodeId
    node_type: Literal["document_sentence"]
    text: str
    source: str
    sentence_id: int
    position: int
```

Important invariants:

- `id == f"m{position}"` for HotpotQA Phase 1.
- `sentence_id` is local to `source`.
- `position` is global within the flattened task memory.

#### Memory Task Input

Purpose: the only task artifact visible to graph construction and retrieval.

```python
class MemoryTaskInput(TypedDict):
    task_id: TaskId
    query: str
    memory_items: list[MemoryItem]
```

Rules:

- No gold fields.
- No answer text.
- No `supporting_facts`.
- Treat as read-only inside algorithms.

#### Memory Task Labels

Purpose: evaluation and dev tuning labels.

```python
class MemoryTaskLabels(TypedDict):
    task_id: TaskId
    gold_answer: str
    gold_evidence_nodes: list[NodeId]
    gold_dependency_edges: list[tuple[NodeId, NodeId, str]]
```

Open detail:

- JSON stores tuples as arrays. The validator can accept arrays and normalize internally when needed.
- For HotpotQA Phase 1, `gold_dependency_edges` should be empty.

#### Graph Node

Purpose: one node in the constructed memory graph.

```python
class QuestionNode(TypedDict):
    id: Literal["q"]
    node_type: Literal["question"]
    text: str

class MemoryGraphNode(MemoryItem):
    pass
```

Alternative:

```python
GraphNode = QuestionNode | MemoryGraphNode
```

Note:

- If inheritance on `TypedDict` feels visually awkward in implementation, use two named `TypedDict`s and a union alias.

#### Graph Edge

Purpose: typed relation between graph nodes.

```python
class GraphEdge(TypedDict):
    source: NodeId
    target: NodeId
    edge_type: EdgeType
    weight: float
    directed: bool
```

Rules:

- Edge endpoints must exist in the graph.
- Weight must be finite and non-negative.
- `source` may be `"q"` for query overlap.

Potential alias:

```python
GraphNodeId = Literal["q"] | NodeId
```

This alias is expressive but may be too clever in Python. Plain `str` with validation may be more readable.

#### Memory Graph

Purpose: graph artifact consumed by rerank and evaluation.

```python
class MemoryGraph(TypedDict):
    task_id: TaskId
    nodes: list[GraphNode]
    edges: list[GraphEdge]
```

Rules:

- Must contain exactly one `q` node.
- Must contain all task memory nodes.
- Must not contain gold label fields.

#### Ranked Node

Purpose: atomic score-bearing result used by retrievers, reranker, and result assembly.

```python
@dataclass(frozen=True)
class RankedNode:
    node_id: NodeId
    score: Score
```

Why dataclass:

- It is created and sorted often.
- It is algorithm output, not just a JSON mirror.
- It makes tests readable.

#### Score Components

Purpose: explain graph rerank decisions without polluting the required ranked result schema.

```python
@dataclass(frozen=True)
class ScoreComponents:
    initial: float
    query: float = 0.0
    neighbor: float = 0.0
    bridge: float = 0.0
    path: float = 0.0
    final: float = 0.0
```

Usage:

```python
ScoreBreakdown = dict[NodeId, ScoreComponents]
```

Note:

- `path` remains `0.0` in HotpotQA Phase 1 unless later explicitly implemented.

#### Rerank Result

Purpose: output of graph reranking before JSON assembly.

```python
@dataclass(frozen=True)
class RerankResult:
    ranked_nodes: list[RankedNode]
    retrieved_subgraph: RetrievedSubgraph
    score_breakdown: ScoreBreakdown | None = None
```

Design note:

- `score_breakdown` is optional because normal runs should keep outputs small.
- Debug mode can persist it separately.

#### Retrieved Subgraph

Purpose: the top-k induced subgraph stored in ranked results.

```python
class RetrievedSubgraph(TypedDict):
    nodes: list[NodeId]
    edges: list[GraphEdge]
```

Open detail:

- For graph edges involving `q`, `nodes` may need to allow `"q"` if query-node connectivity is persisted. For the existing ranked-result schema, retrieved memory nodes are enough; query connectivity can be computed during evaluation from `q + top_k`.

#### Ranked Result

Purpose: persisted per-task output for all retrieval methods.

```python
class RankedResult(TypedDict):
    task_id: TaskId
    method: MethodName
    ranked_nodes: list[RankedNodeRecord]
    retrieved_subgraph: RetrievedSubgraph
    latency_ms: float
    input_tokens: int
```

With:

```python
class RankedNodeRecord(TypedDict):
    node_id: NodeId
    score: float
```

Design note:

- `RankedNode` dataclass is internal.
- `RankedNodeRecord` is the JSON shape.
- Conversion between them should be small and explicit.

#### Config Objects

Purpose: keep algorithm parameters explicit and immutable.

```python
@dataclass(frozen=True)
class GraphBuildConfig:
    max_query_overlap: int = 20
    max_entity_neighbors: int = 10
    max_bridge_edges: int = 50
    use_spacy: bool = False
```

```python
@dataclass(frozen=True)
class DenseConfig:
    model_name: str = "intfloat/e5-base-v2"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = 64
```

```python
@dataclass(frozen=True)
class GraphRerankConfig:
    lambda_init: float = 1.0
    lambda_query: float = 0.1
    lambda_neighbor: float = 0.2
    lambda_bridge: float = 0.1
    lambda_path: float = 0.0
    seed_top_s: int = 30
    max_hops: int = 2
    type_weights: dict[EdgeType, float] = field(default_factory=dict)
```

Implementation note:

- Avoid mutable default dictionaries by using `field(default_factory=...)`.
- Config validation should reject negative lambdas, invalid hop counts, and missing type weights.

#### Evaluation Row

Purpose: one method-level metric row.

```python
class EvaluationRow(TypedDict):
    Method: str
    Recall_at_2: float
    Recall_at_5: float
    Recall_at_10: float
    Evidence_F1_at_5: float
    Evidence_F1_at_10: float
    Full_Support_at_5: float
    Full_Support_at_10: float
    MRR: float
    Connected_Evidence_Recall_at_5: float
    Connected_Evidence_Recall_at_10: float
    Query_Evidence_Connectivity_at_10: float
    Path_Recall_at_10: str | float
    Edge_Recall_at_10: str | float
    Retrieval_Latency_per_Query: float
```

Open detail:

- CSV column names can keep human-readable labels such as `Recall@5`.
- Python keys may use identifier-safe names.
- A small mapping function can translate Python keys to CSV headers.

### Behavior Abstractions Worth Keeping

### Core Behavior Interface Sketch

The behavior layer should separate:

```text
single-task algorithm:
  easy to test, deterministic, no file I/O

batch experiment service:
  loops over tasks, measures latency, assembles artifacts

script adapter:
  parses CLI/config, reads/writes files
```

This separation keeps the real algorithms small and makes experiment orchestration observable.

#### Retriever

Retriever is worth abstracting because BM25, dense, Dense-FT, Memory Stream, GraphRAG, and future methods all answer the same question:

```text
Given one input-visible task, return a complete ranking over memory node IDs.
```

Minimal contract:

```python
class Retriever(Protocol):
    method_name: str

    def rank(self, task: MemoryTaskInput) -> list[RankedNode]:
        ...
```

Rules:

- Returns all memory nodes exactly once.
- Does not read labels.
- Does not compute metrics.
- Does not write files.
- May have internal state such as an encoder model, but state must be explicit in construction.

Recommended scope:

- `rank()` handles exactly one task.
- Batch retrieval belongs to `run_retrieval(...)`, not to the retriever object.

Why:

- HotpotQA memory is task-local.
- BM25 is naturally per-task.
- Dense can still keep a loaded encoder as object state.
- Single-task tests can use tiny fixtures.

Expected implementations:

```text
BM25TaskRetriever.rank(task)
DenseTaskRetriever.rank(task)
```

Future compatible implementations:

```text
DenseFineTunedRetriever.rank(task)
MemoryStreamRetriever.rank(task)
GraphRAGRetriever.rank(task)
```

#### Reranker

Reranker is worth separating from Retriever because graph reranking composes with an initial retriever. This avoids duplicating BM25-graph and dense-graph behavior.

Decision:

- Graph rerank should be extracted as an independent reusable module.
- `bm25_graph_rerank` and `dense_graph_rerank` should be composed methods, not separate copies of graph propagation logic.

Composition:

```text
BM25 Retriever
  -> initial ranking
  -> Graph Reranker
  -> final ranked result

Dense Retriever
  -> initial ranking
  -> Graph Reranker
  -> final ranked result
```

Minimal contract:

```python
class Reranker(Protocol):
    def rerank(
        self,
        initial_ranking: list[RankedNode],
        graph: MemoryGraph,
    ) -> RerankResult:
        ...
```

`RerankResult` can include:

- complete reranked nodes
- retrieved subgraph
- optional score components for debug mode

Rules:

- Does not run BM25 or dense retrieval itself.
- Does not read labels.
- Does not compute metrics.
- Uses graph and initial scores only.
- Should expose optional score components so graph effects can be tested and inspected.

Testing implication:

- Graph rerank can be tested once with artificial initial scores and a tiny graph.
- BM25 and dense tests do not need to duplicate graph propagation assertions.

Observability implication:

- Reranker owns score breakdown fields such as `initial`, `query`, `neighbor`, `bridge`, `path`, and `final`.
- Retrieval result assembly decides whether to persist those components in debug artifacts.

Recommended scope:

- `rerank()` handles exactly one task graph and one initial ranking.
- Batch graph rerank belongs to `run_retrieval(...)`.

Expected implementation:

```text
GraphReranker.rerank(initial_ranking, graph)
```

or function form:

```text
graph_rerank(initial_scores, graph, config) -> list[RankedNode]
```

Current leaning:

- Use a function for the core formula.
- Optionally wrap it in a small `GraphReranker` class only if that makes config and debug options cleaner.

The key abstraction is the separation from retrievers, not the existence of a class.

#### Graph Builder

Graph building should remain function-based in Phase 1.

Recommended functions:

```text
build_graph(task_input, config) -> MemoryGraph
build_graphs(task_inputs, config) -> list[MemoryGraph]
graph_statistics(graph) -> GraphStats
```

Rules:

- Reads only `MemoryTaskInput`.
- Does not read labels.
- Does not run retrieval.
- Does not compute evaluation metrics.

Why not a class yet:

- Phase 1 has one deterministic graph construction strategy.
- `GraphBuildConfig` already carries the needed state.
- Functions are easier to test and inspect.

Possible future extraction:

- If Phase 2 adds GraphRAG-style graph construction, random-edge ablation builders, or multiple graph schemas, introduce a `GraphBuilder` protocol then.

#### Evaluator

Evaluator can be mostly functional in Phase 1, but the concept should be named because it owns the scientific metric boundary.

Suggested shape:

```text
evaluate_results(predictions, labels, graphs) -> list[EvaluationRow]
```

Rules:

- Reads labels and graphs.
- Never re-runs retrieval.
- Never reads gold fields from input tasks.
- Fails if task IDs cannot be joined exactly.

Recommended lower-level functions:

```text
recall_at(ranked_nodes, gold_nodes, k) -> float
evidence_f1_at(ranked_nodes, gold_nodes, k) -> float
full_support_at(ranked_nodes, gold_nodes, k) -> float
mrr(ranked_nodes, gold_nodes) -> float
connected_evidence_at(ranked_nodes, gold_nodes, graph, k) -> float
query_evidence_connectivity_at(ranked_nodes, gold_nodes, graph, k) -> float
```

Recommended higher-level function:

```text
evaluate_results(predictions, labels, graphs) -> list[EvaluationRow]
```

Why:

- Metric primitives are easy to test.
- The higher-level evaluator owns task joins and aggregation.
- This keeps scientific definitions visible.

#### Validator

Validator should remain function-based in Phase 1.

Suggested shape:

```text
validate_memory_task_inputs(records)
validate_memory_task_labels(records, inputs_by_task_id)
validate_graphs(graphs, inputs_by_task_id)
validate_ranked_results(predictions, inputs_by_task_id)
```

Rules:

- Raises exceptions.
- Does not repair data.
- Does not infer missing fields.

Recommended validator style:

```text
validate_xxx(records) -> None
```

not:

```text
validate_xxx(records) -> cleaned_records
```

Reason:

- Validation should not silently transform the scientific artifact.
- If normalization is needed, it should be a named conversion step.

#### IO And Config Loading

I/O should stay boring and explicit.

Recommended functions:

```text
read_json(path) -> object
write_json(path, data) -> None
read_csv(path) -> list[dict[str, str]]
write_csv(path, rows, fieldnames) -> None
load_config(path) -> dict
merge_config(defaults, config_file, cli_overrides) -> dict
```

Rules:

- I/O helpers do not validate domain contracts by themselves.
- Scripts call I/O, then validators.
- Config merge must follow `CLI > config > defaults`.

#### Retrieval Experiment Service

This is the orchestration function beneath `scripts/run_retrieval.py`.

Recommended shape:

```text
run_retrieval(
    method,
    tasks,
    graphs,
    retriever_config,
    rerank_config,
    top_k,
    debug,
) -> list[RankedResult]
```

Responsibilities:

- Select or construct the retriever.
- For graph methods, apply the graph reranker after initial retrieval.
- Measure per-task latency.
- Build retrieved subgraphs.
- Assemble ranked-result artifacts.
- Optionally emit debug records through observability helpers.

Non-responsibilities:

- Does not read files.
- Does not write files.
- Does not evaluate metrics.
- Does not tune parameters.

#### Tuning Service

Tuning is a service function because it composes retrieval and evaluation on dev data.

Recommended shape:

```text
tune_graph_rerank(
    method,
    tasks,
    labels,
    graphs,
    grid,
    objective,
) -> GraphRerankConfig
```

Rules:

- Uses labels because dev tuning is allowed to use labels.
- Never runs on test labels for parameter selection.
- Writes no files directly; the script writes the selected config.

#### Observability Helpers

Observability should be helper functions, not a framework.

Recommended functions:

```text
build_run_summary(...)
graph_stats(graphs) -> dict
build_score_debug_record(task_id, ranked_nodes, score_breakdown) -> dict
build_failure_cases(predictions, labels, graphs) -> list[dict]
```

Rules:

- Observability helpers should consume outputs from algorithms.
- They should not change retrieval, rerank, or evaluation behavior.
- Debug outputs should be optional except for compact run summaries and graph stats.

### Behavior Boundary Summary

| Behavior | Shape | Reason |
|---|---|---|
| BM25/Dense retrieval | `Retriever` protocol | Multiple replaceable implementations. |
| Graph rerank | separate function or small class | Composes with initial retrievers. |
| Graph construction | functions + config | One deterministic strategy in Phase 1. |
| Evaluation | metric functions + aggregate function | Scientific definitions stay visible. |
| Validation | fail-fast functions | No silent repair. |
| I/O | simple utility functions | Keep scripts thin and explicit. |
| Run retrieval | service function | Batch orchestration without file I/O. |
| Tuning | service function | Dev-only composition of retrieval and metrics. |
| Observability | helper functions | Inspectability without framework overhead. |

### Behavior That Should Stay Function-Based For Now

| Behavior | Reason |
|---|---|
| HotpotQA conversion | One deterministic conversion path in Phase 1. |
| Split sampling | Pure deterministic utility. |
| Text normalization | Pure functions are easier to test. |
| Entity extraction | Function with optional spaCy dependency is enough. |
| Graph construction | Phase 1 has one graph builder; use functions plus `GraphBuildConfig`. |
| Metric primitives | Pure functions are clearest for precision, recall, full support, connectivity. |
| JSON/CSV I/O | Utility functions, not a repository abstraction. |

### Composite Services

Some top-level functions should orchestrate behavior without becoming heavy framework objects:

```text
run_retrieval(method, tasks, graphs, config) -> list[RankedResult]
tune_graph_rerank(method, dev_tasks, dev_labels, dev_graphs, grid) -> GraphRerankConfig
aggregate_tables(result_files) -> table artifacts
```

These functions are useful because scripts can call them directly and tests can bypass CLI parsing.

### Anti-Abstractions

Avoid these in Phase 1:

- Dataset plugin registry.
- Method plugin discovery.
- Abstract base classes with many lifecycle hooks.
- Generic pipeline engine.
- Repository objects for local JSON files.
- Large object graph mirroring every JSON field.
- Implicit global config singleton.

These may look neat early but make the experiment harder to inspect.

### Score Component Boundary

Graph rerank should optionally expose score components without forcing every retriever to do the same.

Suggested component names:

- `initial`
- `query`
- `neighbor`
- `bridge`
- `path`
- `final`

This supports observability while keeping the required ranked result schema simple.

## Testing Implications

This architecture enables tests by layer:

| Layer | Test target |
|---|---|
| Contracts | Invalid artifacts fail fast. |
| Dataset conversion | Supporting facts map to node IDs correctly. |
| Graph building | Edge types and weights obey rules. |
| Retrieval | All methods emit the same ranked schema. |
| Reranking | Score propagation changes rankings as expected. |
| Evaluation | Metrics use labels and shared graphs correctly. |
| CLI | One small smoke test per script, if needed. |

## Observability Implications

Observability should attach to boundaries, not invade algorithms.

Potential outputs:

- `run_summary.json` for each script.
- Graph stats from graph builder.
- Score component breakdown from graph reranker.
- Failure-case exports from evaluator.

This keeps the algorithm functions mostly pure while still making experiments inspectable.

## Current Decisions

- Architecture style: library-core with thin CLI.
- Physical module structure: mostly flat package, matching the Phase 1 plan and original experiment command surface.
- Supporting modules `types.py`, `validation.py`, `io.py`, and `observability.py` should be added because they make contracts, I/O, and run records explicit without introducing deep subpackages.
- Internal representation: hybrid, with validated dicts at JSON boundaries and dataclasses for configs/results.
- Graph rerank: separate scoring transform composed with an initial retriever. This is now a confirmed decision.
- Observability: mandatory run summaries and graph stats; optional per-task score breakdowns.
