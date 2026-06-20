# Request-First Dataset Projection Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 第一部分重构要删除 `MemoryTaskInput` / `MemoryTaskLabels` / `MemoryItem` 这套中心契约，改成 dataset-specific record 通过 projector 直接产出各消费者需要的 request。

**Architecture:** 稳定接口放在消费者侧：retriever 声明 `TextRankingRequest`、`GraphRankingRequest`、`TemporalMemoryRankingRequest`，graph builder 声明 `GraphBuildRequest`，evaluator 声明 `EvidenceEvaluationRequest`。HotpotQA 只保留自己的 dataset record 和 projector，例如 `HotpotQAToTextRankingRequest`、`HotpotQAToGraphBuildRequest`、`HotpotQAToEvidenceEvaluationRequest`；未来 2Wiki 或 LongMemEval 新增自己的 projector，不要求复用一个中间 `EvidenceRankingView`。

**Tech Stack:** Python `TypedDict`/dataclass/Protocol，现有 `graph_memory` package，pytest，ruff，basedpyright，OpenSpec validation。Windows host 上所有 `uv` 命令从第一次就必须在 Codex sandbox 外执行。

---

## Core Decision

本计划采用 request-first 设计：

```text
dataset-specific processed records
  -> dataset-specific projector functions
  -> consumer-owned request contracts
  -> graph builder / retriever / evaluator / training
```

不引入通用 `EvidenceRankingView` 作为第一阶段核心对象。原因：

- `View` 只有在多个 projection 共享同一个稳定中间语义时才有价值；当前第一阶段没有证明这个中间层必要。
- HotpotQA 的 `source/sentence_id/position/document_sentence` 不应该成为跨 dataset 的稳定字段。
- 稳定接口应该由消费者声明：BM25/Dense 只要文本 ranking request；GraphBuilder 只要 graph build request；Evaluator 只要 evaluation request。
- Dataset 到 request 的映射就是 projection。projection 可以是简单函数或小类，不需要通用 registry 起步。

## Target Data Flow

```text
HotpotQA raw
  -> HotpotQA parser
  -> HotpotQA converter
  -> HotpotQARankingRecord + HotpotQALabelRecord

HotpotQARankingRecord
  -> HotpotQAToTextRankingRequest
  -> TextRankingRequest
  -> BM25 / Dense / Dense-FT

HotpotQARankingRecord
  -> HotpotQAToGraphBuildRequest
  -> GraphBuildRequest
  -> GraphBuilder
  -> MemoryGraph

HotpotQARankingRecord + seed scores + MemoryGraph
  -> HotpotQAToGraphRankingRequest
  -> GraphRankingRequest
  -> GraphRerank / R-GCN inference

Prediction + HotpotQALabelRecord + optional MemoryGraph
  -> HotpotQAToEvidenceEvaluationRequest
  -> EvidenceEvaluationRequest
  -> evidence metrics
```

## Stable vs Dataset-Specific

| Layer | Stable? | Owner | Examples |
|---|---:|---|---|
| Dataset record | No | `graph_memory.datasets.<dataset>` | `HotpotQARankingRecord`, future `TwoWikiRankingRecord`, future `LongMemEvalContextRecord` |
| Projector | No | `graph_memory.datasets.<dataset>.projectors` | `HotpotQAToTextRankingRequest`, `HotpotQAToGraphBuildRequest` |
| Request | Yes | Consumer package | `TextRankingRequest`, `GraphBuildRequest`, `GraphRankingRequest`, `EvidenceEvaluationRequest` |
| Method implementation | Yes by request | Retrieval/graph/evaluation packages | BM25 consumes `TextRankingRequest`; GraphBuilder consumes `GraphBuildRequest` |

## Request Contracts

第一阶段需要建立这些稳定 consumer request：

```python
@dataclass(frozen=True)
class TextCandidate:
    item_id: str
    text: str
    metadata: Mapping[str, JsonScalar]

@dataclass(frozen=True)
class TextRankingRequest:
    task_id: str
    query_text: str
    candidates: Sequence[TextCandidate]
```

```python
@dataclass(frozen=True)
class GraphBuildNode:
    node_id: str
    text: str
    node_kind: str
    source_label: str | None
    group_key: str | None
    sequence_index: int | None
    metadata: Mapping[str, JsonScalar]

@dataclass(frozen=True)
class GraphBuildRequest:
    task_id: str
    query_text: str
    nodes: Sequence[GraphBuildNode]
    input_visible_edges: Sequence[GraphBuildEdge]
```

```python
@dataclass(frozen=True)
class GraphRankingRequest:
    task_id: str
    query_text: str
    candidates: Sequence[TextCandidate]
    graph: MemoryGraph
    initial_scores: Mapping[str, float]

@dataclass(frozen=True)
class EvidenceEvaluationRequest:
    predictions: Sequence[RankedResult]
    labels: Sequence[EvidenceLabel]
    graphs: Sequence[MemoryGraph]
```

HotpotQA 的 `title/sentence_id/position` 只在 `HotpotQARankingRecord` 和 `HotpotQAToGraphBuildRequest` 内部出现；它们不属于 `TextRankingRequest` 的 required field。

## File Structure

- Modify `graph_memory/contracts/tasks.py`: remove old memory task names; either delete the module after imports are migrated or leave only non-old-name re-exports.
- Create/modify `graph_memory/retrieval/requests.py`: keep `DenseRuntime` and add `TextCandidate`, `TextRankingRequest`, `GraphRankingRequest`, `TemporalMemoryRankingRequest`.
- Create `graph_memory/graphs/requests.py`: add `GraphBuildNode`, `GraphBuildEdge`, `GraphBuildRequest`.
- Create `graph_memory/evaluation/requests.py`: add `EvidenceLabel`, `EvidenceEvaluationRequest`.
- Modify `graph_memory/datasets/hotpotqa/records.py`: define `HotpotQACandidateSentence`, `HotpotQARankingRecord`, `HotpotQALabelRecord`, `CombinedHotpotQARecord`.
- Modify `graph_memory/datasets/hotpotqa/converter.py`: output HotpotQA dataset records, not generic evidence views.
- Create `graph_memory/datasets/hotpotqa/projectors.py`: implement HotpotQA-to-request projectors.
- Modify `graph_memory/datasets/hotpotqa/compatibility.py`: replace old combined helper with `combined_hotpotqa_records`.
- Modify `scripts/prepare_hotpotqa.py`: write HotpotQA dataset records and labels.
- Modify `graph_memory/validation/tasks.py`: rename validators to `validate_hotpotqa_ranking_records` and `validate_hotpotqa_label_records`.
- Modify retrieval, graph, evaluation, training, registry, and stage modules so they consume requests and assemble them via HotpotQA projectors.
- Modify tests and durable docs so active production code no longer references `MemoryTaskInput`, `MemoryTaskLabels`, `MemoryItem`, or `CombinedMemoryTask`.

## Task 1: Add Failing Boundary Tests

**Files:**
- Create: `tests/test_request_first_projection_boundaries.py`
- Modify: `tests/test_core_refactor_batch1_boundaries.py`

- [ ] **Step 1: Add source guard for old memory task names**

Create `tests/test_request_first_projection_boundaries.py`:

```python
from pathlib import Path

FORBIDDEN_NAMES = ("MemoryTaskInput", "MemoryTaskLabels", "MemoryItem", "CombinedMemoryTask")


def test_production_and_test_code_do_not_reference_old_memory_task_contracts() -> None:
    offenders: list[str] = []
    for root in (Path("graph_memory"), Path("scripts"), Path("tests")):
        for path in root.rglob("*.py"):
            if path == Path(__file__):
                continue
            source = path.read_text(encoding="utf-8")
            for name in FORBIDDEN_NAMES:
                if name in source:
                    offenders.append(f"{path.as_posix()}:{name}")
    assert offenders == []
```

- [ ] **Step 2: Add request-first projection tests**

Append to `tests/test_request_first_projection_boundaries.py`:

```python
from graph_memory.datasets.hotpotqa.projectors import (
    HotpotQAToEvidenceEvaluationRequest,
    HotpotQAToGraphBuildRequest,
    HotpotQAToTextRankingRequest,
)


def _hotpotqa_record() -> dict[str, object]:
    return {
        "task_id": "hotpot_1",
        "question": "Where was Ada born?",
        "candidate_sentences": [{"sentence_id": "m0", "title": "Ada Lovelace", "sentence_index": 0, "position": 0, "text": "Ada Lovelace was born in London."}],
    }


def _hotpotqa_label() -> dict[str, object]:
    return {"task_id": "hotpot_1", "gold_answer": "London", "gold_evidence_sentence_ids": ["m0"], "gold_dependency_edges": []}


def test_hotpotqa_text_projection_outputs_retriever_request_only() -> None:
    request = HotpotQAToTextRankingRequest().project(_hotpotqa_record())
    assert request.task_id == "hotpot_1"
    assert request.query_text == "Where was Ada born?"
    assert request.candidates[0].item_id == "m0"
    assert request.candidates[0].text == "Ada Lovelace. Ada Lovelace was born in London."
    assert request.candidates[0].metadata == {"title": "Ada Lovelace"}
    assert not hasattr(request.candidates[0], "sentence_index")


def test_hotpotqa_graph_projection_outputs_graph_builder_request() -> None:
    request = HotpotQAToGraphBuildRequest().project(_hotpotqa_record())
    assert request.task_id == "hotpot_1"
    assert request.nodes[0].node_id == "m0"
    assert request.nodes[0].source_label == "Ada Lovelace"
    assert request.nodes[0].group_key == "document:Ada Lovelace"
    assert request.nodes[0].sequence_index == 0


def test_hotpotqa_evaluation_projection_outputs_evaluator_request() -> None:
    ranked_result = {"task_id": "hotpot_1", "method": "bm25", "ranked_nodes": [{"node_id": "m0", "score": 1.0}], "retrieved_subgraph": {"nodes": ["m0"], "edges": []}, "latency_ms": 1.0, "input_tokens": 8}
    graph = {"task_id": "hotpot_1", "nodes": [{"id": "q", "node_type": "question", "text": "Where?"}], "edges": []}
    request = HotpotQAToEvidenceEvaluationRequest().project(predictions=[ranked_result], labels=[_hotpotqa_label()], graphs=[graph])
    assert request.labels[0].task_id == "hotpot_1"
    assert request.labels[0].gold_evidence_item_ids == ("m0",)
```

- [ ] **Step 3: Update foundation export expectations**

In `tests/test_core_refactor_batch1_boundaries.py`, remove old task contract names from `MIGRATED_CONTRACT_NAMES`. Add and assert exports for:

```python
"TextCandidate", "TextRankingRequest", "GraphRankingRequest", "TemporalMemoryRankingRequest",
"GraphBuildNode", "GraphBuildEdge", "GraphBuildRequest",
"EvidenceLabel", "EvidenceEvaluationRequest",
```

- [ ] **Step 4: Run focused tests and verify RED**

Run outside the Codex filesystem sandbox:

```powershell
uv run pytest tests/test_request_first_projection_boundaries.py tests/test_core_refactor_batch1_boundaries.py -q
```

Expected result: FAIL because request modules and HotpotQA projectors do not exist yet.

## Task 2: Define Stable Consumer Request Contracts

**Files:**
- Modify: `graph_memory/retrieval/requests.py`
- Create: `graph_memory/graphs/requests.py`
- Create: `graph_memory/evaluation/requests.py`
- Modify: `graph_memory/contracts/tasks.py`
- Modify: `graph_memory/contracts/graphs.py`

- [ ] **Step 1: Add retrieval request contracts**

In `graph_memory/retrieval/requests.py`, keep existing `DenseRuntime` and add `TextCandidate`, `TextRankingRequest`, `GraphRankingRequest`, and `TemporalMemoryRankingRequest` exactly as described in `Request Contracts`.

- [ ] **Step 2: Add graph build request contracts**

Create `graph_memory/graphs/requests.py` with `GraphBuildNode`, `GraphBuildEdge`, and `GraphBuildRequest` exactly as described in `Request Contracts`.

- [ ] **Step 3: Add evaluation request contracts**

Create `graph_memory/evaluation/requests.py` with:

```python
@dataclass(frozen=True)
class EvidenceLabel:
    task_id: TaskId
    gold_answer: str
    gold_evidence_item_ids: tuple[NodeId, ...]
    gold_dependency_edges: tuple[tuple[NodeId, NodeId], ...]

@dataclass(frozen=True)
class EvidenceEvaluationRequest:
    predictions: Sequence[RankedResult]
    labels: Sequence[EvidenceLabel]
    graphs: Sequence[MemoryGraph]
```

- [ ] **Step 4: Remove old task contracts**

Replace `graph_memory/contracts/tasks.py` with:

```python
from __future__ import annotations

__all__: list[str] = []
```

- [ ] **Step 5: Decouple graph node contracts from old task contracts**

In `graph_memory/contracts/graphs.py`, remove `from graph_memory.contracts.tasks import MemoryItem`. Define graph nodes explicitly and export `GraphItemNode` instead of `GraphMemoryNode`.

- [ ] **Step 6: Run focused export tests**

Run outside sandbox:

```powershell
uv run pytest tests/test_core_refactor_batch1_boundaries.py -q
```

Expected result: PASS for request module exports after expectations are updated.

## Task 3: Replace MemoryTask Artifacts With HotpotQA Dataset Records

**Files:**
- Modify: `graph_memory/datasets/hotpotqa/records.py`
- Modify: `graph_memory/datasets/hotpotqa/converter.py`
- Modify: `graph_memory/datasets/hotpotqa/compatibility.py`
- Modify: `graph_memory/datasets/hotpotqa/__init__.py`
- Modify: `scripts/prepare_hotpotqa.py`
- Modify: HotpotQA conversion and validation tests

- [ ] **Step 1: Define HotpotQA processed record contracts**

In `graph_memory/datasets/hotpotqa/records.py`, add dataset-specific prepared artifact shapes:

```python
class HotpotQACandidateSentence(TypedDict):
    sentence_id: str
    title: str
    sentence_index: int
    position: int
    text: str

class HotpotQARankingRecord(TypedDict):
    task_id: str
    question: str
    candidate_sentences: list[HotpotQACandidateSentence]

class HotpotQALabelRecord(TypedDict):
    task_id: str
    gold_answer: str
    gold_evidence_sentence_ids: list[str]
    gold_dependency_edges: list[list[str]]

class CombinedHotpotQARecord(HotpotQARankingRecord, HotpotQALabelRecord):
    """Combined HotpotQA inspection artifact; retrieval code must not consume it."""
```

Update conversion result dataclasses to `ranking_record` / `label_record` and `ranking_records` / `label_records`.

- [ ] **Step 2: Update HotpotQA converter**

In `graph_memory/datasets/hotpotqa/converter.py`, replace memory item construction with `HotpotQACandidateSentence`:

```python
candidate_sentence: HotpotQACandidateSentence = {
    "sentence_id": sentence_id_from_position,
    "title": document.title,
    "sentence_index": sentence_index,
    "position": position,
    "text": sentence,
}
```

Return `HotpotQARankingRecord` and `HotpotQALabelRecord`:

```python
ranking_record = {"task_id": task_id, "question": example.question, "candidate_sentences": candidate_sentences}
label_record = {"task_id": task_id, "gold_answer": example.answer, "gold_evidence_sentence_ids": gold_evidence_sentence_ids, "gold_dependency_edges": []}
```

- [ ] **Step 3: Replace combined helper**

In `graph_memory/datasets/hotpotqa/compatibility.py`, replace old helper with:

```python
def combined_hotpotqa_records(
    ranking_records: Sequence[HotpotQARankingRecord],
    label_records: Sequence[HotpotQALabelRecord],
) -> list[CombinedHotpotQARecord]:
    labels_by_task_id = {record["task_id"]: record for record in label_records}
    return [{**record, **labels_by_task_id[record["task_id"]]} for record in ranking_records]
```

- [ ] **Step 4: Update prepare script**

In `scripts/prepare_hotpotqa.py`, rename `PreparedTasks` to `PreparedHotpotQARecords` and keep CLI path flags stable:

```text
--output_input   writes HotpotQARankingRecord[]
--output_labels  writes HotpotQALabelRecord[]
```

- [ ] **Step 5: Run HotpotQA converter tests**

Run outside sandbox:

```powershell
uv run pytest tests/test_core_refactor_batch2_boundaries.py tests/test_phase1_real_validation.py -q
```

Expected result: PASS after test fixtures are updated to `question`, `candidate_sentences`, and `gold_evidence_sentence_ids`.

## Task 4: Add HotpotQA Projectors

**Files:**
- Create: `graph_memory/datasets/hotpotqa/projectors.py`
- Modify: `graph_memory/datasets/hotpotqa/__init__.py`
- Test: `tests/test_request_first_projection_boundaries.py`

- [ ] **Step 1: Implement text request projector**

Create `graph_memory/datasets/hotpotqa/projectors.py` with:

```python
class HotpotQAToTextRankingRequest:
    def project(self, record: HotpotQARankingRecord) -> TextRankingRequest:
        return TextRankingRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            candidates=tuple(
                TextCandidate(
                    item_id=sentence["sentence_id"],
                    text=f'{sentence["title"]}. {sentence["text"]}',
                    metadata={"title": sentence["title"]},
                )
                for sentence in record["candidate_sentences"]
            ),
        )
```

- [ ] **Step 2: Implement graph build request projector**

Append:

```python
class HotpotQAToGraphBuildRequest:
    def project(self, record: HotpotQARankingRecord) -> GraphBuildRequest:
        return GraphBuildRequest(
            task_id=record["task_id"],
            query_text=record["question"],
            nodes=tuple(
                GraphBuildNode(
                    node_id=sentence["sentence_id"],
                    text=sentence["text"],
                    node_kind="document_sentence",
                    source_label=sentence["title"],
                    group_key=f'document:{sentence["title"]}',
                    sequence_index=sentence["sentence_index"],
                    metadata={"title": sentence["title"], "position": sentence["position"]},
                )
                for sentence in record["candidate_sentences"]
            ),
            input_visible_edges=(),
        )
```

- [ ] **Step 3: Implement graph ranking and evaluation request projectors**

Append:

```python
class HotpotQAToGraphRankingRequest:
    def project(self, record: HotpotQARankingRecord, graph: MemoryGraph, initial_scores: Mapping[str, float]) -> GraphRankingRequest:
        text_request = HotpotQAToTextRankingRequest().project(record)
        return GraphRankingRequest(task_id=record["task_id"], query_text=record["question"], candidates=text_request.candidates, graph=graph, initial_scores=initial_scores)

class HotpotQAToEvidenceEvaluationRequest:
    def project(self, *, predictions: Sequence[RankedResult], labels: Sequence[HotpotQALabelRecord], graphs: Sequence[MemoryGraph]) -> EvidenceEvaluationRequest:
        return EvidenceEvaluationRequest(
            predictions=predictions,
            labels=tuple(
                EvidenceLabel(
                    task_id=label["task_id"],
                    gold_answer=label["gold_answer"],
                    gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
                    gold_dependency_edges=tuple(tuple(edge) for edge in label["gold_dependency_edges"]),
                )
                for label in labels
            ),
            graphs=graphs,
        )
```

- [ ] **Step 4: Export projectors and run tests**

Export projector classes from `graph_memory/datasets/hotpotqa/__init__.py`, then run outside sandbox:

```powershell
uv run pytest tests/test_request_first_projection_boundaries.py -q
```

Expected result: PASS for projector behavior after request contracts exist.

## Task 5: Migrate Validation To Dataset Records

**Files:**
- Modify: `graph_memory/validation/tasks.py`
- Modify: `graph_memory/validation/common.py`
- Modify: `graph_memory/validation/__init__.py`
- Modify: validation tests

- [ ] **Step 1: Rename validator fields and functions**

In `graph_memory/validation/tasks.py`, define HotpotQA-specific field sets and validators:

```python
HOTPOTQA_RANKING_RECORD_FIELDS = {"task_id", "question", "candidate_sentences", "metadata", "debug"}
HOTPOTQA_CANDIDATE_SENTENCE_FIELDS = {"sentence_id", "title", "sentence_index", "position", "text"}
HOTPOTQA_LABEL_RECORD_FIELDS = {"task_id", "gold_answer", "gold_evidence_sentence_ids", "gold_dependency_edges", "metadata", "debug"}

def validate_hotpotqa_ranking_records(records: object) -> None:
    ...

def validate_hotpotqa_label_records(records: object, records_by_task_id: object) -> None:
    ...
```

Error messages should say `HotpotQA ranking record`, `HotpotQA candidate sentence`, and `HotpotQA label record`.

- [ ] **Step 2: Update validation exports and run tests**

Export the new validator names from `graph_memory/validation/__init__.py`, remove old validator exports, then run outside sandbox:

```powershell
uv run pytest tests/test_phase1_real_validation.py tests/test_request_first_projection_boundaries.py -q
```

Expected result: PASS.

## Task 6: Migrate Graph Construction To GraphBuildRequest

**Files:**
- Modify: `graph_memory/graphs/construction/builder.py`
- Modify: `graph_memory/graphs/construction/context.py`
- Modify: `graph_memory/graphs/construction/rules/*.py`
- Modify: `scripts/build_graphs.py`
- Modify: graph construction tests

- [ ] **Step 1: Make graph builder consume GraphBuildRequest**

In `graph_memory/graphs/construction/builder.py`, use `GraphBuildRequest` as the build input and construct current `MemoryGraph` nodes from `GraphBuildNode`.

```python
def build(self, request: GraphBuildRequest) -> MemoryGraph:
    prepared_input = prepare_graph_input(request, self.config)
    nodes: list[GraphNode] = [
        {"id": "q", "node_type": "question", "text": request.query_text},
        *[
            {"id": node.node_id, "node_type": node.node_kind, "text": node.text, "source": node.source_label or "", "sentence_id": node.sequence_index or 0, "position": int(node.metadata.get("position", 0))}
            for node in request.nodes
        ],
    ]
```

- [ ] **Step 2: Update graph rules and script projection**

Update sequential/bridge/query/entity rules to read `graph_input.request.nodes`. In `scripts/build_graphs.py`, load HotpotQA records and project:

```python
projector = HotpotQAToGraphBuildRequest()
graph_requests = [projector.project(record) for record in ranking_records]
graphs = build_graphs(graph_requests, config)
```

- [ ] **Step 3: Run graph tests**

Run outside sandbox:

```powershell
uv run pytest tests/test_core_refactor_batch3_boundaries.py tests/test_phase1_real_graphs.py -q
```

Expected result: PASS.

## Task 7: Migrate Text Retrieval To TextRankingRequest

**Files:**
- Modify: `graph_memory/retrieval/contracts.py`
- Modify: `graph_memory/retrieval/methods/flat/bm25.py`
- Modify: `graph_memory/retrieval/methods/flat/dense.py`
- Modify: `graph_memory/retrieval/methods/flat/method.py`
- Modify: `graph_memory/embeddings/dense.py`
- Modify: `graph_memory/retrieval/bulk.py`
- Modify: `graph_memory/retrieval/signals.py`
- Modify: retrieval tests

- [ ] **Step 1: Update SeedRanker protocol and flat methods**

In `graph_memory/retrieval/contracts.py`, make `SeedRanker.rank()` accept `TextRankingRequest`. Update BM25 to rank candidates from the request:

```python
def rank(self, request: TextRankingRequest) -> list[RankedNode]:
    corpus_tokens = [content_tokens(candidate.text) for candidate in request.candidates]
    query_tokens = content_tokens(request.query_text)
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)
    ranked_nodes = [RankedNode(node_id=candidate.item_id, score=float(score)) for candidate, score in zip(request.candidates, scores, strict=True)]
    return sorted(ranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))
```

- [ ] **Step 2: Update Dense encoding**

Make `DenseTaskEncodingRequest` hold `TextRankingRequest`, and use `request.query_text` / `request.candidates` when constructing texts.

- [ ] **Step 3: Run retrieval tests**

Run outside sandbox:

```powershell
uv run pytest tests/test_phase1_real_retrieval.py tests/test_batched_dense_encoding.py tests/test_batched_collection_consumers.py -q
```

Expected result: PASS.

## Task 8: Migrate Retrieval Stage And Graph Methods

**Files:**
- Modify: `graph_memory/retrieval/execution/service.py`
- Modify: `graph_memory/retrieval/execution/results.py`
- Modify: `graph_memory/stages/retrieve.py`
- Modify: `graph_memory/retrieval/methods/graph_rerank/method.py`
- Modify: `graph_memory/retrieval/methods/trainable_graph.py`
- Modify: `graph_memory/retrieval/methods/memory_stream/*.py`
- Modify: `graph_memory/retrieval/tuning/*.py`
- Modify: retrieval registry tests

- [ ] **Step 1: Project HotpotQA records inside retrieve stage**

In `graph_memory/stages/retrieve.py`, replace `task_inputs` with `ranking_records`. For flat jobs:

```python
text_requests = [HotpotQAToTextRankingRequest().project(record) for record in ranking_records]
```

For graph rerank jobs:

```python
text_request = HotpotQAToTextRankingRequest().project(record)
initial_ranking = seed_ranker.rank(text_request)
initial_scores = {node.node_id: node.score for node in initial_ranking}
graph = graph_index.get_required(record["task_id"])
request = HotpotQAToGraphRankingRequest().project(record, graph, initial_scores)
```

- [ ] **Step 2: Make graph rerank consume GraphRankingRequest**

In `graph_memory/retrieval/methods/graph_rerank/method.py`, change `rank_task()` to accept `GraphRankingRequest` and call `rank_graph_from_initial_scores(dict(request.initial_scores), request.graph, ...)`.

- [ ] **Step 3: Run retrieval registry and method tests**

Run outside sandbox:

```powershell
uv run pytest tests/test_retrieval_registry_builders.py tests/test_retrieval_method_results.py tests/test_retrieval_provenance.py tests/test_memory_stream_method.py tests/test_memory_stream_tuning.py -q
```

Expected result: PASS.

## Task 9: Migrate Training, Models, And Evaluation

**Files:**
- Modify: `graph_memory/training_pairs/*.py`
- Modify: `graph_memory/models/dense_finetune/*.py`
- Modify: `graph_memory/models/graph_retriever/*.py`
- Modify: `graph_memory/evaluation/service.py`
- Modify: `graph_memory/evaluation/failure_cases.py`
- Modify: `scripts/train_method.py`
- Modify: `scripts/tune_memory_stream.py`

- [ ] **Step 1: Migrate train pair and model inputs**

Change train pair builder signatures to accept `list[HotpotQARankingRecord]` and `list[HotpotQALabelRecord]`. Use `HotpotQAToTextRankingRequest` for BM25/dense hard negatives. Dense-FT data code should read `record["question"]`, `record["candidate_sentences"]`, and `label["gold_evidence_sentence_ids"]`.

- [ ] **Step 2: Migrate evaluator**

Change evaluation service to:

```python
def evaluate_results(request: EvidenceEvaluationRequest) -> list[MetricRow]:
    ...
```

Callers use `HotpotQAToEvidenceEvaluationRequest().project(...)`.

- [ ] **Step 3: Run training and evaluation tests**

Run outside sandbox:

```powershell
uv run pytest tests/test_dense_finetune_data.py tests/test_dense_finetune_training.py tests/test_phase2_rgcn_pairs.py tests/test_phase2_rgcn_training.py tests/test_phase1_real_evaluation.py -q
```

Expected result: PASS.

## Task 10: Update Workflow, Docs, And Remove Old Names

**Files:**
- Modify: `scripts/workflow/*.py`
- Modify: `docs/20-contracts/data-contracts.md`
- Modify: `docs/20-contracts/retrieval-contracts.md`
- Modify: `docs/30-design/abstractions.md`
- Modify: `docs/30-design/cross-dataset-refactor-design.md`
- Modify: `docs/40-operations/implementation-handoff.md`
- Modify: remaining tests from source guard output

- [ ] **Step 1: Keep CLI file-role flags stable**

Do not rename `--output_input`, `--output_labels`, `--tasks`, or `--labels` in this task. Update descriptions so they say HotpotQA ranking records and HotpotQA label records.

- [ ] **Step 2: Update contracts and architecture docs**

Document active HotpotQA prepared artifacts separately from stable requests:

```text
HotpotQARankingRecord -> HotpotQA-owned prepared dataset artifact
HotpotQALabelRecord -> HotpotQA-owned label artifact
TextRankingRequest / GraphBuildRequest / GraphRankingRequest / TemporalMemoryRankingRequest / EvidenceEvaluationRequest -> stable consumer contracts
```

Replace first-implementation wording with:

```text
Dataset-specific record -> dataset-specific projector -> consumer request
```

Keep a note that a shared task view can be introduced later only if multiple projectors genuinely share one stable intermediate semantic model.

- [ ] **Step 3: Run source guard and workflow tests**

Run:

```powershell
rg "\bMemoryTaskInput\b|\bMemoryTaskLabels\b|\bMemoryItem\b|\bCombinedMemoryTask\b" graph_memory scripts tests
uv run pytest tests/test_workflow_orchestration.py tests/test_config_run_retrieval.py tests/test_retrieval_domain_boundaries.py -q
```

Expected result: `rg` has no output and tests PASS.

## Task 11: Full Verification

**Files:**
- No planned code edits in this task.

- [ ] **Step 1: Run focused changed-area tests**

Run outside sandbox:

```powershell
uv run pytest tests/test_request_first_projection_boundaries.py tests/test_phase1_real_validation.py tests/test_phase1_real_graphs.py tests/test_phase1_real_retrieval.py tests/test_phase1_real_evaluation.py tests/test_dense_finetune_data.py tests/test_phase2_rgcn_training.py tests/test_memory_stream_method.py -q
```

Expected result: PASS.

- [ ] **Step 2: Run full verification**

Run outside sandbox unless the command is `openspec`:

```powershell
uv run pytest tests -q
uv run ruff check
uv run basedpyright --outputjson --level error
openspec validate --all --strict
```

Expected result: all commands PASS.

- [ ] **Step 3: Final scans**

Run:

```powershell
rg "\bMemoryTaskInput\b|\bMemoryTaskLabels\b|\bMemoryItem\b|\bCombinedMemoryTask\b" graph_memory scripts tests
rg "validate_memory_task_inputs|validate_memory_task_labels|combined_memory_tasks" graph_memory scripts tests
rg "EvidenceRankingView|EvidenceEvalView|EvidenceCandidate" graph_memory scripts tests
```

Expected result: all three commands produce no output.

## Review Checklist

- Old `MemoryTask*` names are absent from `graph_memory`, `scripts`, and `tests`.
- No `EvidenceRankingView` intermediate layer is introduced.
- Dataset-side structures are explicitly HotpotQA-owned.
- Projection functions live under `graph_memory.datasets.hotpotqa.projectors`.
- BM25 and Dense consume `TextRankingRequest`.
- Graph builder consumes `GraphBuildRequest`.
- Graph Rerank and R-GCN consume `GraphRankingRequest`.
- Memory Stream consumes `TemporalMemoryRankingRequest`.
- Evaluation consumes `EvidenceEvaluationRequest`.
- Label-only fields do not enter retrieval or graph build requests.
- Future datasets can add their own records and projectors without modifying BM25/Dense/GraphRerank core logic.

