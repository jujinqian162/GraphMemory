from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import graph_memory.models.graph_retriever.inference as inference_module
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.models.graph_retriever.config.defaults import default_model_config
from graph_memory.models.graph_retriever.inference import GraphRetrieverInference
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult
from graph_memory.retrieval.execution.requests import RetrievalExecutionTask
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.requests import (
    GraphRankingRequest,
    RankingMethodRequest,
    TemporalMemoryRankingRequest,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.retrieval.signals import SeedSignal
from graph_memory.validation import ContractValidationError
from graph_memory.validation.metrics import validate_metric_rows


def _text_request(task_id: str = "task_1") -> TextRankingRequest:
    return TextRankingRequest(
        task_id=task_id,
        query_text="Where is the answer?",
        candidates=(
            TextCandidate(item_id="m0", text="Answer evidence.", metadata={}),
            TextCandidate(item_id="m1", text="Distractor.", metadata={}),
        ),
    )


def _graph(task_id: str = "task_1", *, edge_target: str = "m0") -> MemoryGraph:
    return {
        "task_id": task_id,
        "nodes": [
            {"id": "q", "node_type": "question", "text": "Where is the answer?"},
            {
                "id": "m0",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": "Answer evidence.",
                "source_ref": "A",
                "group_key": "document:A",
                "sequence_index": 0,
                "metadata": {},
            },
            {
                "id": "m1",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": "Distractor.",
                "source_ref": "B",
                "group_key": "document:B",
                "sequence_index": 0,
                "metadata": {},
            },
        ],
        "edges": [
            {"source": "q", "target": edge_target, "edge_type": "query_overlap", "weight": 1.0, "directed": True}
        ],
    }


def test_retrieval_execution_service_has_no_concrete_method_dispatch() -> None:
    source = Path("graph_memory/retrieval/execution/service.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []
    isinstance_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            isinstance_lines.append(node.lineno)

    forbidden_modules = {
        "graph_memory.retrieval.methods.graph_rerank.method",
        "graph_memory.retrieval.methods.memory_stream.method",
        "graph_memory.retrieval.methods.trainable_graph",
    }
    assert forbidden_modules.isdisjoint(imported_modules)
    assert isinstance_lines == []


class CapturingGraphMethod:
    name = "bm25_graph_rerank"

    def __init__(self) -> None:
        self.seen_request: GraphRankingRequest | None = None

    def rank_task(self, request: RankingMethodRequest, *, top_k: int) -> RetrievalMethodResult:
        assert isinstance(request, GraphRankingRequest)
        self.seen_request = request
        return RetrievalMethodResult(ranked_nodes=[RankedNode(node_id="m0", score=1.0), RankedNode(node_id="m1", score=0.0)])


def test_retrieval_execution_passes_prebuilt_graph_request() -> None:
    text_request = _text_request()
    graph_request = GraphRankingRequest(
        task_id=text_request.task_id,
        query_text=text_request.query_text,
        candidates=text_request.candidates,
        graph=_graph(),
        initial_scores={"m0": 0.8, "m1": 0.2},
    )
    method = CapturingGraphMethod()

    predictions = run_retrieval(
        retrieval_method=method,
        tasks=[RetrievalExecutionTask(text_request=text_request, method_request=graph_request)],
        top_k=1,
    )

    assert method.seen_request is graph_request
    assert predictions[0]["method"] == "bm25_graph_rerank"


class CapturingTemporalMethod:
    name = "memory_stream"

    def __init__(self) -> None:
        self.seen_request: TemporalMemoryRankingRequest | None = None

    def rank_task(self, request: RankingMethodRequest, *, top_k: int) -> RetrievalMethodResult:
        assert isinstance(request, TemporalMemoryRankingRequest)
        self.seen_request = request
        return RetrievalMethodResult(ranked_nodes=[RankedNode(node_id="m1", score=2.0), RankedNode(node_id="m0", score=1.0)])


def test_retrieval_execution_passes_prebuilt_temporal_request() -> None:
    text_request = _text_request()
    temporal_request = TemporalMemoryRankingRequest(
        task_id=text_request.task_id,
        query_text=text_request.query_text,
        candidates=text_request.candidates,
        importance_by_item_id={"m0": 0.25, "m1": 0.75},
        metadata={"turn_index": 3},
    )
    method = CapturingTemporalMethod()

    predictions = run_retrieval(
        retrieval_method=method,
        tasks=[RetrievalExecutionTask(text_request=text_request, method_request=temporal_request)],
        top_k=1,
    )

    assert method.seen_request is temporal_request
    assert predictions[0]["ranked_nodes"][0]["node_id"] == "m1"


class _StopTensorization(Exception):
    pass


class StubTextEmbeddingProvider:
    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_task_nodes(self, request: TextRankingRequest, node_ids: list[str]) -> torch.Tensor:
        _ = request
        return torch.zeros((len(node_ids), self.embedding_dim), dtype=torch.float32)


class StubSeedSignalProvider:
    def score_task(self, request: TextRankingRequest) -> list[SeedSignal]:
        _ = request
        return []


def test_rgcn_inference_uses_request_graph_instead_of_cached_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    request_graph = _graph("task_1", edge_target="m0")
    cached_graph = _graph("task_1", edge_target="m1")
    captured: dict[str, object] = {}

    def fake_build_full_ranking_batches(**kwargs: object) -> object:
        captured["graphs"] = kwargs["graphs"]
        raise _StopTensorization

    monkeypatch.setattr(inference_module, "build_full_ranking_batches", fake_build_full_ranking_batches)
    inference = GraphRetrieverInference(
        name="dense_rgcn_graph_retriever",
        model=torch.nn.Identity(),
        model_config=default_model_config(
            encoder_model="fake-encoder",
            encoder_dim=4,
            query_prefix="query: ",
            passage_prefix="passage: ",
            encoder_batch_size=64,
        ),
        graph_by_task_id={"task_1": cached_graph},
        text_embedding_provider=StubTextEmbeddingProvider(),
        seed_signal_provider=StubSeedSignalProvider(),
        device=torch.device("cpu"),
    )

    with pytest.raises(_StopTensorization):
        inference.rank_task(
            GraphRankingRequest(
                task_id="task_1",
                query_text="Where is the answer?",
                candidates=_text_request().candidates,
                graph=request_graph,
                initial_scores={"m0": 1.0, "m1": 0.0},
            ),
            top_k=1,
        )

    assert captured["graphs"] == [request_graph]


def _evidence_metric_row() -> dict[str, object]:
    return {
        "Method": "bm25",
        "Recall@2": 1.0,
        "Recall@5": 1.0,
        "Recall@10": 1.0,
        "Evidence F1@5": 1.0,
        "Evidence F1@10": 1.0,
        "Full Support@5": 1.0,
        "Full Support@10": 1.0,
        "MRR": 1.0,
        "Connected Evidence Recall@5": 1.0,
        "Connected Evidence Recall@10": 1.0,
        "Query-Evidence Connectivity@10": 1.0,
        "Path Recall@10": "N/A",
        "Edge Recall@10": "N/A",
        "Retrieval Latency / Query": 0.0,
    }


def test_evidence_metric_validation_remains_strict() -> None:
    row = _evidence_metric_row()
    del row["Evidence F1@10"]

    with pytest.raises(ContractValidationError, match="Evidence F1@10"):
        validate_metric_rows([row])


class TurnSupportMetricSuite:
    name = "turn_support"

    def validate_metric_rows(self, rows: object) -> None:
        if not isinstance(rows, list):
            raise ContractValidationError("Invalid turn metric rows: expected list.")
        for row in rows:
            if not isinstance(row, dict) or "Turn Support@5" not in row:
                raise ContractValidationError("Invalid turn metric rows: missing Turn Support@5.")


def test_non_evidence_metric_suite_defines_its_own_columns() -> None:
    validate_metric_rows(
        [{"Method": "longmem_dense", "Turn Support@5": 1.0}],
        metric_suite=TurnSupportMetricSuite(),
    )
