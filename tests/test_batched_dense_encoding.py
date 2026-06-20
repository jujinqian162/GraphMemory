from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTextRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord
from graph_memory.embeddings import DenseEncodingService, DenseTaskEncodingRequest
from graph_memory.models.graph_retriever.contracts import encode_task_node_groups
from graph_memory.retrieval.bulk import rank_tasks
from graph_memory.retrieval.contracts import (
    RankedNode,
    RetrievalMethodResult,
)
from graph_memory.retrieval.execution import service as retrieval_service
from graph_memory.retrieval.execution.requests import RetrievalExecutionTask
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.requests import RankingMethodRequest, TextRankingRequest
from graph_memory.retrieval.signals import SeedSignal, score_tasks


def _record(task_id: str, *, query: str, node_ids: Sequence[str]) -> HotpotQARankingRecord:
    return {
        "task_id": task_id,
        "question": query,
        "candidate_sentences": [
            {
                "sentence_id": node_id,
                "text": f"text-{node_id}",
                "title": f"source-{node_id}",
                "sentence_index": index,
                "position": index,
            }
            for index, node_id in enumerate(node_ids)
        ],
    }


def _request(record: HotpotQARankingRecord) -> TextRankingRequest:
    return HotpotQAToTextRankingRequest().project(record)


class RecordingEncoder:
    def __init__(self, vectors_by_text: Mapping[str, Sequence[float]] | None = None) -> None:
        self.vectors_by_text = vectors_by_text
        self.calls: list[tuple[list[str], int, bool]] = []

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> object:
        text_list = list(texts)
        self.calls.append((text_list, batch_size, normalize_embeddings))
        if self.vectors_by_text is not None:
            return np.asarray([self.vectors_by_text[text] for text in text_list], dtype=float)
        return np.asarray([[float(index), float(len(text))] for index, text in enumerate(text_list)], dtype=float)

    def get_sentence_embedding_dimension(self) -> int:
        return 2


def test_dense_encoding_service_flattens_variable_length_tasks_and_restores_order() -> None:
    encoder = RecordingEncoder()
    service = DenseEncodingService(
        encoder=encoder,
        query_prefix="Q: ",
        passage_prefix="P: ",
        batch_size=7,
    )
    first = _request(_record("t1", query="first", node_ids=["m0", "m1"]))
    second = _request(_record("t2", query="second", node_ids=["m2", "m3", "m4"]))

    results = service.encode_tasks(
        [
            DenseTaskEncodingRequest(ranking_request=first, node_ids=("q", "m1")),
            DenseTaskEncodingRequest(ranking_request=second, node_ids=("m4", "q", "m2")),
        ]
    )

    assert encoder.calls == [
        (
            [
                "Q: first",
                "P: source-m1. text-m1",
                "P: source-m4. text-m4",
                "Q: second",
                "P: source-m2. text-m2",
            ],
            7,
            True,
        )
    ]
    assert [result.task_id for result in results] == ["t1", "t2"]
    assert [result.node_ids for result in results] == [("q", "m1"), ("m4", "q", "m2")]
    np.testing.assert_array_equal(results[0].embeddings, np.asarray([[0.0, 8.0], [1.0, 21.0]]))
    np.testing.assert_array_equal(
        results[1].embeddings,
        np.asarray([[2.0, 21.0], [3.0, 9.0], [4.0, 21.0]]),
    )
    assert service.embedding_dim == 2


@pytest.mark.parametrize(
    "returned",
    [
        np.zeros((1, 2), dtype=float),
        np.zeros((2,), dtype=float),
        np.zeros((2, 0), dtype=float),
    ],
)
def test_dense_encoding_service_rejects_invalid_encoder_shapes(returned: np.ndarray) -> None:
    class InvalidShapeEncoder(RecordingEncoder):
        def encode(
            self,
            texts: Sequence[str],
            batch_size: int = 64,
            normalize_embeddings: bool = True,
        ) -> object:
            self.calls.append((list(texts), batch_size, normalize_embeddings))
            return returned

    service = DenseEncodingService(encoder=InvalidShapeEncoder(), batch_size=3)
    request = _request(_record("t", query="question", node_ids=["m0"]))

    with pytest.raises(ValueError, match="embedding shape"):
        service.encode_tasks([DenseTaskEncodingRequest(ranking_request=request, node_ids=("q", "m0"))])


def test_dense_encoding_service_prefers_current_embedding_dimension_api() -> None:
    class CurrentDimensionEncoder(RecordingEncoder):
        def get_embedding_dimension(self) -> int:
            return 2

        def get_sentence_embedding_dimension(self) -> int:
            raise AssertionError("deprecated dimension API should not be called")

    service = DenseEncodingService(encoder=CurrentDimensionEncoder(), batch_size=3)

    assert service.embedding_dim == 2


def test_dense_retriever_bulk_and_single_paths_preserve_scores_order_and_tie_breaks() -> None:
    first = _request(_record("t1", query="first", node_ids=["m1", "m0", "m2"]))
    second = _request(_record("t2", query="second", node_ids=["m3"]))
    vectors = {
        "query: first": [1.0, 0.0],
        "passage: source-m1. text-m1": [0.5, 0.0],
        "passage: source-m0. text-m0": [0.5, 0.0],
        "passage: source-m2. text-m2": [0.0, 1.0],
        "query: second": [0.0, 1.0],
        "passage: source-m3. text-m3": [0.0, 0.75],
    }
    bulk_encoder = RecordingEncoder(vectors)
    retriever = DenseTaskRetriever(config=DenseConfig(batch_size=5), encoder=bulk_encoder)

    bulk_results = retriever.rank_many([first, second])

    assert bulk_encoder.calls == [
        (
            [
                "query: first",
                "passage: source-m1. text-m1",
                "passage: source-m0. text-m0",
                "passage: source-m2. text-m2",
                "query: second",
                "passage: source-m3. text-m3",
            ],
            5,
            True,
        )
    ]
    assert [[node.node_id for node in result] for result in bulk_results] == [["m0", "m1", "m2"], ["m3"]]
    assert [[node.score for node in result] for result in bulk_results] == [[0.5, 0.5, 0.0], [0.75]]

    single_encoder = RecordingEncoder(vectors)
    single_retriever = DenseTaskRetriever(config=DenseConfig(batch_size=5), encoder=single_encoder)
    assert single_retriever.rank(first) == bulk_results[0]
    assert single_encoder.calls == [
        (
            [
                "query: first",
                "passage: source-m1. text-m1",
                "passage: source-m0. text-m0",
                "passage: source-m2. text-m2",
            ],
            5,
            True,
        )
    ]


def test_bulk_capability_helpers_fall_back_in_deterministic_input_order() -> None:
    requests = [
        _request(_record("t1", query="first", node_ids=["m0"])),
        _request(_record("t2", query="second", node_ids=["m1"])),
    ]

    class SingleRanker:
        method_name = "single"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def rank(self, request: TextRankingRequest) -> list[RankedNode]:
            self.calls.append(request.task_id)
            return [RankedNode(node_id=request.candidates[0].item_id, score=1.0)]

    class SingleTextProvider:
        embedding_dim = 2

        def __init__(self) -> None:
            self.calls: list[str] = []

        def encode_task_nodes(self, request: TextRankingRequest, node_ids: list[str]) -> torch.Tensor:
            self.calls.append(request.task_id)
            return torch.full((len(node_ids), 2), float(len(self.calls)), dtype=torch.float32)

    class SingleSeedProvider:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def score_task(self, request: TextRankingRequest) -> list[SeedSignal]:
            self.calls.append(request.task_id)
            node_id = request.candidates[0].item_id
            return [SeedSignal(node_id=node_id, score=1.0, rank=1, rank_percentile=0.0)]

    ranker = SingleRanker()
    text_provider = SingleTextProvider()
    seed_provider = SingleSeedProvider()
    encoding_requests = [
        DenseTaskEncodingRequest(ranking_request=requests[0], node_ids=("q", "m0")),
        DenseTaskEncodingRequest(ranking_request=requests[1], node_ids=("q", "m1")),
    ]

    assert [[node.node_id for node in rows] for rows in rank_tasks(ranker, requests)] == [["m0"], ["m1"]]
    assert ranker.calls == ["t1", "t2"]
    embeddings = encode_task_node_groups(text_provider, encoding_requests)
    assert [tensor[:, 0].tolist() for tensor in embeddings] == [[1.0, 1.0], [2.0, 2.0]]
    assert text_provider.calls == ["t1", "t2"]
    assert [[signal.node_id for signal in rows] for rows in score_tasks(seed_provider, requests)] == [["m0"], ["m1"]]
    assert seed_provider.calls == ["t1", "t2"]


def test_bulk_capability_helpers_prefer_bulk_methods() -> None:
    requests = [
        _request(_record("t1", query="first", node_ids=["m0"])),
        _request(_record("t2", query="second", node_ids=["m1"])),
    ]

    class BulkRanker:
        method_name = "bulk"

        def __init__(self) -> None:
            self.bulk_calls = 0

        def rank(self, request: TextRankingRequest) -> list[RankedNode]:
            raise AssertionError("single rank fallback must not run")

        def rank_many(self, request_batch: list[TextRankingRequest]) -> list[list[RankedNode]]:
            self.bulk_calls += 1
            return [
                [RankedNode(node_id=request.candidates[0].item_id, score=1.0)]
                for request in request_batch
            ]

    class BulkTextProvider:
        embedding_dim = 2

        def encode_task_nodes(self, request: TextRankingRequest, node_ids: list[str]) -> torch.Tensor:
            raise AssertionError("single text fallback must not run")

        def encode_task_node_groups(
            self,
            requests: Sequence[DenseTaskEncodingRequest],
        ) -> list[torch.Tensor]:
            return [torch.ones((len(request.node_ids), 2), dtype=torch.float32) for request in requests]

    class BulkSeedProvider:
        def score_task(self, request: TextRankingRequest) -> list[SeedSignal]:
            raise AssertionError("single seed fallback must not run")

        def score_tasks(self, request_batch: Sequence[TextRankingRequest]) -> list[list[SeedSignal]]:
            return [
                [
                    SeedSignal(
                        node_id=request.candidates[0].item_id,
                        score=1.0,
                        rank=1,
                        rank_percentile=0.0,
                    )
                ]
                for request in request_batch
            ]

    ranker = BulkRanker()
    encoding_requests = [
        DenseTaskEncodingRequest(ranking_request=requests[0], node_ids=("q", "m0")),
        DenseTaskEncodingRequest(ranking_request=requests[1], node_ids=("q", "m1")),
    ]

    assert len(rank_tasks(ranker, requests)) == 2
    assert ranker.bulk_calls == 1
    assert len(encode_task_node_groups(BulkTextProvider(), encoding_requests)) == 2
    assert len(score_tasks(BulkSeedProvider(), requests)) == 2


def test_embeddings_package_stays_low_level_and_encoder_protocol_is_shared() -> None:
    forbidden_import_fragments = (
        "graph_memory.retrieval",
        "graph_memory.models.graph_retriever",
        "graph_memory.registry",
        "graph_memory.stages",
        "scripts",
        "workflow",
    )
    for path in Path("graph_memory/embeddings").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert all(fragment not in source for fragment in forbidden_import_fragments)

    retrieval_contracts = Path("graph_memory/retrieval/contracts.py").read_text(encoding="utf-8")
    graph_contracts = Path("graph_memory/models/graph_retriever/contracts.py").read_text(encoding="utf-8")
    assert "class DenseEncoder" not in retrieval_contracts
    assert "class SentenceEncoder" not in graph_contracts


def test_run_retrieval_keeps_per_task_rank_and_latency_boundaries(monkeypatch) -> None:
    records = [
        _record("t1", query="first", node_ids=["m0"]),
        _record("t2", query="second", node_ids=["m0"]),
    ]

    class TaskOrientedMethod:
        name = "bm25"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def rank_task(self, request: RankingMethodRequest, *, top_k: int) -> RetrievalMethodResult:
            assert isinstance(request, TextRankingRequest)
            self.calls.append(request.task_id)
            return RetrievalMethodResult(
                ranked_nodes=[RankedNode(node_id="m0", score=float(top_k))]
            )

    times = iter([1.0, 1.01, 2.0, 2.03])
    monkeypatch.setattr(retrieval_service.time, "perf_counter", lambda: next(times))
    method = TaskOrientedMethod()

    requests = [_request(record) for record in records]
    predictions = retrieval_service.run_retrieval(
        retrieval_method=method,
        tasks=[RetrievalExecutionTask(text_request=request, method_request=request) for request in requests],
        top_k=1,
    )

    assert method.calls == ["t1", "t2"]
    assert [prediction["latency_ms"] for prediction in predictions] == pytest.approx([10.0, 30.0])
