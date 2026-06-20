from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTextRankingRequest
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.datasets.hotpotqa.records import HotpotQALabelRecord, HotpotQARankingRecord
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.requests import DenseRuntime, TextRankingRequest
from graph_memory.retrieval.signals import SeedSignal
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.retrieval.tuning import seed_scores
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.requests import TrainPairBuildTask
from graph_memory.training_pairs.config import NegativeSamplingConfig


def _task(task_id: str) -> HotpotQARankingRecord:
    return {
        "task_id": task_id,
        "question": f"query-{task_id}",
        "candidate_sentences": [
            {
                "sentence_id": f"m{index}",
                "title": f"source-{index}",
                "sentence_index": index,
                "position": index,
                "text": f"text-{index}",
            }
            for index in range(3)
        ],
    }


def _graph(task: HotpotQARankingRecord) -> MemoryGraph:
    return {
        "task_id": task["task_id"],
        "nodes": [
            {"id": "q", "node_type": "question", "text": task["question"]},
            *[
                {
                    "id": sentence["sentence_id"],
                    "node_type": "graph_item",
                    "node_kind": "document_sentence",
                    "text": sentence["text"],
                    "source_ref": sentence["title"],
                    "group_key": f"document:{sentence['title']}",
                    "sequence_index": sentence["sentence_index"],
                    "metadata": {"title": sentence["title"], "position": sentence["position"]},
                }
                for sentence in task["candidate_sentences"]
            ],
        ],
        "edges": [],
    }


def _ranking_requests(tasks: Sequence[HotpotQARankingRecord]) -> list[TextRankingRequest]:
    projector = HotpotQAToTextRankingRequest()
    return [projector.project(task) for task in tasks]


def _evidence_labels(labels: Sequence[HotpotQALabelRecord]) -> list[EvidenceLabel]:
    return [
        EvidenceLabel(
            task_id=label["task_id"],
            gold_answer=label["gold_answer"],
            gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
            gold_dependency_edges=tuple((edge[0], edge[1]) for edge in label["gold_dependency_edges"]),
        )
        for label in labels
    ]


def _pair_tasks(
    tasks: Sequence[HotpotQARankingRecord],
    labels: Sequence[HotpotQALabelRecord],
    graphs: Sequence[MemoryGraph],
) -> list[TrainPairBuildTask]:
    requests = _ranking_requests(tasks)
    evidence_labels = _evidence_labels(labels)
    return [
        TrainPairBuildTask(text_request=request, label=label, graph=graph)
        for request, label, graph in zip(requests, evidence_labels, graphs, strict=True)
    ]


class BulkRanker:
    method_name = "dense"

    def __init__(self) -> None:
        self.bulk_calls: list[list[str]] = []

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        raise AssertionError("single rank path must not run")

    def rank_many(self, requests: list[TextRankingRequest]) -> list[list[RankedNode]]:
        self.bulk_calls.append([request.task_id for request in requests])
        return [
            [
                RankedNode(node_id="m1", score=0.9),
                RankedNode(node_id="m2", score=0.8),
                RankedNode(node_id="m0", score=0.1),
            ]
            for _ in requests
        ]


def test_seed_score_precompute_uses_bounded_bulk_ranking_and_preserves_maps(monkeypatch) -> None:
    tasks = [_task(f"t{index:02d}") for index in range(33)]
    ranker = BulkRanker()
    registry = SimpleNamespace(
        retrieval=SimpleNamespace(build_seed=lambda settings, payload: ranker),
    )
    monkeypatch.setattr(seed_scores, "Registry", registry)

    cache = seed_scores.precompute_seed_score_cache(
        seed_method=RetrievalMethodId.DENSE,
        ranking_requests=_ranking_requests(tasks),
        dense_runtime=DenseRuntime(config=DenseConfig()),
    )

    assert [len(task_ids) for task_ids in ranker.bulk_calls] == [32, 1]
    assert list(cache.scores_by_task_id) == [task["task_id"] for task in tasks]
    assert list(cache.latency_ms_by_task_id) == [task["task_id"] for task in tasks]
    assert cache.scores_by_task_id["t00"] == {"m1": 0.9, "m2": 0.8, "m0": 0.1}
    assert all(latency >= 0.0 for latency in cache.latency_ms_by_task_id.values())


def test_hard_dense_pairs_precompute_bulk_signals_before_sampling_and_preserve_order() -> None:
    tasks = [_task("t0"), _task("t1")]
    labels: list[HotpotQALabelRecord] = [
        {
            "task_id": task["task_id"],
            "gold_answer": "answer",
            "gold_evidence_sentence_ids": ["m0"],
            "gold_dependency_edges": [],
        }
        for task in tasks
    ]
    graphs = [_graph(task) for task in tasks]

    class BulkSeedProvider:
        def __init__(self) -> None:
            self.bulk_calls: list[list[str]] = []

        def score_task(self, request: TextRankingRequest) -> list[SeedSignal]:
            raise AssertionError("single seed path must not run")

        def score_tasks(self, requests: Sequence[TextRankingRequest]) -> list[list[SeedSignal]]:
            self.bulk_calls.append([request.task_id for request in requests])
            return [
                [
                    SeedSignal(node_id="m1", score=0.9, rank=1, rank_percentile=0.0),
                    SeedSignal(node_id="m1", score=0.9, rank=1, rank_percentile=0.0),
                    SeedSignal(node_id="m2", score=0.8, rank=2, rank_percentile=0.5),
                    SeedSignal(node_id="m0", score=0.1, rank=3, rank_percentile=1.0),
                ]
                for _ in requests
            ]

    provider = BulkSeedProvider()
    result = build_train_pairs(
        _pair_tasks(tasks, labels, graphs),
        NegativeSamplingConfig(
            random_seed=7,
            easy_random_per_positive=0,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=1,
            hard_graph_neighbor_per_positive=0,
            hard_pool_size=2,
        ),
        dense_seed_signal_provider=provider,
    )

    assert provider.bulk_calls == [["t0", "t1"]]
    assert result.pairs == [
        {"task_id": "t0", "node_id": "m0", "label": 1, "sample_type": "positive"},
        {"task_id": "t0", "node_id": "m1", "label": 0, "sample_type": "hard_dense"},
        {"task_id": "t1", "node_id": "m0", "label": 1, "sample_type": "positive"},
        {"task_id": "t1", "node_id": "m1", "label": 0, "sample_type": "hard_dense"},
    ]
    assert result.summary["negative_count_by_type"] == {"hard_dense": 2}
