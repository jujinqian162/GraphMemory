from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.requests import DenseRuntime
from graph_memory.retrieval.signals import SeedSignal
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.retrieval.tuning import seed_scores
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.config import NegativeSamplingConfig


def _task(task_id: str) -> MemoryTaskInput:
    return {
        "task_id": task_id,
        "query": f"query-{task_id}",
        "memory_items": [
            {
                "id": f"m{index}",
                "node_type": "document_sentence",
                "text": f"text-{index}",
                "source": f"source-{index}",
                "sentence_id": index,
                "position": index,
            }
            for index in range(3)
        ],
    }


class BulkRanker:
    method_name = "dense"

    def __init__(self) -> None:
        self.bulk_calls: list[list[str]] = []

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        raise AssertionError("single rank path must not run")

    def rank_many(self, task_inputs: list[MemoryTaskInput]) -> list[list[RankedNode]]:
        self.bulk_calls.append([task_input["task_id"] for task_input in task_inputs])
        return [
            [
                RankedNode(node_id="m1", score=0.9),
                RankedNode(node_id="m2", score=0.8),
                RankedNode(node_id="m0", score=0.1),
            ]
            for _ in task_inputs
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
        task_inputs=tasks,
        dense_runtime=DenseRuntime(config=DenseConfig()),
    )

    assert [len(task_ids) for task_ids in ranker.bulk_calls] == [32, 1]
    assert list(cache.scores_by_task_id) == [task["task_id"] for task in tasks]
    assert list(cache.latency_ms_by_task_id) == [task["task_id"] for task in tasks]
    assert cache.scores_by_task_id["t00"] == {"m1": 0.9, "m2": 0.8, "m0": 0.1}
    assert all(latency >= 0.0 for latency in cache.latency_ms_by_task_id.values())


def test_hard_dense_pairs_precompute_bulk_signals_before_sampling_and_preserve_order() -> None:
    tasks = [_task("t0"), _task("t1")]
    labels: list[MemoryTaskLabels] = [
        {
            "task_id": task["task_id"],
            "gold_answer": "answer",
            "gold_evidence_nodes": ["m0"],
            "gold_dependency_edges": [],
        }
        for task in tasks
    ]
    graphs: list[MemoryGraph] = [
        {
            "task_id": task["task_id"],
            "nodes": [{"id": "q", "node_type": "question", "text": task["query"]}, *task["memory_items"]],
            "edges": [],
        }
        for task in tasks
    ]

    class BulkSeedProvider:
        def __init__(self) -> None:
            self.bulk_calls: list[list[str]] = []

        def score_task(self, task_input: MemoryTaskInput) -> list[SeedSignal]:
            raise AssertionError("single seed path must not run")

        def score_tasks(self, task_inputs: Sequence[MemoryTaskInput]) -> list[list[SeedSignal]]:
            self.bulk_calls.append([task_input["task_id"] for task_input in task_inputs])
            return [
                [
                    SeedSignal(node_id="m1", score=0.9, rank=1, rank_percentile=0.0),
                    SeedSignal(node_id="m1", score=0.9, rank=1, rank_percentile=0.0),
                    SeedSignal(node_id="m2", score=0.8, rank=2, rank_percentile=0.5),
                    SeedSignal(node_id="m0", score=0.1, rank=3, rank_percentile=1.0),
                ]
                for _ in task_inputs
            ]

    provider = BulkSeedProvider()
    result = build_train_pairs(
        tasks,
        labels,
        graphs,
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
