from __future__ import annotations

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult, RetrievalTrace, SeedRanker
from graph_memory.retrieval.methods.flat.method import ScorePipelineMethod


class FakeSeedRanker:
    method_name = "fake"

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        return [RankedNode(node_id="m0", score=1.0)]


def tiny_task_input() -> MemoryTaskInput:
    return {
        "task_id": "task",
        "query": "question",
        "memory_items": [
            {
                "id": "m0",
                "node_type": "document_sentence",
                "text": "answer",
                "source": "A",
                "sentence_id": 0,
                "position": 0,
            }
        ],
    }


def test_seed_ranker_protocol_names_full_ranking_boundary() -> None:
    ranker: SeedRanker = FakeSeedRanker()

    assert ranker.rank(tiny_task_input()) == [RankedNode(node_id="m0", score=1.0)]


def test_flat_score_pipeline_method_returns_structured_result_with_empty_trace() -> None:
    method = ScorePipelineMethod(name="fake", retriever=FakeSeedRanker())

    result = method.rank_task(tiny_task_input(), top_k=1)

    assert result == RetrievalMethodResult(
        ranked_nodes=[RankedNode(node_id="m0", score=1.0)],
        trace=RetrievalTrace(),
    )
    assert result.trace.retrieved_edges == []
