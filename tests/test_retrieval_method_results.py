from __future__ import annotations

from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult, RetrievalTrace, SeedRanker
from graph_memory.retrieval.methods.flat.method import ScorePipelineMethod
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest


class FakeSeedRanker:
    method_name = "fake"

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        return [RankedNode(node_id="m0", score=1.0)]


def tiny_text_request() -> TextRankingRequest:
    return TextRankingRequest(
        task_id="task",
        query_text="question",
        candidates=(TextCandidate(item_id="m0", text="answer", metadata={"title": "A"}),),
    )


def test_seed_ranker_protocol_names_full_ranking_boundary() -> None:
    ranker: SeedRanker = FakeSeedRanker()

    assert ranker.rank(tiny_text_request()) == [RankedNode(node_id="m0", score=1.0)]


def test_flat_score_pipeline_method_returns_structured_result_with_empty_trace() -> None:
    method = ScorePipelineMethod(name="fake", retriever=FakeSeedRanker())

    result = method.rank_task(tiny_text_request(), top_k=1)

    assert result == RetrievalMethodResult(
        ranked_nodes=[RankedNode(node_id="m0", score=1.0)],
        trace=RetrievalTrace(),
    )
    assert result.trace.retrieved_edges == []
