from __future__ import annotations


from graph_memory.analysis.efficiency import benchmark_retrieval
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult
from graph_memory.retrieval.requests import (
    RankingMethodRequest,
    TextCandidate,
    TextRankingRequest,
)


class _FixedRetriever:
    name = "fixed"

    def rank_task(
        self, request: RankingMethodRequest, *, top_k: int
    ) -> RetrievalMethodResult:
        del top_k
        assert isinstance(request, TextRankingRequest)
        return _ranking(request)


def _ranking(request: TextRankingRequest) -> RetrievalMethodResult:
    return RetrievalMethodResult(
        ranked_nodes=tuple(
            RankedNode(node_id=candidate.item_id, score=float(3 - index))
            for index, candidate in enumerate(request.candidates)
        )
    )


def _requests() -> list[TextRankingRequest]:
    return [
        TextRankingRequest(
            task_id=f"q{task_index}",
            query_text="query",
            candidates=tuple(
                TextCandidate(
                    item_id=f"q{task_index}-c{index}", text="text", metadata={}
                )
                for index in range(3)
            ),
        )
        for task_index in range(2)
    ]


def test_benchmark_retrieval_reports_latency_and_throughput() -> None:
    result = benchmark_retrieval(
        retrieval_method=_FixedRetriever(),
        requests=_requests(),
        top_k=2,
        warmup_queries=1,
        repeats=2,
        device="cpu",
    )

    assert result.task_count == 2
    assert result.warmup_queries == 1
    assert result.repeats == 2
    assert result.latency.p95_ms >= 0.0
    assert len(result.throughput.repeat_seconds) == 2
    assert result.throughput.mean_queries_per_second > 0.0
    assert result.device_memory.peak_allocated_bytes == 0
