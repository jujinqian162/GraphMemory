from __future__ import annotations

import time

from tqdm.auto import tqdm

from graph_memory.retrieval.contracts import RetrievalMethod
from graph_memory.retrieval.requests import RankingMethodRequest
from graph_memory.retrieval.execution.results import assemble_ranked_result
from graph_memory.retrieval.results import RankedResult


def run_retrieval(
    *,
    retrieval_method: RetrievalMethod,
    requests: list[RankingMethodRequest],
    top_k: int,
) -> list[RankedResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    predictions: list[RankedResult] = []
    for request in tqdm(requests, desc="retrieval", unit="query"):
        started = time.perf_counter()
        result = retrieval_method.rank_task(request, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        prediction = assemble_ranked_result(
            request=request,
            method=retrieval_method.name,
            ranked_nodes=result.ranked_nodes,
            top_k=top_k,
            latency_ms=latency_ms,
            retrieved_edges=result.trace.retrieved_edges,
            native_trace=result.trace.native_trace,
        )
        predictions.append(prediction)

    return predictions
