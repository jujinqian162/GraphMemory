from __future__ import annotations

import time

from graph_memory.contracts.ranking import RankedResult
from graph_memory.retrieval.contracts import RetrievalMethod
from graph_memory.retrieval.execution.requests import RetrievalExecutionTask
from graph_memory.retrieval.execution.results import assemble_ranked_result
from graph_memory.validation import validate_ranked_results


def run_retrieval(
    *,
    retrieval_method: RetrievalMethod,
    tasks: list[RetrievalExecutionTask],
    top_k: int,
) -> list[RankedResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    predictions: list[RankedResult] = []
    for task in tasks:
        started = time.perf_counter()
        result = retrieval_method.rank_task(task.method_request, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        predictions.append(
            assemble_ranked_result(
                text_request=task.text_request,
                method=retrieval_method.name,
                ranked_nodes=result.ranked_nodes,
                top_k=top_k,
                latency_ms=latency_ms,
                retrieved_edges=result.trace.retrieved_edges,
            )
        )

    validate_ranked_results(predictions, [task.text_request for task in tasks])
    return predictions