from __future__ import annotations

import time

from tqdm.auto import tqdm

from graph_memory.retrieval.contracts import RetrievalMethod
from graph_memory.retrieval.execution.requests import RetrievalExecutionTask
from graph_memory.retrieval.execution.results import assemble_ranked_result
from graph_memory.retrieval.results import RankedResult


def run_retrieval(
    *,
    retrieval_method: RetrievalMethod,
    tasks: list[RetrievalExecutionTask],
    top_k: int,
) -> list[RankedResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    predictions: list[RankedResult] = []
    for task in tqdm(tasks, desc="retrieval", unit="query"):
        started = time.perf_counter()
        result = retrieval_method.rank_task(task.method_request, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        prediction = assemble_ranked_result(
            text_request=task.text_request,
            method=retrieval_method.name,
            ranked_nodes=result.ranked_nodes,
            top_k=top_k,
            latency_ms=latency_ms,
            retrieved_edges=result.trace.retrieved_edges,
            native_trace=result.trace.native_trace,
        )
        predictions.append(prediction)

    return predictions
