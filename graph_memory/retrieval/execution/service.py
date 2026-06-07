from __future__ import annotations

import time

from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.contracts import RetrievalMethod
from graph_memory.retrieval.execution.results import assemble_ranked_result
from graph_memory.validation import (
    validate_memory_task_inputs,
    validate_ranked_results,
)


def run_retrieval(
    *,
    retrieval_method: RetrievalMethod,
    task_inputs: list[MemoryTaskInput],
    top_k: int,
) -> list[RankedResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    validate_memory_task_inputs(task_inputs)

    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    predictions: list[RankedResult] = []
    for task_input in task_inputs:
        started = time.perf_counter()
        result = retrieval_method.rank_task(task_input, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        predictions.append(
            assemble_ranked_result(
                task_input=task_input,
                method=retrieval_method.name,
                ranked_nodes=result.ranked_nodes,
                top_k=top_k,
                latency_ms=latency_ms,
                retrieved_edges=result.trace.retrieved_edges,
            )
        )

    validate_ranked_results(predictions, inputs_by_task_id)
    return predictions
