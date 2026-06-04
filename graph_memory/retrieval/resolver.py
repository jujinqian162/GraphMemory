from __future__ import annotations

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.graphs.index import GraphIndex
from graph_memory.retrieval.methods.graph_rerank.config import ensure_graph_rerank_config
from graph_memory.retrieval.requests import (
    FlatMethodBuildRequest,
    GraphRerankMethodBuildRequest,
    MethodBuildRequest,
    RetrievalMethodResolveRequest,
    SeedRetrieverBuildRequest,
    TrainableGraphMethodBuildRequest,
)
from graph_memory.retrieval.catalog import get_method_spec
from graph_memory.validation import validate_graphs, validate_task_id_alignment


def resolve_method_build_request(request: RetrievalMethodResolveRequest) -> MethodBuildRequest:
    spec = get_method_spec(request.method)
    if spec.builder_id in {"bm25", "dense"}:
        return FlatMethodBuildRequest(
            method=request.method,
            seed_retriever=SeedRetrieverBuildRequest(method=request.method, dense_runtime=request.dense_runtime),
        )
    if spec.builder_id == "graph_rerank":
        if spec.seed_method is None:
            raise ValueError(f"Graph rerank method={request.method} requires a seed method.")
        graph_index = _validated_graph_index(
            method=request.method,
            task_inputs=request.task_inputs,
            graphs=request.graphs,
        )
        return GraphRerankMethodBuildRequest(
            method=request.method,
            seed_retriever=SeedRetrieverBuildRequest(method=spec.seed_method, dense_runtime=request.dense_runtime),
            graphs=graph_index,
            config=ensure_graph_rerank_config(request.graph_config),
        )
    if spec.builder_id == "trainable_graph":
        if request.trainable_runtime is None:
            raise ValueError(f"Trainable graph method={request.method} requires a checkpoint path.")
        return TrainableGraphMethodBuildRequest(
            method=request.method,
            graphs=_validated_graph_index(method=request.method, task_inputs=request.task_inputs, graphs=request.graphs),
            runtime=request.trainable_runtime,
        )
    raise ValueError(f"Unsupported retrieval method builder={spec.builder_id} for method={spec.name}.")


def seed_method_for(method: str) -> str:
    spec = get_method_spec(method)
    return spec.seed_method or method


def _validated_graph_index(
    *,
    method: str,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph] | None,
) -> GraphIndex:
    if not graphs:
        raise ValueError(f"Graph rerank method={method} requires graph inputs.")
    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    validate_graphs(graphs, inputs_by_task_id)
    validate_task_id_alignment(
        "retrieval graph inputs",
        set(inputs_by_task_id),
        {graph["task_id"] for graph in graphs},
    )
    return GraphIndex.from_graphs(graphs)
