from __future__ import annotations

from graph_memory.retrieval.contracts import RetrievalMethod, Retriever
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.methods.flat.method import ScorePipelineMethod
from graph_memory.retrieval.methods.graph_rerank.method import GraphRerankMethod
from graph_memory.retrieval.requests import (
    FlatMethodBuildRequest,
    GraphRerankMethodBuildRequest,
    MethodBuildRequest,
    SeedRetrieverBuildRequest,
    TrainableGraphMethodBuildRequest,
)


def build_retrieval_method(request: MethodBuildRequest) -> RetrievalMethod:
    if isinstance(request, FlatMethodBuildRequest):
        return ScorePipelineMethod(
            name=request.method,
            retriever=build_seed_retriever(request.seed_retriever),
        )
    if isinstance(request, GraphRerankMethodBuildRequest):
        return GraphRerankMethod(
            name=request.method,
            retriever=build_seed_retriever(request.seed_retriever),
            graphs=request.graphs,
            graph_config=request.config,
        )
    if isinstance(request, TrainableGraphMethodBuildRequest):
        from graph_memory.retrieval.methods.trainable_graph import TrainableGraphRetrievalMethod

        dense_runtime = request.runtime.dense_runtime
        return TrainableGraphRetrievalMethod.from_checkpoint(
            request.runtime.checkpoint_path,
            graphs=list(request.graphs.graph_by_task_id.values()),
            text_embedding_provider=request.runtime.text_embedding_provider,
            seed_signal_provider=request.runtime.seed_signal_provider,
            dense_encoder=None if dense_runtime is None else dense_runtime.encoder,
            device=request.runtime.device,
        )
    raise ValueError(f"Unsupported retrieval build request: {type(request).__name__}.")


def build_seed_retriever(request: SeedRetrieverBuildRequest) -> Retriever:
    if request.method == "bm25":
        return BM25TaskRetriever()
    if request.method == "dense":
        dense_config = request.dense_runtime.config
        return DenseTaskRetriever(
            config=dense_config,
            encoder=request.dense_runtime.encoder,
        )
    raise ValueError(f"Unsupported retrieval method: {request.method}")
