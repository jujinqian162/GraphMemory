from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.graphs.views import induced_retrieved_subgraph, model_visible_graph
from graph_memory.models.graph_retriever.batching import build_full_ranking_batches, move_training_batch
from graph_memory.models.graph_retriever.checkpoint import load_rgcn_checkpoint
from graph_memory.models.graph_retriever.config.records import RgcnModelConfig
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.factory import GraphScoringModelFactory
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult, RetrievalTrace
from graph_memory.retrieval.requests import GraphRankingRequest, TextRankingRequest
from graph_memory.retrieval.signals import SeedSignal, SeedSignalProvider, seed_signals_from_ranked_nodes


@dataclass(frozen=True)
class GraphRetrieverInference:
    """
    Checkpoint-backed trainable graph retriever inference runtime.
    基于 checkpoint 的可训练图检索推理 runtime。
    """

    name: str
    model: torch.nn.Module
    model_config: RgcnModelConfig
    graph_by_task_id: dict[str, MemoryGraph]
    text_embedding_provider: TextEmbeddingProvider
    seed_signal_provider: SeedSignalProvider
    device: torch.device

    def rank_task(self, request: GraphRankingRequest, *, top_k: int) -> RetrievalMethodResult:
        graph = request.graph
        text_request = TextRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
        )
        batches = build_full_ranking_batches(
            ranking_requests=[text_request],
            graphs=[graph],
            model_config=self.model_config,
            text_embedding_provider=self.text_embedding_provider,
            seed_signal_provider=_PrecomputedGraphRankingSignalProvider(request),
            batch_size=1,
        )
        if len(batches) != 1:
            raise RuntimeError("Expected exactly one full ranking batch.")
        with torch.no_grad():
            batch = move_training_batch(batches[0], self.device)
            logits = self.model(batch).detach().cpu().tolist()
        ranked_nodes = sorted(
            [
                RankedNode(node_id=node_id, score=float(score))
                for node_id, score in zip(batches[0].sample_node_ids, logits)
            ],
            key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id),
        )
        top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:top_k]]
        visible_graph = model_visible_graph(graph, frozenset(self.model_config.enabled_edge_types))
        retrieved_subgraph = induced_retrieved_subgraph(visible_graph, top_node_ids)
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(retrieved_edges=retrieved_subgraph["edges"]),
        )


@dataclass(frozen=True)
class _PrecomputedGraphRankingSignalProvider:
    request: GraphRankingRequest

    def score_task(self, request: TextRankingRequest) -> list[SeedSignal]:
        if request.task_id != self.request.task_id:
            raise ValueError(f"Unexpected graph ranking task_id={request.task_id}.")
        ranked_nodes = [
            RankedNode(node_id=candidate.item_id, score=float(self.request.initial_scores.get(candidate.item_id, 0.0)))
            for candidate in request.candidates
        ]
        return seed_signals_from_ranked_nodes(request, ranked_nodes)

    def score_tasks(self, requests: Sequence[TextRankingRequest]) -> list[list[SeedSignal]]:
        return [self.score_task(request) for request in requests]


@dataclass(frozen=True)
class CheckpointGraphRetrieverLoader:
    """
    Loads checkpoint-backed graph retriever inference without depending on training.
    加载 checkpoint-backed graph retriever inference，不依赖 training 生命周期。
    """

    model_factory: GraphScoringModelFactory = GraphScoringModelFactory()

    def load(
        self,
        checkpoint_path: str | Path,
        *,
        graphs: list[MemoryGraph],
        text_embedding_provider: TextEmbeddingProvider,
        seed_signal_provider: SeedSignalProvider,
        device: str | torch.device = "cpu",
    ) -> GraphRetrieverInference:
        """
        Load a trainable graph retriever inference runtime from `best.pt`.
        从 `best.pt` 加载可训练图检索推理 runtime。
        """

        checkpoint = load_rgcn_checkpoint(
            checkpoint_path,
            expected_method="dense_rgcn_graph_retriever",
            map_location=device,
        )
        model = self.model_factory.build(checkpoint.model_config).to(device)
        model.load_state_dict(checkpoint.payload["model_state_dict"])
        model.eval()

        return GraphRetrieverInference(
            name=checkpoint.model_config.method_name,
            model=model,
            model_config=checkpoint.model_config,
            graph_by_task_id={graph["task_id"]: graph for graph in graphs},
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
            device=torch.device(device),
        )
