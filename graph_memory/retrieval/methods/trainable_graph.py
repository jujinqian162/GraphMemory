from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from graph_memory.models.graph_retriever.batching import (
    collate_evidence_tasks,
    move_training_batch,
    split_batch_node_scores,
)
from graph_memory.models.graph_retriever.config.records import RgcnModelConfig
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.inference import (
    CheckpointGraphRetrieverLoader,
    GraphRetrieverInference,
)
from graph_memory.models.graph_retriever.internals.neural import EvidenceScoringModel
from graph_memory.models.graph_retriever.provenance import (
    PROVENANCE_RELATION_VOCAB,
    tensorize_provenance_ranking_task,
)
from graph_memory.retrieval.contracts import (
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
)
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.retrieval.requests import (
    EvidenceGraphRankingRequest,
    ProvenanceRgcnRequest,
    RankingMethodRequest,
)
from graph_memory.retrieval.signals import SeedSignalProvider


@dataclass(frozen=True)
class TrainableGraphRetrievalMethod:
    """
    Retrieval-owned adapter for checkpoint-backed graph retriever inference.
    checkpoint-backed graph retriever inference 的 retrieval 适配器。
    """

    name: str
    inference: GraphRetrieverInference

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        text_embedding_provider: TextEmbeddingProvider,
        seed_signal_provider: SeedSignalProvider,
        device: str | torch.device,
        expected_method: RetrievalMethodId = (
            RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER
        ),
    ) -> "TrainableGraphRetrievalMethod":
        inference = CheckpointGraphRetrieverLoader().load(
            checkpoint_path,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
            device=device,
            expected_method=expected_method,
        )
        return cls(name=inference.name, inference=inference)

    def rank_task(
        self, request: RankingMethodRequest, *, top_k: int
    ) -> RetrievalMethodResult:
        if not isinstance(request, EvidenceGraphRankingRequest):
            raise TypeError(
                f"{self.name} requires EvidenceGraphRankingRequest, got {type(request).__name__}."
            )
        return self.inference.rank_task(request, top_k=top_k)


@dataclass(frozen=True)
class ProvenanceRgcnRetrievalMethod:
    name: str
    model: EvidenceScoringModel
    model_config: RgcnModelConfig
    text_embedding_provider: TextEmbeddingProvider
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        text_embedding_provider: TextEmbeddingProvider,
        device: str | torch.device,
    ) -> "ProvenanceRgcnRetrievalMethod":
        from graph_memory.models.graph_retriever.checkpoint import (
            load_rgcn_checkpoint,
        )

        run_device = torch.device(device)
        checkpoint = load_rgcn_checkpoint(
            checkpoint_path,
            expected_method=RetrievalMethodId.PROVENANCE_RGCN,
            map_location=run_device,
        )
        config = checkpoint.model_config
        if config.relation_vocab != PROVENANCE_RELATION_VOCAB:
            raise ValueError("provenance checkpoint relation vocabulary is invalid")
        if (
            config.edge_weight_policy != "uniform"
            or config.enabled_edge_types
            or config.feature_config.node_feature_names
            or config.feature_config.scorer_feature_names
        ):
            raise ValueError("provenance checkpoint enables forbidden graph features")
        model = build_model_from_config(config).to(run_device)
        model.load_state_dict(checkpoint.payload["model_state_dict"])
        model.eval()
        return cls(
            name=RetrievalMethodId.PROVENANCE_RGCN.value,
            model=model,
            model_config=config,
            text_embedding_provider=text_embedding_provider,
            device=run_device,
        )

    def rank_task(
        self, request: RankingMethodRequest, *, top_k: int
    ) -> RetrievalMethodResult:
        del top_k
        if not isinstance(request, ProvenanceRgcnRequest):
            raise TypeError(
                f"{self.name} requires ProvenanceRgcnRequest, "
                f"got {type(request).__name__}."
            )
        task = tensorize_provenance_ranking_task(
            request,
            model_config=self.model_config,
            text_embedding_provider=self.text_embedding_provider,
        )
        batch = collate_evidence_tasks([task])
        with torch.no_grad():
            logits = self.model(move_training_batch(batch, self.device))
        rows = split_batch_node_scores(batch, logits)[request.task_id]
        ranked_nodes = tuple(
            RankedNode(node_id=node_id, score=score)
            for node_id, score in sorted(rows, key=lambda row: (-row[1], row[0]))
        )
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(),
        )


__all__ = ["ProvenanceRgcnRetrievalMethod", "TrainableGraphRetrievalMethod"]
