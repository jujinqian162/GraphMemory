from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from graph_memory.indexes.dense import DenseTaskRetriever
from graph_memory.learned.batching import build_full_ranking_batches, move_training_batch
from graph_memory.learned.checkpoint import load_trainable_checkpoint
from graph_memory.learned.features import (
    DenseTextEmbeddingProvider,
    RetrieverSeedSignalProvider,
    SeedSignalProvider,
    SentenceEncoder,
    TextEmbeddingProvider,
)
from graph_memory.learned.training import build_model_from_config
from graph_memory.rerank import induced_retrieved_subgraph
from graph_memory.types import GraphEdge, MemoryGraph, MemoryTaskInput, RankedNode, TrainableModelConfig


@dataclass(frozen=True)
class TrainableGraphRetriever:
    """
    Checkpoint-backed trainable graph retrieval method.
    基于 checkpoint 的可训练图检索方法。

    Fields / 字段:
    - name: Public retrieval method name.
      name：公开检索方法名。
    - model: Loaded evidence scoring model.
      model：已加载的 evidence scoring model。
    - model_config: Model reconstruction and feature config.
      model_config：模型重建和特征配置。
    - graph_by_task_id: Graph artifacts keyed by task id.
      graph_by_task_id：按 task id 索引的 graph artifact。
    - text_embedding_provider: Frozen text embedding provider.
      text_embedding_provider：冻结文本 embedding provider。
    - seed_signal_provider: Frozen seed signal provider.
      seed_signal_provider：冻结 seed signal provider。
    - device: Torch device used for inference.
      device：推理使用的 torch device。
    """

    name: str
    model: torch.nn.Module
    model_config: TrainableModelConfig
    graph_by_task_id: dict[str, MemoryGraph]
    text_embedding_provider: TextEmbeddingProvider
    seed_signal_provider: SeedSignalProvider
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        graphs: list[MemoryGraph],
        text_embedding_provider: TextEmbeddingProvider | None = None,
        seed_signal_provider: SeedSignalProvider | None = None,
        dense_encoder: object | None = None,
        device: str | torch.device = "cpu",
    ) -> "TrainableGraphRetriever":
        """
        Load a trainable graph retriever from `best.pt`.
        从 `best.pt` 加载可训练图检索器。
        """

        checkpoint = load_trainable_checkpoint(
            checkpoint_path,
            expected_method="dense_rgcn_graph_retriever",
            map_location=device,
        )
        model = build_model_from_config(checkpoint.model_config).to(device)
        model.load_state_dict(checkpoint.payload["model_state_dict"])
        model.eval()

        resolved_text_provider = text_embedding_provider
        if resolved_text_provider is None:
            resolved_text_provider = DenseTextEmbeddingProvider(
                model_name=checkpoint.model_config.encoder_model,
                query_prefix=checkpoint.model_config.query_prefix,
                passage_prefix=checkpoint.model_config.passage_prefix,
                encoder=cast(SentenceEncoder | None, dense_encoder),
            )
        resolved_seed_provider = seed_signal_provider
        if resolved_seed_provider is None:
            encoder = getattr(resolved_text_provider, "encoder", dense_encoder)
            resolved_seed_provider = RetrieverSeedSignalProvider(
                DenseTaskRetriever(
                    model_name=checkpoint.model_config.encoder_model,
                    query_prefix=checkpoint.model_config.query_prefix,
                    passage_prefix=checkpoint.model_config.passage_prefix,
                    encoder=encoder,
                )
            )
        return cls(
            name=checkpoint.model_config.method_name,
            model=model,
            model_config=checkpoint.model_config,
            graph_by_task_id={graph["task_id"]: graph for graph in graphs},
            text_embedding_provider=resolved_text_provider,
            seed_signal_provider=resolved_seed_provider,
            device=torch.device(device),
        )

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        graph = self.graph_by_task_id.get(task_input["task_id"])
        if graph is None:
            raise ValueError(f"Missing graph for task_id={task_input['task_id']}.")
        batches = build_full_ranking_batches(
            task_inputs=[task_input],
            graphs=[graph],
            model_config=self.model_config,
            text_embedding_provider=self.text_embedding_provider,
            seed_signal_provider=self.seed_signal_provider,
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
        retrieved_subgraph = induced_retrieved_subgraph(graph, top_node_ids)
        return ranked_nodes, retrieved_subgraph["edges"]
