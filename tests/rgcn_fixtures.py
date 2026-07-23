from __future__ import annotations

import torch

from graph_memory.contracts.graphs import GraphItemNode, EvidenceGraph
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord
from graph_memory.models.graph_retriever.config.records import (
    NodeFeatureConfig,
    RgcnModelConfig,
    RgcnTrainingConfig,
)
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import TextRankingRequest


class FakeRetriever:
    method_name = "dense"

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        scores = {"m0": 0.9, "m1": 0.2, "m2": 0.7}
        return sorted(
            [
                RankedNode(node_id=candidate.item_id, score=scores[candidate.item_id])
                for candidate in request.candidates
            ],
            key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id),
        )


class FakeTextEmbeddingProvider(TextEmbeddingProvider):
    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_task_nodes(
        self, request: TextRankingRequest, node_ids: list[str]
    ) -> torch.Tensor:
        rows: list[list[float]] = []
        for node_id in node_ids:
            if node_id == "q":
                rows.append([1.0, 0.0, 0.0, 0.0])
            else:
                position = int(node_id[1:])
                rows.append(
                    [
                        0.0,
                        1.0 if position == 0 else 0.0,
                        1.0 if position == 1 else 0.0,
                        1.0 if position == 2 else 0.0,
                    ]
                )
        return torch.tensor(rows, dtype=torch.float32)


def tiny_task_inputs() -> list[HotpotQARankingRecord]:
    return [
        {
            "task_id": "hotpot_rgcn_train",
            "question": "Which evidence mentions Alpha?",
            "candidate_sentences": [
                {
                    "sentence_id": "m0",
                    "text": "Alpha is the answer evidence.",
                    "title": "A",
                    "sentence_index": 0,
                    "position": 0,
                },
                {
                    "sentence_id": "m1",
                    "text": "Beta is unrelated.",
                    "title": "B",
                    "sentence_index": 0,
                    "position": 1,
                },
                {
                    "sentence_id": "m2",
                    "text": "Gamma connects to Alpha.",
                    "title": "C",
                    "sentence_index": 0,
                    "position": 2,
                },
            ],
        }
    ]


def _graph_nodes(task: HotpotQARankingRecord) -> list[GraphItemNode]:
    return [
        {
            "id": sentence["sentence_id"],
            "node_type": "graph_item",
            "node_kind": "document_sentence",
            "text": sentence["text"],
            "source_ref": sentence["title"],
            "group_key": f"document:{sentence['title']}",
            "sequence_index": sentence["sentence_index"],
            "metadata": {
                "title": sentence["title"],
                "position": sentence["position"],
            },
        }
        for sentence in task["candidate_sentences"]
    ]


def tiny_graphs() -> list[EvidenceGraph]:
    task = tiny_task_inputs()[0]
    return [
        {
            "task_id": task["task_id"],
            "nodes": [
                {"id": "q", "node_type": "question", "text": task["question"]},
                *_graph_nodes(task),
            ],
            "edges": [
                {
                    "source": "q",
                    "target": "m0",
                    "edge_type": "query_overlap",
                    "weight": 1.0,
                    "directed": True,
                },
                {
                    "source": "m0",
                    "target": "m2",
                    "edge_type": "bridge",
                    "weight": 0.8,
                    "directed": False,
                },
            ],
        }
    ]


def tiny_model_config() -> RgcnModelConfig:
    return RgcnModelConfig(
        method_name="dense_rgcn_graph_retriever",
        encoder_model="fake-encoder",
        encoder_dim=4,
        query_prefix="query: ",
        passage_prefix="passage: ",
        encoder_batch_size=64,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        feature_config=NodeFeatureConfig(),
        relation_vocab=(
            "query_overlap_forward",
            "sequential_forward",
            "sequential_reverse",
            "entity_overlap_forward",
            "entity_overlap_reverse",
            "bridge_forward",
            "bridge_reverse",
        ),
        graph_encoder_type="rgcn",
        message_transform_type="typed",
        edge_weight_policy="artifact",
        enabled_edge_types=(
            "bridge",
            "entity_overlap",
            "query_overlap",
            "sequential",
        ),
        ablation_name="full_rgcn",
    )


def tiny_training_config() -> RgcnTrainingConfig:
    return RgcnTrainingConfig(
        optimizer_name="AdamW",
        learning_rate=0.01,
        per_device_graph_batch_size=1,
        max_grad_norm=1.0,
        random_seed=13,
        pos_weight_enabled=False,
        epochs=2,
    )
