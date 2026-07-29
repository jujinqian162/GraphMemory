from __future__ import annotations

import torch
from torch import nn

from graph_memory.models.graph_retriever.internals.neural import (
    RGCNGraphEncoder,
    SharedRelationTransform,
    TypedRelationTransform,
)
from graph_memory.models.provenance_rgcn.config import ProvenanceRgcnModelConfig
from graph_memory.models.provenance_rgcn.contracts import (
    ProvenanceGraphTensor,
    ProvenanceModelOutput,
)


class ExecutionProvenanceRGCN(nn.Module):
    def __init__(self, config: ProvenanceRgcnModelConfig) -> None:
        super().__init__()
        self.config = config
        self.node_type_embedding = nn.Embedding(
            len(config.node_type_vocab), config.node_type_dim
        )
        self.input_projection = nn.Sequential(
            nn.Linear(config.encoder_dim + config.node_type_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        if config.message_transform_type == "typed":
            self.graph_encoder = RGCNGraphEncoder(
                hidden_dim=config.hidden_dim,
                num_relations=len(config.relation_vocab),
                num_layers=config.num_layers,
                message_transform_factory=lambda: TypedRelationTransform(
                    hidden_dim=config.hidden_dim,
                    num_relations=len(config.relation_vocab),
                ),
                dropout=config.dropout,
            )
        else:
            self.graph_encoder = RGCNGraphEncoder(
                hidden_dim=config.hidden_dim,
                num_relations=len(config.relation_vocab),
                num_layers=config.num_layers,
                message_transform_factory=lambda: SharedRelationTransform(
                    hidden_dim=config.hidden_dim
                ),
                dropout=config.dropout,
            )
        scorer_dim = config.hidden_dim * 3
        self.candidate_scorer = nn.Sequential(
            nn.Linear(scorer_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, tensor: ProvenanceGraphTensor) -> ProvenanceModelOutput:
        batch = tensor.graph_batch
        initial = self.input_projection(
            torch.cat(
                [
                    batch.node_embeddings,
                    self.node_type_embedding(tensor.node_type_ids),
                ],
                dim=1,
            )
        )
        states = self.graph_encoder.forward(batch, initial)
        candidates = states[tensor.candidate_node_indices]
        candidate_queries = states[tensor.candidate_query_indices]
        candidate_logits = self.candidate_scorer(
            torch.cat(
                [
                    candidates,
                    candidate_queries,
                    candidates * candidate_queries,
                ],
                dim=1,
            )
        ).squeeze(-1)
        if tensor.transition_node_indices.shape[1] > 0:
            source_states = states[tensor.transition_node_indices[0]]
            target_states = states[tensor.transition_node_indices[1]]
            edge_queries = states[tensor.transition_query_indices]
            edge_logits = self.edge_scorer(
                torch.cat(
                    [
                        source_states,
                        target_states,
                        source_states * target_states,
                        edge_queries,
                    ],
                    dim=1,
                )
            ).squeeze(-1)
        else:
            edge_logits = states.new_empty((0,))
        return ProvenanceModelOutput(
            node_states=states,
            candidate_logits=candidate_logits,
            edge_logits=edge_logits,
        )


__all__ = ["ExecutionProvenanceRGCN"]
