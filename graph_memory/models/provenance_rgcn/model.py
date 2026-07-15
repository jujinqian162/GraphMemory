from __future__ import annotations

import torch
from torch import nn

from graph_memory.models.graph_retriever.internals.neural import (
    RGCNGraphEncoder,
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
        query = states[tensor.query_node_index]
        candidates = states[tensor.candidate_node_indices]
        query_rows = query.unsqueeze(0).expand_as(candidates)
        candidate_logits = self.candidate_scorer(
            torch.cat([candidates, query_rows, candidates * query_rows], dim=1)
        ).squeeze(-1)
        if tensor.logical_transitions:
            source_states = torch.stack(
                [states[item.source_node_index] for item in tensor.logical_transitions]
            )
            target_states = torch.stack(
                [states[item.target_node_index] for item in tensor.logical_transitions]
            )
            edge_query = query.unsqueeze(0).expand_as(source_states)
            edge_logits = self.edge_scorer(
                torch.cat(
                    [
                        source_states,
                        target_states,
                        source_states * target_states,
                        edge_query,
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
