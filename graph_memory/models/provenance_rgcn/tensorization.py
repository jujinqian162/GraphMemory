from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from graph_memory.embeddings import SentenceEncoder
from graph_memory.graphs.provenance import (
    ProvenanceEdgeType,
    ProvenanceNodeType,
    binding_matches_endpoints,
    binding_relation_key,
)
from graph_memory.models.graph_retriever.internals.contracts import GraphBatch
from graph_memory.models.provenance_rgcn.config import ProvenanceRgcnModelConfig
from graph_memory.models.provenance_rgcn.contracts import (
    LogicalProvenanceTransition,
    ProvenanceGraphTensor,
)
from graph_memory.retrieval.requests import ExecutionProvenanceRankingRequest


def tensorize_provenance_request(
    request: ExecutionProvenanceRankingRequest,
    *,
    encoder: SentenceEncoder,
    config: ProvenanceRgcnModelConfig,
) -> ProvenanceGraphTensor:
    nodes = list(request.graph.nodes)
    node_index = {node.node_id: index for index, node in enumerate(nodes)}
    node_type_id = {name: index for index, name in enumerate(config.node_type_vocab)}
    relation_id = {name: index for index, name in enumerate(config.relation_vocab)}
    task_indices = [
        index
        for index, node in enumerate(nodes)
        if node.node_type is ProvenanceNodeType.TASK
    ]
    if len(task_indices) != 1:
        raise ValueError(
            "Provenance R-GCN requires exactly one Task query anchor per graph."
        )
    candidate_ids = tuple(candidate.item_id for candidate in request.candidates)
    if any(
        request.graph.node(candidate_id).node_type is not ProvenanceNodeType.TOOL_OUTPUT
        for candidate_id in candidate_ids
    ):
        raise ValueError("Provenance R-GCN candidates must all be ToolOutput nodes.")

    formatted_texts = [
        (
            config.query_prefix + node.text
            if node.node_type is ProvenanceNodeType.TASK
            else config.passage_prefix + node.text
        )
        for node in nodes
    ]
    encoded = np.asarray(
        encoder.encode(
            formatted_texts,
            batch_size=config.encoder_batch_size,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if encoded.ndim != 2 or encoded.shape != (len(nodes), config.encoder_dim):
        raise ValueError(
            "Provenance encoder shape mismatch: "
            f"expected={(len(nodes), config.encoder_dim)} observed={encoded.shape}."
        )

    sources: list[int] = []
    targets: list[int] = []
    relation_ids: list[int] = []
    weights: list[float] = []
    node_by_id = {node.node_id: node for node in nodes}
    for edge in request.graph.edges:
        source = node_index[edge.source]
        target = node_index[edge.target]
        for message_source, message_target, suffix in (
            (source, target, "forward"),
            (target, source, "reverse"),
        ):
            relation_base = edge.edge_type.value
            if edge.edge_type is ProvenanceEdgeType.FEEDS:
                if not binding_matches_endpoints(edge, node_by_id):
                    raise ValueError(
                        "Provenance R-GCN refuses an inconsistent field binding: "
                        f"{edge.source}->{edge.target}."
                    )
                exact_relation = binding_relation_key(edge)
                relation_base = (
                    exact_relation
                    if f"{exact_relation}_{suffix}" in relation_id
                    else "feeds:bound_other"
                )
            relation_name = f"{relation_base}_{suffix}"
            if relation_name not in relation_id:
                raise ValueError(
                    f"Provenance relation missing from vocabulary: {relation_name}."
                )
            sources.append(message_source)
            targets.append(message_target)
            relation_ids.append(relation_id[relation_name])
            weights.append(
                edge.weight if config.edge_weight_policy == "artifact" else 1.0
            )
    edge_index = (
        torch.tensor([sources, targets], dtype=torch.long)
        if sources
        else torch.empty((2, 0), dtype=torch.long)
    )
    graph_batch = GraphBatch(
        node_embeddings=torch.from_numpy(encoded),
        node_features=torch.empty((len(nodes), 0), dtype=torch.float32),
        edge_index=edge_index,
        relation_ids=torch.tensor(relation_ids, dtype=torch.long),
        edge_weights=torch.tensor(weights, dtype=torch.float32),
        query_node_indices=torch.tensor(task_indices, dtype=torch.long),
        task_node_offsets=[0],
        task_ids=[request.task_id],
        node_ids_by_task=[[node.node_id for node in nodes]],
    )
    return ProvenanceGraphTensor(
        graph_batch=graph_batch,
        node_type_ids=torch.tensor(
            [node_type_id[node.node_type.value] for node in nodes], dtype=torch.long
        ),
        query_node_index=task_indices[0],
        candidate_node_indices=torch.tensor(
            [node_index[candidate_id] for candidate_id in candidate_ids],
            dtype=torch.long,
        ),
        candidate_ids=candidate_ids,
        logical_transitions=_logical_transitions(
            request, node_index, set(candidate_ids)
        ),
    )


def _logical_transitions(
    request: ExecutionProvenanceRankingRequest,
    node_index: dict[str, int],
    candidate_ids: set[str],
) -> tuple[LogicalProvenanceTransition, ...]:
    returns_by_call = defaultdict(list)
    for edge in request.graph.edges:
        if edge.edge_type is ProvenanceEdgeType.RETURNS:
            returns_by_call[edge.source].append(edge)
    transitions: dict[tuple[str, str], LogicalProvenanceTransition] = {}
    for feeds in request.graph.edges:
        if feeds.edge_type is not ProvenanceEdgeType.FEEDS:
            continue
        for returns in returns_by_call.get(feeds.target, ()):
            if feeds.source not in candidate_ids or returns.target not in candidate_ids:
                continue
            key = (feeds.source, returns.target)
            if key in transitions:
                raise ValueError(
                    f"Duplicate logical provenance transition: {feeds.source}->{returns.target}."
                )
            transitions[key] = LogicalProvenanceTransition(
                source_id=feeds.source,
                target_id=returns.target,
                source_node_index=node_index[feeds.source],
                target_node_index=node_index[returns.target],
                native_edges=(feeds, returns),
            )
    return tuple(transitions[key] for key in sorted(transitions))


def move_provenance_tensor(
    tensor: ProvenanceGraphTensor, device: torch.device | str
) -> ProvenanceGraphTensor:
    target = torch.device(device)
    batch = tensor.graph_batch
    return ProvenanceGraphTensor(
        graph_batch=GraphBatch(
            node_embeddings=batch.node_embeddings.to(target),
            node_features=batch.node_features.to(target),
            edge_index=batch.edge_index.to(target),
            relation_ids=batch.relation_ids.to(target),
            edge_weights=batch.edge_weights.to(target),
            query_node_indices=batch.query_node_indices.to(target),
            task_node_offsets=batch.task_node_offsets,
            task_ids=batch.task_ids,
            node_ids_by_task=batch.node_ids_by_task,
        ),
        node_type_ids=tensor.node_type_ids.to(target),
        query_node_index=tensor.query_node_index,
        candidate_node_indices=tensor.candidate_node_indices.to(target),
        candidate_ids=tensor.candidate_ids,
        logical_transitions=tensor.logical_transitions,
    )


__all__ = ["move_provenance_tensor", "tensorize_provenance_request"]
