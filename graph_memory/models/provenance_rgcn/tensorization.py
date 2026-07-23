from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import torch

from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.embeddings import SentenceEncoder
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.provenance import (
    ProvenanceEdgeType,
    ProvenanceNodeType,
    binding_matches_endpoints,
    binding_relation_key,
)
from graph_memory.models.graph_batching import (
    TaskGraphTensor,
    collate_task_graphs,
    move_graph_batch,
)
from graph_memory.models.provenance_rgcn.config import ProvenanceRgcnModelConfig
from graph_memory.models.provenance_rgcn.contracts import (
    LogicalProvenanceTransition,
    ProvenanceGraphTensor,
    ProvenanceModelOutput,
    ProvenanceTaskTensor,
    ProvenanceTrainingBatch,
    ProvenanceTrainingTask,
)
from graph_memory.retrieval.requests import ExecutionProvenanceRankingRequest


def tensorize_provenance_task(
    request: ExecutionProvenanceRankingRequest,
    *,
    encoder: SentenceEncoder,
    config: ProvenanceRgcnModelConfig,
) -> ProvenanceTaskTensor:
    """Materialize one provenance graph with only task-local indices."""

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
    feed_weights_by_source: defaultdict[str, list[float]] = defaultdict(list)
    for edge in request.graph.edges:
        if edge.edge_type is ProvenanceEdgeType.FEEDS:
            feed_weights_by_source[edge.source].append(edge.weight)
    feed_source_means = {
        source: sum(source_weights) / len(source_weights)
        for source, source_weights in feed_weights_by_source.items()
    }
    shuffled_feed_targets = _shuffled_feed_targets(request, node_index, config)
    feed_index = 0
    for edge in request.graph.edges:
        source = node_index[edge.source]
        target = node_index[edge.target]
        if edge.edge_type is ProvenanceEdgeType.FEEDS:
            target = shuffled_feed_targets[feed_index]
            feed_index += 1
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
            resolved_weight = edge.weight
            if (
                config.edge_weight_policy == "uniform"
                and edge.edge_type is ProvenanceEdgeType.FEEDS
            ):
                resolved_weight = feed_source_means[edge.source]
            weights.append(resolved_weight)
    edge_index = (
        torch.tensor([sources, targets], dtype=torch.long)
        if sources
        else torch.empty((2, 0), dtype=torch.long)
    )
    return ProvenanceTaskTensor(
        graph_tensor=TaskGraphTensor(
            node_embeddings=torch.from_numpy(encoded),
            node_features=torch.empty((len(nodes), 0), dtype=torch.float32),
            edge_index=edge_index,
            relation_ids=torch.tensor(relation_ids, dtype=torch.long),
            edge_weights=torch.tensor(weights, dtype=torch.float32),
            query_node_index=task_indices[0],
            task_id=request.task_id,
            node_ids=[node.node_id for node in nodes],
        ),
        node_type_ids=torch.tensor(
            [node_type_id[node.node_type.value] for node in nodes], dtype=torch.long
        ),
        candidate_node_indices=torch.tensor(
            [node_index[candidate_id] for candidate_id in candidate_ids],
            dtype=torch.long,
        ),
        candidate_ids=candidate_ids,
        logical_transitions=_logical_transitions(
            request, node_index, set(candidate_ids)
        ),
    )


def tensorize_provenance_request(
    request: ExecutionProvenanceRankingRequest,
    *,
    encoder: SentenceEncoder,
    config: ProvenanceRgcnModelConfig,
) -> ProvenanceGraphTensor:
    """Compatibility one-task entry point using the same batched contract."""

    return collate_provenance_tasks(
        [tensorize_provenance_task(request, encoder=encoder, config=config)]
    )


def collate_provenance_tasks(
    tasks: Sequence[ProvenanceTaskTensor],
) -> ProvenanceGraphTensor:
    """Collate provenance tasks and preserve explicit task ownership."""

    if not tasks:
        raise ValueError("Provenance collation requires at least one task.")
    graph_batch = collate_task_graphs([task.graph_tensor for task in tasks])
    node_type_ids: list[torch.Tensor] = []
    candidate_node_indices: list[torch.Tensor] = []
    candidate_query_indices: list[torch.Tensor] = []
    candidate_offsets = [0]
    candidate_ids_by_task: list[tuple[str, ...]] = []
    transition_node_indices: list[torch.Tensor] = []
    transition_query_indices: list[torch.Tensor] = []
    transition_offsets = [0]
    transitions_by_task: list[tuple[LogicalProvenanceTransition, ...]] = []

    for task_index, task in enumerate(tasks):
        _validate_provenance_task(task)
        node_offset = graph_batch.task_node_offsets[task_index]
        query_index = int(graph_batch.query_node_indices[task_index])
        node_type_ids.append(task.node_type_ids)
        candidate_count = len(task.candidate_ids)
        candidate_node_indices.append(task.candidate_node_indices + node_offset)
        candidate_query_indices.append(
            torch.full((candidate_count,), query_index, dtype=torch.long)
        )
        candidate_offsets.append(candidate_offsets[-1] + candidate_count)
        candidate_ids_by_task.append(task.candidate_ids)

        global_transitions = tuple(
            replace(
                transition,
                source_node_index=transition.source_node_index + node_offset,
                target_node_index=transition.target_node_index + node_offset,
            )
            for transition in task.logical_transitions
        )
        transition_count = len(global_transitions)
        if global_transitions:
            transition_node_indices.append(
                torch.tensor(
                    [
                        [item.source_node_index for item in global_transitions],
                        [item.target_node_index for item in global_transitions],
                    ],
                    dtype=torch.long,
                )
            )
            transition_query_indices.append(
                torch.full((transition_count,), query_index, dtype=torch.long)
            )
        transition_offsets.append(transition_offsets[-1] + transition_count)
        transitions_by_task.append(task.logical_transitions)

    return ProvenanceGraphTensor(
        graph_batch=graph_batch,
        node_type_ids=torch.cat(node_type_ids),
        candidate_node_indices=torch.cat(candidate_node_indices),
        candidate_query_indices=torch.cat(candidate_query_indices),
        candidate_offsets=candidate_offsets,
        candidate_ids_by_task=tuple(candidate_ids_by_task),
        transition_node_indices=(
            torch.cat(transition_node_indices, dim=1)
            if transition_node_indices
            else torch.empty((2, 0), dtype=torch.long)
        ),
        transition_query_indices=(
            torch.cat(transition_query_indices)
            if transition_query_indices
            else torch.empty((0,), dtype=torch.long)
        ),
        transition_offsets=transition_offsets,
        logical_transitions_by_task=tuple(transitions_by_task),
    )


def materialize_provenance_training_task(
    request: ExecutionProvenanceRankingRequest,
    label: EvidenceLabel,
    train_pairs: list[TrainPairRecord],
    *,
    encoder: SentenceEncoder,
    config: ProvenanceRgcnModelConfig,
) -> ProvenanceTrainingTask:
    """Materialize one provenance task and aligned `1/0/-1` v2 targets."""

    if label.task_id != request.task_id:
        raise ValueError("Provenance training request and label task IDs must match.")
    task = tensorize_provenance_task(request, encoder=encoder, config=config)
    candidate_targets = torch.full((len(task.candidate_ids),), -1, dtype=torch.int8)
    candidate_index = {
        candidate_id: index for index, candidate_id in enumerate(task.candidate_ids)
    }
    seen_pair_nodes: set[str] = set()
    for pair in train_pairs:
        if pair["task_id"] != label.task_id:
            raise ValueError(
                "Provenance train pair task mismatch: "
                f"expected={label.task_id!r} observed={pair['task_id']!r}."
            )
        node_id = pair["node_id"]
        if node_id not in candidate_index:
            raise ValueError(
                f"Provenance train pair node_id={node_id!r} is not a candidate."
            )
        if node_id in seen_pair_nodes:
            raise ValueError(
                "Provenance train pairs must contain each candidate at most once: "
                f"task_id={label.task_id!r} node_id={node_id!r}."
            )
        seen_pair_nodes.add(node_id)
        candidate_targets[candidate_index[node_id]] = int(pair["label"])
    if not bool((candidate_targets == 1).any()):
        raise ValueError(
            f"Provenance task_id={label.task_id!r} has no positive train pairs."
        )
    gold_edges = set(label.gold_dependency_edges)
    edge_targets = torch.tensor(
        [
            float((item.source_id, item.target_id) in gold_edges)
            for item in task.logical_transitions
        ],
        dtype=torch.float32,
    )
    return ProvenanceTrainingTask(
        tensor=task,
        candidate_targets=candidate_targets,
        edge_targets=edge_targets,
    )


def collate_provenance_training_tasks(
    tasks: Sequence[ProvenanceTrainingTask],
) -> ProvenanceTrainingBatch:
    """Collate provenance tasks and their aligned loss targets."""

    if not tasks:
        raise ValueError("Provenance training collation requires at least one task.")
    return ProvenanceTrainingBatch(
        tensor=collate_provenance_tasks([task.tensor for task in tasks]),
        candidate_targets=torch.cat([task.candidate_targets for task in tasks]),
        edge_targets=torch.cat([task.edge_targets for task in tasks]),
    )


def move_provenance_tensor(
    tensor: ProvenanceGraphTensor, device: torch.device | str
) -> ProvenanceGraphTensor:
    target = torch.device(device)
    return ProvenanceGraphTensor(
        graph_batch=move_graph_batch(tensor.graph_batch, target),
        node_type_ids=tensor.node_type_ids.to(target),
        candidate_node_indices=tensor.candidate_node_indices.to(target),
        candidate_query_indices=tensor.candidate_query_indices.to(target),
        candidate_offsets=tensor.candidate_offsets,
        candidate_ids_by_task=tensor.candidate_ids_by_task,
        transition_node_indices=tensor.transition_node_indices.to(target),
        transition_query_indices=tensor.transition_query_indices.to(target),
        transition_offsets=tensor.transition_offsets,
        logical_transitions_by_task=tensor.logical_transitions_by_task,
    )


def move_provenance_training_batch(
    batch: ProvenanceTrainingBatch, device: torch.device | str
) -> ProvenanceTrainingBatch:
    target = torch.device(device)
    return ProvenanceTrainingBatch(
        tensor=move_provenance_tensor(batch.tensor, target),
        candidate_targets=batch.candidate_targets.to(target),
        edge_targets=batch.edge_targets.to(target),
    )


def split_provenance_output(
    tensor: ProvenanceGraphTensor,
    output: ProvenanceModelOutput,
) -> tuple[ProvenanceModelOutput, ...]:
    """Split batched model output back into declared task order."""

    if len(output.node_states) != tensor.graph_batch.task_node_offsets[-1]:
        raise ValueError("Node-state count does not match the graph batch.")
    if len(output.candidate_logits) != tensor.candidate_offsets[-1]:
        raise ValueError("Candidate-logit count does not match candidate offsets.")
    if len(output.edge_logits) != tensor.transition_offsets[-1]:
        raise ValueError("Edge-logit count does not match transition offsets.")
    results: list[ProvenanceModelOutput] = []
    for task_index in range(tensor.task_count):
        node_start = tensor.graph_batch.task_node_offsets[task_index]
        node_end = tensor.graph_batch.task_node_offsets[task_index + 1]
        candidate_start = tensor.candidate_offsets[task_index]
        candidate_end = tensor.candidate_offsets[task_index + 1]
        transition_start = tensor.transition_offsets[task_index]
        transition_end = tensor.transition_offsets[task_index + 1]
        results.append(
            ProvenanceModelOutput(
                node_states=output.node_states[node_start:node_end],
                candidate_logits=output.candidate_logits[candidate_start:candidate_end],
                edge_logits=output.edge_logits[transition_start:transition_end],
            )
        )
    return tuple(results)


def _validate_provenance_task(task: ProvenanceTaskTensor) -> None:
    node_count = len(task.graph_tensor.node_ids)
    if task.node_type_ids.shape != (node_count,):
        raise ValueError("node_type_ids must contain one value per task node.")
    if task.candidate_node_indices.shape != (len(task.candidate_ids),):
        raise ValueError("Candidate IDs and indices must have equal lengths.")
    if len(set(task.candidate_ids)) != len(task.candidate_ids):
        raise ValueError("Provenance candidate IDs must be unique within a task.")
    if len(task.candidate_node_indices):
        minimum = int(task.candidate_node_indices.min().item())
        maximum = int(task.candidate_node_indices.max().item())
        if minimum < 0 or maximum >= node_count:
            raise ValueError("A candidate index crosses its task node interval.")
    for transition in task.logical_transitions:
        if not (
            0 <= transition.source_node_index < node_count
            and 0 <= transition.target_node_index < node_count
        ):
            raise ValueError("A logical transition crosses its task node interval.")


def _shuffled_feed_targets(
    request: ExecutionProvenanceRankingRequest,
    node_index: dict[str, int],
    config: ProvenanceRgcnModelConfig,
) -> list[int]:
    targets = [
        node_index[edge.target]
        for edge in request.graph.edges
        if edge.edge_type is ProvenanceEdgeType.FEEDS
    ]
    if config.feed_message_topology == "native" or len(targets) < 2:
        return targets
    material = (
        f"{config.feed_message_shuffle_seed}|{request.task_id}|"
        + "|".join(candidate.item_id for candidate in request.candidates)
    ).encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
    shuffled = list(targets)
    rng.shuffle(shuffled)
    if shuffled == targets:
        shuffled = [*targets[1:], targets[0]]
    return shuffled


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
                    "Duplicate logical provenance transition: "
                    f"{feeds.source}->{returns.target}."
                )
            transitions[key] = LogicalProvenanceTransition(
                source_id=feeds.source,
                target_id=returns.target,
                source_node_index=node_index[feeds.source],
                target_node_index=node_index[returns.target],
                native_edges=(feeds, returns),
            )
    return tuple(transitions[key] for key in sorted(transitions))


__all__ = [
    "collate_provenance_tasks",
    "collate_provenance_training_tasks",
    "materialize_provenance_training_task",
    "move_provenance_tensor",
    "move_provenance_training_batch",
    "split_provenance_output",
    "tensorize_provenance_request",
    "tensorize_provenance_task",
]
