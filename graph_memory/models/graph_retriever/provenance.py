from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

from graph_memory.graphs.provenance import (
    ARGUMENT_CHUNK_NODE,
    FEEDS_EDGE,
    HAS_ARGUMENT_EDGE,
    HAS_CONTENT_EDGE,
    NEXT_CHUNK_EDGE,
    OUTPUT_CHUNK_NODE,
    PRECEDES_EDGE,
    READS_EDGE,
    RETURNS_EDGE,
    WRITES_EDGE,
    ProvenanceGraph,
    logical_output_dependencies,
)
from graph_memory.models.graph_batching import TaskGraphTensor
from graph_memory.models.graph_retriever.batching import EvidenceTaskTensor
from graph_memory.models.graph_retriever.config.records import (
    NodeFeatureConfig,
    RgcnModelConfig,
)
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.internals.tensorization import (
    MessageEdgeTensors,
)
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.query_synthesis.provenance.contracts import (
    TemplateSupervisionRecord,
)
from graph_memory.retrieval.requests import (
    ProvenanceRgcnRequest,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.training_pairs.contracts import TrainPairRecord
from graph_memory.training_pairs.requests import (
    CandidateNeighborEdge,
    TrainPairBuildTask,
)
from graph_memory.trajectories import SourceSpan

PROVENANCE_PHYSICAL_RELATIONS: tuple[str, ...] = (
    RETURNS_EDGE,
    HAS_ARGUMENT_EDGE,
    HAS_CONTENT_EDGE,
    FEEDS_EDGE,
    READS_EDGE,
    WRITES_EDGE,
    NEXT_CHUNK_EDGE,
)
PROVENANCE_RELATION_VOCAB: tuple[str, ...] = tuple(
    direction
    for relation in PROVENANCE_PHYSICAL_RELATIONS
    for direction in (f"{relation}_forward", f"{relation}_reverse")
)


def provenance_rgcn_model_config(
    *,
    encoder_model: str,
    encoder_dim: int,
    query_prefix: str,
    passage_prefix: str,
    encoder_batch_size: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    ablation_name: str = "full_rgcn",
) -> RgcnModelConfig:
    layer_count = 0 if ablation_name == "wo_graph" or num_layers == 0 else num_layers
    return RgcnModelConfig(
        method_name="provenance_rgcn",
        encoder_model=encoder_model,
        encoder_dim=encoder_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        encoder_batch_size=encoder_batch_size,
        hidden_dim=hidden_dim,
        num_layers=layer_count,
        dropout=dropout,
        feature_config=NodeFeatureConfig(
            node_feature_names=(),
            scorer_feature_names=(),
        ),
        relation_vocab=PROVENANCE_RELATION_VOCAB,
        graph_encoder_type="identity" if layer_count == 0 else "rgcn",
        message_transform_type="typed",
        edge_weight_policy="uniform",
        enabled_edge_types=(),
        ablation_name="wo_graph" if layer_count == 0 else "full_rgcn",
    )


def provenance_training_label(
    request: ProvenanceRgcnRequest,
    *,
    gold_spans: Sequence[SourceSpan],
    template: TemplateSupervisionRecord | None = None,
) -> EvidenceLabel:
    """Map natural exact spans or explicit template focus to candidate positives."""

    if template is not None:
        if template.task_id != request.task_id or template.graph_id != request.graph.graph_id:
            raise ValueError("template supervision does not align with provenance task")
        if template.query_text != request.query_text:
            raise ValueError("template supervision query text changed")
        if template.graph_fingerprint != request.graph.fingerprint():
            raise ValueError("template supervision graph fingerprint changed")
        focused_candidate_ids = {
            edge.target
            for edge in request.graph.edges
            if edge.relation == HAS_CONTENT_EDGE
            and edge.source in set(template.focus_output_ids)
        }
        if set(template.positive_candidate_ids) != focused_candidate_ids:
            raise ValueError(
                "template positives must be exactly the focused output content"
            )
        positive_ids = template.positive_candidate_ids
    else:
        positive_ids = tuple(
            candidate.item_id
            for candidate in request.candidates
            if any(
                _source_spans_overlap(candidate_span, gold_span)
                for candidate_span in candidate.source_spans
                for gold_span in gold_spans
            )
        )
    if not positive_ids:
        raise ValueError(
            f"provenance task={request.task_id!r} has no positive candidates"
        )
    candidate_ids = {candidate.item_id for candidate in request.candidates}
    missing = set(positive_ids) - candidate_ids
    if missing:
        raise ValueError(
            f"provenance task={request.task_id!r} positives are not candidates: "
            f"{sorted(missing)}"
        )
    return EvidenceLabel(
        task_id=request.task_id,
        gold_answer="",
        gold_evidence_item_ids=positive_ids,
        gold_dependency_edges=(),
    )


def provenance_candidate_neighbor_edges(
    graph: ProvenanceGraph,
) -> tuple[CandidateNeighborEdge, ...]:
    """Project physical connectivity to neutral candidate-neighbor pairs."""

    candidate_ids = {
        node.node_id
        for node in graph.nodes
        if node.kind in {ARGUMENT_CHUNK_NODE, OUTPUT_CHUNK_NODE}
    }
    output_content_by_owner: dict[str, list[str]] = {}
    argument_content_by_call: dict[str, list[str]] = {}
    output_by_call: dict[str, str] = {}
    pairs: set[tuple[str, str]] = set()
    for edge in graph.edges:
        if edge.relation == HAS_CONTENT_EDGE:
            output_content_by_owner.setdefault(edge.source, []).append(edge.target)
        elif edge.relation == HAS_ARGUMENT_EDGE:
            argument_content_by_call.setdefault(edge.source, []).append(edge.target)
        elif edge.relation == RETURNS_EDGE:
            output_by_call[edge.source] = edge.target
        elif edge.relation == NEXT_CHUNK_EDGE:
            _add_candidate_pair(pairs, edge.source, edge.target, candidate_ids)
    for call_id, argument_ids in argument_content_by_call.items():
        output_id = output_by_call.get(call_id)
        if output_id is None:
            continue
        for argument_id in argument_ids:
            for content_id in output_content_by_owner.get(output_id, ()):
                _add_candidate_pair(pairs, argument_id, content_id, candidate_ids)
    for dependency in logical_output_dependencies(graph):
        for source_id in output_content_by_owner.get(
            dependency.source_output_id, ()
        ):
            for target_id in output_content_by_owner.get(
                dependency.target_output_id, ()
            ):
                _add_candidate_pair(pairs, source_id, target_id, candidate_ids)
    return tuple(
        CandidateNeighborEdge(source=source, target=target)
        for source, target in sorted(pairs)
    )


def provenance_train_pair_task(
    request: ProvenanceRgcnRequest,
    label: EvidenceLabel,
) -> TrainPairBuildTask:
    if request.task_id != label.task_id:
        raise ValueError("provenance pair request and label task IDs must align")
    return TrainPairBuildTask(
        text_request=TextRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
        ),
        label=label,
        candidate_neighbor_edges=provenance_candidate_neighbor_edges(request.graph),
    )


def tensorize_provenance_edges(graph: ProvenanceGraph) -> MessageEdgeTensors:
    """Map the fixed physical relation policy to uniform forward/reverse messages."""

    node_index_by_id = {
        node.node_id: index for index, node in enumerate(graph.nodes)
    }
    relation_id_by_name = {
        name: index for index, name in enumerate(PROVENANCE_RELATION_VOCAB)
    }
    sources: list[int] = []
    targets: list[int] = []
    relation_ids: list[int] = []
    for edge in graph.edges:
        if edge.relation == PRECEDES_EDGE:
            continue
        if edge.relation not in PROVENANCE_PHYSICAL_RELATIONS:
            continue
        source = node_index_by_id[edge.source]
        target = node_index_by_id[edge.target]
        sources.extend((source, target))
        targets.extend((target, source))
        relation_ids.extend(
            (
                relation_id_by_name[f"{edge.relation}_forward"],
                relation_id_by_name[f"{edge.relation}_reverse"],
            )
        )
    return MessageEdgeTensors(
        edge_index=(
            torch.tensor((sources, targets), dtype=torch.long)
            if sources
            else torch.empty((2, 0), dtype=torch.long)
        ),
        relation_ids=torch.tensor(relation_ids, dtype=torch.long),
        edge_weights=torch.ones(len(relation_ids), dtype=torch.float32),
    )


def _source_spans_overlap(left: SourceSpan, right: SourceSpan) -> bool:
    if left.event_id != right.event_id or left.json_pointer != right.json_pointer:
        return False
    if (
        left.char_start is None
        or left.char_end is None
        or right.char_start is None
        or right.char_end is None
    ):
        return False
    return max(left.char_start, right.char_start) < min(left.char_end, right.char_end)


def _add_candidate_pair(
    pairs: set[tuple[str, str]],
    left: str,
    right: str,
    candidate_ids: set[str],
) -> None:
    if left == right or left not in candidate_ids or right not in candidate_ids:
        return
    pairs.add((left, right) if left < right else (right, left))


def provenance_embedding_request(
    request: ProvenanceRgcnRequest,
) -> tuple[TextRankingRequest, list[str]]:
    node_ids = [node.node_id for node in request.graph.nodes]
    if "q" in node_ids:
        raise ValueError("persisted provenance graph cannot contain the ephemeral q node")
    node_ids.append("q")
    return (
        TextRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=tuple(
                TextCandidate(
                    item_id=node.node_id,
                    text=node.text,
                    metadata={},
                    source_spans=node.source_spans,
                )
                for node in request.graph.nodes
            ),
        ),
        node_ids,
    )


def tensorize_provenance_ranking_task(
    request: ProvenanceRgcnRequest,
    *,
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
) -> EvidenceTaskTensor:
    """Adapt one physical provenance graph to the maintained task tensor contract."""

    if model_config.method_name != "provenance_rgcn":
        raise ValueError("provenance tensorization requires method_name='provenance_rgcn'")
    if model_config.relation_vocab != PROVENANCE_RELATION_VOCAB:
        raise ValueError("provenance R-GCN relation vocabulary does not match fixed policy")
    if model_config.feature_config != NodeFeatureConfig(
        node_feature_names=(), scorer_feature_names=()
    ):
        raise ValueError("provenance R-GCN forbids metadata-derived numeric features")

    graph = request.graph
    fingerprint = graph.fingerprint()
    embedding_request, node_ids = provenance_embedding_request(request)
    node_embeddings = text_embedding_provider.encode_task_nodes(
        embedding_request,
        node_ids,
    )
    expected_shape = (len(node_ids), model_config.encoder_dim)
    if tuple(node_embeddings.shape) != expected_shape:
        raise ValueError(
            "provenance node embedding shape mismatch: "
            f"expected={expected_shape} observed={tuple(node_embeddings.shape)}"
        )

    message_edges = tensorize_provenance_edges(graph)
    candidate_indices = [
        node_ids.index(candidate.item_id) for candidate in request.candidates
    ]
    graph_tensor = TaskGraphTensor(
        node_embeddings=node_embeddings,
        node_features=torch.empty((len(node_ids), 0), dtype=torch.float32),
        edge_index=message_edges.edge_index,
        relation_ids=message_edges.relation_ids,
        edge_weights=message_edges.edge_weights,
        query_node_index=len(node_ids) - 1,
        task_id=request.task_id,
        node_ids=node_ids,
    )
    if graph.fingerprint() != fingerprint:
        raise RuntimeError("provenance tensorization mutated the persisted graph")
    return EvidenceTaskTensor(
        graph_tensor=graph_tensor,
        sample_node_indices=torch.tensor(candidate_indices, dtype=torch.long),
        sample_node_features=torch.empty(
            (len(candidate_indices), 0), dtype=torch.float32
        ),
        labels=torch.zeros(len(candidate_indices), dtype=torch.float32),
        sample_node_ids=[candidate.item_id for candidate in request.candidates],
        sample_types=["easy_random"] * len(candidate_indices),
    )


def tensorize_provenance_dev_task(
    request: ProvenanceRgcnRequest,
    label: EvidenceLabel,
    *,
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
) -> EvidenceTaskTensor:
    task = tensorize_provenance_ranking_task(
        request,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
    )
    if label.task_id != request.task_id:
        raise ValueError("provenance dev request and label task IDs must align")
    gold_ids = set(label.gold_evidence_item_ids)
    missing = gold_ids - set(task.sample_node_ids)
    if missing:
        raise ValueError(
            f"provenance dev positives are not candidates: {sorted(missing)}"
        )
    return replace(
        task,
        labels=torch.tensor(
            [float(node_id in gold_ids) for node_id in task.sample_node_ids],
            dtype=torch.float32,
        ),
        sample_types=[
            "positive" if node_id in gold_ids else "easy_random"
            for node_id in task.sample_node_ids
        ],
    )


def tensorize_provenance_training_task(
    request: ProvenanceRgcnRequest,
    pairs: Sequence[TrainPairRecord],
    *,
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
) -> EvidenceTaskTensor:
    task = tensorize_provenance_ranking_task(
        request,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
    )
    node_index = {
        node_id: index for index, node_id in enumerate(task.graph_tensor.node_ids)
    }
    rows = list(pairs)
    if not rows:
        raise ValueError("provenance training task requires supervised pairs")
    for pair in rows:
        if pair.task_id != request.task_id:
            raise ValueError("provenance training pair crosses task ownership")
        if pair.node_id not in {
            candidate.item_id for candidate in request.candidates
        }:
            raise ValueError("provenance training pair is not a ranking candidate")
    return replace(
        task,
        sample_node_indices=torch.tensor(
            [node_index[pair.node_id] for pair in rows], dtype=torch.long
        ),
        sample_node_features=torch.empty((len(rows), 0), dtype=torch.float32),
        labels=torch.tensor([float(pair.label) for pair in rows], dtype=torch.float32),
        sample_node_ids=[pair.node_id for pair in rows],
        sample_types=[pair.sample_type for pair in rows],
    )


def tensorize_provenance_ranking_tasks(
    requests: Sequence[ProvenanceRgcnRequest],
    *,
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
) -> list[EvidenceTaskTensor]:
    return [
        tensorize_provenance_ranking_task(
            request,
            model_config=model_config,
            text_embedding_provider=text_embedding_provider,
        )
        for request in requests
    ]


__all__ = [
    "PROVENANCE_PHYSICAL_RELATIONS",
    "PROVENANCE_RELATION_VOCAB",
    "provenance_candidate_neighbor_edges",
    "provenance_embedding_request",
    "provenance_rgcn_model_config",
    "provenance_train_pair_task",
    "provenance_training_label",
    "tensorize_provenance_dev_task",
    "tensorize_provenance_edges",
    "tensorize_provenance_ranking_task",
    "tensorize_provenance_ranking_tasks",
    "tensorize_provenance_training_task",
]
