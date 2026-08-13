from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import random
from typing import Literal

import torch

from graph_memory.graphs.provenance import (
    ARGUMENT_CHUNK_NODE,
    FEEDS_EDGE,
    HAS_ARGUMENT_EDGE,
    HAS_CONTENT_EDGE,
    NEXT_CHUNK_EDGE,
    OUTPUT_CHUNK_NODE,
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
from graph_memory.embeddings import DenseTaskEncodingRequest
from graph_memory.models.graph_retriever.contracts import (
    TextEmbeddingProvider,
    build_task_feature_groups,
)
from graph_memory.models.graph_retriever.internals.features import NodeFeatureBuilder
from graph_memory.models.graph_retriever.internals.tensorization import (
    MessageEdgeTensors,
)
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.retrieval.signals import SeedSignalProvider
from graph_memory.training_pairs.contracts import TrainPairRecord
from graph_memory.training_pairs.requests import (
    CandidateNeighborEdge,
    TrainPairBuildTask,
)
from graph_memory.trajectories import SourceSpan, source_spans_overlap

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
HOMOGENEOUS_PROVENANCE_RELATION_VOCAB: tuple[str, ...] = ("homogeneous",)
PROVENANCE_RANDOM_EDGE_SEED = 13
PROVENANCE_CONTROL_VARIANTS = frozenset(
    {
        "full_rgcn",
        "wo_graph",
        "homogeneous_gcn",
        "wo_feeds",
        "wo_execution_ownership",
        "wo_artifact_io",
        "wo_chunk_adjacency",
        "random_edges",
    }
)
_RELATIONS_REMOVED_BY_VARIANT: dict[str, frozenset[str]] = {
    "wo_feeds": frozenset({FEEDS_EDGE}),
    "wo_execution_ownership": frozenset(
        {RETURNS_EDGE, HAS_ARGUMENT_EDGE, HAS_CONTENT_EDGE}
    ),
    "wo_artifact_io": frozenset({READS_EDGE, WRITES_EDGE}),
    "wo_chunk_adjacency": frozenset({NEXT_CHUNK_EDGE}),
}


@dataclass(frozen=True)
class _ProvenanceControlPolicy:
    ablation_name: str
    enabled_relations: tuple[str, ...]
    relation_vocab: tuple[str, ...]
    message_transform_type: Literal["typed", "shared"]
    message_topology: Literal["native", "degree_preserving_random_v1"]
    message_topology_seed: int


@dataclass(frozen=True)
class _PhysicalMessageEdge:
    relation: str
    source: str
    target: str


@dataclass(frozen=True)
class ProvenanceTopologyDiagnostics:
    graph_id: str
    physical_edge_count: int
    active_edge_count: int
    transformed_edge_count: int
    rewiring_candidate_edge_count: int
    rewired_edge_count: int
    successful_swaps: int
    relation_counts: dict[str, int]
    rewired_relation_counts: dict[str, int]


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
    message_topology_seed: int = PROVENANCE_RANDOM_EDGE_SEED,
) -> RgcnModelConfig:
    canonical_ablation = (
        "wo_graph" if ablation_name == "wo_graph" or num_layers == 0 else ablation_name
    )
    layer_count = 0 if canonical_ablation == "wo_graph" else num_layers
    policy = _provenance_control_policy(
        canonical_ablation, message_topology_seed=message_topology_seed
    )
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
            node_feature_names=("seed_score",),
            scorer_feature_names=("seed_score",),
        ),
        relation_vocab=policy.relation_vocab,
        graph_encoder_type="identity" if layer_count == 0 else "rgcn",
        message_transform_type=policy.message_transform_type,
        edge_weight_policy="uniform",
        enabled_edge_types=(),
        enabled_provenance_relations=policy.enabled_relations,
        message_topology=policy.message_topology,
        message_topology_seed=policy.message_topology_seed,
        ablation_name=canonical_ablation,
        scoring_mode="seed_passthrough" if layer_count == 0 else "seed_residual",
    )


def _provenance_control_policy(
    ablation_name: str,
    *,
    message_topology_seed: int,
) -> _ProvenanceControlPolicy:
    if ablation_name not in PROVENANCE_CONTROL_VARIANTS:
        raise ValueError(f"unsupported provenance R-GCN variant={ablation_name!r}")
    removed = _RELATIONS_REMOVED_BY_VARIANT.get(ablation_name, frozenset())
    enabled = tuple(
        relation for relation in PROVENANCE_PHYSICAL_RELATIONS if relation not in removed
    )
    homogeneous = ablation_name == "homogeneous_gcn"
    return _ProvenanceControlPolicy(
        ablation_name=ablation_name,
        enabled_relations=enabled,
        relation_vocab=(
            HOMOGENEOUS_PROVENANCE_RELATION_VOCAB
            if homogeneous
            else PROVENANCE_RELATION_VOCAB
        ),
        message_transform_type="shared" if homogeneous else "typed",
        message_topology=(
            "degree_preserving_random_v1"
            if ablation_name == "random_edges"
            else "native"
        ),
        message_topology_seed=message_topology_seed,
    )


def require_provenance_rgcn_model_config(model_config: RgcnModelConfig) -> None:
    if model_config.method_name != "provenance_rgcn":
        raise ValueError("provenance tensorization requires method_name='provenance_rgcn'")
    policy = _provenance_control_policy(
        model_config.ablation_name,
        message_topology_seed=model_config.message_topology_seed,
    )
    # Empty enabled_provenance_relations is accepted only for historical full/wo-graph
    # schema-v5 checkpoints written before the explicit policy fields existed.
    legacy_full_policy = (
        not model_config.enabled_provenance_relations
        and model_config.ablation_name in {"full_rgcn", "wo_graph"}
    )
    if not legacy_full_policy and (
        model_config.enabled_provenance_relations != policy.enabled_relations
    ):
        raise ValueError(
            "provenance checkpoint enabled relations do not match its ablation"
        )
    if model_config.relation_vocab != policy.relation_vocab:
        raise ValueError(
            "provenance checkpoint relation vocabulary does not match its ablation"
        )
    if model_config.message_transform_type != policy.message_transform_type:
        raise ValueError(
            "provenance checkpoint message transform does not match its ablation"
        )
    if model_config.message_topology != policy.message_topology:
        raise ValueError(
            "provenance checkpoint message topology does not match its ablation"
        )
    if (
        model_config.edge_weight_policy != "uniform"
        or model_config.enabled_edge_types
        or model_config.feature_config
        != NodeFeatureConfig(
            node_feature_names=("seed_score",),
            scorer_feature_names=("seed_score",),
        )
    ):
        raise ValueError("provenance checkpoint enables invalid graph features")


def provenance_training_label(
    request: ExecutionProvenanceRankingRequest,
    *,
    gold_spans: Sequence[SourceSpan],
) -> EvidenceLabel:
    """Map natural-query exact spans to provenance candidate positives."""

    positive_ids = tuple(
        candidate.item_id
        for candidate in request.candidates
        if any(
            source_spans_overlap(candidate_span, gold_span)
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
        for source_id in output_content_by_owner.get(dependency.source_output_id, ()):
            for target_id in output_content_by_owner.get(
                dependency.target_output_id, ()
            ):
                _add_candidate_pair(pairs, source_id, target_id, candidate_ids)
    return tuple(
        CandidateNeighborEdge(source=source, target=target)
        for source, target in sorted(pairs)
    )


def provenance_train_pair_task(
    request: ExecutionProvenanceRankingRequest,
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


def tensorize_provenance_edges(
    graph: ProvenanceGraph,
    *,
    model_config: RgcnModelConfig | None = None,
) -> MessageEdgeTensors:
    """Build a variant-specific message view without mutating the persisted graph."""

    if model_config is None:
        policy = _provenance_control_policy(
            "full_rgcn", message_topology_seed=PROVENANCE_RANDOM_EDGE_SEED
        )
    else:
        require_provenance_rgcn_model_config(model_config)
        policy = _provenance_control_policy(
            model_config.ablation_name,
            message_topology_seed=model_config.message_topology_seed,
        )
    physical_edges, _diagnostics = _provenance_edge_view(graph, policy)
    node_index_by_id = {node.node_id: index for index, node in enumerate(graph.nodes)}
    relation_id_by_name = {
        name: index for index, name in enumerate(policy.relation_vocab)
    }
    sources: list[int] = []
    targets: list[int] = []
    relation_ids: list[int] = []
    for edge in physical_edges:
        source = node_index_by_id[edge.source]
        target = node_index_by_id[edge.target]
        sources.extend((source, target))
        targets.extend((target, source))
        if policy.relation_vocab == HOMOGENEOUS_PROVENANCE_RELATION_VOCAB:
            relation_ids.extend((0, 0))
        else:
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


def _provenance_edge_view(
    graph: ProvenanceGraph,
    policy: _ProvenanceControlPolicy,
) -> tuple[tuple[_PhysicalMessageEdge, ...], ProvenanceTopologyDiagnostics]:
    active_relations = set(policy.enabled_relations)
    original = tuple(
        _PhysicalMessageEdge(
            relation=edge.relation,
            source=edge.source,
            target=edge.target,
        )
        for edge in graph.edges
        if edge.relation in active_relations
    )
    transformed = original
    rewiring_candidate_edge_count = 0
    successful_swaps = 0
    if policy.message_topology == "degree_preserving_random_v1":
        transformed, rewiring_candidate_edge_count, successful_swaps = (
            _degree_preserving_random_edges(
                graph,
                original,
                seed=policy.message_topology_seed,
            )
        )
    rewired_by_relation: Counter[str] = Counter()
    for before, after in zip(original, transformed, strict=True):
        if (before.source, before.target) != (after.source, after.target):
            rewired_by_relation[before.relation] += 1
    physical_edge_count = sum(
        edge.relation in PROVENANCE_PHYSICAL_RELATIONS for edge in graph.edges
    )
    diagnostics = ProvenanceTopologyDiagnostics(
        graph_id=graph.graph_id,
        physical_edge_count=physical_edge_count,
        active_edge_count=len(original),
        transformed_edge_count=len(transformed),
        rewiring_candidate_edge_count=rewiring_candidate_edge_count,
        rewired_edge_count=sum(rewired_by_relation.values()),
        successful_swaps=successful_swaps,
        relation_counts=dict(sorted(Counter(edge.relation for edge in original).items())),
        rewired_relation_counts=dict(sorted(rewired_by_relation.items())),
    )
    return transformed, diagnostics


def _degree_preserving_random_edges(
    graph: ProvenanceGraph,
    edges: tuple[_PhysicalMessageEdge, ...],
    *,
    seed: int,
) -> tuple[tuple[_PhysicalMessageEdge, ...], int, int]:
    node_kind = {node.node_id: node.kind for node in graph.nodes}
    grouped_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        grouped_indices[
            (edge.relation, node_kind[edge.source], node_kind[edge.target])
        ].append(index)

    graph_fingerprint = graph.fingerprint()
    transformed = list(edges)
    candidate_count = 0
    successful_swaps = 0
    for group, indices in sorted(grouped_indices.items()):
        if len(indices) < 2:
            continue
        candidate_count += len(indices)
        relation, source_kind, target_kind = group
        group_seed = int.from_bytes(
            hashlib.sha256(
                "\0".join(
                    (
                        "degree-preserving-random-v1",
                        str(seed),
                        graph_fingerprint,
                        relation,
                        source_kind,
                        target_kind,
                    )
                ).encode("utf-8")
            ).digest()[:8],
            byteorder="big",
        )
        rng = random.Random(group_seed)
        occupied = {
            (transformed[index].source, transformed[index].target) for index in indices
        }
        attempts_remaining = max(100, len(indices) * 50)
        target_swaps = max(1, len(indices) * 5)
        group_swaps = 0
        while attempts_remaining > 0 and group_swaps < target_swaps:
            attempts_remaining -= 1
            left_index, right_index = rng.sample(indices, 2)
            left = transformed[left_index]
            right = transformed[right_index]
            if left.source == right.source or left.target == right.target:
                continue
            left_rewired = (left.source, right.target)
            right_rewired = (right.source, left.target)
            if left_rewired[0] == left_rewired[1] or right_rewired[0] == right_rewired[1]:
                continue
            left_original = (left.source, left.target)
            right_original = (right.source, right.target)
            occupied.remove(left_original)
            occupied.remove(right_original)
            if (
                left_rewired in occupied
                or right_rewired in occupied
                or left_rewired == right_rewired
            ):
                occupied.add(left_original)
                occupied.add(right_original)
                continue
            occupied.add(left_rewired)
            occupied.add(right_rewired)
            transformed[left_index] = _PhysicalMessageEdge(
                relation=left.relation,
                source=left_rewired[0],
                target=left_rewired[1],
            )
            transformed[right_index] = _PhysicalMessageEdge(
                relation=right.relation,
                source=right_rewired[0],
                target=right_rewired[1],
            )
            group_swaps += 1
        successful_swaps += group_swaps

    _validate_degree_preserving_rewire(graph, edges, tuple(transformed))
    return tuple(transformed), candidate_count, successful_swaps


def _validate_degree_preserving_rewire(
    graph: ProvenanceGraph,
    original: tuple[_PhysicalMessageEdge, ...],
    transformed: tuple[_PhysicalMessageEdge, ...],
) -> None:
    if len(original) != len(transformed):
        raise RuntimeError("random-edge control changed the physical edge count")
    node_kind = {node.node_id: node.kind for node in graph.nodes}

    def degree_signature(
        values: tuple[_PhysicalMessageEdge, ...],
    ) -> Counter[tuple[str, str, str]]:
        signature: Counter[tuple[str, str, str]] = Counter()
        for edge in values:
            signature[(edge.relation, "out", edge.source)] += 1
            signature[(edge.relation, "in", edge.target)] += 1
        return signature

    if degree_signature(original) != degree_signature(transformed):
        raise RuntimeError("random-edge control changed per-relation node degrees")
    if Counter(edge.relation for edge in original) != Counter(
        edge.relation for edge in transformed
    ):
        raise RuntimeError("random-edge control changed the relation histogram")
    for before, after in zip(original, transformed, strict=True):
        if before.relation != after.relation:
            raise RuntimeError("random-edge control changed an edge relation")
        if (
            node_kind[before.source] != node_kind[after.source]
            or node_kind[before.target] != node_kind[after.target]
        ):
            raise RuntimeError("random-edge control changed endpoint node kinds")
    if len({(edge.relation, edge.source, edge.target) for edge in transformed}) != len(
        transformed
    ):
        raise RuntimeError("random-edge control created duplicate physical edges")


def provenance_control_summary(
    graphs: Sequence[ProvenanceGraph],
    *,
    model_config: RgcnModelConfig,
) -> dict[str, object]:
    """Aggregate relation/topology diagnostics over task graphs for delivery."""

    require_provenance_rgcn_model_config(model_config)
    policy = _provenance_control_policy(
        model_config.ablation_name,
        message_topology_seed=model_config.message_topology_seed,
    )
    graph_list = list(graphs)
    unique_graphs: dict[str, ProvenanceGraph] = {}
    for graph in graph_list:
        unique_graphs.setdefault(graph.fingerprint(), graph)
    diagnostics = [
        _provenance_edge_view(graph, policy)[1] for graph in unique_graphs.values()
    ]
    relation_counts: Counter[str] = Counter()
    rewired_relation_counts: Counter[str] = Counter()
    for diagnostic in diagnostics:
        relation_counts.update(diagnostic.relation_counts)
        rewired_relation_counts.update(diagnostic.rewired_relation_counts)
    return {
        "control_schema": "provenance-relation-control-v1",
        "ablation_name": policy.ablation_name,
        "enabled_relations": list(policy.enabled_relations),
        "relation_vocab": list(policy.relation_vocab),
        "message_transform_type": policy.message_transform_type,
        "message_topology": policy.message_topology,
        "message_topology_seed": policy.message_topology_seed,
        "task_count": len(graph_list),
        "unique_graph_count": len(unique_graphs),
        "physical_edge_count": sum(row.physical_edge_count for row in diagnostics),
        "active_edge_count": sum(row.active_edge_count for row in diagnostics),
        "transformed_edge_count": sum(
            row.transformed_edge_count for row in diagnostics
        ),
        "rewiring_candidate_edge_count": sum(
            row.rewiring_candidate_edge_count for row in diagnostics
        ),
        "rewired_edge_count": sum(row.rewired_edge_count for row in diagnostics),
        "successful_swaps": sum(row.successful_swaps for row in diagnostics),
        "changed_graph_count": sum(row.rewired_edge_count > 0 for row in diagnostics),
        "relation_counts": dict(sorted(relation_counts.items())),
        "rewired_relation_counts": dict(sorted(rewired_relation_counts.items())),
        "graphs": [asdict(row) for row in diagnostics],
    }


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
    request: ExecutionProvenanceRankingRequest,
) -> tuple[TextRankingRequest, list[str]]:
    node_ids = [node.node_id for node in request.graph.nodes]
    if "q" in node_ids:
        raise ValueError(
            "persisted provenance graph cannot contain the ephemeral q node"
        )
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
    request: ExecutionProvenanceRankingRequest,
    *,
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
) -> EvidenceTaskTensor:
    """Adapt one physical provenance graph to the maintained task tensor contract."""

    require_provenance_rgcn_model_config(model_config)

    graph = request.graph
    fingerprint = graph.fingerprint()
    embedding_request, node_ids = provenance_embedding_request(request)
    groups = build_task_feature_groups(
        text_embedding_provider,
        seed_signal_provider,
        (
            DenseTaskEncodingRequest(
                ranking_request=embedding_request,
                node_ids=tuple(node_ids),
            ),
        ),
    )
    if len(groups) != 1:
        raise RuntimeError("provenance tensorization expected one feature group")
    dense_features = groups[0]
    node_embeddings = dense_features.node_embeddings
    expected_shape = (len(node_ids), model_config.encoder_dim)
    if tuple(node_embeddings.shape) != expected_shape:
        raise ValueError(
            "provenance node embedding shape mismatch: "
            f"expected={expected_shape} observed={tuple(node_embeddings.shape)}"
        )
    numeric_features = NodeFeatureBuilder(model_config.feature_config).build_node_features(
        node_ids=node_ids,
        seed_signals=dense_features.seed_signals,
    )

    message_edges = tensorize_provenance_edges(graph, model_config=model_config)
    candidate_indices = [
        node_ids.index(candidate.item_id) for candidate in request.candidates
    ]
    graph_tensor = TaskGraphTensor(
        node_embeddings=node_embeddings,
        node_features=numeric_features.node_features,
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
        sample_node_features=numeric_features.scorer_features[candidate_indices],
        labels=torch.zeros(len(candidate_indices), dtype=torch.float32),
        sample_node_ids=[candidate.item_id for candidate in request.candidates],
        sample_types=["easy_random"] * len(candidate_indices),
    )


def tensorize_provenance_dev_task(
    request: ExecutionProvenanceRankingRequest,
    label: EvidenceLabel,
    *,
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
) -> EvidenceTaskTensor:
    task = tensorize_provenance_ranking_task(
        request,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
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
    request: ExecutionProvenanceRankingRequest,
    pairs: Sequence[TrainPairRecord],
    *,
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
) -> EvidenceTaskTensor:
    task = tensorize_provenance_ranking_task(
        request,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
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
        if pair.node_id not in {candidate.item_id for candidate in request.candidates}:
            raise ValueError("provenance training pair is not a ranking candidate")
    candidate_position = {
        node_id: index for index, node_id in enumerate(task.sample_node_ids)
    }
    return replace(
        task,
        sample_node_indices=torch.tensor(
            [node_index[pair.node_id] for pair in rows], dtype=torch.long
        ),
        sample_node_features=torch.stack(
            [task.sample_node_features[candidate_position[pair.node_id]] for pair in rows]
        ),
        labels=torch.tensor([float(pair.label) for pair in rows], dtype=torch.float32),
        sample_node_ids=[pair.node_id for pair in rows],
        sample_types=[pair.sample_type for pair in rows],
    )


def tensorize_provenance_ranking_tasks(
    requests: Sequence[ExecutionProvenanceRankingRequest],
    *,
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
) -> list[EvidenceTaskTensor]:
    return [
        tensorize_provenance_ranking_task(
            request,
            model_config=model_config,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
        )
        for request in requests
    ]


__all__ = [
    "HOMOGENEOUS_PROVENANCE_RELATION_VOCAB",
    "PROVENANCE_CONTROL_VARIANTS",
    "PROVENANCE_PHYSICAL_RELATIONS",
    "PROVENANCE_RANDOM_EDGE_SEED",
    "PROVENANCE_RELATION_VOCAB",
    "ProvenanceTopologyDiagnostics",
    "provenance_control_summary",
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
    "require_provenance_rgcn_model_config",
]
