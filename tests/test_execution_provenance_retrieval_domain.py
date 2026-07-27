from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from graph_memory.embeddings import SentenceEncoder
from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceGraph,
    ExecutionProvenanceNode,
    FieldBinding,
    ProvenanceEdgeType,
    ProvenanceNodeType,
)
from graph_memory.registry import Registry
from graph_memory.registry.methods import RetrievalTaskFamily
from graph_memory.registry.retrieval import (
    DenseEncoderSettings,
    ExecutionProvenanceBuildPayload,
    ExecutionProvenanceRetrievalSettings,
    RetrievalMethodId,
)
from graph_memory.retrieval.methods.epgm import (
    DEFAULT_RELATION_DESCRIPTIONS,
    EpgmRetriever,
    EpgmRetrieverConfig,
    PartnerProposal,
    RelationAffinity,
    TypedArc,
    apply_promotions,
    build_typed_adjacency,
    candidate_dependency,
    propose_partners,
    query_relation_affinities,
    select_promotions,
)
from graph_memory.retrieval.contracts import (
    GraphRAGTrace,
    TypedPartnerCompletionTrace,
)
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.methods.graphrag import (
    GraphRAGConfig,
    GraphRAGMethod,
    build_graphrag_request,
)
from graph_memory.retrieval.requests import (
    EvidenceGraphRankingRequest,
    ExecutionProvenanceRankingRequest,
    TextCandidate,
    TextRankingRequest,
)
from tests.rgcn_fixtures import tiny_graphs


EXPECTED_METHOD_IDS = {
    "bm25",
    "dense",
    "dense_ft",
    "graphrag",
    "dense_rgcn_graph_retriever",
    "dense_ft_rgcn_graph_retriever",
    "execution_provenance_retriever",
    "execution_provenance_rgcn_retriever",
}


class KeywordEncoder:
    _vocab = ("alpha", "lookup", "result", "answer")

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        _ = batch_size, normalize_embeddings, show_progress_bar
        rows = []
        for text in texts:
            lowered = text.lower()
            vector = np.asarray(
                [float(lowered.count(token)) for token in self._vocab],
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(vector))
            rows.append(vector / norm if norm else vector)
        return np.asarray(rows, dtype=np.float32)


class LocalPromotionEncoder:
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        _ = batch_size, show_progress_bar
        rows: list[np.ndarray] = []
        for text in texts:
            lowered = text.casefold()
            if lowered.startswith("q:") and "source evidence:" not in lowered:
                vector = np.asarray([1.0, 0.0], dtype=np.float32)
            elif "source evidence:" in lowered:
                vector = np.asarray([0.0, 1.0], dtype=np.float32)
            elif "alpha" in lowered:
                vector = np.asarray([1.0, 0.1], dtype=np.float32)
            elif "noise one" in lowered:
                vector = np.asarray([0.9, 0.0], dtype=np.float32)
            elif "noise two" in lowered:
                vector = np.asarray([0.8, 0.0], dtype=np.float32)
            elif "bridge city" in lowered:
                vector = np.asarray([0.6, 1.0], dtype=np.float32)
            else:
                vector = np.asarray([0.1, 0.0], dtype=np.float32)
            _ = normalize_embeddings
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


class EqualEncoder:
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        _ = batch_size, normalize_embeddings, show_progress_bar
        return np.asarray([[1.0, 0.0] for _text in texts], dtype=np.float32)


class RevisionEncoder:
    """Ranks the verification first and the overturned claim outside the top-1.

    Mirrors the RQ3 failure shape: the query names the check, not the claim it
    invalidated, so the partner must be recovered structurally.
    """

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        _ = batch_size, normalize_embeddings, show_progress_bar
        rows: list[np.ndarray] = []
        for text in texts:
            lowered = text.casefold()
            if "verification" in lowered:
                vector = np.asarray([1.0, 0.0], dtype=np.float32)
            elif "noise" in lowered:
                vector = np.asarray([0.5, 0.0], dtype=np.float32)
            elif "obsolete" in lowered:
                vector = np.asarray([0.2, 0.0], dtype=np.float32)
            else:
                vector = np.asarray([0.9, 0.0], dtype=np.float32)
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


def _candidates() -> tuple[TextCandidate, ...]:
    return (
        TextCandidate("call-1", "lookup Alpha", {}),
        TextCandidate("out-1", "Alpha result", {}),
        TextCandidate("call-2", "use Alpha result", {}),
        TextCandidate("out-2", "final Alpha answer", {}),
    )


def _dense_ranker() -> DenseTaskRetriever:
    return DenseTaskRetriever(
        encoder=KeywordEncoder(), query_prefix="", passage_prefix=""
    )


def _local_dense_ranker(
    encoder: SentenceEncoder | None = None,
) -> DenseTaskRetriever:
    return DenseTaskRetriever(
        encoder=encoder or LocalPromotionEncoder(),
        query_prefix="Q:",
        passage_prefix="P:",
    )


def _local_graphrag_candidates() -> tuple[TextCandidate, ...]:
    return (
        TextCandidate(
            "a",
            "Alpha. Alpha was born in Bridge City.",
            {"title": "Alpha", "source_ref": "Alpha"},
        ),
        TextCandidate("x", "Noise One. Distractor.", {"title": "Noise One"}),
        TextCandidate("y", "Noise Two. Distractor.", {"title": "Noise Two"}),
        TextCandidate(
            "b",
            "Bridge City. Bridge City is located in Country Z.",
            {"title": "Bridge City"},
        ),
    )


def _local_provenance_request(
    *, valid_binding: bool = True
) -> ExecutionProvenanceRankingRequest:
    binding_hash = "alpha-hash" if valid_binding else "wrong-hash"
    nodes = (
        ExecutionProvenanceNode(
            "call-a",
            ProvenanceNodeType.TOOL_CALL,
            "Alpha call",
            {"input_parameters": ["context"]},
        ),
        ExecutionProvenanceNode(
            "a",
            ProvenanceNodeType.TOOL_OUTPUT,
            "Alpha source",
            {"output_field_hashes": {"evidence": "alpha-hash"}},
        ),
        ExecutionProvenanceNode("x", ProvenanceNodeType.TOOL_OUTPUT, "Noise One"),
        ExecutionProvenanceNode("y", ProvenanceNodeType.TOOL_OUTPUT, "Noise Two"),
        ExecutionProvenanceNode(
            "call-b",
            ProvenanceNodeType.TOOL_CALL,
            "Bridge call",
            {"input_parameters": ["context"]},
        ),
        ExecutionProvenanceNode(
            "b", ProvenanceNodeType.TOOL_OUTPUT, "Bridge City target"
        ),
    )
    edges = (
        ExecutionProvenanceEdge("call-a", "a", ProvenanceEdgeType.RETURNS),
        ExecutionProvenanceEdge(
            "a",
            "call-b",
            ProvenanceEdgeType.FEEDS,
            binding=FieldBinding(
                "evidence", "context", binding_hash, "semantic_reference"
            ),
            weight=0.8,
            metadata={
                "semantic_scorer": "frozen_dense",
                "semantic_rank": 1,
                "semantic_score": 0.7,
            },
        ),
        ExecutionProvenanceEdge("call-b", "b", ProvenanceEdgeType.RETURNS),
    )
    return ExecutionProvenanceRankingRequest(
        "local-path",
        "base query",
        (
            TextCandidate("a", "Alpha source", {}),
            TextCandidate("x", "Noise One", {}),
            TextCandidate("y", "Noise Two", {}),
            TextCandidate("b", "Bridge City target", {}),
        ),
        ExecutionProvenanceGraph("local-path", nodes, edges),
    )


def _provenance_graph(
    *, include_untraversed_edge: bool = False
) -> ExecutionProvenanceGraph:
    nodes = (
        ExecutionProvenanceNode("task", ProvenanceNodeType.TASK, "find Alpha"),
        ExecutionProvenanceNode("agent", ProvenanceNodeType.AGENT, "assistant"),
        ExecutionProvenanceNode("call-1", ProvenanceNodeType.TOOL_CALL, "lookup Alpha"),
        ExecutionProvenanceNode(
            "out-1",
            ProvenanceNodeType.TOOL_OUTPUT,
            "Alpha result",
            {"output_field_hashes": {"result": "alpha-hash"}},
        ),
        ExecutionProvenanceNode(
            "call-2",
            ProvenanceNodeType.TOOL_CALL,
            "use Alpha result",
            {"input_parameters": ["input"]},
        ),
        ExecutionProvenanceNode(
            "out-2", ProvenanceNodeType.TOOL_OUTPUT, "final Alpha answer"
        ),
        ExecutionProvenanceNode("answer", ProvenanceNodeType.ANSWER, "Alpha"),
    )
    edges = [
        ExecutionProvenanceEdge("agent", "call-1", ProvenanceEdgeType.INVOKES),
        ExecutionProvenanceEdge("call-1", "out-1", ProvenanceEdgeType.RETURNS),
        ExecutionProvenanceEdge(
            "out-1",
            "call-2",
            ProvenanceEdgeType.FEEDS,
            binding=FieldBinding("result", "input", "alpha-hash", "exact"),
            weight=0.9,
            metadata={
                "semantic_scorer": "frozen_dense",
                "semantic_rank": 1,
                "semantic_score": 0.8,
            },
        ),
        ExecutionProvenanceEdge("call-2", "out-2", ProvenanceEdgeType.RETURNS),
        ExecutionProvenanceEdge("out-2", "answer", ProvenanceEdgeType.GROUNDS),
    ]
    if include_untraversed_edge:
        edges.append(
            ExecutionProvenanceEdge("call-1", "call-2", ProvenanceEdgeType.PRECEDES)
        )
    return ExecutionProvenanceGraph("task-1", nodes, tuple(edges))


def test_registry_exposes_exact_method_matrix_and_semantic_inputs() -> None:
    assert {
        method.value for method in Registry.methods.list_ids()
    } == EXPECTED_METHOD_IDS

    rgcn = Registry.methods.get(RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER)
    assert rgcn.request_type is EvidenceGraphRankingRequest
    assert rgcn.supported_families == frozenset(
        {RetrievalTaskFamily.EVIDENCE_RETRIEVAL}
    )

    dense_ft = Registry.methods.get(RetrievalMethodId.DENSE_FT)
    assert dense_ft.request_type is TextRankingRequest
    assert dense_ft.supported_families == frozenset(
        {
            RetrievalTaskFamily.EVIDENCE_RETRIEVAL,
            RetrievalTaskFamily.EXECUTION_PROVENANCE,
        }
    )


def test_retired_method_id_is_explicitly_unsupported() -> None:
    legacy_method = "memory" + "_stream"

    with pytest.raises(ValueError, match="Unsupported retrieval method"):
        Registry.methods.get(legacy_method)


def test_registry_rejects_cross_graph_method_requests() -> None:
    candidates = _candidates()
    provenance_request = ExecutionProvenanceRankingRequest(
        "task-1", "find Alpha", candidates, _provenance_graph()
    )
    evidence_request = EvidenceGraphRankingRequest(
        "hotpot_rgcn_train", "find Alpha", candidates[:3], tiny_graphs()[0], {}
    )

    with pytest.raises(TypeError, match="EvidenceGraphRankingRequest"):
        Registry.methods.validate_request(
            RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            provenance_request,
            RetrievalTaskFamily.EXECUTION_PROVENANCE,
        )
    with pytest.raises(TypeError, match="ExecutionProvenanceRankingRequest"):
        Registry.methods.validate_request(
            RetrievalMethodId.EXECUTION_PROVENANCE_RETRIEVER,
            evidence_request,
            RetrievalTaskFamily.EVIDENCE_RETRIEVAL,
        )


def test_feeds_requires_binding_but_precedes_does_not() -> None:
    with pytest.raises(ValueError, match="binding"):
        ExecutionProvenanceEdge("out-1", "call-2", ProvenanceEdgeType.FEEDS)

    chronological = ExecutionProvenanceEdge(
        "out-1", "call-2", ProvenanceEdgeType.PRECEDES
    )
    assert chronological.binding is None


def test_graphrag_builds_typed_mentions_without_evidence_graph() -> None:
    config = GraphRAGConfig(seed_top_s=2)
    request = build_graphrag_request(
        TextRankingRequest("task-1", "Alpha answer", _candidates()), config
    )
    method = GraphRAGMethod(
        dense_ranker=_dense_ranker(),
        config=config,
    )

    result = method.rank_task(request, top_k=2)

    assert len(result.ranked_nodes) == len(request.candidates)
    assert isinstance(result.trace.native_trace, GraphRAGTrace)
    assert result.trace.native_trace.exact_dense_fallback
    assert result.trace.native_trace.dense_ranks
    assert result.trace.retrieved_edges == []


def test_graphrag_promotes_only_resolved_partner_after_protected_prefix() -> None:
    candidates = _local_graphrag_candidates()
    config = GraphRAGConfig(
        seed_top_s=1,
        max_entity_document_frequency_ratio=0.75,
        min_sentence_score_margin=0.02,
        min_bridge_confidence=0.0,
        preserve_dense_top_n=2,
    )
    ranker = _local_dense_ranker()
    request = build_graphrag_request(
        TextRankingRequest("local-bridge", "Which country?", candidates), config
    )

    result = GraphRAGMethod(ranker, config).rank_task(request, top_k=4)

    assert [node.node_id for node in result.ranked_nodes] == ["a", "x", "b", "y"]
    assert [node.score for node in result.ranked_nodes] == sorted(
        [node.score for node in result.ranked_nodes], reverse=True
    )
    trace = result.trace.native_trace
    assert isinstance(trace, GraphRAGTrace)
    accepted = next(item for item in trace.bridges if item.accepted)
    assert result.trace.retrieved_edges == [
        {
            "source": "a",
            "target": "b",
            "edge_type": "bridge_to",
            "weight": accepted.bridge.confidence,
            "directed": True,
        }
    ]
    assert not trace.exact_dense_fallback


def test_graphrag_low_margin_abstains_to_exact_dense_objects() -> None:
    candidates = (
        *_local_graphrag_candidates(),
        TextCandidate(
            "b2",
            "Bridge City. Another possible sentence.",
            {"title": "Bridge City"},
        ),
    )
    config = GraphRAGConfig(
        seed_top_s=1,
        max_entity_document_frequency_ratio=0.8,
        min_sentence_score_margin=0.02,
        min_bridge_confidence=0.0,
        preserve_dense_top_n=2,
    )
    ranker = _local_dense_ranker(EqualEncoder())
    request = build_graphrag_request(
        TextRankingRequest("ambiguous-bridge", "Which country?", candidates), config
    )
    dense = ranker.rank(
        TextRankingRequest(request.task_id, request.query_text, request.candidates)
    )

    result = GraphRAGMethod(ranker, config).rank_task(request, top_k=5)

    assert result.ranked_nodes == dense
    trace = result.trace.native_trace
    assert isinstance(trace, GraphRAGTrace)
    assert trace.exact_dense_fallback
    assert any(
        evidence.rejection_reason == "ambiguous_title_sentence"
        for evidence in trace.resolver_evidence
    )


def _revision_request() -> ExecutionProvenanceRankingRequest:
    """An audit graph with no feeds/returns backbone at all.

    A verification contradicts an earlier claim. This is the structure the
    dependency-path preset cannot use: its walk is restricted to dependency
    edges and its gate demands one feeds plus one returns edge, so the
    overturned claim is unreachable.
    """

    nodes = (
        ExecutionProvenanceNode(
            "verify", ProvenanceNodeType.VERIFICATION, "verification alpha check"
        ),
        ExecutionProvenanceNode(
            "claim-old",
            ProvenanceNodeType.CLAIM,
            "obsolete unrelated wording",
            {"lifecycle_state": "invalidated"},
        ),
        ExecutionProvenanceNode("noise", ProvenanceNodeType.CLAIM, "noise one"),
    )
    edges = (
        ExecutionProvenanceEdge(
            "verify", "claim-old", ProvenanceEdgeType.CONTRADICTS
        ),
    )
    return ExecutionProvenanceRankingRequest(
        "revision-task",
        "alpha",
        (
            TextCandidate("verify", "verification alpha check", {}),
            TextCandidate("claim-old", "obsolete unrelated wording", {}),
            TextCandidate("noise", "noise one", {}),
        ),
        ExecutionProvenanceGraph("revision-task", nodes, edges),
    )


def test_provenance_proposals_cover_typed_neighbours_with_reasons() -> None:
    """Every enumerated proposal is auditable, including the rejected ones.

    In this fixture Dense already ranks each partner at or above its anchor, so
    no promotion is admissible. The correct outcome is exact Dense fallback with
    a stated reason per proposal, not silence.
    """

    graph = _provenance_graph(include_untraversed_edge=True)
    request = ExecutionProvenanceRankingRequest(
        "task-1", "Alpha answer", _candidates(), graph
    )
    ranker = _dense_ranker()
    dense = ranker.rank(
        TextRankingRequest(request.task_id, request.query_text, request.candidates)
    )
    method = EpgmRetriever(
        dense_ranker=ranker,
        config=EpgmRetrieverConfig(
            anchor_top_a=4, preserve_dense_top_n=0, min_partner_confidence=0.0
        ),
    )

    result = method.rank_task(request, top_k=4)

    assert result.ranked_nodes == dense
    trace = result.trace.native_trace
    assert isinstance(trace, TypedPartnerCompletionTrace)
    assert trace.proposals
    assert trace.exact_dense_fallback
    assert all(
        not proposal.accepted and proposal.rejection_reason
        for proposal in trace.proposals
    )
    # `precedes` is traversable for ranking, but never emits a dependency edge.
    assert result.trace.retrieved_edges == []


def test_promoted_arcs_resolve_to_stored_provenance_edges() -> None:
    request = _local_provenance_request()
    method = EpgmRetriever(
        dense_ranker=_local_dense_ranker(),
        config=EpgmRetrieverConfig(
            anchor_top_a=1, preserve_dense_top_n=0, min_partner_confidence=0.2
        ),
    )

    result = method.rank_task(request, top_k=4)
    trace = result.trace.native_trace

    assert isinstance(trace, TypedPartnerCompletionTrace)
    assert not trace.exact_dense_fallback
    stored = {
        (edge.source, edge.target, edge.edge_type.value)
        for edge in request.graph.edges
    }
    assert trace.promoted_native_edges
    assert all(
        (edge.source, edge.target, edge.edge_type.value) in stored
        for edge in trace.promoted_native_edges
    )


def test_contains_edges_are_never_traversed() -> None:
    """Scope membership must not make co-scoped candidates partners."""

    nodes = (
        ExecutionProvenanceNode("task", ProvenanceNodeType.TASK, "scope"),
        ExecutionProvenanceNode("a", ProvenanceNodeType.TOOL_OUTPUT, "alpha one"),
        ExecutionProvenanceNode("b", ProvenanceNodeType.TOOL_OUTPUT, "alpha two"),
    )
    edges = (
        ExecutionProvenanceEdge("task", "a", ProvenanceEdgeType.CONTAINS),
        ExecutionProvenanceEdge("task", "b", ProvenanceEdgeType.CONTAINS),
    )
    graph = ExecutionProvenanceGraph("scope-task", nodes, edges)
    adjacency = build_typed_adjacency(graph)

    assert adjacency == {}

    proposals = propose_partners(
        graph,
        anchor_ids=("a",),
        candidate_ids=frozenset({"a", "b"}),
        dense_relevance={"a": 1.0, "b": 0.5},
        relation_affinities=(
            RelationAffinity("contains", 1.0, 1.0, 1.0, 1.0),
        ),
        config=EpgmRetrieverConfig(),
    )

    assert proposals == ()


def test_zero_weight_edge_is_not_traversable() -> None:
    nodes = (
        ExecutionProvenanceNode("a", ProvenanceNodeType.TOOL_OUTPUT, "alpha"),
        ExecutionProvenanceNode("b", ProvenanceNodeType.TOOL_OUTPUT, "beta"),
    )
    edges = (
        ExecutionProvenanceEdge(
            "a", "b", ProvenanceEdgeType.DEPENDS_ON, weight=0.0
        ),
    )
    graph = ExecutionProvenanceGraph("zero-task", nodes, edges)

    assert build_typed_adjacency(graph) == {}


def test_unbound_feeds_edge_remains_traversable() -> None:
    """A binding gate is deliberately absent; only the edge type matters."""

    request = _local_provenance_request(valid_binding=False)
    method = EpgmRetriever(
        dense_ranker=_local_dense_ranker(),
        config=EpgmRetrieverConfig(
            anchor_top_a=1, preserve_dense_top_n=0, min_partner_confidence=0.2
        ),
    )

    result = method.rank_task(request, top_k=4)
    trace = result.trace.native_trace

    assert isinstance(trace, TypedPartnerCompletionTrace)
    assert any(
        proposal.partner_id == "b" and proposal.accepted
        for proposal in trace.proposals
    )


def test_constant_recorded_weights_do_not_change_confidence() -> None:
    """Confidence must not read weight magnitude, so RQ3 graphs behave."""

    def confidences(weight: float) -> list[float]:
        nodes = (
            ExecutionProvenanceNode("a", ProvenanceNodeType.TOOL_OUTPUT, "alpha"),
            ExecutionProvenanceNode("b", ProvenanceNodeType.TOOL_OUTPUT, "beta"),
        )
        edges = (
            ExecutionProvenanceEdge(
                "a", "b", ProvenanceEdgeType.DEPENDS_ON, weight=weight
            ),
        )
        graph = ExecutionProvenanceGraph("w-task", nodes, edges)
        proposals = propose_partners(
            graph,
            anchor_ids=("a",),
            candidate_ids=frozenset({"a", "b"}),
            dense_relevance={"a": 1.0, "b": 0.7},
            relation_affinities=(
                RelationAffinity("depends_on", 1.0, 1.0, 1.0, 1.0),
            ),
            config=EpgmRetrieverConfig(),
        )
        return [proposal.confidence for proposal in proposals]

    assert confidences(1.0) == confidences(0.4)


def test_partner_relevance_drives_confidence_not_anchor_relevance() -> None:
    """Measured on RQ3: partner relevance predicts gold, anchor relevance does not."""

    nodes = (
        ExecutionProvenanceNode("a", ProvenanceNodeType.TOOL_OUTPUT, "anchor"),
        ExecutionProvenanceNode("strong", ProvenanceNodeType.TOOL_OUTPUT, "strong"),
        ExecutionProvenanceNode("weak", ProvenanceNodeType.TOOL_OUTPUT, "weak"),
    )
    edges = (
        ExecutionProvenanceEdge("a", "strong", ProvenanceEdgeType.DEPENDS_ON),
        ExecutionProvenanceEdge("a", "weak", ProvenanceEdgeType.DEPENDS_ON),
    )
    graph = ExecutionProvenanceGraph("rel-task", nodes, edges)
    proposals = propose_partners(
        graph,
        anchor_ids=("a",),
        candidate_ids=frozenset({"a", "strong", "weak"}),
        dense_relevance={"a": 1.0, "strong": 0.9, "weak": 0.1},
        relation_affinities=(RelationAffinity("depends_on", 1.0, 1.0, 1.0, 1.0),),
        config=EpgmRetrieverConfig(),
    )

    by_partner = {proposal.partner_id: proposal.confidence for proposal in proposals}
    assert by_partner["strong"] > by_partner["weak"]
    assert by_partner["strong"] == pytest.approx(0.9)


def test_rare_relation_outranks_bulk_relation_at_equal_query_match() -> None:
    """Inverse frequency is what separates evidence relations from the backbone."""

    counts = {"depends_on": 40, "invalidates": 1}
    vectors = np.zeros((len(DEFAULT_RELATION_DESCRIPTIONS), 2), dtype=float)
    order = sorted(DEFAULT_RELATION_DESCRIPTIONS)
    for edge_type in ("depends_on", "invalidates"):
        vectors[order.index(edge_type)] = np.asarray([1.0, 0.0])
    query = np.asarray([1.0, 0.0])

    relations = query_relation_affinities(
        query,
        vectors,
        frozenset(counts),
        EpgmRetrieverConfig(),
        edge_type_counts=counts,
    )
    weight = {item.edge_type: item.traversal_weight for item in relations}

    # Identical query similarity, so preference alone cannot separate them.
    assert weight["invalidates"] > weight["depends_on"]


def test_specificity_defaults_to_one_when_counts_are_absent() -> None:
    """Omitting counts isolates the pure query-similarity signal for ablation."""

    vectors = np.zeros((len(DEFAULT_RELATION_DESCRIPTIONS), 2), dtype=float)
    order = sorted(DEFAULT_RELATION_DESCRIPTIONS)
    vectors[order.index("depends_on")] = np.asarray([1.0, 0.0])
    vectors[order.index("supports")] = np.asarray([0.0, 1.0])

    relations = query_relation_affinities(
        np.asarray([1.0, 0.0]),
        vectors,
        frozenset({"depends_on", "supports"}),
        EpgmRetrieverConfig(),
    )

    assert all(item.specificity == pytest.approx(1.0) for item in relations)
    assert all(
        item.traversal_weight == pytest.approx(item.preference) for item in relations
    )


def test_relation_affinity_is_a_distribution_over_present_types() -> None:
    vectors = np.zeros((len(DEFAULT_RELATION_DESCRIPTIONS), 2), dtype=float)
    order = sorted(DEFAULT_RELATION_DESCRIPTIONS)
    vectors[order.index("depends_on")] = np.asarray([1.0, 0.0])
    vectors[order.index("supports")] = np.asarray([0.6, 0.8])

    relations = query_relation_affinities(
        np.asarray([1.0, 0.0]),
        vectors,
        frozenset({"depends_on", "supports"}),
        EpgmRetrieverConfig(),
    )

    assert {item.edge_type for item in relations} == {"depends_on", "supports"}
    assert sum(item.affinity for item in relations) == pytest.approx(1.0)
    assert max(item.preference for item in relations) == pytest.approx(1.0)
    # The query matches depends_on exactly, so it must win on preference.
    best = max(relations, key=lambda item: item.preference)
    assert best.edge_type == "depends_on"


def test_query_conditioning_uses_embeddings_not_dataset_rules() -> None:
    """Two different queries over one graph must weight relations differently."""

    vectors = np.zeros((len(DEFAULT_RELATION_DESCRIPTIONS), 2), dtype=float)
    order = sorted(DEFAULT_RELATION_DESCRIPTIONS)
    vectors[order.index("depends_on")] = np.asarray([1.0, 0.0])
    vectors[order.index("contradicts")] = np.asarray([0.0, 1.0])
    present = frozenset({"depends_on", "contradicts"})

    def winner(query: NDArray[np.float64]) -> str:
        relations = query_relation_affinities(
            query, vectors, present, EpgmRetrieverConfig()
        )
        return max(relations, key=lambda item: item.preference).edge_type

    assert winner(np.asarray([1.0, 0.0])) == "depends_on"
    assert winner(np.asarray([0.0, 1.0])) == "contradicts"


def test_proposals_are_exhaustive_within_the_hop_bound() -> None:
    """A far candidate is proposed even when nearer candidates exist.

    This is the property a budgeted greedy selector lacks: there, one long
    first path could consume the whole budget and starve this partner.
    """

    nodes = tuple(
        ExecutionProvenanceNode(f"c{i}", ProvenanceNodeType.TOOL_OUTPUT, f"node {i}")
        for i in range(4)
    )
    edges = tuple(
        ExecutionProvenanceEdge(
            f"c{i}", f"c{i + 1}", ProvenanceEdgeType.DEPENDS_ON
        )
        for i in range(3)
    )
    graph = ExecutionProvenanceGraph("chain-task", nodes, edges)
    candidate_ids = frozenset(f"c{i}" for i in range(4))
    relations = (RelationAffinity("depends_on", 1.0, 1.0, 1.0, 1.0),)

    proposals = propose_partners(
        graph,
        anchor_ids=("c0",),
        candidate_ids=candidate_ids,
        dense_relevance={node_id: 1.0 for node_id in candidate_ids},
        relation_affinities=relations,
        config=EpgmRetrieverConfig(max_hops=2),
    )

    assert {proposal.partner_id for proposal in proposals} == {"c1", "c2"}

    deeper = propose_partners(
        graph,
        anchor_ids=("c0",),
        candidate_ids=candidate_ids,
        dense_relevance={node_id: 1.0 for node_id in candidate_ids},
        relation_affinities=relations,
        config=EpgmRetrieverConfig(max_hops=3),
    )

    assert {proposal.partner_id for proposal in deeper} == {"c1", "c2", "c3"}


def test_shorter_path_wins_for_the_same_partner() -> None:
    nodes = (
        ExecutionProvenanceNode("a", ProvenanceNodeType.TOOL_OUTPUT, "a"),
        ExecutionProvenanceNode("mid", ProvenanceNodeType.TOOL_OUTPUT, "mid"),
        ExecutionProvenanceNode("b", ProvenanceNodeType.TOOL_OUTPUT, "b"),
    )
    edges = (
        ExecutionProvenanceEdge("a", "b", ProvenanceEdgeType.DEPENDS_ON),
        ExecutionProvenanceEdge("a", "mid", ProvenanceEdgeType.DEPENDS_ON),
        ExecutionProvenanceEdge("mid", "b", ProvenanceEdgeType.DEPENDS_ON),
    )
    graph = ExecutionProvenanceGraph("dual-task", nodes, edges)

    proposals = propose_partners(
        graph,
        anchor_ids=("a",),
        candidate_ids=frozenset({"a", "mid", "b"}),
        dense_relevance={"a": 1.0, "mid": 1.0, "b": 1.0},
        relation_affinities=(RelationAffinity("depends_on", 1.0, 1.0, 1.0, 1.0),),
        config=EpgmRetrieverConfig(max_hops=2, hop_decay=0.6),
    )

    to_b = next(item for item in proposals if item.partner_id == "b")
    assert to_b.hops == 1
    assert to_b.confidence == pytest.approx(1.0)


def test_non_candidate_connector_is_reported_but_never_ranked() -> None:
    request = _local_provenance_request()
    method = EpgmRetriever(
        dense_ranker=_local_dense_ranker(),
        config=EpgmRetrieverConfig(
            anchor_top_a=1, preserve_dense_top_n=0, min_partner_confidence=0.2
        ),
    )

    result = method.rank_task(request, top_k=4)
    trace = result.trace.native_trace

    assert isinstance(trace, TypedPartnerCompletionTrace)
    ranked_ids = {node.node_id for node in result.ranked_nodes}
    assert ranked_ids == {"a", "x", "y", "b"}
    # `call-b` joins the two candidate outputs without consuming a slot.
    assert "call-b" in trace.connector_node_ids
    assert not set(trace.connector_node_ids) & ranked_ids


def test_promotion_moves_partner_directly_after_anchor() -> None:
    dense_ids = ["a", "n1", "n2", "n3", "b"]
    dense_rank = {node_id: i for i, node_id in enumerate(dense_ids, start=1)}
    proposal = PartnerProposal(
        anchor_id="a",
        partner_id="b",
        path_node_ids=("a", "b"),
        arcs=(TypedArc("a", "b", "depends_on", "forward"),),
        confidence=0.9,
    )
    outcomes: dict[PartnerProposal, tuple[bool, str | None]] = {
        proposal: (True, None)
    }

    final_ids, promoted = apply_promotions(
        dense_ids,
        outcomes=outcomes,
        dense_rank=dense_rank,
        config=EpgmRetrieverConfig(preserve_dense_top_n=0),
    )

    assert final_ids == ["a", "b", "n1", "n2", "n3"]
    assert promoted == (proposal,)


def test_protected_prefix_blocks_promotion_of_an_early_partner() -> None:
    dense_ids = ["a", "b", "c"]
    dense_rank = {node_id: i for i, node_id in enumerate(dense_ids, start=1)}
    proposal = PartnerProposal(
        anchor_id="a",
        partner_id="b",
        path_node_ids=("a", "b"),
        arcs=(TypedArc("a", "b", "depends_on", "forward"),),
        confidence=0.9,
    )

    outcomes = select_promotions(
        [proposal],
        dense_rank=dense_rank,
        config=EpgmRetrieverConfig(preserve_dense_top_n=2),
    )

    assert outcomes[proposal] == (False, "protected_partner")


def test_partner_already_above_anchor_is_rejected() -> None:
    dense_rank = {"b": 1, "a": 2}
    proposal = PartnerProposal(
        anchor_id="a",
        partner_id="b",
        path_node_ids=("a", "b"),
        arcs=(TypedArc("a", "b", "depends_on", "forward"),),
        confidence=0.9,
    )

    outcomes = select_promotions(
        [proposal],
        dense_rank=dense_rank,
        config=EpgmRetrieverConfig(preserve_dense_top_n=0),
    )

    assert outcomes[proposal] == (False, "partner_not_after_anchor")


def test_below_threshold_proposal_is_rejected_with_a_reason() -> None:
    dense_rank = {"a": 1, "b": 5}
    proposal = PartnerProposal(
        anchor_id="a",
        partner_id="b",
        path_node_ids=("a", "b"),
        arcs=(TypedArc("a", "b", "depends_on", "forward"),),
        confidence=0.05,
    )

    outcomes = select_promotions(
        [proposal],
        dense_rank=dense_rank,
        config=EpgmRetrieverConfig(min_partner_confidence=0.2),
    )

    assert outcomes[proposal] == (False, "below_partner_confidence")


def test_one_partner_per_anchor_and_one_anchor_per_partner() -> None:
    dense_rank = {"a1": 1, "a2": 2, "p1": 6, "p2": 7}

    def make(anchor: str, partner: str, confidence: float) -> PartnerProposal:
        return PartnerProposal(
            anchor_id=anchor,
            partner_id=partner,
            path_node_ids=(anchor, partner),
            arcs=(TypedArc(anchor, partner, "depends_on", "forward"),),
            confidence=confidence,
        )

    strong = make("a1", "p1", 0.9)
    weaker = make("a1", "p2", 0.5)
    conflict = make("a2", "p1", 0.8)

    outcomes = select_promotions(
        [strong, weaker, conflict],
        dense_rank=dense_rank,
        config=EpgmRetrieverConfig(preserve_dense_top_n=0),
    )

    assert outcomes[strong] == (True, None)
    assert outcomes[weaker] == (False, "lower_confidence_for_anchor")
    assert outcomes[conflict] == (False, "partner_conflict")


def test_promotion_preserves_the_dense_score_multiset() -> None:
    request = _local_provenance_request()
    ranker = _local_dense_ranker()
    dense = ranker.rank(
        TextRankingRequest(request.task_id, request.query_text, request.candidates)
    )
    method = EpgmRetriever(
        dense_ranker=ranker,
        config=EpgmRetrieverConfig(
            anchor_top_a=1, preserve_dense_top_n=0, min_partner_confidence=0.2
        ),
    )

    result = method.rank_task(request, top_k=4)

    assert [node.score for node in result.ranked_nodes] == [
        node.score for node in dense
    ]
    assert {node.node_id for node in result.ranked_nodes} == {
        node.node_id for node in dense
    }
    assert [node.node_id for node in result.ranked_nodes] != [
        node.node_id for node in dense
    ]


def test_exact_dense_fallback_when_no_partner_is_admissible() -> None:
    nodes = (
        ExecutionProvenanceNode("a", ProvenanceNodeType.TOOL_OUTPUT, "Alpha source"),
        ExecutionProvenanceNode("x", ProvenanceNodeType.TOOL_OUTPUT, "Noise One"),
    )
    graph = ExecutionProvenanceGraph("empty-task", nodes, ())
    request = ExecutionProvenanceRankingRequest(
        "empty-task",
        "base query",
        (
            TextCandidate("a", "Alpha source", {}),
            TextCandidate("x", "Noise One", {}),
        ),
        graph,
    )
    ranker = _local_dense_ranker()
    dense = ranker.rank(
        TextRankingRequest(request.task_id, request.query_text, request.candidates)
    )
    method = EpgmRetriever(dense_ranker=ranker, config=EpgmRetrieverConfig())

    result = method.rank_task(request, top_k=2)
    trace = result.trace.native_trace

    assert result.ranked_nodes == dense
    assert result.trace.retrieved_edges == []
    assert isinstance(trace, TypedPartnerCompletionTrace)
    assert trace.exact_dense_fallback
    assert trace.connector_node_ids == ()
    assert trace.emitted_edges == ()
    assert trace.promoted_native_edges == ()


def test_reverse_walk_emits_stored_orientation() -> None:
    proposal = PartnerProposal(
        anchor_id="downstream",
        partner_id="upstream",
        path_node_ids=("downstream", "upstream"),
        arcs=(TypedArc("downstream", "upstream", "depends_on", "reverse"),),
        confidence=0.7,
    )

    dependency = candidate_dependency(proposal)

    assert dependency is not None
    # The walk went against the recorded direction, so the emitted edge flips.
    assert (dependency.source, dependency.target) == ("upstream", "downstream")
    assert dependency.edge_type == "depends_on"


def test_divergent_path_emits_no_fabricated_dependency() -> None:
    """`A <- X -> B` is connected but implies no dependency either way."""

    proposal = PartnerProposal(
        anchor_id="a",
        partner_id="b",
        path_node_ids=("a", "x", "b"),
        arcs=(
            TypedArc("a", "x", "depends_on", "reverse"),
            TypedArc("x", "b", "depends_on", "forward"),
        ),
        confidence=0.5,
    )

    assert candidate_dependency(proposal) is None


def test_structural_only_path_emits_no_candidate_dependency() -> None:
    """`invokes`/`returns` can rank evidence without being a contracted edge."""

    proposal = PartnerProposal(
        anchor_id="a",
        partner_id="b",
        path_node_ids=("a", "b"),
        arcs=(TypedArc("a", "b", "invokes", "forward"),),
        confidence=0.6,
    )

    assert candidate_dependency(proposal) is None


def test_rq3_shaped_revision_graph_promotes_the_overturned_claim() -> None:
    """No feeds, no bindings, constant weights: the real agent-trace shape."""

    request = _revision_request()
    method = EpgmRetriever(
        dense_ranker=_local_dense_ranker(encoder=RevisionEncoder()),
        config=EpgmRetrieverConfig(
            anchor_top_a=1, preserve_dense_top_n=0, min_partner_confidence=0.1
        ),
    )

    result = method.rank_task(request, top_k=3)
    trace = result.trace.native_trace

    assert isinstance(trace, TypedPartnerCompletionTrace)
    accepted = [item for item in trace.proposals if item.accepted]
    assert [item.partner_id for item in accepted] == ["claim-old"]
    assert [node.node_id for node in result.ranked_nodes][:2] == [
        "verify",
        "claim-old",
    ]


def test_upstream_partner_is_reachable_without_directed_gating() -> None:
    """RQ3 gold often sits upstream of the anchor, so the walk is undirected."""

    nodes = (
        ExecutionProvenanceNode("source", ProvenanceNodeType.TOOL_OUTPUT, "source"),
        ExecutionProvenanceNode("claim", ProvenanceNodeType.CLAIM, "claim"),
    )
    edges = (
        ExecutionProvenanceEdge("source", "claim", ProvenanceEdgeType.GROUNDS),
    )
    graph = ExecutionProvenanceGraph("upstream-task", nodes, edges)

    proposals = propose_partners(
        graph,
        anchor_ids=("claim",),
        candidate_ids=frozenset({"source", "claim"}),
        dense_relevance={"claim": 1.0, "source": 0.8},
        relation_affinities=(RelationAffinity("grounds", 1.0, 1.0, 1.0, 1.0),),
        config=EpgmRetrieverConfig(),
    )

    assert [item.partner_id for item in proposals] == ["source"]
    assert proposals[0].arcs[0].direction == "reverse"


def test_proposal_enumeration_is_deterministic() -> None:
    graph = _provenance_graph(include_untraversed_edge=True)
    candidate_ids = frozenset(
        candidate.item_id for candidate in _candidates()
    )
    relevance = {node_id: 0.5 for node_id in candidate_ids}
    relations = (
        RelationAffinity("returns", 1.0, 0.5, 1.0, 1.0),
        RelationAffinity("feeds", 1.0, 0.3, 0.8, 0.9),
        RelationAffinity("precedes", 1.0, 0.2, 0.4, 0.7),
    )
    config = EpgmRetrieverConfig(anchor_top_a=4)

    runs = [
        propose_partners(
            graph,
            anchor_ids=("out-1", "call-2"),
            candidate_ids=candidate_ids,
            dense_relevance=relevance,
            relation_affinities=relations,
            config=config,
        )
        for _ in range(3)
    ]

    assert runs[0] == runs[1] == runs[2]


def test_registry_builds_provenance_method_from_native_payload() -> None:
    request = ExecutionProvenanceRankingRequest(
        "task-1", "Alpha answer", _candidates(), _provenance_graph()
    )
    settings = ExecutionProvenanceRetrievalSettings(
        top_k=3,
        encoder=DenseEncoderSettings("keyword", "", "", 8),
    )

    built = Registry.retrieval.build(
        settings,
        ExecutionProvenanceBuildPayload(
            provenance_requests=[request],
            dense_encoder=KeywordEncoder(),
        ),
    )

    assert built.provenance.method is RetrievalMethodId.EXECUTION_PROVENANCE_RETRIEVER
    assert built.execution_tasks[0].method_request is request


def test_registry_default_epgm_config_has_no_variant_axis() -> None:
    settings = ExecutionProvenanceRetrievalSettings(
        top_k=3,
        encoder=DenseEncoderSettings("keyword", "", "", 8),
    )

    assert settings.config == EpgmRetrieverConfig()
    assert not hasattr(settings.config, "variant")


def test_behavior_parameters_change_the_cache_fingerprint() -> None:
    base = EpgmRetrieverConfig()

    for changed in (
        EpgmRetrieverConfig(anchor_top_a=5),
        EpgmRetrieverConfig(max_hops=1),
        EpgmRetrieverConfig(hop_decay=0.9),
        EpgmRetrieverConfig(min_partner_confidence=0.55),
        EpgmRetrieverConfig(preserve_dense_top_n=0),
        EpgmRetrieverConfig(relation_temperature=0.5),
    ):
        assert changed.cache_fingerprint() != base.cache_fingerprint()


def test_config_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="anchor_top_a"):
        EpgmRetrieverConfig(anchor_top_a=0)
    with pytest.raises(ValueError, match="max_hops"):
        EpgmRetrieverConfig(max_hops=0)
    with pytest.raises(ValueError, match="hop_decay"):
        EpgmRetrieverConfig(hop_decay=0.0)
    with pytest.raises(ValueError, match="preserve_dense_top_n"):
        EpgmRetrieverConfig(preserve_dense_top_n=-1)
    with pytest.raises(ValueError, match="relation_temperature"):
        EpgmRetrieverConfig(relation_temperature=0.0)


def test_provenance_graph_rejects_parallel_edges_with_the_same_typed_key() -> None:
    nodes = (
        ExecutionProvenanceNode("a", ProvenanceNodeType.TOOL_OUTPUT, "a"),
        ExecutionProvenanceNode("b", ProvenanceNodeType.TOOL_OUTPUT, "b"),
    )
    edges = (
        ExecutionProvenanceEdge("a", "b", ProvenanceEdgeType.DEPENDS_ON),
        ExecutionProvenanceEdge("a", "b", ProvenanceEdgeType.DEPENDS_ON),
    )

    with pytest.raises(ValueError, match="duplicate typed edge"):
        ExecutionProvenanceGraph("dup-task", nodes, edges)
