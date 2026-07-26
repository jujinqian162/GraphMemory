from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest

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
    EpgmRetriever,
    EpgmRetrieverConfig,
    EpgmVariant,
    effective_edge_weights,
    search_epgm_paths,
)
from graph_memory.retrieval.contracts import (
    GraphRAGTrace,
    StatelessExecutionProvenanceTrace,
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


def test_provenance_retriever_returns_only_actual_traversed_edges() -> None:
    graph = _provenance_graph(include_untraversed_edge=True)
    request = ExecutionProvenanceRankingRequest(
        "task-1", "Alpha answer", _candidates(), graph
    )
    method = EpgmRetriever(
        dense_ranker=_dense_ranker(),
        config=EpgmRetrieverConfig.for_variant(
            "dependency_path",
            seed_top_s=4,
            max_hops=2,
            preserve_dense_top_n=0,
            min_path_confidence=0.0,
        ),
    )

    result = method.rank_task(request, top_k=4)

    assert len(result.ranked_nodes) == len(request.candidates)
    assert isinstance(result.trace.native_trace, StatelessExecutionProvenanceTrace)
    assert result.trace.native_trace.paths
    assert result.trace.native_trace.edges
    assert all(
        edge.edge_type is not ProvenanceEdgeType.PRECEDES
        for edge in result.trace.native_trace.edges
    )
    assert all(
        edge["source"] == "out-1" and edge["target"] == "out-2"
        for edge in result.trace.retrieved_edges
    )


def test_provenance_local_path_promotes_partner_and_uses_confidence_once() -> None:
    request = _local_provenance_request()
    method = EpgmRetriever(
        dense_ranker=_local_dense_ranker(),
        config=EpgmRetrieverConfig.for_variant(
            "dependency_path",
            seed_top_s=1,
            min_path_confidence=0.2,
            preserve_dense_top_n=2,
        ),
    )

    result = method.rank_task(request, top_k=4)

    assert [node.node_id for node in result.ranked_nodes] == ["a", "x", "b", "y"]
    assert result.trace.retrieved_edges == [
        {
            "source": "a",
            "target": "b",
            "edge_type": "feeds",
            "weight": pytest.approx(0.8),
            "directed": True,
        }
    ]
    trace = result.trace.native_trace
    assert isinstance(trace, StatelessExecutionProvenanceTrace)
    accepted = next(path for path in trace.paths if path.accepted)
    assert accepted.path_confidence == pytest.approx(0.8)
    assert not trace.exact_dense_fallback


def test_provenance_binding_failure_abstains_to_exact_dense_objects() -> None:
    request = _local_provenance_request(valid_binding=False)
    ranker = _local_dense_ranker()
    dense = ranker.rank(
        TextRankingRequest(request.task_id, request.query_text, request.candidates)
    )
    method = EpgmRetriever(
        dense_ranker=ranker,
        config=EpgmRetrieverConfig.for_variant(
            "dependency_path",
            seed_top_s=1,
            min_path_confidence=0.0,
            preserve_dense_top_n=2,
        ),
    )

    result = method.rank_task(request, top_k=4)

    assert result.ranked_nodes == dense
    assert result.trace.retrieved_edges == []
    trace = result.trace.native_trace
    assert isinstance(trace, StatelessExecutionProvenanceTrace)
    assert trace.exact_dense_fallback
    assert any(path.rejection_reason == "binding_mismatch" for path in trace.paths)


def test_provenance_expansion_does_not_reverse_incoming_dependencies() -> None:
    request = ExecutionProvenanceRankingRequest(
        "task-1", "Alpha answer", _candidates(), _provenance_graph()
    )

    paths = search_epgm_paths(
        request.graph,
        ("out-2",),
        {"out-2": 1.0},
        candidate_ids=frozenset(
            candidate.item_id for candidate in request.candidates
        ),
        config=EpgmRetrieverConfig.for_variant(
            "dependency_path", beam_width=2, max_hops=2
        ),
    )

    assert all("out-1" not in path.node_ids for path in paths)


def test_provenance_config_rejects_empty_beam() -> None:
    with pytest.raises(ValueError, match="beam_width"):
        EpgmRetrieverConfig(beam_width=0)


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


def test_registry_default_epgm_config_is_the_typed_beam_preset() -> None:
    settings = ExecutionProvenanceRetrievalSettings(
        top_k=3,
        encoder=DenseEncoderSettings("keyword", "", "", 8),
    )

    assert settings.config == EpgmRetrieverConfig.for_variant("typed_beam")
    assert settings.config.gating == "none"
    assert settings.config.fusion == "additive"


def test_epgm_variant_presets_differ_only_by_declared_axes() -> None:
    typed_beam = EpgmRetrieverConfig.for_variant("typed_beam")
    dependency_path = EpgmRetrieverConfig.for_variant("dependency_path")

    assert (typed_beam.traversal, typed_beam.edge_scope) == (
        "bidirectional",
        "typed",
    )
    assert (dependency_path.traversal, dependency_path.edge_scope) == (
        "directed",
        "dependency",
    )
    assert dependency_path.gating == "schema"
    assert dependency_path.fusion == "stable_insert"
    with pytest.raises(ValueError, match="unknown EPGM variant"):
        EpgmRetrieverConfig.for_variant(cast(EpgmVariant, cast(object, "does_not_exist")))


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


def test_typed_beam_reaches_invalidated_claim_that_dependency_path_cannot() -> None:
    request = _revision_request()

    def graph_scores(variant: EpgmVariant) -> dict[str, float]:
        method = EpgmRetriever(
            dense_ranker=_dense_ranker(),
            config=EpgmRetrieverConfig.for_variant(variant),
        )
        return {
            node.node_id: node.graph_score for node in method.rank(request).ranked_nodes
        }

    typed_beam = graph_scores("typed_beam")
    dependency_path = graph_scores("dependency_path")

    # the overturned claim is only reachable when audit edges are traversable
    assert typed_beam["claim-old"] > 0.0
    assert dependency_path["claim-old"] == 0.0
    assert all(score == 0.0 for score in dependency_path.values())


def test_dependency_path_gate_rejects_audit_only_paths_with_a_reason() -> None:
    request = _revision_request()
    method = EpgmRetriever(
        dense_ranker=_dense_ranker(),
        config=EpgmRetrieverConfig.for_variant(
            "dependency_path", edge_scope="typed", traversal="bidirectional"
        ),
    )

    result = method.rank_task(request, top_k=3)
    trace = result.trace.native_trace

    assert isinstance(trace, StatelessExecutionProvenanceTrace)
    assert trace.variant == "dependency_path"
    assert trace.paths
    assert not any(path.accepted for path in trace.paths)
    assert {path.rejection_reason for path in trace.paths} == {"incomplete_path"}
    assert result.trace.retrieved_edges == []


def test_stable_insert_preserves_dense_score_multiset_but_additive_rescores() -> None:
    request = _local_provenance_request()
    ranker = _local_dense_ranker()
    dense = ranker.rank(
        TextRankingRequest(request.task_id, request.query_text, request.candidates)
    )
    dense_scores = sorted((node.score for node in dense), reverse=True)

    stable = EpgmRetriever(
        dense_ranker=ranker,
        config=EpgmRetrieverConfig.for_variant("dependency_path", seed_top_s=1),
    ).rank_task(request, top_k=4)
    additive = EpgmRetriever(
        dense_ranker=ranker,
        config=EpgmRetrieverConfig.for_variant("typed_beam"),
    ).rank_task(request, top_k=4)

    # stable insertion only permutes node ids; scores are the dense slots
    assert sorted(
        (node.score for node in stable.ranked_nodes), reverse=True
    ) == pytest.approx(dense_scores)
    # additive fusion emits fused scores, so it is not multiset-preserving
    assert sorted(
        (node.score for node in additive.ranked_nodes), reverse=True
    ) != pytest.approx(dense_scores)
    for result in (stable, additive):
        assert len(result.ranked_nodes) == len(request.candidates)
        assert [node.score for node in result.ranked_nodes] == sorted(
            (node.score for node in result.ranked_nodes), reverse=True
        )


def test_additive_fusion_never_demotes_a_dense_hit() -> None:
    request = _local_provenance_request()
    ranker = _local_dense_ranker()
    dense = ranker.rank(
        TextRankingRequest(request.task_id, request.query_text, request.candidates)
    )
    dense_rank = {node.node_id: index for index, node in enumerate(dense, start=1)}

    result = EpgmRetriever(
        dense_ranker=ranker, config=EpgmRetrieverConfig.for_variant("typed_beam")
    ).rank_task(request, top_k=4)

    top_dense = dense[0].node_id
    final_rank = {
        node.node_id: index for index, node in enumerate(result.ranked_nodes, start=1)
    }
    # graph propagation is additive on top of dense relevance, so the strongest
    # dense hit cannot be pushed below a node it already outranked
    assert final_rank[top_dense] == dense_rank[top_dense] == 1


def _weighted_feeds_graph(
    weights: tuple[float, float, float],
) -> ExecutionProvenanceGraph:
    """A feeds/returns backbone whose recorded feeds weights are configurable.

    This is the shape a scorer-calibrated synthetic provenance graph has: the
    `feeds` weights carry a real measurement while the structural `returns`
    edges are a constant 1.0.
    """

    binding = FieldBinding(
        output_field="evidence",
        input_parameter="context",
        binding_value_hash="a" * 8,
        binding_kind="semantic_reference",
    )
    nodes = tuple(
        ExecutionProvenanceNode(node_id, node_type, f"text for {node_id}", {})
        for node_id, node_type in (
            ("out_a", ProvenanceNodeType.TOOL_OUTPUT),
            ("out_b", ProvenanceNodeType.TOOL_OUTPUT),
            ("out_c", ProvenanceNodeType.TOOL_OUTPUT),
            ("call_a", ProvenanceNodeType.TOOL_CALL),
            ("call_b", ProvenanceNodeType.TOOL_CALL),
            ("call_c", ProvenanceNodeType.TOOL_CALL),
        )
    )
    feeds = tuple(
        ExecutionProvenanceEdge(
            source=source,
            target=target,
            edge_type=ProvenanceEdgeType.FEEDS,
            binding=binding,
            weight=weight,
            metadata={},
        )
        for (source, target), weight in zip(
            (("out_a", "call_a"), ("out_b", "call_b"), ("out_c", "call_c")),
            weights,
            strict=True,
        )
    )
    returns = tuple(
        ExecutionProvenanceEdge(
            source=source,
            target=target,
            edge_type=ProvenanceEdgeType.RETURNS,
            binding=None,
            weight=1.0,
            metadata={},
        )
        for source, target in (("call_a", "out_b"), ("call_b", "out_c"))
    )
    return ExecutionProvenanceGraph(
        task_id="weighted", nodes=nodes, edges=feeds + returns
    )


def test_effective_weights_are_identity_when_recorded_weights_are_constant() -> None:
    """A recorded agent trace stores weight=1.0 everywhere as a placeholder.

    w_eff must then be an exact identity factor, so weight sensitivity needs no
    dataset-specific switch: the same scoring formula degenerates to pure type
    priors on its own.
    """

    graph = _weighted_feeds_graph((1.0, 1.0, 1.0))
    config = EpgmRetrieverConfig.for_variant("typed_beam")

    assert effective_edge_weights(graph, config) == {}


def test_effective_weights_normalise_only_informative_edge_types() -> None:
    graph = _weighted_feeds_graph((0.5, 0.75, 1.0))
    config = EpgmRetrieverConfig.for_variant("typed_beam")

    effective = effective_edge_weights(graph, config)

    # `feeds` varies, so it is normalised into [min_effective_weight, 1].
    feeds = {
        source: value
        for (source, _target, edge_type), value in effective.items()
        if edge_type == ProvenanceEdgeType.FEEDS.value
    }
    assert feeds["out_a"] == pytest.approx(config.min_effective_weight)
    assert feeds["out_c"] == pytest.approx(1.0)
    assert feeds["out_a"] < feeds["out_b"] < feeds["out_c"]
    # `returns` is constant, so it stays absent and is treated as w_eff = 1.
    assert not any(
        edge_type == ProvenanceEdgeType.RETURNS.value
        for _source, _target, edge_type in effective
    )


def test_recorded_weights_reorder_paths_only_when_they_carry_information() -> None:
    """The same search must be weight-sensitive or weight-blind by the data.

    On a graph whose feeds weights vary, a weaker recorded weight has to lower
    the path score; on the constant-weight version of the same topology the
    scores must be identical, because there is nothing to learn from them.
    """

    config = EpgmRetrieverConfig.for_variant("typed_beam")
    seeds = ("out_a",)
    relevance = {"out_a": 1.0}
    candidates = frozenset({"call_a", "out_b", "call_b", "out_c", "call_c"})

    def score_of(weights: tuple[float, float, float], target: str) -> float:
        paths = search_epgm_paths(
            _weighted_feeds_graph(weights),
            seeds,
            relevance,
            candidate_ids=candidates,
            config=config,
        )
        return next(path.score for path in paths if path.target_id == target)

    # constant weights: w_eff cancels, so both topologies score identically
    constant_low = score_of((1.0, 1.0, 1.0), "call_a")
    constant_high = score_of((1.0, 1.0, 1.0), "call_a")
    assert constant_low == constant_high

    # varied weights: the weakest recorded feeds edge must score strictly lower
    weak = score_of((0.5, 1.0, 1.0), "call_a")
    strong = score_of((1.0, 0.5, 1.0), "call_a")
    assert weak < strong
    assert weak < constant_low


def test_weight_aware_can_be_disabled_without_touching_other_axes() -> None:
    graph = _weighted_feeds_graph((0.5, 0.75, 1.0))
    blind = EpgmRetrieverConfig.for_variant("typed_beam", weight_aware=False)

    assert effective_edge_weights(graph, blind) == {}
    assert blind.traversal == "bidirectional"
    assert blind.edge_scope == "typed"
