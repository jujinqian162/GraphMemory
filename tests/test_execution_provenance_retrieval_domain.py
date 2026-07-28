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
    PprResult,
    RelationAffinity,
    TypedTransition,
    build_typed_transitions,
    dense_teleport,
    effective_edge_weights,
    personalized_pagerank,
    query_relation_affinities,
    search_epgm_paths,
    select_budgeted_subgraph,
)
from graph_memory.retrieval.contracts import (
    GraphRAGTrace,
    QueryConditionedExecutionProvenanceTrace,
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
        TextCandidate(item_id="call-1", text="lookup Alpha", metadata={}),
        TextCandidate(item_id="out-1", text="Alpha result", metadata={}),
        TextCandidate(item_id="call-2", text="use Alpha result", metadata={}),
        TextCandidate(item_id="out-2", text="final Alpha answer", metadata={}),
    )


def _dense_ranker() -> DenseTaskRetriever:
    return DenseTaskRetriever(
        encoder=KeywordEncoder(), query_prefix="", passage_prefix="", device="cpu"
    )


def _local_dense_ranker(
    encoder: SentenceEncoder | None = None,
) -> DenseTaskRetriever:
    return DenseTaskRetriever(
        encoder=encoder or LocalPromotionEncoder(),
        query_prefix="Q:",
        passage_prefix="P:",
        device="cpu",
    )


def _local_graphrag_candidates() -> tuple[TextCandidate, ...]:
    return (
        TextCandidate(
            item_id="a",
            text="Alpha. Alpha was born in Bridge City.",
            metadata={"title": "Alpha", "source_ref": "Alpha"},
        ),
        TextCandidate(item_id="x", text="Noise One. Distractor.", metadata={"title": "Noise One"}),
        TextCandidate(item_id="y", text="Noise Two. Distractor.", metadata={"title": "Noise Two"}),
        TextCandidate(
            item_id="b",
            text="Bridge City. Bridge City is located in Country Z.",
            metadata={"title": "Bridge City"},
        ),
    )


def _local_provenance_request(
    *, valid_binding: bool = True
) -> ExecutionProvenanceRankingRequest:
    binding_hash = "alpha-hash" if valid_binding else "wrong-hash"
    nodes = (
        ExecutionProvenanceNode(
            node_id="call-a",
            node_type=ProvenanceNodeType.TOOL_CALL,
            text="Alpha call",
            metadata={"input_parameters": ["context"]},
        ),
        ExecutionProvenanceNode(
            node_id="a",
            node_type=ProvenanceNodeType.TOOL_OUTPUT,
            text="Alpha source",
            metadata={"output_field_hashes": {"evidence": "alpha-hash"}},
        ),
        ExecutionProvenanceNode(node_id="x", node_type=ProvenanceNodeType.TOOL_OUTPUT, text="Noise One"),
        ExecutionProvenanceNode(node_id="y", node_type=ProvenanceNodeType.TOOL_OUTPUT, text="Noise Two"),
        ExecutionProvenanceNode(
            node_id="call-b",
            node_type=ProvenanceNodeType.TOOL_CALL,
            text="Bridge call",
            metadata={"input_parameters": ["context"]},
        ),
        ExecutionProvenanceNode(
            node_id="b", node_type=ProvenanceNodeType.TOOL_OUTPUT, text="Bridge City target"
        ),
    )
    edges = (
        ExecutionProvenanceEdge(source="call-a", target="a", edge_type=ProvenanceEdgeType.RETURNS),
        ExecutionProvenanceEdge(
            source="a",
            target="call-b",
            edge_type=ProvenanceEdgeType.FEEDS,
            binding=FieldBinding(
                output_field="evidence", input_parameter="context", binding_value_hash=binding_hash, binding_kind="semantic_reference"
            ),
            weight=0.8,
            metadata={
                "semantic_scorer": "frozen_dense",
                "semantic_rank": 1,
                "semantic_score": 0.7,
            },
        ),
        ExecutionProvenanceEdge(source="call-b", target="b", edge_type=ProvenanceEdgeType.RETURNS),
    )
    return ExecutionProvenanceRankingRequest(
        task_id="local-path",
        query_text="base query",
        candidates=(
            TextCandidate(item_id="a", text="Alpha source", metadata={}),
            TextCandidate(item_id="x", text="Noise One", metadata={}),
            TextCandidate(item_id="y", text="Noise Two", metadata={}),
            TextCandidate(item_id="b", text="Bridge City target", metadata={}),
        ),
        graph=ExecutionProvenanceGraph(task_id="local-path", nodes=nodes, edges=edges),
    )


def _provenance_graph(
    *, include_untraversed_edge: bool = False
) -> ExecutionProvenanceGraph:
    nodes = (
        ExecutionProvenanceNode(node_id="task", node_type=ProvenanceNodeType.TASK, text="find Alpha"),
        ExecutionProvenanceNode(node_id="agent", node_type=ProvenanceNodeType.AGENT, text="assistant"),
        ExecutionProvenanceNode(node_id="call-1", node_type=ProvenanceNodeType.TOOL_CALL, text="lookup Alpha"),
        ExecutionProvenanceNode(
            node_id="out-1",
            node_type=ProvenanceNodeType.TOOL_OUTPUT,
            text="Alpha result",
            metadata={"output_field_hashes": {"result": "alpha-hash"}},
        ),
        ExecutionProvenanceNode(
            node_id="call-2",
            node_type=ProvenanceNodeType.TOOL_CALL,
            text="use Alpha result",
            metadata={"input_parameters": ["input"]},
        ),
        ExecutionProvenanceNode(
            node_id="out-2", node_type=ProvenanceNodeType.TOOL_OUTPUT, text="final Alpha answer"
        ),
        ExecutionProvenanceNode(node_id="answer", node_type=ProvenanceNodeType.ANSWER, text="Alpha"),
    )
    edges = [
        ExecutionProvenanceEdge(source="agent", target="call-1", edge_type=ProvenanceEdgeType.INVOKES),
        ExecutionProvenanceEdge(source="call-1", target="out-1", edge_type=ProvenanceEdgeType.RETURNS),
        ExecutionProvenanceEdge(
            source="out-1",
            target="call-2",
            edge_type=ProvenanceEdgeType.FEEDS,
            binding=FieldBinding(output_field="result", input_parameter="input", binding_value_hash="alpha-hash", binding_kind="exact"),
            weight=0.9,
            metadata={
                "semantic_scorer": "frozen_dense",
                "semantic_rank": 1,
                "semantic_score": 0.8,
            },
        ),
        ExecutionProvenanceEdge(source="call-2", target="out-2", edge_type=ProvenanceEdgeType.RETURNS),
        ExecutionProvenanceEdge(source="out-2", target="answer", edge_type=ProvenanceEdgeType.GROUNDS),
    ]
    if include_untraversed_edge:
        edges.append(
            ExecutionProvenanceEdge(source="call-1", target="call-2", edge_type=ProvenanceEdgeType.PRECEDES)
        )
    return ExecutionProvenanceGraph(task_id="task-1", nodes=nodes, edges=tuple(edges))


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
        task_id="task-1", query_text="find Alpha", candidates=candidates, graph=_provenance_graph()
    )
    evidence_request = EvidenceGraphRankingRequest(
        task_id="hotpot_rgcn_train",
        query_text="find Alpha",
        candidates=tuple(TextCandidate(item_id=f"m{index}", text=f"candidate {index}", metadata={}) for index in range(3)),
        graph=tiny_graphs()[0],
        initial_scores={f"m{index}": 0.0 for index in range(3)},
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
        ExecutionProvenanceEdge(source="out-1", target="call-2", edge_type=ProvenanceEdgeType.FEEDS)

    chronological = ExecutionProvenanceEdge(
        source="out-1", target="call-2", edge_type=ProvenanceEdgeType.PRECEDES
    )
    assert chronological.binding is None


def test_graphrag_builds_typed_mentions_without_evidence_graph() -> None:
    config = GraphRAGConfig(seed_top_s=2)
    request = build_graphrag_request(
        TextRankingRequest(task_id="task-1", query_text="Alpha answer", candidates=_candidates()), config
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
    assert result.trace.retrieved_edges == ()


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
        TextRankingRequest(task_id="local-bridge", query_text="Which country?", candidates=candidates), config
    )

    result = GraphRAGMethod(ranker, config).rank_task(request, top_k=4)

    assert [node.node_id for node in result.ranked_nodes] == ["a", "x", "b", "y"]
    assert [node.score for node in result.ranked_nodes] == sorted(
        [node.score for node in result.ranked_nodes], reverse=True
    )
    trace = result.trace.native_trace
    assert isinstance(trace, GraphRAGTrace)
    accepted = next(item for item in trace.bridges if item.accepted)
    assert [
        edge.model_dump(mode="json") for edge in result.trace.retrieved_edges
    ] == [
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
            item_id="b2",
            text="Bridge City. Another possible sentence.",
            metadata={"title": "Bridge City"},
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
        TextRankingRequest(task_id="ambiguous-bridge", query_text="Which country?", candidates=candidates), config
    )
    dense = ranker.rank(
        TextRankingRequest(task_id=request.task_id, query_text=request.query_text, candidates=request.candidates)
    )

    result = GraphRAGMethod(ranker, config).rank_task(request, top_k=5)

    assert list(result.ranked_nodes) == dense
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
        task_id="task-1", query_text="Alpha answer", candidates=_candidates(), graph=graph
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
        edge.source == "out-1" and edge.target == "out-2"
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
    # Path confidence is the recorded weight scaled by the relation's frozen
    # type prior (feeds = 0.6), so a graph whose weights are all identical
    # placeholders still yields relation-discriminative path scores.
    assert [
        edge.model_dump(mode="json") for edge in result.trace.retrieved_edges
    ] == [
        {
            "source": "a",
            "target": "b",
            "edge_type": "feeds",
            "weight": pytest.approx(0.8 * 0.6),
            "directed": True,
        }
    ]
    trace = result.trace.native_trace
    assert isinstance(trace, StatelessExecutionProvenanceTrace)
    accepted = next(path for path in trace.paths if path.accepted)
    assert accepted.path_confidence == pytest.approx(0.8 * 0.6)
    assert not trace.exact_dense_fallback


def test_provenance_binding_failure_abstains_to_exact_dense_objects() -> None:
    request = _local_provenance_request(valid_binding=False)
    ranker = _local_dense_ranker()
    dense = ranker.rank(
        TextRankingRequest(task_id=request.task_id, query_text=request.query_text, candidates=request.candidates)
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

    assert list(result.ranked_nodes) == dense
    assert result.trace.retrieved_edges == ()
    trace = result.trace.native_trace
    assert isinstance(trace, StatelessExecutionProvenanceTrace)
    assert trace.exact_dense_fallback
    assert any(path.rejection_reason == "binding_mismatch" for path in trace.paths)


def test_provenance_expansion_does_not_reverse_incoming_dependencies() -> None:
    request = ExecutionProvenanceRankingRequest(
        task_id="task-1", query_text="Alpha answer", candidates=_candidates(), graph=_provenance_graph()
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
        task_id="task-1", query_text="Alpha answer", candidates=_candidates(), graph=_provenance_graph()
    )
    settings = ExecutionProvenanceRetrievalSettings(
        top_k=3,
        encoder=DenseEncoderSettings("keyword", "", "", 8),
        device="cpu",
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


def test_registry_default_epgm_config_is_the_ppr_steiner_preset() -> None:
    settings = ExecutionProvenanceRetrievalSettings(
        top_k=3,
        encoder=DenseEncoderSettings("keyword", "", "", 8),
        device="cpu",
    )

    assert settings.config == EpgmRetrieverConfig.for_variant("ppr_steiner")
    assert settings.config.variant == "ppr_steiner"
    assert settings.config.weight_aware is False


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
    default = EpgmRetrieverConfig.for_variant("ppr_steiner")
    changed = EpgmRetrieverConfig.for_variant("ppr_steiner", ppr_alpha=0.8)
    assert default.cache_fingerprint() != changed.cache_fingerprint()
    assert default.cache_fingerprint() != typed_beam.cache_fingerprint()
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
            node_id="verify", node_type=ProvenanceNodeType.VERIFICATION, text="verification alpha check"
        ),
        ExecutionProvenanceNode(
            node_id="claim-old",
            node_type=ProvenanceNodeType.CLAIM,
            text="obsolete unrelated wording",
            metadata={"lifecycle_state": "invalidated"},
        ),
        ExecutionProvenanceNode(node_id="noise", node_type=ProvenanceNodeType.CLAIM, text="noise one"),
    )
    edges = (
        ExecutionProvenanceEdge(
            source="verify", target="claim-old", edge_type=ProvenanceEdgeType.CONTRADICTS
        ),
    )
    return ExecutionProvenanceRankingRequest(
        task_id="revision-task",
        query_text="alpha",
        candidates=(
            TextCandidate(item_id="verify", text="verification alpha check", metadata={}),
            TextCandidate(item_id="claim-old", text="obsolete unrelated wording", metadata={}),
            TextCandidate(item_id="noise", text="noise one", metadata={}),
        ),
        graph=ExecutionProvenanceGraph(task_id="revision-task", nodes=nodes, edges=edges),
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
    assert result.trace.retrieved_edges == ()


def test_stable_insert_preserves_dense_score_multiset_but_additive_rescores() -> None:
    request = _local_provenance_request()
    ranker = _local_dense_ranker()
    dense = ranker.rank(
        TextRankingRequest(task_id=request.task_id, query_text=request.query_text, candidates=request.candidates)
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
        TextRankingRequest(task_id=request.task_id, query_text=request.query_text, candidates=request.candidates)
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
        ExecutionProvenanceNode(node_id=node_id, node_type=node_type, text=f"text for {node_id}", metadata={})
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
    config = EpgmRetrieverConfig.for_variant("typed_beam", weight_aware=True)

    assert effective_edge_weights(graph, config) == {}


def test_effective_weights_normalise_only_informative_edge_types() -> None:
    graph = _weighted_feeds_graph((0.5, 0.75, 1.0))
    config = EpgmRetrieverConfig.for_variant("typed_beam", weight_aware=True)

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

    config = EpgmRetrieverConfig.for_variant("typed_beam", weight_aware=True)
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


def _sibling_feeds_graph(weights: tuple[float, float]) -> ExecutionProvenanceGraph:
    binding = FieldBinding(
        output_field="evidence",
        input_parameter="context",
        binding_value_hash="b" * 8,
        binding_kind="semantic_reference",
    )
    nodes = (
        ExecutionProvenanceNode(
            node_id="out",
            node_type=ProvenanceNodeType.TOOL_OUTPUT,
            text="shared source",
            metadata={"output_field_hashes": {"evidence": "b" * 8}},
        ),
        ExecutionProvenanceNode(
            node_id="call-a",
            node_type=ProvenanceNodeType.TOOL_CALL,
            text="first branch",
            metadata={"input_parameters": ["context"]},
        ),
        ExecutionProvenanceNode(
            node_id="call-b",
            node_type=ProvenanceNodeType.TOOL_CALL,
            text="second branch",
            metadata={"input_parameters": ["context"]},
        ),
    )
    edges = tuple(
        ExecutionProvenanceEdge(
            source="out",
            target=target,
            edge_type=ProvenanceEdgeType.FEEDS,
            binding=binding,
            weight=weight,
            metadata={},
        )
        for target, weight in zip(("call-a", "call-b"), weights, strict=True)
    )
    return ExecutionProvenanceGraph(task_id="siblings", nodes=nodes, edges=edges)


def _uniform_relations(graph: ExecutionProvenanceGraph) -> tuple[RelationAffinity, ...]:
    edge_types = sorted({edge.edge_type.value for edge in graph.edges})
    mass = 1.0 / len(edge_types)
    return tuple(RelationAffinity(edge_type, 0.0, mass) for edge_type in edge_types)


def test_source_local_transition_preserves_calibrated_feeds_ratio() -> None:
    graph = _sibling_feeds_graph((0.8, 0.6))
    transitions = build_typed_transitions(
        graph, _uniform_relations(graph), EpgmRetrieverConfig()
    )
    forward = {
        item.target: item.probability
        for item in transitions
        if item.source == "out" and item.direction == "forward"
    }

    assert sum(forward.values()) == pytest.approx(1.0)
    assert forward["call-a"] / forward["call-b"] == pytest.approx(0.8 / 0.6)


def test_constant_recorded_weights_are_transition_identity() -> None:
    graph = _sibling_feeds_graph((1.0, 1.0))
    transitions = build_typed_transitions(
        graph, _uniform_relations(graph), EpgmRetrieverConfig()
    )
    forward = [
        item.probability
        for item in transitions
        if item.source == "out" and item.direction == "forward"
    ]

    assert forward == pytest.approx([0.5, 0.5])


def test_personalized_pagerank_conserves_mass_and_is_deterministic() -> None:
    graph = _sibling_feeds_graph((0.8, 0.6))
    config = EpgmRetrieverConfig(ppr_tolerance=1e-12)
    teleport = {"out": 1.0}
    relations = _uniform_relations(graph)

    first = personalized_pagerank(graph, teleport, relations, config)
    second = personalized_pagerank(graph, teleport, relations, config)

    assert first == second
    assert sum(first.node_scores.values()) == pytest.approx(1.0)
    assert first.residual <= config.ppr_tolerance
    assert first.converged


def test_budgeted_selection_keeps_connector_outside_candidate_budget() -> None:
    request = _local_provenance_request()
    dense_scores = {
        candidate.item_id: float(len(request.candidates) - index)
        for index, candidate in enumerate(request.candidates)
    }
    config = EpgmRetrieverConfig(
        candidate_inclusion_cost=0.0, selection_edge_cost_weight=0.01
    )
    teleport = dense_teleport(dense_scores, request.graph)
    ppr = personalized_pagerank(
        request.graph,
        teleport,
        _uniform_relations(request.graph),
        config,
    )

    selected, _prizes = select_budgeted_subgraph(
        dense_scores, ppr, top_k=4, config=config
    )

    assert len(selected.candidate_ids) <= 4
    assert not set(selected.connector_ids) & set(dense_scores)
    assert selected.exact_dense_fallback is False
    assert selected.candidate_edges


def test_ppr_steiner_default_returns_closed_subgraph_trace() -> None:
    request = _local_provenance_request()
    method = EpgmRetriever(_local_dense_ranker(), EpgmRetrieverConfig())

    result = method.rank_task(request, top_k=4)
    trace = result.trace.native_trace

    assert isinstance(trace, QueryConditionedExecutionProvenanceTrace)
    assert trace.variant == "ppr_steiner"
    assert trace.relation_description_version
    assert trace.ppr_converged
    assert len(trace.selected_candidate_ids) <= 4
    assert trace.exact_dense_fallback == (not trace.selection_steps)
    assert len(result.ranked_nodes) == len(request.candidates)


def test_query_relation_affinity_uses_embedding_similarity_not_dataset_rules() -> None:
    config = EpgmRetrieverConfig(relation_temperature=0.1)
    edge_types = sorted(config.relation_descriptions)
    dimension = len(edge_types)
    relation_vectors = np.eye(dimension, dtype=float)
    target = ProvenanceEdgeType.INVALIDATES.value
    query_vector = relation_vectors[edge_types.index(target)]

    affinities = query_relation_affinities(
        query_vector,
        relation_vectors,
        frozenset(
            {
                target,
                ProvenanceEdgeType.FEEDS.value,
                ProvenanceEdgeType.CONTAINS.value,
            }
        ),
        config,
    )

    by_type = {item.edge_type: item.affinity for item in affinities}
    assert by_type[target] > by_type[ProvenanceEdgeType.FEEDS.value]
    assert sum(by_type.values()) == pytest.approx(1.0)


def test_invalid_field_binding_is_excluded_from_typed_transitions() -> None:
    graph = _sibling_feeds_graph((0.8, 0.6))
    invalid_edges = tuple(
        ExecutionProvenanceEdge(
            source=edge.source,
            target=edge.target,
            edge_type=edge.edge_type,
            binding=FieldBinding(
                output_field="evidence",
                input_parameter="context",
                binding_value_hash="wrong-hash",
                binding_kind="semantic_reference",
            ),
            weight=edge.weight,
            metadata=edge.metadata,
        )
        for edge in graph.edges
    )
    invalid = ExecutionProvenanceGraph(task_id=graph.task_id, nodes=graph.nodes, edges=invalid_edges)

    transitions = build_typed_transitions(
        invalid, _uniform_relations(invalid), EpgmRetrieverConfig()
    )

    assert transitions == ()


def test_ppr_steiner_uses_byte_for_byte_dense_fallback_without_dependencies() -> None:
    nodes = (
        ExecutionProvenanceNode(node_id="a", node_type=ProvenanceNodeType.CLAIM, text="alpha", metadata={}),
        ExecutionProvenanceNode(node_id="b", node_type=ProvenanceNodeType.CLAIM, text="beta", metadata={}),
    )
    request = ExecutionProvenanceRankingRequest(
        task_id="edgeless",
        query_text="alpha",
        candidates=(TextCandidate(item_id="a", text="alpha", metadata={}), TextCandidate(item_id="b", text="beta", metadata={})),
        graph=ExecutionProvenanceGraph(task_id="edgeless", nodes=nodes, edges=()),
    )
    ranker = _dense_ranker()
    dense = ranker.rank(
        TextRankingRequest(task_id=request.task_id, query_text=request.query_text, candidates=request.candidates)
    )

    result = EpgmRetriever(ranker, EpgmRetrieverConfig()).rank_task(request, top_k=2)
    trace = result.trace.native_trace

    assert isinstance(trace, QueryConditionedExecutionProvenanceTrace)
    assert trace.exact_dense_fallback
    assert list(result.ranked_nodes) == dense
    assert result.trace.retrieved_edges == ()
    assert trace.selected_candidate_ids == ()
    assert trace.emitted_edges == ()


def test_connected_selection_emits_stored_orientation_after_reverse_walk() -> None:
    nodes = (
        ExecutionProvenanceNode(node_id="a", node_type=ProvenanceNodeType.CLAIM, text="alpha", metadata={}),
        ExecutionProvenanceNode(node_id="b", node_type=ProvenanceNodeType.CLAIM, text="beta", metadata={}),
    )
    graph = ExecutionProvenanceGraph(
        task_id="reverse",
        nodes=nodes,
        edges=(
            ExecutionProvenanceEdge(
                source="b", target="a", edge_type=ProvenanceEdgeType.DEPENDS_ON, binding=None, weight=1.0, metadata={}
            ),
        ),
    )
    request = ExecutionProvenanceRankingRequest(
        task_id="reverse",
        query_text="alpha",
        candidates=(TextCandidate(item_id="a", text="alpha", metadata={}), TextCandidate(item_id="b", text="beta", metadata={})),
        graph=graph,
    )

    result = EpgmRetriever(_dense_ranker(), EpgmRetrieverConfig()).rank_task(
        request, top_k=2
    )

    assert [(edge.source, edge.target) for edge in result.trace.retrieved_edges] == [
        ("b", "a")
    ]


def test_rq3_shaped_revision_graph_is_selected_without_feeds_weights() -> None:
    request = _revision_request()

    result = EpgmRetriever(_dense_ranker(), EpgmRetrieverConfig()).rank_task(
        request, top_k=3
    )
    trace = result.trace.native_trace

    assert isinstance(trace, QueryConditionedExecutionProvenanceTrace)
    assert not trace.exact_dense_fallback
    assert "claim-old" in trace.selected_candidate_ids
    assert any(edge.edge_type == "contradicts" for edge in trace.emitted_edges)


def test_greedy_selector_first_step_matches_tiny_exhaustive_best_gain() -> None:
    transitions = (
        TypedTransition("a", "b", "depends_on", "forward", 1.0, 1.0, 0.8, -np.log(0.8)),
        TypedTransition("a", "c", "depends_on", "forward", 1.0, 1.0, 0.2, -np.log(0.2)),
    )
    ppr = PprResult(
        node_scores={"a": 0.5, "b": 0.25, "c": 0.25},
        teleport={"a": 0.5, "b": 0.25, "c": 0.25},
        relation_affinities=(RelationAffinity("depends_on", 1.0, 1.0),),
        transitions=transitions,
        iterations=1,
        residual=0.0,
        converged=True,
    )
    dense_scores = {"a": 1.0, "b": 0.5, "c": 0.5}

    selected, _ = select_budgeted_subgraph(
        dense_scores,
        ppr,
        top_k=3,
        config=EpgmRetrieverConfig(
            candidate_inclusion_cost=0.0, selection_edge_cost_weight=0.1
        ),
    )

    assert selected.steps[0].target_id == "b"
    assert selected.steps[0].marginal_gain > selected.steps[1].marginal_gain


def test_greedy_selector_reuses_selected_tree_edges_at_zero_residual_cost() -> None:
    """A shared connector must expose its cheap residual branch.

    The fresh A-Y-C path is cheaper than the *full* A-X-C path, but after
    A-X-B has entered the tree only X-C is new. Charging A-X again causes the
    selector to reject C even though its residual marginal is positive.
    """

    def transition(source: str, target: str, cost: float) -> TypedTransition:
        return TypedTransition(
            source,
            target,
            "depends_on",
            "forward",
            1.0,
            1.0,
            0.5,
            cost,
        )

    ppr = PprResult(
        node_scores={"a": 1.0, "b": 0.6, "c": 0.3, "x": 0.0, "y": 0.0},
        teleport={"a": 1.0, "b": 0.0, "c": 0.0, "x": 0.0, "y": 0.0},
        relation_affinities=(),
        transitions=(
            transition("a", "x", 5.0),
            transition("x", "b", 0.1),
            transition("x", "c", 0.1),
            transition("a", "y", 2.2),
            transition("y", "c", 2.2),
        ),
        iterations=1,
        residual=0.0,
        converged=True,
    )
    selected, _ = select_budgeted_subgraph(
        {"a": 1.0, "b": 0.6, "c": 0.3},
        ppr,
        top_k=3,
        config=EpgmRetrieverConfig(
            dense_prize_weight=0.0,
            ppr_prize_weight=1.0,
            candidate_inclusion_cost=0.0,
            selection_edge_cost_weight=0.08,
        ),
    )

    assert selected.candidate_ids == ("a", "b", "c")
    assert selected.steps[1].path_node_ids == ("a", "x", "c")
    assert selected.steps[1].edge_cost == pytest.approx(0.1)
    assert selected.steps[1].marginal_gain == pytest.approx(0.292)


def test_connected_candidate_pays_for_the_dense_incumbent_it_displaces() -> None:
    transition = TypedTransition(
        "a", "c", "depends_on", "forward", 1.0, 1.0, 1.0, 0.05
    )
    ppr = PprResult(
        node_scores={"a": 1.0, "b": 0.0, "c": 1.0},
        teleport={"a": 1.0, "b": 0.0, "c": 0.0},
        relation_affinities=(),
        transitions=(transition,),
        iterations=1,
        residual=0.0,
        converged=True,
    )

    selected, _ = select_budgeted_subgraph(
        {"a": 1.0, "b": 0.9, "c": 0.1},
        ppr,
        top_k=2,
        config=EpgmRetrieverConfig(),
    )

    # C has positive graph prize, but not enough to replace the stronger Dense
    # incumbent B after edge and displacement opportunity costs are included.
    assert selected.exact_dense_fallback


def test_dense_incumbent_prize_cannot_subsidize_weaker_replacement() -> None:
    def transition(source: str, target: str) -> TypedTransition:
        return TypedTransition(
            source, target, "depends_on", "forward", 1.0, 1.0, 1.0, 0.1
        )

    ppr = PprResult(
        node_scores={"a": 1.0, "b": 0.6, "c": 0.1, "d": 0.2},
        teleport={"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0},
        relation_affinities=(),
        transitions=(transition("a", "b"), transition("b", "c")),
        iterations=1,
        residual=0.0,
        converged=True,
    )
    selected, _ = select_budgeted_subgraph(
        {"a": 1.0, "b": 0.9, "d": 0.8, "c": 0.1},
        ppr,
        top_k=3,
        config=EpgmRetrieverConfig(
            dense_prize_weight=0.0,
            ppr_prize_weight=1.0,
            candidate_inclusion_cost=0.0,
        ),
    )

    assert "b" in selected.candidate_ids
    assert "c" not in selected.candidate_ids


def test_dense_teleport_preserves_query_score_dynamic_range() -> None:
    request = _local_provenance_request()
    teleport = dense_teleport(
        {"a": 0.9, "b": 0.8, "x": 0.7, "y": 0.6}, request.graph
    )

    assert sum(teleport.values()) == pytest.approx(1.0)
    assert all(value > 0.0 for value in teleport.values())
    assert teleport["a"] > 100.0 * teleport["y"]


def test_typed_transition_penalizes_high_degree_hub_targets() -> None:
    nodes = tuple(
        ExecutionProvenanceNode(node_id=node_id, node_type=ProvenanceNodeType.CLAIM, text=node_id, metadata={})
        for node_id in ("source", "hub", "leaf", "x1", "x2", "x3")
    )
    edges = (
        ExecutionProvenanceEdge(source="source", target="hub", edge_type=ProvenanceEdgeType.SUPPORTS),
        ExecutionProvenanceEdge(source="source", target="leaf", edge_type=ProvenanceEdgeType.SUPPORTS),
        ExecutionProvenanceEdge(source="hub", target="x1", edge_type=ProvenanceEdgeType.SUPPORTS),
        ExecutionProvenanceEdge(source="hub", target="x2", edge_type=ProvenanceEdgeType.SUPPORTS),
        ExecutionProvenanceEdge(source="hub", target="x3", edge_type=ProvenanceEdgeType.SUPPORTS),
    )
    graph = ExecutionProvenanceGraph(task_id="hub", nodes=nodes, edges=edges)

    transitions = build_typed_transitions(
        graph, _uniform_relations(graph), EpgmRetrieverConfig()
    )
    row = {
        item.target: item.probability
        for item in transitions
        if item.source == "source" and item.direction == "forward"
    }

    assert row["leaf"] > row["hub"]


def test_provenance_graph_rejects_parallel_edges_with_the_same_typed_key() -> None:
    nodes = (
        ExecutionProvenanceNode(node_id="a", node_type=ProvenanceNodeType.CLAIM, text="a", metadata={}),
        ExecutionProvenanceNode(node_id="b", node_type=ProvenanceNodeType.CLAIM, text="b", metadata={}),
    )
    duplicate = ExecutionProvenanceEdge(
        source="a", target="b", edge_type=ProvenanceEdgeType.SUPPORTS, binding=None, weight=1.0, metadata={}
    )

    with pytest.raises(ValueError, match="duplicate typed edge"):
        ExecutionProvenanceGraph(task_id="duplicate", nodes=nodes, edges=(duplicate, duplicate))


def test_zero_weight_edge_has_no_forward_or_reverse_transition() -> None:
    nodes = (
        ExecutionProvenanceNode(node_id="a", node_type=ProvenanceNodeType.CLAIM, text="a", metadata={}),
        ExecutionProvenanceNode(node_id="b", node_type=ProvenanceNodeType.CLAIM, text="b", metadata={}),
    )
    graph = ExecutionProvenanceGraph(
        task_id="zero",
        nodes=nodes,
        edges=(
            ExecutionProvenanceEdge(
                source="a", target="b", edge_type=ProvenanceEdgeType.DEPENDS_ON, binding=None, weight=0.0, metadata={}
            ),
        ),
    )

    assert build_typed_transitions(
        graph, _uniform_relations(graph), EpgmRetrieverConfig()
    ) == ()


def test_mixed_direction_connector_does_not_fabricate_candidate_dependency() -> None:
    ppr = PprResult(
        node_scores={"a": 0.5, "x": 0.25, "b": 0.25},
        teleport={"a": 0.6, "x": 0.0, "b": 0.4},
        relation_affinities=(RelationAffinity("depends_on", 1.0, 1.0),),
        transitions=(
            TypedTransition(
                "a", "x", "depends_on", "reverse", 1.0, 1.0, 1.0, 0.05
            ),
            TypedTransition(
                "x", "b", "depends_on", "forward", 1.0, 1.0, 1.0, 0.05
            ),
        ),
        iterations=1,
        residual=0.0,
        converged=True,
    )

    selected, _ = select_budgeted_subgraph(
        {"a": 1.0, "b": 0.9},
        ppr,
        top_k=2,
        config=EpgmRetrieverConfig(
            candidate_inclusion_cost=0.0, selection_edge_cost_weight=0.01
        ),
    )

    assert selected.steps
    assert not selected.exact_dense_fallback
    assert selected.candidate_edges == ()
