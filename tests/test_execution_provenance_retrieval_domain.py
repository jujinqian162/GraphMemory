from __future__ import annotations

from collections.abc import Sequence

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
from graph_memory.registry.methods import (
    ArtifactKind,
    RequiredArtifact,
    RetrievalLifecycle,
    RetrievalTaskFamily,
)
from graph_memory.registry.retrieval import (
    DenseEncoderSettings,
    ExecutionProvenanceBuildPayload,
    ExecutionProvenanceRetrievalSettings,
    RetrievalMethodId,
)
from graph_memory.retrieval.methods.execution_provenance import (
    ExecutionProvenanceConfig,
    ExecutionProvenanceRetriever,
)
from graph_memory.retrieval.methods.execution_provenance.search import (
    search_provenance_paths,
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
    ) -> object:
        _ = batch_size, normalize_embeddings
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
    ) -> object:
        _ = batch_size
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
    ) -> object:
        _ = batch_size, normalize_embeddings
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
    assert rgcn.input_spec.request_type is EvidenceGraphRankingRequest
    assert rgcn.input_spec.required_artifact is RequiredArtifact.EVIDENCE_GRAPH
    assert rgcn.input_spec.supported_families == frozenset(
        {RetrievalTaskFamily.EVIDENCE_RETRIEVAL}
    )

    dense_ft = Registry.methods.get(RetrievalMethodId.DENSE_FT)
    assert dense_ft.lifecycle is RetrievalLifecycle.DENSE_FINETUNE
    assert dense_ft.input_spec.request_type is TextRankingRequest
    assert dense_ft.input_spec.required_artifact is RequiredArtifact.NONE
    assert dense_ft.input_spec.supported_families == frozenset(
        {
            RetrievalTaskFamily.EVIDENCE_RETRIEVAL,
            RetrievalTaskFamily.EXECUTION_PROVENANCE,
        }
    )
    assert dense_ft.train_artifact is not None
    assert dense_ft.train_artifact.basename == "best_model"
    assert dense_ft.train_artifact.kind is ArtifactKind.DIRECTORY
    assert dense_ft.capabilities.produces_ranked_nodes
    assert not dense_ft.capabilities.produces_native_edge_trace
    assert dense_ft.capabilities.trainable

    assert {
        method.value
        for method in Registry.methods.list_by_family(
            RetrievalTaskFamily.EXECUTION_PROVENANCE
        )
    } == {
        "bm25",
        "dense",
        "dense_ft",
        "graphrag",
        "execution_provenance_retriever",
        "execution_provenance_rgcn_retriever",
    }


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


def test_graphrag_semantic_fallback_keeps_complete_ranking() -> None:
    candidates = (
        TextCandidate("x", "x", {}),
        TextCandidate("y", "y", {}),
    )
    config = GraphRAGConfig()
    method = GraphRAGMethod(dense_ranker=_dense_ranker(), config=config)

    result = method.rank_task(
        build_graphrag_request(
            TextRankingRequest("task", "answer", candidates), config
        ),
        top_k=1,
    )

    assert len(result.ranked_nodes) == len(candidates)
    assert isinstance(result.trace.native_trace, GraphRAGTrace)
    assert result.trace.native_trace.bridges == ()
    assert result.trace.native_trace.exact_dense_fallback


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
    method = ExecutionProvenanceRetriever(
        dense_ranker=_dense_ranker(),
        config=ExecutionProvenanceConfig(
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
    method = ExecutionProvenanceRetriever(
        dense_ranker=_local_dense_ranker(),
        config=ExecutionProvenanceConfig(
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
    method = ExecutionProvenanceRetriever(
        dense_ranker=ranker,
        config=ExecutionProvenanceConfig(
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


def test_provenance_graph_does_not_require_claim_or_verification_nodes() -> None:
    graph = _provenance_graph()

    assert all(
        node.node_type
        not in {ProvenanceNodeType.CLAIM, ProvenanceNodeType.VERIFICATION}
        for node in graph.nodes
    )


def test_provenance_expansion_does_not_reverse_incoming_dependencies() -> None:
    request = ExecutionProvenanceRankingRequest(
        "task-1", "Alpha answer", _candidates(), _provenance_graph()
    )

    paths = search_provenance_paths(
        request,
        ("out-2",),
        config=ExecutionProvenanceConfig(beam_width=2, max_hops=2),
    )

    assert all("out-1" not in path.path.node_ids for path in paths)


def test_provenance_config_rejects_empty_beam() -> None:
    with pytest.raises(ValueError, match="beam_width"):
        ExecutionProvenanceConfig(beam_width=0)


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
