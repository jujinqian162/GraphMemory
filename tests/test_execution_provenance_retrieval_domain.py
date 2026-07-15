from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceGraph,
    ExecutionProvenanceNode,
    FieldBinding,
    ProvenanceEdgeType,
    ProvenanceNodeType,
)
from graph_memory.registry import Registry
from graph_memory.registry.methods import RequiredArtifact, RetrievalTaskFamily
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
    enumerate_provenance_paths,
)
from graph_memory.retrieval.contracts import ExecutionProvenanceTrace, GraphRAGTrace
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


def _provenance_graph(
    *, include_untraversed_edge: bool = False
) -> ExecutionProvenanceGraph:
    nodes = (
        ExecutionProvenanceNode("task", ProvenanceNodeType.TASK, "find Alpha"),
        ExecutionProvenanceNode("agent", ProvenanceNodeType.AGENT, "assistant"),
        ExecutionProvenanceNode("call-1", ProvenanceNodeType.TOOL_CALL, "lookup Alpha"),
        ExecutionProvenanceNode(
            "out-1", ProvenanceNodeType.TOOL_OUTPUT, "Alpha result"
        ),
        ExecutionProvenanceNode(
            "call-2", ProvenanceNodeType.TOOL_CALL, "use Alpha result"
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

    assert {
        method.value
        for method in Registry.methods.list_by_family(
            RetrievalTaskFamily.EXECUTION_PROVENANCE
        )
    } == {
        "bm25",
        "dense",
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


def test_graphrag_builds_entity_search_without_evidence_graph() -> None:
    config = GraphRAGConfig(seed_top_s=2, max_iterations=20)
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
    assert result.trace.native_trace.entity_ids
    assert result.trace.native_trace.linked_entity_ids
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
    assert result.trace.native_trace.relations == ()


def test_provenance_retriever_returns_only_actual_traversed_edges() -> None:
    graph = _provenance_graph(include_untraversed_edge=True)
    request = ExecutionProvenanceRankingRequest(
        "task-1", "Alpha answer", _candidates(), graph
    )
    method = ExecutionProvenanceRetriever(
        dense_ranker=_dense_ranker(),
        config=ExecutionProvenanceConfig(seed_top_s=1, max_hops=4, top_paths=2),
    )

    result = method.rank_task(request, top_k=4)

    assert len(result.ranked_nodes) == len(request.candidates)
    assert isinstance(result.trace.native_trace, ExecutionProvenanceTrace)
    assert result.trace.native_trace.paths
    assert result.trace.native_trace.edges
    assert all(
        edge.edge_type is not ProvenanceEdgeType.PRECEDES
        for edge in result.trace.native_trace.edges
    )
    assert result.trace.retrieved_edges == [
        {
            "source": "out-1",
            "target": "out-2",
            "edge_type": "sequential",
            "weight": 1.0,
            "directed": True,
        }
    ]


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

    paths = enumerate_provenance_paths(
        request,
        ("out-2",),
        ExecutionProvenanceConfig(beam_width=2, max_hops=4),
    )

    assert all("out-1" not in path.node_ids for path in paths)


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
