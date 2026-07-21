from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
from graph_memory.registry.methods import RequiredArtifact
from graph_memory.registry.retrieval import (
    DenseEncoderSettings,
    ExecutionProvenanceBuildPayload,
    ExecutionProvenanceRetrievalSettings,
    GraphRAGBuildPayload,
    GraphRAGRetrievalSettings,
    RetrievalMethodId,
)
from graph_memory.retrieval.methods.execution_provenance import (
    ExecutionProvenanceConfig,
)
from graph_memory.retrieval.methods.execution_provenance.search import (
    DEPENDENCY_EDGE_TYPES,
    invalidated_node_ids,
    search_provenance_paths,
)
from graph_memory.retrieval.methods.graphrag import GraphRAGConfig
from graph_memory.retrieval.methods.graphrag.index import build_graphrag_request
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    GraphRAGRequest,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.validation import ContractValidationError, validate_ranked_results


ROOT = Path(__file__).resolve().parents[1]


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        _ = batch_size, normalize_embeddings, show_progress_bar
        self.calls.append(list(texts))
        rows = []
        for index, _text in enumerate(texts):
            row = np.zeros(16, dtype=np.float32)
            row[index % row.shape[0]] = 1.0
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)


def _graphrag_text_request() -> TextRankingRequest:
    return TextRankingRequest(
        task_id="graph-task",
        query_text="What did Ada design?",
        candidates=(
            TextCandidate(
                "c1",
                "Ada Lovelace designed the Analytical Engine.",
                {"title": "Ada Lovelace", "aliases": "Ada"},
            ),
            TextCandidate(
                "c2",
                "Ada and the Analytical Engine influenced early computing.",
                {"title": "Analytical Engine"},
            ),
        ),
    )


def _binding(name: str) -> FieldBinding:
    return FieldBinding(name, name, f"{name}-hash", "exact")


def _alternative_path_request() -> ExecutionProvenanceRankingRequest:
    nodes = (
        ExecutionProvenanceNode(
            "seed",
            ProvenanceNodeType.TOOL_OUTPUT,
            "seed result",
            {"output_field_hashes": {"seed": "seed-hash"}},
        ),
        ExecutionProvenanceNode(
            "call-a",
            ProvenanceNodeType.TOOL_CALL,
            "use seed",
            {"input_parameters": ["seed"]},
        ),
        ExecutionProvenanceNode(
            "out-a",
            ProvenanceNodeType.TOOL_OUTPUT,
            "bound result",
            {"output_field_hashes": {"result": "result-hash"}},
        ),
        ExecutionProvenanceNode(
            "target",
            ProvenanceNodeType.TOOL_CALL,
            "final call",
            {"input_parameters": ["result"]},
        ),
    )
    edges = (
        ExecutionProvenanceEdge("seed", "target", ProvenanceEdgeType.DEPENDS_ON),
        ExecutionProvenanceEdge(
            "seed", "call-a", ProvenanceEdgeType.FEEDS, binding=_binding("seed")
        ),
        ExecutionProvenanceEdge("call-a", "out-a", ProvenanceEdgeType.RETURNS),
        ExecutionProvenanceEdge(
            "out-a", "target", ProvenanceEdgeType.FEEDS, binding=_binding("result")
        ),
    )
    graph = ExecutionProvenanceGraph("path-task", nodes, edges)
    return ExecutionProvenanceRankingRequest(
        "path-task",
        "find result",
        (
            TextCandidate("seed", "seed result", {}),
            TextCandidate("target", "final call", {}),
        ),
        graph,
    )


def test_graphrag_builder_assembles_explicit_alias_aware_graph() -> None:
    first = build_graphrag_request(_graphrag_text_request(), GraphRAGConfig())
    second = build_graphrag_request(_graphrag_text_request(), GraphRAGConfig())

    assert isinstance(first, GraphRAGRequest)
    assert first == second
    ada_title = next(
        mention
        for mention in first.knowledge_graph.mentions
        if mention.normalized_surface == "ada lovelace"
        and mention.mention_type == "TITLE_ENTITY"
    )
    assert any(
        mention.candidate_id == "c2"
        and mention.entity_id == ada_title.entity_id
        and mention.mention_type == "MENTIONS"
        and mention.alias_confidence == pytest.approx(0.9)
        for mention in first.knowledge_graph.mentions
    )
    analytical_group = next(
        group
        for group in first.knowledge_graph.title_groups
        if group.normalized_title_entity == "analytical engine"
    )
    assert analytical_group.candidate_ids == ("c2",)


def test_new_dense_methods_preserve_query_and_passage_prefixes() -> None:
    graph_encoder = RecordingEncoder()
    graph_built = Registry.retrieval.build(
        GraphRAGRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings("recording", "Q::", "P::", 7),
        ),
        GraphRAGBuildPayload(
            text_requests=[_graphrag_text_request()], dense_encoder=graph_encoder
        ),
    )
    graph_built.method.rank_task(graph_built.execution_tasks[0].method_request, top_k=2)

    provenance_encoder = RecordingEncoder()
    provenance_request = _alternative_path_request()
    provenance_built = Registry.retrieval.build(
        ExecutionProvenanceRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings("recording", "Q::", "P::", 7),
        ),
        ExecutionProvenanceBuildPayload(
            provenance_requests=[provenance_request],
            dense_encoder=provenance_encoder,
        ),
    )
    provenance_built.method.rank_task(provenance_request, top_k=2)

    for encoder in (graph_encoder, provenance_encoder):
        assert encoder.calls
        assert all(call[0].startswith("Q::") for call in encoder.calls)
        assert all(
            text.startswith("P::") for call in encoder.calls for text in call[1:]
        )


def test_provenance_rejects_untyped_support_transition() -> None:
    with pytest.raises(ValueError, match="Invalid supports transition"):
        ExecutionProvenanceGraph(
            "invalid",
            (
                ExecutionProvenanceNode("answer", ProvenanceNodeType.ANSWER, "answer"),
                ExecutionProvenanceNode("call", ProvenanceNodeType.TOOL_CALL, "call"),
            ),
            (ExecutionProvenanceEdge("answer", "call", ProvenanceEdgeType.SUPPORTS),),
        )


def test_provenance_rejects_incomplete_and_multi_semantic_paths() -> None:
    request = _alternative_path_request()
    config = ExecutionProvenanceConfig(
        max_hops=3,
        max_path_expansions=32,
        hop_penalty=0.01,
    )
    evaluations = search_provenance_paths(request, ("seed",), config=config)
    target_paths = [
        evaluation for evaluation in evaluations if evaluation.partner_id == "target"
    ]

    assert {evaluation.path.node_ids for evaluation in target_paths} == {
        ("seed", "target"),
        ("seed", "call-a", "out-a", "target"),
    }
    assert all(not evaluation.score.valid for evaluation in target_paths)
    assert {evaluation.score.rejection_reason for evaluation in target_paths} == {
        "incomplete_path"
    }


def test_provenance_invalidation_uses_revision_edges_and_lifecycle_metadata() -> None:
    graph = ExecutionProvenanceGraph(
        "revision-task",
        (
            ExecutionProvenanceNode(
                "verification", ProvenanceNodeType.VERIFICATION, "new check"
            ),
            ExecutionProvenanceNode(
                "edge-invalidated", ProvenanceNodeType.CLAIM, "old claim"
            ),
            ExecutionProvenanceNode(
                "metadata-invalidated",
                ProvenanceNodeType.CLAIM,
                "obsolete claim",
                {"lifecycle_state": "superseded"},
            ),
        ),
        (
            ExecutionProvenanceEdge(
                "verification",
                "edge-invalidated",
                ProvenanceEdgeType.INVALIDATES,
            ),
        ),
    )
    request = ExecutionProvenanceRankingRequest(
        "revision-task",
        "current claim",
        (
            TextCandidate("edge-invalidated", "old claim", {}),
            TextCandidate("metadata-invalidated", "obsolete claim", {}),
        ),
        graph,
    )

    assert invalidated_node_ids(request) == frozenset(
        {"edge-invalidated", "metadata-invalidated"}
    )
    assert ProvenanceEdgeType.INVALIDATES not in DEPENDENCY_EDGE_TYPES
    assert ProvenanceEdgeType.PRECEDES not in DEPENDENCY_EDGE_TYPES


def test_ranked_result_validation_rejects_malformed_native_trace() -> None:
    request = TextRankingRequest(
        "trace-task", "query", (TextCandidate("c1", "candidate", {}),)
    )
    prediction = {
        "task_id": "trace-task",
        "method": "execution_provenance_retriever",
        "ranked_nodes": [{"node_id": "c1", "score": 1.0}],
        "retrieved_subgraph": {"nodes": ["c1"], "edges": []},
        "latency_ms": 1.0,
        "input_tokens": 1,
        "metadata": {
            "native_trace": {
                "trace_kind": "execution_provenance",
                "paths": [],
                "edges": [
                    {
                        "source": "c1",
                        "edge_type": "feeds",
                        "weight": float("nan"),
                    }
                ],
            }
        },
    }

    with pytest.raises(ContractValidationError, match="native trace"):
        validate_ranked_results([prediction], [request])


def test_registry_required_artifact_query_is_authoritative() -> None:
    assert Registry.methods.requires_artifact(
        RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
        RequiredArtifact.EVIDENCE_GRAPH,
    )
    assert not Registry.methods.requires_artifact(
        RetrievalMethodId.GRAPHRAG,
        RequiredArtifact.EVIDENCE_GRAPH,
    )
