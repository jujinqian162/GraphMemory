from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from graph_memory.contracts.ranking import RankedResult
from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceGraph,
    ExecutionProvenanceNode,
    FieldBinding,
    ProvenanceEdgeType,
    ProvenanceNodeType,
)
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    DenseEncoderSettings,
    ExecutionProvenanceBuildPayload,
    ExecutionProvenanceRetrievalSettings,
    GraphRAGBuildPayload,
    GraphRAGRetrievalSettings,
)
from graph_memory.retrieval.execution.results import assemble_ranked_result
from graph_memory.retrieval.methods.epgm import (
    DEPENDENCY_EDGE_TYPES,
    EpgmRetrieverConfig,
    invalidated_node_ids,
    search_epgm_paths,
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
        # Ranking batches contain one query followed by passages. EPGM also
        # caches a passage-only batch of frozen relation descriptions.
        assert all(
            (
                call[0].startswith("Q::")
                and all(text.startswith("P::") for text in call[1:])
            )
            or all(text.startswith("P::") for text in call)
            for call in encoder.calls
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
    config = EpgmRetrieverConfig.for_variant(
        "dependency_path",
        max_hops=3,
        max_path_expansions=32,
        hop_penalty=0.01,
    )
    paths = search_epgm_paths(
        request.graph,
        ("seed",),
        {"seed": 1.0},
        candidate_ids=frozenset(
            candidate.item_id for candidate in request.candidates
        ),
        config=config,
    )
    target_paths = [path for path in paths if path.target_id == "target"]

    assert {path.node_ids for path in target_paths} == {
        ("seed", "target"),
        ("seed", "call-a", "out-a", "target"),
    }
    assert all(not path.gate.valid for path in target_paths)
    assert {path.gate.rejection_reason for path in target_paths} == {
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

    assert invalidated_node_ids(request.graph) == frozenset(
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


def _ppr_steiner_prediction() -> tuple[RankedResult, TextRankingRequest]:
    request = _alternative_path_request()
    built = Registry.retrieval.build(
        ExecutionProvenanceRetrievalSettings(
            top_k=3,
            encoder=DenseEncoderSettings("recording", "Q::", "P::", 7),
        ),
        ExecutionProvenanceBuildPayload(
            provenance_requests=[request], dense_encoder=RecordingEncoder()
        ),
    )
    result = built.method.rank_task(request, top_k=3)
    text_request = TextRankingRequest(
        request.task_id, request.query_text, request.candidates
    )
    prediction = assemble_ranked_result(
        text_request=text_request,
        method=built.method.name,
        ranked_nodes=result.ranked_nodes,
        top_k=3,
        latency_ms=1.0,
        retrieved_edges=result.trace.retrieved_edges,
        native_trace=result.trace.native_trace,
    )
    return prediction, text_request


def test_ppr_steiner_trace_round_trips_through_ranked_result_validation() -> None:
    prediction, request = _ppr_steiner_prediction()

    validate_ranked_results([prediction], [request])

    plain = cast(dict[str, Any], cast(object, prediction))
    metadata = cast(dict[str, Any], plain["metadata"])
    trace = cast(dict[str, Any], metadata["native_trace"])
    assert trace["trace_kind"] == "execution_provenance_subgraph"
    assert trace["variant"] == "ppr_steiner"


def test_ppr_steiner_trace_rejects_non_normalized_transition_row() -> None:
    prediction, request = _ppr_steiner_prediction()
    malformed = cast(dict[str, Any], cast(object, copy.deepcopy(prediction)))
    metadata = cast(dict[str, Any], malformed["metadata"])
    trace = cast(dict[str, Any], metadata["native_trace"])
    transitions = cast(list[dict[str, Any]], trace["transitions"])
    source = transitions[0]["source"]
    for transition in transitions:
        if transition["source"] == source:
            transition["probability"] *= 0.5

    with pytest.raises(ContractValidationError, match="does not sum to one"):
        validate_ranked_results(
            [cast(RankedResult, cast(object, malformed))], [request]
        )


def test_ppr_steiner_reuses_dense_query_vector_and_caches_relation_vectors() -> None:
    encoder = RecordingEncoder()
    request = _alternative_path_request()
    built = Registry.retrieval.build(
        ExecutionProvenanceRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings("recording", "Q::", "P::", 7),
        ),
        ExecutionProvenanceBuildPayload(
            provenance_requests=[request], dense_encoder=encoder
        ),
    )

    built.method.rank_task(request, top_k=2)
    built.method.rank_task(request, top_k=2)

    query_batches = [call for call in encoder.calls if call[0].startswith("Q::")]
    relation_batches = [
        call for call in encoder.calls if all(text.startswith("P::") for text in call)
    ]
    assert len(query_batches) == 2
    assert len(relation_batches) == 1


def test_ppr_steiner_trace_rejects_selected_native_edge_reorientation() -> None:
    prediction, request = _ppr_steiner_prediction()
    malformed = cast(dict[str, Any], cast(object, copy.deepcopy(prediction)))
    metadata = cast(dict[str, Any], malformed["metadata"])
    trace = cast(dict[str, Any], metadata["native_trace"])
    edges = cast(list[dict[str, Any]], trace["selected_native_edges"])
    assert edges
    edges[0]["source"], edges[0]["target"] = edges[0]["target"], edges[0]["source"]

    with pytest.raises(ContractValidationError, match="orientation is inconsistent"):
        validate_ranked_results(
            [cast(RankedResult, cast(object, malformed))], [request]
        )
