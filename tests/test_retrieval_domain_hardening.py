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
from graph_memory.retrieval.methods.epgm import EpgmRetrieverConfig
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


class PromotionEncoder:
    """Ranks the anchor first and the structural partner last.

    The partner must therefore be recovered by typed partner completion rather
    than by Dense alone, which is what makes a promoting trace available for
    round-trip and rejection tests.
    """

    _scores = {
        "seed result": 1.0,
        "noise one": 0.9,
        "noise two": 0.8,
        "final result": 0.5,
    }

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        _ = batch_size, normalize_embeddings, show_progress_bar
        rows = [
            np.asarray(
                [self._scores.get(text.split("::", 1)[-1], 1.0), 0.0],
                dtype=np.float32,
            )
            for text in texts
        ]
        return np.asarray(rows, dtype=np.float32)


def _promotion_request(
    *, repeated_relation_path: bool = False
) -> ExecutionProvenanceRankingRequest:
    """A graph whose lowest-ranked candidate is a typed neighbour of the top one.

    With ``repeated_relation_path`` the anchor reaches its partner over two hops
    of the *same* relation type through a non-candidate connector.
    """

    nodes = [
        ExecutionProvenanceNode("seed", ProvenanceNodeType.TOOL_OUTPUT, "seed result"),
        ExecutionProvenanceNode("n1", ProvenanceNodeType.TOOL_OUTPUT, "noise one"),
        ExecutionProvenanceNode("n2", ProvenanceNodeType.TOOL_OUTPUT, "noise two"),
        ExecutionProvenanceNode(
            "target", ProvenanceNodeType.TOOL_OUTPUT, "final result"
        ),
    ]
    if repeated_relation_path:
        nodes.append(
            ExecutionProvenanceNode("mid", ProvenanceNodeType.TOOL_CALL, "mid hop")
        )
        edges = (
            ExecutionProvenanceEdge("seed", "mid", ProvenanceEdgeType.DEPENDS_ON),
            ExecutionProvenanceEdge("mid", "target", ProvenanceEdgeType.DEPENDS_ON),
        )
    else:
        nodes.append(
            ExecutionProvenanceNode("call-x", ProvenanceNodeType.TOOL_CALL, "aux call")
        )
        edges = (
            ExecutionProvenanceEdge("seed", "target", ProvenanceEdgeType.DEPENDS_ON),
            ExecutionProvenanceEdge("call-x", "n2", ProvenanceEdgeType.RETURNS),
        )
    graph = ExecutionProvenanceGraph("promo-task", tuple(nodes), edges)
    return ExecutionProvenanceRankingRequest(
        "promo-task",
        "find result",
        (
            TextCandidate("seed", "seed result", {}),
            TextCandidate("n1", "noise one", {}),
            TextCandidate("n2", "noise two", {}),
            TextCandidate("target", "final result", {}),
        ),
        graph,
    )


def _promotion_prediction(
    *, repeated_relation_path: bool = False
) -> tuple[RankedResult, TextRankingRequest]:
    request = _promotion_request(repeated_relation_path=repeated_relation_path)
    built = Registry.retrieval.build(
        ExecutionProvenanceRetrievalSettings(
            top_k=4,
            encoder=DenseEncoderSettings("promotion", "Q::", "P::", 7),
            config=EpgmRetrieverConfig(
                preserve_dense_top_n=0, min_partner_confidence=0.1
            ),
        ),
        ExecutionProvenanceBuildPayload(
            provenance_requests=[request], dense_encoder=PromotionEncoder()
        ),
    )
    result = built.method.rank_task(request, top_k=4)
    text_request = TextRankingRequest(
        request.task_id, request.query_text, request.candidates
    )
    prediction = assemble_ranked_result(
        text_request=text_request,
        method=built.method.name,
        ranked_nodes=result.ranked_nodes,
        top_k=4,
        latency_ms=1.0,
        retrieved_edges=result.trace.retrieved_edges,
        native_trace=result.trace.native_trace,
    )
    return prediction, text_request


def _native_trace_of(prediction: RankedResult) -> dict[str, Any]:
    plain = cast(dict[str, Any], cast(object, prediction))
    metadata = cast(dict[str, Any], plain["metadata"])
    return cast(dict[str, Any], metadata["native_trace"])


def test_typed_partner_completion_trace_round_trips_through_validation() -> None:
    prediction, request = _promotion_prediction()

    validate_ranked_results([prediction], [request])

    trace = _native_trace_of(prediction)
    assert trace["trace_kind"] == "typed_partner_completion"
    assert not trace["exact_dense_fallback"]
    accepted = [item for item in trace["proposals"] if item["accepted"]]
    assert [(item["anchor_id"], item["partner_id"]) for item in accepted] == [
        ("seed", "target")
    ]
    assert trace["promoted_native_edges"]


def test_typed_partner_completion_trace_rejects_non_normalized_affinities() -> None:
    prediction, request = _promotion_prediction()
    malformed = cast(dict[str, Any], cast(object, copy.deepcopy(prediction)))
    trace = _native_trace_of(cast(RankedResult, cast(object, malformed)))
    relations = cast(list[dict[str, Any]], trace["relations"])
    assert relations
    relations[0]["affinity"] *= 0.5

    with pytest.raises(ContractValidationError, match="do not sum to one"):
        validate_ranked_results(
            [cast(RankedResult, cast(object, malformed))], [request]
        )


def test_typed_partner_completion_trace_rejects_promoted_edge_reorientation() -> None:
    prediction, request = _promotion_prediction()
    malformed = cast(dict[str, Any], cast(object, copy.deepcopy(prediction)))
    trace = _native_trace_of(cast(RankedResult, cast(object, malformed)))
    edges = cast(list[dict[str, Any]], trace["promoted_native_edges"])
    assert edges
    edges[0]["source"], edges[0]["target"] = edges[0]["target"], edges[0]["source"]

    with pytest.raises(
        ContractValidationError, match="do not match accepted proposal paths"
    ):
        validate_ranked_results(
            [cast(RankedResult, cast(object, malformed))], [request]
        )


def test_typed_partner_completion_trace_rejects_candidate_as_connector() -> None:
    prediction, request = _promotion_prediction()
    malformed = cast(dict[str, Any], cast(object, copy.deepcopy(prediction)))
    trace = _native_trace_of(cast(RankedResult, cast(object, malformed)))
    connectors = cast(list[str], trace["connector_node_ids"])
    connectors.append("n1")

    with pytest.raises(
        ContractValidationError, match="connector must not be a request candidate"
    ):
        validate_ranked_results(
            [cast(RankedResult, cast(object, malformed))], [request]
        )


def test_repeated_relation_path_survives_trace_validation() -> None:
    """path_edge_types/path_directions are ordered sequences, not sets.

    A two-hop path may legitimately traverse one relation type twice, so the
    validator must not apply its set-uniqueness rule to these fields.
    """

    prediction, request = _promotion_prediction(repeated_relation_path=True)
    trace = _native_trace_of(prediction)
    accepted = [item for item in trace["proposals"] if item["accepted"]]
    assert [item["path_edge_types"] for item in accepted] == [
        ["depends_on", "depends_on"]
    ]

    validate_ranked_results([prediction], [request])


def test_epgm_reuses_dense_query_vector_and_caches_relation_vectors() -> None:
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
