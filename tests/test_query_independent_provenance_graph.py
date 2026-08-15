from __future__ import annotations

import pytest
from pydantic import ValidationError

from graph_memory.datasets.isetrace import adapt_isetrace_record, parse_isetrace_record
from graph_memory.graphs.provenance import (
    CORE_NODE_KINDS,
    CORE_RELATIONS,
    FEEDS_EDGE,
    ProvenanceGraph,
    build_provenance_graph,
)
from tests.isetrace_fixtures import isetrace_record

REVISION = "e40e04d41c04e4eb4bae181ebdd41b61c688081b"


def _graph() -> ProvenanceGraph:
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(isetrace_record()), source_revision=REVISION
    )
    return build_provenance_graph(trajectory)


def test_graph_uses_minimal_query_independent_core() -> None:
    graph = _graph()

    assert {node.kind for node in graph.nodes} == CORE_NODE_KINDS
    assert {edge.relation for edge in graph.edges}.issubset(CORE_RELATIONS)
    assert FEEDS_EDGE in {edge.relation for edge in graph.edges}
    serialized = graph.model_dump_json()
    for forbidden in ("query_id", "query_text", "answer_output_ids", "motif_type"):
        assert forbidden not in serialized
    assert "weight" not in serialized
    assert graph.fingerprint() == _graph().fingerprint()


def test_graph_rejects_unknown_core_relation_and_illegal_endpoints() -> None:
    graph = _graph()
    bad_relation = graph.edges[0].model_copy(update={"relation": "data.semantic_guess"})
    with pytest.raises(ValidationError, match="unsupported core relation"):
        _ = ProvenanceGraph.model_validate(
            graph.model_copy(
                update={"edges": (bad_relation, *graph.edges[1:])}
            ).model_dump(mode="python")
        )

    output_nodes = [
        node for node in graph.nodes if node.kind == "execution.tool_output"
    ]
    bad_endpoint = graph.edges[0].model_copy(
        update={
            "source": output_nodes[0].node_id,
            "target": output_nodes[1].node_id,
        }
    )
    with pytest.raises(ValidationError, match="expects endpoints"):
        _ = ProvenanceGraph.model_validate(
            graph.model_copy(
                update={"edges": (bad_endpoint, *graph.edges[1:])}
            ).model_dump(mode="python")
        )
