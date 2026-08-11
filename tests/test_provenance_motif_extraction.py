from __future__ import annotations

from graph_memory.datasets.isetrace import adapt_isetrace_record, parse_isetrace_record
from graph_memory.graphs.provenance import build_provenance_graph
from graph_memory.query_synthesis.provenance import extract_motifs
from graph_memory.query_synthesis.provenance.authoring import (
    parse_task_intents,
    parse_task_sources,
    render_task_text,
    source_aliases,
)
from tests.isetrace_fixtures import isetrace_record

REVISION = "e40e04d41c04e4eb4bae181ebdd41b61c688081b"


def _trajectory_graph_and_motifs():
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(isetrace_record()), source_revision=REVISION
    )
    graph = build_provenance_graph(trajectory)
    return trajectory, graph, extract_motifs(graph)


def test_extracts_diverse_internal_authoring_motifs() -> None:
    _trajectory, graph, motifs = _trajectory_graph_and_motifs()

    assert {
        "call_result",
        "value_flow",
        "artifact_lifecycle",
        "multi_hop_flow",
        "multi_source_join",
    }.issubset({motif.motif_type for motif in motifs})
    assert all(motif.graph_id == graph.graph_id for motif in motifs)
    for motif in motifs:
        participants = set(motif.participant_output_ids)
        assert all(
            set(target.focus_output_ids).issubset(target.participant_output_ids)
            for target in motif.targets
        )
        assert all(
            set(target.participant_output_ids).issubset(participants)
            for target in motif.targets
        )
        assert not any(
            field in type(target).model_fields
            for target in motif.targets
            for field in (
                "answer_output_ids",
                "support_output_ids",
                "answer_evidence_spans",
                "support_evidence_spans",
            )
        )


def test_motif_planning_is_deterministic_and_does_not_mutate_graph() -> None:
    _trajectory, graph, motifs = _trajectory_graph_and_motifs()
    fingerprint = graph.fingerprint()

    assert motifs == extract_motifs(graph)
    assert graph.fingerprint() == fingerprint


def test_authoring_helpers_render_and_parse_one_canonical_task_text() -> None:
    trajectory, graph, motifs = _trajectory_graph_and_motifs()
    motif = next(item for item in motifs if item.motif_type == "artifact_lifecycle")
    target = motif.targets[0]
    aliases = source_aliases(graph, target.participant_output_ids)
    text = render_task_text(trajectory, graph, aliases)
    sources = parse_task_sources(text)

    assert parse_task_intents(text) == tuple(
        intent.text for intent in trajectory.intents
    )
    assert {source.handle for source in sources} == set(aliases.values())
    assert {source.kind for source in sources} == {"tool_call", "tool_output"}
