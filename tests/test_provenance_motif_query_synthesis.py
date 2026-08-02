from __future__ import annotations

from graph_memory.datasets.isetrace import adapt_isetrace_record, parse_isetrace_record
import pytest

from graph_memory.graphs.provenance import HAS_CONTENT_EDGE, build_provenance_graph
from graph_memory.query_synthesis.provenance import (
    enumerate_template_supervision,
    extract_motifs,
    render_template_supervision,
    select_template_supervision,
    template_count_for_mix,
)
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


def test_template_renderer_uses_existing_motifs_and_focused_output_content() -> None:
    _trajectory, graph, motifs = _trajectory_graph_and_motifs()
    motif = next(item for item in motifs if item.motif_type == "artifact_lifecycle")
    target = next(
        item
        for item in motif.targets
        if len(item.focus_output_ids) < len(item.participant_output_ids)
    )
    fingerprint = graph.fingerprint()

    record = render_template_supervision(graph, motif, target)

    expected_positive_ids = tuple(
        sorted(
            edge.target
            for edge in graph.edges
            if edge.relation == HAS_CONTENT_EDGE
            and edge.source in set(target.focus_output_ids)
        )
    )
    participant_only = set(target.participant_output_ids) - set(
        target.focus_output_ids
    )
    participant_candidate_ids = {
        edge.target
        for edge in graph.edges
        if edge.relation == HAS_CONTENT_EDGE and edge.source in participant_only
    }
    assert record.graph_id == graph.graph_id
    assert record.focus_output_ids == target.focus_output_ids
    assert record.positive_candidate_ids == expected_positive_ids
    assert not (set(record.positive_candidate_ids) & participant_candidate_ids)
    assert record.renderer_version == "provenance-template-v1"
    assert record.graph_fingerprint == fingerprint
    assert graph.fingerprint() == fingerprint
    assert record == render_template_supervision(graph, motif, target)


def test_template_enumeration_skips_targets_with_empty_focused_outputs() -> None:
    _trajectory, graph, motifs = _trajectory_graph_and_motifs()
    motif, target = next(
        (motif, target)
        for motif in motifs
        if motif.motif_type == "artifact_lifecycle"
        for target in motif.targets
        if len(target.focus_output_ids) == 1
    )
    focused_output_id = target.focus_output_ids[0]
    removed_node_ids = {
        edge.target
        for edge in graph.edges
        if edge.relation == HAS_CONTENT_EDGE and edge.source == focused_output_id
    }
    assert removed_node_ids
    graph_with_empty_output = graph.model_copy(
        update={
            "nodes": tuple(
                node for node in graph.nodes if node.node_id not in removed_node_ids
            ),
            "edges": tuple(
                edge
                for edge in graph.edges
                if edge.source not in removed_node_ids
                and edge.target not in removed_node_ids
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="template focused outputs have no output-content candidates",
    ):
        _ = render_template_supervision(graph_with_empty_output, motif, target)

    records = enumerate_template_supervision((graph_with_empty_output,))

    assert records
    assert all(
        focused_output_id not in record.focus_output_ids for record in records
    )
    assert records == enumerate_template_supervision((graph_with_empty_output,))


def test_template_queries_do_not_leak_internal_graph_or_hidden_identifiers() -> None:
    _trajectory, graph, _motifs = _trajectory_graph_and_motifs()
    records = enumerate_template_supervision((graph,))

    assert records
    assert not any(record.motif_type == "call_result" for record in records)
    assert len({record.task_id for record in records}) == len(records)
    assert len(
        {
            (
                record.graph_id,
                record.query_text,
                record.focus_output_ids,
                record.participant_output_ids,
                record.positive_candidate_ids,
            )
            for record in records
        }
    ) == len(records)
    binding_hashes: set[str] = set()
    for edge in graph.edges:
        bindings = (edge.attributes or {}).get("bindings")
        if not isinstance(bindings, list):
            continue
        binding_hashes.update(
            str(binding["value_hash"])
            for binding in bindings
            if isinstance(binding, dict) and "value_hash" in binding
        )
    forbidden = {
        graph.graph_id,
        graph.trajectory_fingerprint,
        *(node.node_id for node in graph.nodes),
        *(edge.edge_id for edge in graph.edges),
        *(record.motif_id for record in records),
        *binding_hashes,
    }
    assert all(
        not any(value and value in record.query_text for value in forbidden)
        for record in records
    )


def test_template_selection_is_owned_deterministic_and_fails_when_insufficient() -> (
    None
):
    _trajectory, graph, _motifs = _trajectory_graph_and_motifs()
    records = enumerate_template_supervision((graph,))

    first = select_template_supervision(
        records,
        eligible_graph_ids=frozenset({graph.graph_id}),
        requested_count=2,
        split="train",
        split_seed=13,
    )
    second = select_template_supervision(
        records,
        eligible_graph_ids=frozenset({graph.graph_id}),
        requested_count=2,
        split="train",
        split_seed=13,
    )
    assert first == second
    assert all(record.graph_id == graph.graph_id for record in first)
    assert template_count_for_mix(4, {"natural": 1, "template": 3}) == 12
    assert template_count_for_mix(4, {"natural": 1, "template": 0}) == 0

    with pytest.raises(ValueError, match=r"requested=\d+ available=\d+"):
        select_template_supervision(
            records,
            eligible_graph_ids=frozenset({graph.graph_id}),
            requested_count=len(records) + 1,
            split="dev",
            split_seed=13,
        )
    with pytest.raises(ValueError, match="natural-only"):
        select_template_supervision(
            records,
            eligible_graph_ids=frozenset({graph.graph_id}),
            requested_count=1,
            split="test",
            split_seed=13,
        )


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
