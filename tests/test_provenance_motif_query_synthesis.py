from __future__ import annotations

from collections import defaultdict

import pytest
from pydantic import ValidationError

from graph_memory.datasets.isetrace import adapt_isetrace_record, parse_isetrace_record
from graph_memory.graphs.provenance import build_provenance_graph
from graph_memory.query_synthesis.provenance import (
    DEFAULT_TEMPLATE_CATALOG,
    QueryTemplate,
    SyntheticQueryExample,
    extract_motifs,
    verbalize_all_templates,
    verbalize_motif,
)
from tests.isetrace_fixtures import isetrace_record

REVISION = "e40e04d41c04e4eb4bae181ebdd41b61c688081b"


def _graph_and_motifs():
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(isetrace_record()), source_revision=REVISION
    )
    graph = build_provenance_graph(trajectory)
    return graph, extract_motifs(graph)


def test_extracts_diverse_native_and_composite_motifs() -> None:
    graph, motifs = _graph_and_motifs()
    motif_types = {motif.motif_type for motif in motifs}

    assert {
        "call_result",
        "value_flow",
        "artifact_lifecycle",
        "multi_hop_flow",
        "multi_source_join",
    }.issubset(motif_types)
    assert all(motif.graph_id == graph.graph_id for motif in motifs)
    for motif in motifs:
        participants = set(motif.participant_output_ids)
        assert all(
            set(target.answer_output_ids).issubset(target.support_output_ids)
            for target in motif.targets
        )
        complete_chain_targets = [
            target
            for target in motif.targets
            if target.query_intent == "complete_chain"
        ]
        assert all(
            set(target.support_output_ids) == participants
            for target in complete_chain_targets
        )
        assert all(
            {
                dependency.source_output_id,
                dependency.target_output_id,
            }.issubset(participants)
            for dependency in motif.dependencies
        )


def test_template_catalog_enforces_pair_level_diversity() -> None:
    grouped: defaultdict[tuple[str, str], list[QueryTemplate]] = defaultdict(list)
    for template in DEFAULT_TEMPLATE_CATALOG.templates:
        grouped[(template.motif_type, template.query_intent)].append(template)

    assert len(DEFAULT_TEMPLATE_CATALOG.templates) == 78
    for templates in grouped.values():
        assert len(templates) >= 6
        assert len({template.text.lower() for template in templates}) == len(templates)
        assert len({tag for template in templates for tag in template.style_tags}) >= 3


def test_one_motif_supports_multiple_intents_and_wordings_without_graph_mutation() -> None:
    graph, motifs = _graph_and_motifs()
    motif = next(motif for motif in motifs if motif.motif_type == "value_flow")
    graph_fingerprint = graph.fingerprint()

    generated = {
        target.query_intent: verbalize_motif(motif, target.query_intent, seed=7)
        for target in motif.targets
    }
    assert set(generated) == {
        "upstream_source",
        "downstream_result",
        "complete_chain",
    }
    assert len({item.query.query_text for item in generated.values()}) == 3
    assert graph.fingerprint() == graph_fingerprint

    all_forms = verbalize_all_templates(motif, "upstream_source", seed=7)
    assert len(all_forms) >= 6
    assert len({item.query.query_text for item in all_forms}) == len(all_forms)
    assert len({item.label.style_tags for item in all_forms}) >= 3
    assert graph.fingerprint() == graph_fingerprint


def test_verbalization_is_deterministic_and_rejects_internal_id_leakage() -> None:
    _, motifs = _graph_and_motifs()
    motif = next(motif for motif in motifs if motif.motif_type == "call_result")
    first = verbalize_motif(motif, "call_result", seed=31)
    second = verbalize_motif(motif, "call_result", seed=31)
    assert first == second

    with pytest.raises(ValidationError, match="leaks an internal"):
        _ = SyntheticQueryExample(
            query=first.query.model_copy(
                update={
                    "query_text": (
                        first.query.query_text
                        + " "
                        + first.label.answer_output_ids[0]
                    )
                }
            ),
            label=first.label,
        )
