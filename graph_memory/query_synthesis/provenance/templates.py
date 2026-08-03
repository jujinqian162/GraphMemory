from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from graph_memory.graphs.provenance import (
    ARTIFACT_NODE,
    HAS_CONTENT_EDGE,
    TOOL_OUTPUT_NODE,
    ProvenanceGraph,
)
from graph_memory.query_synthesis.provenance.contracts import (
    MotifAuthoringTarget,
    MotifSpec,
    TemplateSupervisionRecord,
)
from graph_memory.query_synthesis.provenance.motifs import extract_motifs

_ELIGIBLE_TEMPLATE_MOTIFS = frozenset(
    {"value_flow", "artifact_lifecycle", "multi_hop_flow", "multi_source_join"}
)
_SAFE_SPACE = re.compile(r"\s+")


class _MissingFocusedOutputContent(ValueError):
    """A valid motif target cannot provide source-backed template labels."""


def render_template_supervision(
    graph: ProvenanceGraph,
    motif: MotifSpec,
    target: MotifAuthoringTarget,
) -> TemplateSupervisionRecord:
    if motif.graph_id != graph.graph_id:
        raise ValueError("template motif and provenance graph IDs must align")
    if motif.motif_type not in _ELIGIBLE_TEMPLATE_MOTIFS:
        raise ValueError(f"motif_type={motif.motif_type!r} is not template-eligible")
    if target not in motif.targets:
        raise ValueError("template target must belong to the supplied motif")

    node_by_id = graph.node_by_id
    for output_id in target.participant_output_ids:
        node = node_by_id.get(output_id)
        if node is None or node.kind != TOOL_OUTPUT_NODE:
            raise ValueError(f"template participant={output_id!r} is not a ToolOutput")
    candidate_ids_by_output: dict[str, list[str]] = {
        output_id: [] for output_id in target.focus_output_ids
    }
    for edge in graph.edges:
        if edge.relation == HAS_CONTENT_EDGE and edge.source in candidate_ids_by_output:
            candidate_ids_by_output[edge.source].append(edge.target)
    if any(not candidate_ids for candidate_ids in candidate_ids_by_output.values()):
        raise _MissingFocusedOutputContent(
            "template focused outputs have no output-content candidates"
        )
    positive_candidate_ids = tuple(
        sorted(
            candidate_id
            for candidate_ids in candidate_ids_by_output.values()
            for candidate_id in candidate_ids
        )
    )

    query_text = _render_query(graph, motif, target)
    identity = "\0".join(
        (
            "provenance-template-v1",
            graph.graph_id,
            graph.fingerprint(),
            motif.motif_id,
            target.query_intent,
        )
    )
    task_id = f"template:{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    return TemplateSupervisionRecord(
        task_id=task_id,
        graph_id=graph.graph_id,
        query_text=query_text,
        focus_output_ids=target.focus_output_ids,
        participant_output_ids=target.participant_output_ids,
        positive_candidate_ids=positive_candidate_ids,
        motif_id=motif.motif_id,
        motif_type=motif.motif_type,
        query_intent=target.query_intent,
        graph_fingerprint=graph.fingerprint(),
    )


def enumerate_template_supervision(
    graphs: Iterable[ProvenanceGraph],
) -> tuple[TemplateSupervisionRecord, ...]:
    records: list[TemplateSupervisionRecord] = []
    seen_semantics: set[tuple[object, ...]] = set()
    seen_task_ids: set[str] = set()
    for graph in sorted(graphs, key=lambda item: item.graph_id):
        for motif in extract_motifs(graph):
            if motif.motif_type not in _ELIGIBLE_TEMPLATE_MOTIFS:
                continue
            for target in sorted(motif.targets, key=lambda item: item.query_intent):
                try:
                    record = render_template_supervision(graph, motif, target)
                except _MissingFocusedOutputContent:
                    # Empty tool outputs are valid source data but cannot supervise
                    # output-content retrieval, so omit only the unusable target.
                    continue
                semantic_key = (
                    record.graph_id,
                    record.query_text,
                    record.focus_output_ids,
                    record.participant_output_ids,
                    record.positive_candidate_ids,
                )
                if semantic_key in seen_semantics:
                    continue
                if record.task_id in seen_task_ids:
                    raise ValueError(f"duplicate template task ID={record.task_id!r}")
                seen_semantics.add(semantic_key)
                seen_task_ids.add(record.task_id)
                records.append(record)
    return tuple(sorted(records, key=lambda item: item.task_id))


def select_template_supervision(
    records: Iterable[TemplateSupervisionRecord],
    *,
    eligible_graph_ids: frozenset[str],
    requested_count: int,
    split: str,
    split_seed: int,
) -> tuple[TemplateSupervisionRecord, ...]:
    if split == "test":
        if requested_count:
            raise ValueError("ISETrace test split is natural-only")
        return ()
    if split not in {"train", "dev"}:
        raise ValueError(f"unsupported template split={split!r}")
    if requested_count < 0:
        raise ValueError("requested template count cannot be negative")
    eligible = tuple(
        record for record in records if record.graph_id in eligible_graph_ids
    )
    if requested_count > len(eligible):
        raise ValueError(
            "insufficient ISETrace template pool: "
            f"requested={requested_count} available={len(eligible)} split={split}"
        )
    ordered = sorted(
        eligible,
        key=lambda record: hashlib.sha256(
            f"{split_seed}\0{split}\0{record.task_id}".encode()
        ).hexdigest(),
    )
    return tuple(ordered[:requested_count])


def _render_query(
    graph: ProvenanceGraph,
    motif: MotifSpec,
    target: MotifAuthoringTarget,
) -> str:
    tools = tuple(_tool_name(graph, output_id) for output_id in target.participant_output_ids)
    source_tool = tools[0]
    target_tool = tools[-1]
    hop_count = max(1, len(motif.dependencies))
    artifact = _artifact_description(graph, motif)
    if target.query_intent == "upstream_source":
        return (
            f"Which output from {source_tool} supplied the upstream information "
            f"used by {target_tool}?"
        )
    if target.query_intent == "downstream_result":
        return (
            f"What result did {target_tool} produce after using information from "
            f"{source_tool}?"
        )
    if target.query_intent == "complete_chain":
        return (
            f"Retrieve the outputs that show the {hop_count}-step dependency chain "
            f"from {source_tool} to {target_tool}."
        )
    if target.query_intent == "artifact_origin":
        return (
            f"Which output established artifact {artifact} before it was used by "
            f"{target_tool}?"
        )
    if target.query_intent == "artifact_use":
        return (
            f"What output resulted when {target_tool} used artifact {artifact} "
            f"from {source_tool}?"
        )
    if target.query_intent == "contributing_sources":
        contributors = ", ".join(tools[:-1])
        return (
            f"Which outputs from {contributors} jointly contributed to the result "
            f"produced by {target_tool}?"
        )
    raise ValueError(f"unsupported dependency template intent={target.query_intent!r}")


def _tool_name(graph: ProvenanceGraph, output_id: str) -> str:
    node = graph.node_by_id[output_id]
    value = (node.attributes or {}).get("tool_name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ToolOutput={output_id!r} has no safe tool description")
    return _safe_description(value)


def _artifact_description(graph: ProvenanceGraph, motif: MotifSpec) -> str:
    artifact_id = (motif.hidden_metadata or {}).get("artifact_id")
    if isinstance(artifact_id, str):
        node = graph.node_by_id.get(artifact_id)
        if node is not None and node.kind == ARTIFACT_NODE:
            return _safe_description(node.text)
    return "the referenced artifact"


def _safe_description(value: str) -> str:
    compact = _SAFE_SPACE.sub(" ", value).strip()
    if not compact:
        raise ValueError("template description cannot be empty")
    return compact[:160]


__all__ = [
    "enumerate_template_supervision",
    "render_template_supervision",
    "select_template_supervision",
]
