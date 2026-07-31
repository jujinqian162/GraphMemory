from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable

from pydantic import JsonValue

from graph_memory.graphs.provenance import (
    ARTIFACT_NODE,
    FEEDS_EDGE,
    READS_EDGE,
    RESOURCE_FLOW_RELATION,
    RETURNS_EDGE,
    TOOL_CALL_NODE,
    TOOL_OUTPUT_NODE,
    WRITES_EDGE,
    ProvenanceGraph,
    ProvenanceNode,
    logical_output_dependencies,
    output_source_spans,
)
from graph_memory.query_synthesis.provenance.contracts import (
    LogicalDependency,
    MotifQueryTarget,
    MotifSpec,
    QueryIntent,
)


def _motif_id(
    motif_type: str,
    participants: Iterable[str],
    dependencies: Iterable[LogicalDependency],
) -> str:
    payload = {
        "motif_type": motif_type,
        "participants": list(participants),
        "dependencies": [
            dependency.model_dump(mode="json") for dependency in dependencies
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"motif:{motif_type}:{hashlib.sha256(serialized.encode()).hexdigest()[:16]}"


def _attributes(node: ProvenanceNode) -> dict[str, JsonValue]:
    return dict(node.attributes or {})


def _tool_name(node: ProvenanceNode) -> str:
    value = _attributes(node).get("tool_name")
    return value if isinstance(value, str) else "tool"


def _argument_context(node: ProvenanceNode) -> str:
    argument_keys = _attributes(node).get("argument_keys")
    if not isinstance(argument_keys, list) or not argument_keys:
        return "without explicit arguments"
    keys = ", ".join(
        sorted(value for value in argument_keys if isinstance(value, str))
    )
    return f"using the argument fields {keys}" if keys else "with arguments"


def _query_target(
    graph: ProvenanceGraph,
    *,
    query_intent: QueryIntent,
    answer_output_ids: tuple[str, ...],
    support_output_ids: tuple[str, ...],
    safe_slots: dict[str, str],
) -> MotifQueryTarget:
    return MotifQueryTarget(
        query_intent=query_intent,
        answer_output_ids=answer_output_ids,
        support_output_ids=support_output_ids,
        answer_evidence_spans=output_source_spans(graph, answer_output_ids),
        support_evidence_spans=output_source_spans(graph, support_output_ids),
        safe_slots=safe_slots,
    )


def _dependency_targets(
    graph: ProvenanceGraph,
    *,
    source_output_id: str,
    target_output_id: str,
    source_tool: str,
    target_tool: str,
    target_context: str,
    artifact: str | None = None,
) -> tuple[MotifQueryTarget, ...]:
    support = (source_output_id, target_output_id)
    shared = {
        "source_tool": source_tool,
        "target_tool": target_tool,
        "target_context": target_context,
    }
    if artifact is not None:
        shared["artifact"] = artifact
        return (
            _query_target(
                graph,
                query_intent="artifact_origin",
                answer_output_ids=(source_output_id,),
                support_output_ids=support,
                safe_slots=shared,
            ),
            _query_target(
                graph,
                query_intent="artifact_use",
                answer_output_ids=(target_output_id,),
                support_output_ids=support,
                safe_slots=shared,
            ),
            _query_target(
                graph,
                query_intent="complete_chain",
                answer_output_ids=support,
                support_output_ids=support,
                safe_slots=shared,
            ),
        )
    return (
        _query_target(
            graph,
            query_intent="upstream_source",
            answer_output_ids=(source_output_id,),
            support_output_ids=support,
            safe_slots=shared,
        ),
        _query_target(
            graph,
            query_intent="downstream_result",
            answer_output_ids=(target_output_id,),
            support_output_ids=support,
            safe_slots=shared,
        ),
        _query_target(
            graph,
            query_intent="complete_chain",
            answer_output_ids=support,
            support_output_ids=support,
            safe_slots=shared,
        ),
    )


def extract_motifs(graph: ProvenanceGraph) -> tuple[MotifSpec, ...]:
    calls = {
        node.node_id: node
        for node in graph.nodes
        if node.kind == TOOL_CALL_NODE
    }
    outputs = {
        node.node_id: node
        for node in graph.nodes
        if node.kind == TOOL_OUTPUT_NODE
    }
    artifacts = {
        node.node_id: node
        for node in graph.nodes
        if node.kind == ARTIFACT_NODE
    }
    output_for_call: dict[str, str] = {}
    for edge in graph.edges:
        if edge.relation == RETURNS_EDGE:
            output_for_call[edge.source] = edge.target

    motifs: list[MotifSpec] = []
    for call_id, output_id in sorted(output_for_call.items()):
        call = calls[call_id]
        target = _query_target(
            graph,
            query_intent="call_result",
            answer_output_ids=(output_id,),
            support_output_ids=(output_id,),
            safe_slots={
                "tool_name": _tool_name(call),
                "call_context": _argument_context(call),
            },
        )
        motif_id = _motif_id("call_result", (output_id,), ())
        motifs.append(
            MotifSpec(
                motif_id=motif_id,
                motif_type="call_result",
                graph_id=graph.graph_id,
                participant_output_ids=(output_id,),
                dependencies=(),
                targets=(target,),
            )
        )

    base_dependencies: dict[
        tuple[str, str, str], LogicalDependency
    ] = {}
    base_motifs: list[MotifSpec] = []
    call_for_output = {
        output_id: call_id for call_id, output_id in output_for_call.items()
    }
    edge_by_id = {edge.edge_id: edge for edge in graph.edges}
    for projected in logical_output_dependencies(graph):
        dependency = LogicalDependency(
            source_output_id=projected.source_output_id,
            target_output_id=projected.target_output_id,
            relation=projected.relation,
        )
        key = (
            dependency.source_output_id,
            dependency.target_output_id,
            dependency.relation,
        )
        base_dependencies[key] = dependency
        target_call_id = call_for_output[dependency.target_output_id]
        target_call = calls[target_call_id]
        participants = (
            dependency.source_output_id,
            dependency.target_output_id,
        )
        if dependency.relation == FEEDS_EDGE:
            base_motifs.append(
                MotifSpec(
                    motif_id=_motif_id("value_flow", participants, (dependency,)),
                    motif_type="value_flow",
                    graph_id=graph.graph_id,
                    participant_output_ids=participants,
                    dependencies=(dependency,),
                    targets=_dependency_targets(
                        graph,
                        source_output_id=dependency.source_output_id,
                        target_output_id=dependency.target_output_id,
                        source_tool=_tool_name(outputs[dependency.source_output_id]),
                        target_tool=_tool_name(target_call),
                        target_context=_argument_context(target_call),
                    ),
                    hidden_metadata={
                        "source_edge_id": projected.supporting_edge_ids[0]
                    },
                )
            )
            continue
        if dependency.relation != RESOURCE_FLOW_RELATION:
            continue
        resource_edges = [
            edge_by_id[edge_id]
            for edge_id in projected.supporting_edge_ids
            if edge_by_id[edge_id].relation in {READS_EDGE, WRITES_EDGE}
        ]
        artifact_ids = {edge.target for edge in resource_edges}
        if len(artifact_ids) != 1:
            raise ValueError(
                f"resource dependency has inconsistent artifacts={artifact_ids}"
            )
        artifact_id = next(iter(artifact_ids))
        writer_call_id = call_for_output[dependency.source_output_id]
        base_motifs.append(
            MotifSpec(
                motif_id=_motif_id(
                    "artifact_lifecycle", participants, (dependency,)
                ),
                motif_type="artifact_lifecycle",
                graph_id=graph.graph_id,
                participant_output_ids=participants,
                dependencies=(dependency,),
                targets=_dependency_targets(
                    graph,
                    source_output_id=dependency.source_output_id,
                    target_output_id=dependency.target_output_id,
                    source_tool=_tool_name(calls[writer_call_id]),
                    target_tool=_tool_name(target_call),
                    target_context=_argument_context(target_call),
                    artifact=artifacts[artifact_id].text,
                ),
                hidden_metadata={"artifact_id": artifact_id},
            )
        )

    motifs.extend(sorted(base_motifs, key=lambda motif: motif.motif_id))

    dependencies = sorted(
        base_dependencies.values(),
        key=lambda item: (
            item.source_output_id,
            item.target_output_id,
            item.relation,
        ),
    )
    incoming: dict[str, list[LogicalDependency]] = defaultdict(list)
    adjacency: dict[str, list[LogicalDependency]] = defaultdict(list)
    for dependency in dependencies:
        incoming[dependency.target_output_id].append(dependency)
        adjacency[dependency.source_output_id].append(dependency)

    for target_output_id, joined in sorted(incoming.items()):
        unique_sources = sorted({item.source_output_id for item in joined})
        if len(unique_sources) < 2:
            continue
        selected = tuple(
            sorted(
                (
                    item
                    for item in joined
                    if item.source_output_id in unique_sources
                ),
                key=lambda item: (item.source_output_id, item.relation),
            )
        )
        participants = (*unique_sources, target_output_id)
        target_call_id = next(
            call_id
            for call_id, output_id in output_for_call.items()
            if output_id == target_output_id
        )
        source_tools = ", ".join(
            sorted({_tool_name(outputs[source]) for source in unique_sources})
        )
        support = tuple(participants)
        targets = (
            _query_target(
                graph,
                query_intent="contributing_sources",
                answer_output_ids=tuple(unique_sources),
                support_output_ids=support,
                safe_slots={
                    "source_tools": source_tools,
                    "target_tool": _tool_name(calls[target_call_id]),
                    "target_context": _argument_context(calls[target_call_id]),
                },
            ),
            _query_target(
                graph,
                query_intent="downstream_result",
                answer_output_ids=(target_output_id,),
                support_output_ids=support,
                safe_slots={
                    "source_tools": source_tools,
                    "target_tool": _tool_name(calls[target_call_id]),
                    "target_context": _argument_context(calls[target_call_id]),
                },
            ),
            _query_target(
                graph,
                query_intent="complete_chain",
                answer_output_ids=support,
                support_output_ids=support,
                safe_slots={
                    "source_tools": source_tools,
                    "target_tool": _tool_name(calls[target_call_id]),
                    "target_context": _argument_context(calls[target_call_id]),
                },
            ),
        )
        motifs.append(
            MotifSpec(
                motif_id=_motif_id("multi_source_join", participants, selected),
                motif_type="multi_source_join",
                graph_id=graph.graph_id,
                participant_output_ids=participants,
                dependencies=selected,
                targets=targets,
            )
        )

    seen_paths: set[tuple[str, ...]] = set()

    def visit(
        current: str,
        path_nodes: tuple[str, ...],
        path_edges: tuple[LogicalDependency, ...],
    ) -> None:
        if 2 <= len(path_edges) <= 3 and path_nodes not in seen_paths:
            seen_paths.add(path_nodes)
            source_tool = _tool_name(outputs[path_nodes[0]])
            target_tool = _tool_name(outputs[path_nodes[-1]])
            safe_slots = {
                "source_tool": source_tool,
                "target_tool": target_tool,
                "hop_count": str(len(path_edges)),
            }
            targets = (
                _query_target(
                    graph,
                    query_intent="upstream_source",
                    answer_output_ids=(path_nodes[0],),
                    support_output_ids=path_nodes,
                    safe_slots=safe_slots,
                ),
                _query_target(
                    graph,
                    query_intent="downstream_result",
                    answer_output_ids=(path_nodes[-1],),
                    support_output_ids=path_nodes,
                    safe_slots=safe_slots,
                ),
                _query_target(
                    graph,
                    query_intent="complete_chain",
                    answer_output_ids=path_nodes,
                    support_output_ids=path_nodes,
                    safe_slots=safe_slots,
                ),
            )
            motifs.append(
                MotifSpec(
                    motif_id=_motif_id(
                        "multi_hop_flow", path_nodes, path_edges
                    ),
                    motif_type="multi_hop_flow",
                    graph_id=graph.graph_id,
                    participant_output_ids=path_nodes,
                    dependencies=path_edges,
                    targets=targets,
                )
            )
        if len(path_edges) == 3:
            return
        for dependency in adjacency.get(current, ()):
            next_node = dependency.target_output_id
            if next_node in path_nodes:
                continue
            visit(
                next_node,
                (*path_nodes, next_node),
                (*path_edges, dependency),
            )

    for source in sorted(adjacency):
        visit(source, (source,), ())

    unique = {motif.motif_id: motif for motif in motifs}
    return tuple(unique[motif_id] for motif_id in sorted(unique))


__all__ = ["extract_motifs"]
