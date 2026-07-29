from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import cast

from pydantic import JsonValue

from graph_memory.graphs.provenance import (
    ARTIFACT_NODE,
    FEEDS_EDGE,
    READS_EDGE,
    RETURNS_EDGE,
    TOOL_CALL_NODE,
    TOOL_OUTPUT_NODE,
    WRITES_EDGE,
    ProvenanceGraph,
    ProvenanceNode,
)
from graph_memory.query_synthesis.provenance.contracts import (
    LogicalDependency,
    MotifQueryTarget,
    MotifSpec,
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


def _position(node: ProvenanceNode) -> tuple[int, int]:
    attributes = _attributes(node)
    message_index = attributes.get("message_index")
    sub_index = attributes.get("sub_index")
    return (
        message_index if isinstance(message_index, int) else 0,
        sub_index if isinstance(sub_index, int) else 0,
    )


def _argument_context(node: ProvenanceNode) -> str:
    arguments = _attributes(node).get("arguments")
    if not isinstance(arguments, dict) or not arguments:
        return "without explicit arguments"
    argument_map = cast(dict[str, JsonValue], arguments)
    compact = json.dumps(
        argument_map, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if len(compact) > 180:
        keys = ", ".join(sorted(argument_map))
        return f"using the argument fields {keys}"
    return f"with arguments {compact}"


def _dependency_targets(
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
            MotifQueryTarget(
                query_intent="artifact_origin",
                answer_output_ids=(source_output_id,),
                support_output_ids=support,
                safe_slots=shared,
            ),
            MotifQueryTarget(
                query_intent="artifact_use",
                answer_output_ids=(target_output_id,),
                support_output_ids=support,
                safe_slots=shared,
            ),
            MotifQueryTarget(
                query_intent="complete_chain",
                answer_output_ids=support,
                support_output_ids=support,
                safe_slots=shared,
            ),
        )
    return (
        MotifQueryTarget(
            query_intent="upstream_source",
            answer_output_ids=(source_output_id,),
            support_output_ids=support,
            safe_slots=shared,
        ),
        MotifQueryTarget(
            query_intent="downstream_result",
            answer_output_ids=(target_output_id,),
            support_output_ids=support,
            safe_slots=shared,
        ),
        MotifQueryTarget(
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
        target = MotifQueryTarget(
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
    for edge in graph.edges:
        if edge.relation != FEEDS_EDGE:
            continue
        target_output_id = output_for_call.get(edge.target)
        if target_output_id is None:
            continue
        dependency = LogicalDependency(
            source_output_id=edge.source,
            target_output_id=target_output_id,
            relation=FEEDS_EDGE,
        )
        key = (
            dependency.source_output_id,
            dependency.target_output_id,
            dependency.relation,
        )
        base_dependencies[key] = dependency
        source_tool = _tool_name(outputs[edge.source])
        target_call = calls[edge.target]
        participants = (edge.source, target_output_id)
        motif_id = _motif_id("value_flow", participants, (dependency,))
        base_motifs.append(
            MotifSpec(
                motif_id=motif_id,
                motif_type="value_flow",
                graph_id=graph.graph_id,
                participant_output_ids=participants,
                dependencies=(dependency,),
                targets=_dependency_targets(
                    source_output_id=edge.source,
                    target_output_id=target_output_id,
                    source_tool=source_tool,
                    target_tool=_tool_name(target_call),
                    target_context=_argument_context(target_call),
                ),
                hidden_metadata={"source_edge_id": edge.edge_id},
            )
        )

    writes_by_artifact: dict[str, list[str]] = defaultdict(list)
    reads_by_artifact: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.relation == WRITES_EDGE:
            writes_by_artifact[edge.target].append(edge.source)
        elif edge.relation == READS_EDGE:
            reads_by_artifact[edge.target].append(edge.source)
    for artifact_id in sorted(set(writes_by_artifact) & set(reads_by_artifact)):
        writer_calls = sorted(
            writes_by_artifact[artifact_id], key=lambda node_id: _position(calls[node_id])
        )
        reader_calls = sorted(
            reads_by_artifact[artifact_id], key=lambda node_id: _position(calls[node_id])
        )
        for reader_call_id in reader_calls:
            reader_position = _position(calls[reader_call_id])
            prior_writers = [
                writer_call_id
                for writer_call_id in writer_calls
                if _position(calls[writer_call_id]) < reader_position
            ]
            if not prior_writers:
                continue
            writer_call_id = prior_writers[-1]
            source_output_id = output_for_call[writer_call_id]
            target_output_id = output_for_call[reader_call_id]
            dependency = LogicalDependency(
                source_output_id=source_output_id,
                target_output_id=target_output_id,
                relation="resource.flow",
            )
            key = (
                dependency.source_output_id,
                dependency.target_output_id,
                dependency.relation,
            )
            base_dependencies[key] = dependency
            artifact = artifacts[artifact_id].text
            participants = (source_output_id, target_output_id)
            motif_id = _motif_id(
                "artifact_lifecycle", participants, (dependency,)
            )
            base_motifs.append(
                MotifSpec(
                    motif_id=motif_id,
                    motif_type="artifact_lifecycle",
                    graph_id=graph.graph_id,
                    participant_output_ids=participants,
                    dependencies=(dependency,),
                    targets=_dependency_targets(
                        source_output_id=source_output_id,
                        target_output_id=target_output_id,
                        source_tool=_tool_name(calls[writer_call_id]),
                        target_tool=_tool_name(calls[reader_call_id]),
                        target_context=_argument_context(calls[reader_call_id]),
                        artifact=artifact,
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
            MotifQueryTarget(
                query_intent="contributing_sources",
                answer_output_ids=tuple(unique_sources),
                support_output_ids=support,
                safe_slots={
                    "source_tools": source_tools,
                    "target_tool": _tool_name(calls[target_call_id]),
                    "target_context": _argument_context(calls[target_call_id]),
                },
            ),
            MotifQueryTarget(
                query_intent="downstream_result",
                answer_output_ids=(target_output_id,),
                support_output_ids=support,
                safe_slots={
                    "source_tools": source_tools,
                    "target_tool": _tool_name(calls[target_call_id]),
                    "target_context": _argument_context(calls[target_call_id]),
                },
            ),
            MotifQueryTarget(
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
                MotifQueryTarget(
                    query_intent="upstream_source",
                    answer_output_ids=(path_nodes[0],),
                    support_output_ids=path_nodes,
                    safe_slots=safe_slots,
                ),
                MotifQueryTarget(
                    query_intent="downstream_result",
                    answer_output_ids=(path_nodes[-1],),
                    support_output_ids=path_nodes,
                    safe_slots=safe_slots,
                ),
                MotifQueryTarget(
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
