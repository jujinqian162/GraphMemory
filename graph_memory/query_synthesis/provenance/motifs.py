from __future__ import annotations

from collections import defaultdict
from graph_memory.graphs.provenance import (
    ARTIFACT_NODE,
    FEEDS_EDGE,
    READS_EDGE,
    RESOURCE_FLOW_RELATION,
    RETURNS_EDGE,
    WRITES_EDGE,
    ProvenanceGraph,
    logical_output_dependencies,
)
from graph_memory.query_synthesis.provenance.contracts import (
    LogicalDependency,
    MotifAuthoringTarget,
    MotifSpec,
    QueryIntent,
    motif_id,
)


def _query_target(
    *,
    query_intent: QueryIntent,
    focus_output_ids: tuple[str, ...],
    participant_output_ids: tuple[str, ...],
) -> MotifAuthoringTarget:
    return MotifAuthoringTarget(
        query_intent=query_intent,
        focus_output_ids=focus_output_ids,
        participant_output_ids=participant_output_ids,
    )


def _dependency_targets(
    *,
    source_output_id: str,
    target_output_id: str,
    artifact: str | None = None,
) -> tuple[MotifAuthoringTarget, ...]:
    participants = (source_output_id, target_output_id)
    if artifact is not None:
        return (
            _query_target(
                query_intent="artifact_origin",
                focus_output_ids=(source_output_id,),
                participant_output_ids=participants,
            ),
            _query_target(
                query_intent="artifact_use",
                focus_output_ids=(target_output_id,),
                participant_output_ids=participants,
            ),
            _query_target(
                query_intent="complete_chain",
                focus_output_ids=participants,
                participant_output_ids=participants,
            ),
        )
    return (
        _query_target(
            query_intent="upstream_source",
            focus_output_ids=(source_output_id,),
            participant_output_ids=participants,
        ),
        _query_target(
            query_intent="downstream_result",
            focus_output_ids=(target_output_id,),
            participant_output_ids=participants,
        ),
        _query_target(
            query_intent="complete_chain",
            focus_output_ids=participants,
            participant_output_ids=participants,
        ),
    )


def extract_motifs(graph: ProvenanceGraph) -> tuple[MotifSpec, ...]:
    artifacts = {
        node.node_id: node for node in graph.nodes if node.kind == ARTIFACT_NODE
    }
    output_for_call: dict[str, str] = {}
    for edge in graph.edges:
        if edge.relation == RETURNS_EDGE:
            output_for_call[edge.source] = edge.target

    motifs: list[MotifSpec] = []
    for _call_id, output_id in sorted(output_for_call.items()):
        target = _query_target(
            query_intent="call_result",
            focus_output_ids=(output_id,),
            participant_output_ids=(output_id,),
        )
        generated_motif_id = motif_id("call_result", (output_id,), ())
        motifs.append(
            MotifSpec(
                motif_id=generated_motif_id,
                motif_type="call_result",
                graph_id=graph.graph_id,
                participant_output_ids=(output_id,),
                dependencies=(),
                targets=(target,),
            )
        )

    base_dependencies: dict[tuple[str, str, str], LogicalDependency] = {}
    base_motifs: list[MotifSpec] = []
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
        participants = (
            dependency.source_output_id,
            dependency.target_output_id,
        )
        if dependency.relation == FEEDS_EDGE:
            base_motifs.append(
                MotifSpec(
                    motif_id=motif_id("value_flow", participants, (dependency,)),
                    motif_type="value_flow",
                    graph_id=graph.graph_id,
                    participant_output_ids=participants,
                    dependencies=(dependency,),
                    targets=_dependency_targets(
                        source_output_id=dependency.source_output_id,
                        target_output_id=dependency.target_output_id,
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
        base_motifs.append(
            MotifSpec(
                motif_id=motif_id("artifact_lifecycle", participants, (dependency,)),
                motif_type="artifact_lifecycle",
                graph_id=graph.graph_id,
                participant_output_ids=participants,
                dependencies=(dependency,),
                targets=_dependency_targets(
                    source_output_id=dependency.source_output_id,
                    target_output_id=dependency.target_output_id,
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
                (item for item in joined if item.source_output_id in unique_sources),
                key=lambda item: (item.source_output_id, item.relation),
            )
        )
        participants = (*unique_sources, target_output_id)
        targets = (
            _query_target(
                query_intent="contributing_sources",
                focus_output_ids=tuple(unique_sources),
                participant_output_ids=participants,
            ),
            _query_target(
                query_intent="downstream_result",
                focus_output_ids=(target_output_id,),
                participant_output_ids=participants,
            ),
            _query_target(
                query_intent="complete_chain",
                focus_output_ids=participants,
                participant_output_ids=participants,
            ),
        )
        motifs.append(
            MotifSpec(
                motif_id=motif_id("multi_source_join", participants, selected),
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
            targets = (
                _query_target(
                    query_intent="upstream_source",
                    focus_output_ids=(path_nodes[0],),
                    participant_output_ids=path_nodes,
                ),
                _query_target(
                    query_intent="downstream_result",
                    focus_output_ids=(path_nodes[-1],),
                    participant_output_ids=path_nodes,
                ),
                _query_target(
                    query_intent="complete_chain",
                    focus_output_ids=path_nodes,
                    participant_output_ids=path_nodes,
                ),
            )
            motifs.append(
                MotifSpec(
                    motif_id=motif_id("multi_hop_flow", path_nodes, path_edges),
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
