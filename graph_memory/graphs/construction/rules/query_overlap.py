from __future__ import annotations

from dataclasses import dataclass

from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.context import PreparedGraphInput
from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator
from graph_memory.text.entities import extract_entities, title_aliases
from graph_memory.text.lexical import lexical_score


@dataclass(frozen=True)
class QueryOverlapEdgeRule:
    config: GraphBuildConfig

    def add_edges(self, graph_input: PreparedGraphInput, accumulator: EdgeAccumulator) -> None:
        request = graph_input.request
        query_entities = extract_entities(request.query_text, use_spacy=self.config.use_spacy)
        scored_targets: list[tuple[float, str]] = []
        for node in request.nodes:
            passage = f"{node.source_ref}. {node.text}" if node.source_ref else node.text
            score = lexical_score(
                request.query_text,
                passage,
                graph_input.idf,
                title_aliases=title_aliases(node.source_ref or ""),
                query_entities=query_entities,
                passage_entities=graph_input.entities_by_node_id[node.node_id],
            )
            if score > 0.0:
                scored_targets.append((score, node.node_id))

        for score, node_id in sorted(scored_targets, key=lambda scored: (-scored[0], scored[1]))[
            : self.config.max_query_overlap
        ]:
            accumulator.add("q", node_id, "query_overlap", score, directed=True)
