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
        task_input = graph_input.task_input
        query_entities = extract_entities(task_input["query"], use_spacy=self.config.use_spacy)
        scored_targets: list[tuple[float, str]] = []
        for item in task_input["memory_items"]:
            passage = f'{item["source"]}. {item["text"]}'
            score = lexical_score(
                task_input["query"],
                passage,
                graph_input.idf,
                title_aliases=title_aliases(item["source"]),
                query_entities=query_entities,
                passage_entities=graph_input.entities_by_node_id[item["id"]],
            )
            if score > 0.0:
                scored_targets.append((score, item["id"]))

        for score, node_id in sorted(scored_targets, key=lambda scored: (-scored[0], scored[1]))[
            : self.config.max_query_overlap
        ]:
            accumulator.add("q", node_id, "query_overlap", score, directed=True)

