from __future__ import annotations

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGPruningConfig
from graph_memory.retrieval.methods.fast_graphrag.pruning import prune_knowledge_graph
from graph_memory.retrieval.requests import FastGraphRAGEntity, FastGraphRAGKnowledgeGraph, FastGraphRAGRelation


def test_prune_knowledge_graph_removes_low_frequency_entities_and_orphan_relations() -> None:
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:a", "A", "a", "noun_phrase", "A", ("m0", "m1")),
            FastGraphRAGEntity("e:b", "B", "b", "noun_phrase", "B", ("m0",)),
            FastGraphRAGEntity("e:c", "C", "c", "noun_phrase", "C", ("m1", "m2")),
        ),
        relations=(
            FastGraphRAGRelation("r:a:b", "e:a", "e:b", "cooccurs", ("m0",), 1.0),
            FastGraphRAGRelation("r:a:c", "e:a", "e:c", "cooccurs", ("m1",), 1.0),
        ),
    )

    pruned = prune_knowledge_graph(kg, FastGraphRAGPruningConfig(min_node_freq=2))

    assert [entity.entity_id for entity in pruned.entities] == ["e:a", "e:c"]
    assert [relation.relation_id for relation in pruned.relations] == ["r:a:c"]


def test_prune_knowledge_graph_removes_edges_below_weight_percentile() -> None:
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:a", "A", "a", "noun_phrase", "A", ("m0", "m1")),
            FastGraphRAGEntity("e:b", "B", "b", "noun_phrase", "B", ("m0",)),
            FastGraphRAGEntity("e:c", "C", "c", "noun_phrase", "C", ("m1",)),
        ),
        relations=(
            FastGraphRAGRelation("r:a:b", "e:a", "e:b", "cooccurs", ("m0",), 0.1),
            FastGraphRAGRelation("r:a:c", "e:a", "e:c", "cooccurs", ("m1",), 0.9),
        ),
    )

    pruned = prune_knowledge_graph(kg, FastGraphRAGPruningConfig(min_edge_weight_pct=50.0))

    assert [relation.relation_id for relation in pruned.relations] == ["r:a:c"]


def test_prune_knowledge_graph_can_remove_highest_degree_ego_node() -> None:
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:a", "A", "a", "noun_phrase", "", ("m0", "m1", "m2")),
            FastGraphRAGEntity("e:b", "B", "b", "noun_phrase", "", ("m0",)),
            FastGraphRAGEntity("e:c", "C", "c", "noun_phrase", "", ("m1",)),
            FastGraphRAGEntity("e:d", "D", "d", "noun_phrase", "", ("m2",)),
        ),
        relations=(
            FastGraphRAGRelation("r:a:b", "e:a", "e:b", "", ("m0",), 1.0),
            FastGraphRAGRelation("r:a:c", "e:a", "e:c", "", ("m1",), 1.0),
            FastGraphRAGRelation("r:a:d", "e:a", "e:d", "", ("m2",), 1.0),
        ),
    )

    pruned = prune_knowledge_graph(
        kg,
        FastGraphRAGPruningConfig(
            min_node_degree=0,
            min_edge_weight_pct=0.0,
            remove_ego_nodes=True,
        ),
    )

    assert "e:a" not in {entity.entity_id for entity in pruned.entities}
    assert pruned.relations == ()
