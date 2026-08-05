from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from graph_memory.experiment.config import DenseEncoderConfig, GraphRAGMethodConfig
from graph_memory.registry.retrieval_builders import build_retrieval
from graph_memory.retrieval.methods.graphrag import GraphRAGConfig
from graph_memory.retrieval.methods.graphrag.index import build_graphrag_request
from graph_memory.retrieval.requests import (
    GraphRAGRequest,
    TextCandidate,
    TextRankingRequest,
)


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        _ = batch_size, normalize_embeddings, show_progress_bar
        self.calls.append(list(texts))
        rows = []
        for index, _text in enumerate(texts):
            row = np.zeros(16, dtype=np.float32)
            row[index % row.shape[0]] = 1.0
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)


def _graphrag_text_request() -> TextRankingRequest:
    return TextRankingRequest(
        task_id="graph-task",
        query_text="What connected Ada Lovelace and the Analytical Engine?",
        candidates=(
            TextCandidate(
                item_id="c1",
                text=(
                    "Ada Lovelace documented the Analytical Engine design. "
                    "Ada Lovelace discussed the Analytical Engine architecture."
                ),
                metadata={},
            ),
            TextCandidate(
                item_id="c2",
                text=(
                    "The Analytical Engine notes credit Ada Lovelace. "
                    "The Analytical Engine archive preserves Ada Lovelace notes."
                ),
                metadata={},
            ),
        ),
    )


def test_graphrag_builder_assembles_deterministic_noun_cooccurrence_graph() -> None:
    config = GraphRAGConfig(
        text_unit_size=12,
        text_unit_overlap=2,
        min_node_frequency=1,
        min_edge_weight_percentile=0.0,
        remove_ego_node=False,
    )
    first = build_graphrag_request(_graphrag_text_request(), config)
    second = build_graphrag_request(_graphrag_text_request(), config)

    assert isinstance(first, GraphRAGRequest)
    assert first == second
    assert first.knowledge_graph.text_units
    names = {entity.name for entity in first.knowledge_graph.entities}
    assert "ada lovelace" in names
    assert "analytical engine" in names
    entity_by_name = {
        entity.name: entity.entity_id for entity in first.knowledge_graph.entities
    }
    expected_pair = {
        entity_by_name["ada lovelace"],
        entity_by_name["analytical engine"],
    }
    assert any(
        {relation.source_entity_id, relation.target_entity_id} == expected_pair
        for relation in first.knowledge_graph.relations
    )


def test_graphrag_preserves_query_and_passage_prefixes() -> None:
    encoder = RecordingEncoder()
    retrieval_method, _provenance, execution_requests = build_retrieval(
        GraphRAGMethodConfig(
            method="graphrag",
            encoder=DenseEncoderConfig(
                model_name="recording",
                query_prefix="Q::",
                passage_prefix="P::",
                batch_size=7,
            ),
            text_unit_size=48,
            text_unit_overlap=8,
            min_node_frequency=1,
            min_edge_weight_percentile=0.0,
            remove_ego_node=False,
            seed_top_s=5,
        ),
        text_requests=[_graphrag_text_request()],
        dense_encoder=encoder,
        device="cpu",
    )
    retrieval_method.rank_task(execution_requests[0], top_k=2)

    assert encoder.calls
    assert all(
        call[0].startswith("Q::")
        and all(text.startswith("P::") for text in call[1:])
        for call in encoder.calls
    )
