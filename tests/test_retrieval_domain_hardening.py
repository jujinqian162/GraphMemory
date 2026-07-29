from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    DenseEncoderSettings,
    GraphRAGBuildPayload,
    GraphRAGRetrievalSettings,
)
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
        query_text="What did Ada design?",
        candidates=(
            TextCandidate(
                item_id="c1",
                text="Ada Lovelace designed the Analytical Engine.",
                metadata={"title": "Ada Lovelace", "aliases": "Ada"},
            ),
            TextCandidate(
                item_id="c2",
                text="Ada and the Analytical Engine influenced early computing.",
                metadata={"title": "Analytical Engine"},
            ),
        ),
    )


def test_graphrag_builder_assembles_explicit_alias_aware_graph() -> None:
    first = build_graphrag_request(_graphrag_text_request(), GraphRAGConfig())
    second = build_graphrag_request(_graphrag_text_request(), GraphRAGConfig())

    assert isinstance(first, GraphRAGRequest)
    assert first == second
    ada_title = next(
        mention
        for mention in first.knowledge_graph.mentions
        if mention.normalized_surface == "ada lovelace"
        and mention.mention_type == "TITLE_ENTITY"
    )
    assert any(
        mention.candidate_id == "c2"
        and mention.entity_id == ada_title.entity_id
        and mention.mention_type == "MENTIONS"
        and mention.alias_confidence == pytest.approx(0.9)
        for mention in first.knowledge_graph.mentions
    )
    analytical_group = next(
        group
        for group in first.knowledge_graph.title_groups
        if group.normalized_title_entity == "analytical engine"
    )
    assert analytical_group.candidate_ids == ("c2",)


def test_graphrag_preserves_query_and_passage_prefixes() -> None:
    encoder = RecordingEncoder()
    built = Registry.retrieval.build(
        GraphRAGRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings("recording", "Q::", "P::", 7),
            device="cpu",
        ),
        GraphRAGBuildPayload(
            text_requests=[_graphrag_text_request()], dense_encoder=encoder
        ),
    )
    built.method.rank_task(built.execution_tasks[0].method_request, top_k=2)

    assert encoder.calls
    assert all(
        call[0].startswith("Q::")
        and all(text.startswith("P::") for text in call[1:])
        for call in encoder.calls
    )
