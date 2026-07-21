from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from graph_memory.experiment.config import (
    Bm25MethodConfig,
    DenseEncoderConfig,
    DenseMethodConfig,
    GraphRAGMethodConfig,
)
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    DenseEncoderSettings,
    FlatRetrievalBuildPayload,
    GraphRAGRetrievalSettings,
)
from graph_memory.retrieval.contracts import GraphRAGTrace
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.methods.graphrag import (
    GraphRAGConfig,
    GraphRAGMethod,
    build_graphrag_request,
)
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest
from graph_memory.stages.retrieve import run_retrieve_stage


class KeywordEncoder:
    vocabulary = ("eiffel", "paris", "seine", "everest")

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        _ = batch_size, normalize_embeddings, show_progress_bar
        rows = []
        for text in texts:
            lowered = text.casefold()
            vector = np.asarray(
                [float(lowered.count(token)) for token in self.vocabulary],
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(vector))
            rows.append(vector / norm if norm else vector)
        return np.asarray(rows, dtype=np.float32)


def _task_inputs() -> list[dict[str, object]]:
    return [
        {
            "task_id": "hotpot_ex1",
            "question": "Which river runs through the city with the Eiffel Tower?",
            "candidate_sentences": [
                {
                    "sentence_id": "m0",
                    "text": "The Eiffel Tower is in Paris.",
                    "title": "Eiffel Tower",
                    "sentence_index": 0,
                    "position": 0,
                },
                {
                    "sentence_id": "m1",
                    "text": "The Seine runs through Paris.",
                    "title": "Paris",
                    "sentence_index": 0,
                    "position": 1,
                },
                {
                    "sentence_id": "m2",
                    "text": "Mount Everest is tall.",
                    "title": "Mount Everest",
                    "sentence_index": 0,
                    "position": 2,
                },
            ],
        }
    ]


def _encoder_config() -> DenseEncoderConfig:
    return DenseEncoderConfig(
        model_name="keyword-encoder",
        query_prefix="",
        passage_prefix="",
        batch_size=8,
    )


def test_bm25_stage_runs_without_evidence_graph_artifact() -> None:
    result = run_retrieve_stage(
        Bm25MethodConfig(
            method="bm25",
        ),
        dataset="hotpotqa",
        top_k=2,
        task_inputs=_task_inputs(),
        evidence_graphs=None,
        model=None,
        encoder_source=None,
        device="cpu",
    )

    assert result.provenance.method.value == "bm25"
    assert len(result.predictions[0]["ranked_nodes"]) == 3
    assert len(result.predictions[0]["retrieved_subgraph"]["nodes"]) == 2


def test_dense_stage_runs_without_evidence_graph_artifact() -> None:
    result = run_retrieve_stage(
        DenseMethodConfig(
            method="dense",
            encoder=_encoder_config(),
        ),
        dataset="hotpotqa",
        top_k=2,
        task_inputs=_task_inputs(),
        evidence_graphs=None,
        model=None,
        encoder_source=None,
        device="cpu",
        dense_encoder=KeywordEncoder(),
    )

    assert result.provenance.method.value == "dense"
    assert len(result.predictions[0]["ranked_nodes"]) == 3
    assert len(result.predictions[0]["retrieved_subgraph"]["nodes"]) == 2


def test_graphrag_stage_builds_method_owned_entity_graph() -> None:
    result = run_retrieve_stage(
        GraphRAGMethodConfig(
            method="graphrag",
            encoder=_encoder_config(),
            seed_top_s=2,
            max_entity_document_frequency_ratio=0.75,
            sentence_resolver="frozen_dense",
            min_sentence_score_margin=0.02,
            min_bridge_confidence=0.2,
            max_partners_per_anchor=1,
            preserve_dense_top_n=2,
        ),
        dataset="hotpotqa",
        top_k=2,
        task_inputs=_task_inputs(),
        evidence_graphs=None,
        model=None,
        encoder_source=None,
        device="cpu",
        dense_encoder=KeywordEncoder(),
    )

    assert result.provenance.method.value == "graphrag"
    assert len(result.predictions[0]["ranked_nodes"]) == 3
    assert len(result.predictions[0]["retrieved_subgraph"]["nodes"]) == 2


def test_graphrag_builds_typed_local_bridge_evidence() -> None:
    candidates = (
        TextCandidate("m0", "Eiffel Tower in Paris", {"title": "Eiffel Tower"}),
        TextCandidate("m1", "Seine runs through Paris", {"title": "Paris"}),
        TextCandidate("m2", "Everest mountain", {"title": "Everest"}),
    )
    config = GraphRAGConfig(
        seed_top_s=2,
        max_entity_document_frequency_ratio=0.75,
        min_bridge_confidence=0.0,
        preserve_dense_top_n=1,
    )
    method = GraphRAGMethod(
        dense_ranker=DenseTaskRetriever(
            encoder=KeywordEncoder(), query_prefix="", passage_prefix=""
        ),
        config=config,
    )

    result = method.rank_task(
        build_graphrag_request(
            TextRankingRequest("task", "Eiffel river city", candidates), config
        ),
        top_k=2,
    )

    assert len(result.ranked_nodes) == len(candidates)
    assert isinstance(result.trace.native_trace, GraphRAGTrace)
    assert result.trace.native_trace.mentions
    assert result.trace.native_trace.title_groups
    assert result.ranked_nodes[-1].node_id == "m2"


def test_graphrag_builder_rejects_flat_payload() -> None:
    settings = GraphRAGRetrievalSettings(
        top_k=2,
        encoder=DenseEncoderSettings("keyword-encoder", "", "", 8),
    )

    with pytest.raises(TypeError, match="GraphRAGBuildPayload"):
        Registry.retrieval.build(
            settings,
            FlatRetrievalBuildPayload(text_requests=[]),
        )
