from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTextRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord, HotpotQALabelRecord
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.embeddings import DenseEncodingService, DenseTaskEncodingRequest
from graph_memory.models.dense_finetune.data import (
    DenseFinetuneDataSettings,
    build_dense_finetune_examples,
    build_ir_evaluator_payload,
)


def _task(task_id: str, *, query: str, nodes: Mapping[str, tuple[str, str]]) -> HotpotQARankingRecord:
    return {
        "task_id": task_id,
        "question": query,
        "candidate_sentences": [
            {
                "sentence_id": node_id,
                "title": source,
                "text": text,
                "sentence_index": index,
                "position": index,
            }
            for index, (node_id, (source, text)) in enumerate(nodes.items())
        ],
    }


def _labels(task_id: str, gold_nodes: list[str]) -> HotpotQALabelRecord:
    return {
        "task_id": task_id,
        "gold_answer": "answer",
        "gold_evidence_sentence_ids": gold_nodes,
        "gold_dependency_edges": [],
    }


def _request(task: HotpotQARankingRecord):
    return HotpotQAToTextRankingRequest().project(task)


def _evidence_label(label: HotpotQALabelRecord) -> EvidenceLabel:
    return EvidenceLabel(
        task_id=label["task_id"],
        gold_answer=label["gold_answer"],
        gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
        gold_dependency_edges=tuple((edge[0], edge[1]) for edge in label["gold_dependency_edges"]),
    )


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int, bool]] = []

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> object:
        text_list = list(texts)
        self.calls.append((text_list, batch_size, normalize_embeddings))
        return np.asarray([[float(index), float(len(text))] for index, text in enumerate(text_list)], dtype=float)

    def get_sentence_embedding_dimension(self) -> int:
        return 2


def test_dense_finetune_uses_same_text_format_as_dense_encoding_service() -> None:
    task = _task(
        "t1",
        query="Who wrote the book?",
        nodes={
            "m0": ("Book", "The book was written by Ada."),
            "m1": ("Distractor", "A different sentence."),
        },
    )
    encoder = RecordingEncoder()
    service = DenseEncodingService(
        encoder=encoder,
        query_prefix="Q: ",
        passage_prefix="P: ",
        batch_size=8,
    )

    text_request = HotpotQAToTextRankingRequest().project(task)
    service.encode_task(DenseTaskEncodingRequest(ranking_request=text_request, node_ids=("q", "m0", "m1")))
    result = build_dense_finetune_examples(
        ranking_requests=[_request(task)],
        train_pairs=[
            TrainPairRecord(task_id="t1", node_id="m0", label=1, sample_type="positive"),
            TrainPairRecord(task_id="t1", node_id="m1", label=0, sample_type="hard_dense"),
        ],
        settings=DenseFinetuneDataSettings(hard_negatives_per_positive=1),
        query_prefix="Q: ",
        passage_prefix="P: ",
    )

    assert encoder.calls == [
        (
            [
                "Q: Who wrote the book?",
                "P: Book. The book was written by Ada.",
                "P: Distractor. A different sentence.",
            ],
            8,
            True,
        )
    ]
    assert len(result.examples) == 1
    assert result.examples[0].anchor == encoder.calls[0][0][0]
    assert result.examples[0].positive == encoder.calls[0][0][1]
    assert result.examples[0].negative == encoder.calls[0][0][2]
    assert result.rows == (
        {
            "anchor": "Q: Who wrote the book?",
            "positive": "P: Book. The book was written by Ada.",
            "negative": "P: Distractor. A different sentence.",
        },
    )


def test_dense_finetune_builds_positive_only_rows_without_negatives() -> None:
    task = _task("t1", query="query", nodes={"m0": ("S", "positive")})

    result = build_dense_finetune_examples(
        ranking_requests=[_request(task)],
        train_pairs=[TrainPairRecord(task_id="t1", node_id="m0", label=1, sample_type="positive")],
        settings=DenseFinetuneDataSettings(),
    )

    assert result.rows == ({"anchor": "query: query", "positive": "passage: S. positive"},)
    assert result.examples[0].negative is None
    assert result.examples[0].negative_sample_type is None


def test_dense_finetune_selects_hard_negatives_by_priority_and_original_order() -> None:
    task = _task(
        "t1",
        query="query",
        nodes={
            "p": ("S", "positive"),
            "easy": ("S", "easy"),
            "graph": ("S", "graph"),
            "bm25": ("S", "bm25"),
            "dense1": ("S", "dense one"),
            "dense2": ("S", "dense two"),
        },
    )
    pairs: list[TrainPairRecord] = [
        TrainPairRecord(task_id="t1", node_id="p", label=1, sample_type="positive"),
        TrainPairRecord(task_id="t1", node_id="easy", label=0, sample_type="easy_random"),
        TrainPairRecord(task_id="t1", node_id="graph", label=0, sample_type="hard_graph_neighbor"),
        TrainPairRecord(task_id="t1", node_id="bm25", label=0, sample_type="hard_bm25"),
        TrainPairRecord(task_id="t1", node_id="dense1", label=0, sample_type="hard_dense"),
        TrainPairRecord(task_id="t1", node_id="dense2", label=0, sample_type="hard_dense"),
    ]

    result = build_dense_finetune_examples(
        ranking_requests=[_request(task)],
        train_pairs=pairs,
        settings=DenseFinetuneDataSettings(hard_negatives_per_positive=3),
    )

    assert [example.negative_node_id for example in result.examples] == ["dense1", "dense2", "bm25"]
    assert [example.negative_sample_type for example in result.examples] == ["hard_dense", "hard_dense", "hard_bm25"]
    assert [row["negative"] for row in result.rows] == [
        "passage: S. dense one",
        "passage: S. dense two",
        "passage: S. bm25",
    ]


def test_dense_finetune_rejects_unknown_pair_node_id() -> None:
    task = _task("t1", query="query", nodes={"m0": ("S", "positive")})

    with pytest.raises(ValueError, match="task_id=t1.*node_id=missing"):
        build_dense_finetune_examples(
            ranking_requests=[_request(task)],
            train_pairs=[
                TrainPairRecord(task_id="t1", node_id="m0", label=1, sample_type="positive"),
                TrainPairRecord(task_id="t1", node_id="missing", label=0, sample_type="hard_dense"),
            ],
            settings=DenseFinetuneDataSettings(),
        )


def test_ir_evaluator_payload_uses_task_qualified_corpus_ids() -> None:
    tasks = [
        _task("t1", query="first", nodes={"m0": ("Shared", "first positive"), "m1": ("Other", "negative")}),
        _task("t2", query="second", nodes={"m0": ("Shared", "second positive")}),
    ]

    payload = build_ir_evaluator_payload(
        ranking_requests=[_request(task) for task in tasks],
        labels=[_evidence_label(_labels("t1", ["m0"])), _evidence_label(_labels("t2", ["m0"]))],
        query_prefix="Q: ",
        passage_prefix="P: ",
    )

    assert payload.queries == {"t1": "Q: first", "t2": "Q: second"}
    assert payload.corpus == {
        "t1::m0": "P: Shared. first positive",
        "t1::m1": "P: Other. negative",
        "t2::m0": "P: Shared. second positive",
    }
    assert payload.relevant_docs == {"t1": {"t1::m0"}, "t2": {"t2::m0"}}
