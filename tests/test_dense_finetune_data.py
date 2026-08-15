from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from graph_memory.datasets.selection import text_ranking_requests_for_dataset
from graph_memory.datasets.hotpotqa.records import (
    HotpotQALabelRecord,
)
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.training_pairs.contracts import TrainPairRecord
from graph_memory.embeddings import DenseEncodingService, DenseTaskEncodingRequest
from graph_memory.models.dense_finetune.data import (
    DenseFinetuneDataSettings,
    build_dense_finetune_examples,
)
from graph_memory.models.dense_finetune.training import (
    _TaskLocalDenseFinetuneEvaluator,
)
from graph_memory.models.dense_finetune.contracts import (
    DenseFinetuneTaskLocalEvaluatorPayload,
)
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest


def _task(
    task_id: str, *, query: str, nodes: Mapping[str, tuple[str, str]]
) -> dict[str, Any]:
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
    return HotpotQALabelRecord(
        task_id=task_id,
        gold_answer="answer",
        gold_evidence_sentence_ids=tuple(gold_nodes),
        gold_dependency_edges=(),
    )


def _request(task: dict[str, Any]) -> TextRankingRequest:
    return TextRankingRequest(
        task_id=task["task_id"],
        query_text=task["question"],
        candidates=tuple(
            TextCandidate(
                item_id=candidate["sentence_id"],
                text=f"{candidate['title']}. {candidate['text']}",
                metadata={"title": candidate["title"]},
            )
            for candidate in task["candidate_sentences"]
        ),
    )


def _evidence_label(label: HotpotQALabelRecord) -> EvidenceLabel:
    return EvidenceLabel(
        task_id=label.task_id,
        gold_answer=label.gold_answer,
        gold_evidence_item_ids=label.gold_evidence_sentence_ids,
        gold_dependency_edges=label.gold_dependency_edges,
    )


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int, bool]] = []

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        text_list = list(texts)
        self.calls.append((text_list, batch_size, normalize_embeddings))
        return np.asarray(
            [[float(index), float(len(text))] for index, text in enumerate(text_list)],
            dtype=float,
        )

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

    text_request = text_ranking_requests_for_dataset("hotpotqa", [task])[0]
    service.encode_task(
        DenseTaskEncodingRequest(
            ranking_request=text_request, node_ids=("q", "m0", "m1")
        )
    )
    result = build_dense_finetune_examples(
        ranking_requests=[_request(task)],
        train_pairs=[
            TrainPairRecord(
                task_id="t1", node_id="m0", label=1, sample_type="positive"
            ),
            TrainPairRecord(
                task_id="t1", node_id="m1", label=0, sample_type="hard_dense"
            ),
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
        TrainPairRecord(
            task_id="t1", node_id="easy", label=0, sample_type="easy_random"
        ),
        TrainPairRecord(
            task_id="t1", node_id="graph", label=0, sample_type="hard_graph_neighbor"
        ),
        TrainPairRecord(task_id="t1", node_id="bm25", label=0, sample_type="hard_bm25"),
        TrainPairRecord(
            task_id="t1", node_id="dense1", label=0, sample_type="hard_dense"
        ),
        TrainPairRecord(
            task_id="t1", node_id="dense2", label=0, sample_type="hard_dense"
        ),
    ]

    result = build_dense_finetune_examples(
        ranking_requests=[_request(task)],
        train_pairs=pairs,
        settings=DenseFinetuneDataSettings(hard_negatives_per_positive=3),
    )

    assert [example.negative_node_id for example in result.examples] == [
        "dense1",
        "dense2",
        "bm25",
    ]
    assert [example.negative_sample_type for example in result.examples] == [
        "hard_dense",
        "hard_dense",
        "hard_bm25",
    ]
    assert [row["negative"] for row in result.rows] == [
        "passage: S. dense one",
        "passage: S. dense two",
        "passage: S. bm25",
    ]


def test_task_local_dense_evaluator_ranks_only_each_tasks_candidates() -> None:
    requests = (
        TextRankingRequest(
            task_id="first",
            query_text="first query",
            candidates=(
                TextCandidate(item_id="shared", text="first positive", metadata={}),
                TextCandidate(item_id="n", text="first negative", metadata={}),
            ),
        ),
        TextRankingRequest(
            task_id="second",
            query_text="second query",
            candidates=(
                TextCandidate(item_id="shared", text="second negative", metadata={}),
                TextCandidate(item_id="t", text="second positive", metadata={}),
            ),
        ),
    )
    labels = (
        EvidenceLabel(
            task_id="first",
            gold_answer="",
            gold_evidence_item_ids=("shared",),
            gold_dependency_edges=(),
        ),
        EvidenceLabel(
            task_id="second",
            gold_answer="",
            gold_evidence_item_ids=("t",),
            gold_dependency_edges=(),
        ),
    )

    class KeywordModel:
        def encode(self, texts, **kwargs):
            del kwargs
            vectors = []
            for text in texts:
                vectors.append(
                    [
                        float("first" in text),
                        float("second" in text and "negative" not in text),
                    ]
                )
            return np.asarray(vectors, dtype=float)

    evaluator = _TaskLocalDenseFinetuneEvaluator(
        payload=DenseFinetuneTaskLocalEvaluatorPayload(
            requests=requests,
            labels=labels,
        ),
        query_prefix="",
        passage_prefix="",
        batch_size=8,
        selected_metric="dev_recall_at_5",
    )

    score = evaluator(KeywordModel())

    assert score == 1.0
    assert evaluator.metric_values["dev_recall_at_5"] == 1.0
