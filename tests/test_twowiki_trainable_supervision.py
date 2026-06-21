from __future__ import annotations

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.datasets.twowiki.projectors import TwoWikiToTextRankingRequest
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.training_pairs.requests import TrainPairBuildTask


def test_twowiki_train_pairs_use_gold_evidence_items_not_dependency_edges() -> None:
    text_request = TwoWikiToTextRankingRequest().project(
        {
            "task_id": "2wiki_abc123",
            "question": "Which evidence supports the answer?",
            "question_type": "compositional",
            "candidate_sentences": [
                {
                    "sentence_id": "m0",
                    "title": "Alpha",
                    "sentence_index": 0,
                    "position": 0,
                    "text": "Alpha introduces the chain.",
                },
                {
                    "sentence_id": "m1",
                    "title": "Beta",
                    "sentence_index": 0,
                    "position": 1,
                    "text": "Beta is one supporting sentence.",
                },
                {
                    "sentence_id": "m2",
                    "title": "Gamma",
                    "sentence_index": 0,
                    "position": 2,
                    "text": "Gamma is only present in the dependency label.",
                },
                {
                    "sentence_id": "m3",
                    "title": "Delta",
                    "sentence_index": 0,
                    "position": 3,
                    "text": "Delta is another supporting sentence.",
                },
            ],
            "metadata": {"dataset": "2wiki", "raw_id": "abc123"},
        }
    )
    label = EvidenceLabel(
        task_id="2wiki_abc123",
        gold_answer="Beta and Delta",
        gold_evidence_item_ids=("m1", "m3"),
        gold_dependency_edges=(("m0", "m2"),),
    )
    graph: MemoryGraph = {
        "task_id": "2wiki_abc123",
        "nodes": [
            {"id": "q", "node_type": "question", "text": text_request.query_text},
            *[
                {
                    "id": candidate.item_id,
                    "node_type": "graph_item",
                    "node_kind": "document_sentence",
                    "text": candidate.text,
                }
                for candidate in text_request.candidates
            ],
        ],
        "edges": [],
    }
    result = build_train_pairs(
        [TrainPairBuildTask(text_request=text_request, label=label, graph=graph)],
        NegativeSamplingConfig(
            random_seed=13,
            easy_random_per_positive=0,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=0,
            hard_graph_neighbor_per_positive=0,
            hard_pool_size=10,
        ),
    )

    positives = {pair["node_id"] for pair in result.pairs if pair["label"] == 1}
    assert positives == {"m1", "m3"}
    assert {"m0", "m2"}.isdisjoint(positives)
