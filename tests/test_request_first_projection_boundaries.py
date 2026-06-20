from pathlib import Path

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.hotpotqa.records import HotpotQALabelRecord, HotpotQARankingRecord
from graph_memory.datasets.hotpotqa.projectors import (
    HotpotQAToEvidenceEvaluationRequest,
    HotpotQAToGraphBuildRequest,
    HotpotQAToTextRankingRequest,
)


def _old_name(prefix: str, suffix: str) -> str:
    return prefix + suffix


FORBIDDEN_NAMES = (
    _old_name("Memory", "TaskInput"),
    _old_name("Memory", "TaskLabels"),
    _old_name("Memory", "Item"),
    _old_name("Combined", "MemoryTask"),
)


def test_production_and_test_code_do_not_reference_old_memory_task_contracts() -> None:
    offenders: list[str] = []
    for root in (Path("graph_memory"), Path("scripts"), Path("tests")):
        for path in root.rglob("*.py"):
            if path == Path(__file__):
                continue
            source = path.read_text(encoding="utf-8")
            for name in FORBIDDEN_NAMES:
                if name in source:
                    offenders.append(f"{path.as_posix()}:{name}")
    assert offenders == []


def _hotpotqa_record() -> HotpotQARankingRecord:
    return {
        "task_id": "hotpot_1",
        "question": "Where was Ada born?",
        "candidate_sentences": [
            {
                "sentence_id": "m0",
                "title": "Ada Lovelace",
                "sentence_index": 0,
                "position": 0,
                "text": "Ada Lovelace was born in London.",
            }
        ],
    }


def _hotpotqa_label() -> HotpotQALabelRecord:
    return {
        "task_id": "hotpot_1",
        "gold_answer": "London",
        "gold_evidence_sentence_ids": ["m0"],
        "gold_dependency_edges": [],
    }


def test_hotpotqa_text_projection_outputs_retriever_request_only() -> None:
    request = HotpotQAToTextRankingRequest().project(_hotpotqa_record())
    assert request.task_id == "hotpot_1"
    assert request.query_text == "Where was Ada born?"
    assert request.candidates[0].item_id == "m0"
    assert request.candidates[0].text == "Ada Lovelace. Ada Lovelace was born in London."
    assert request.candidates[0].metadata == {
        "title": "Ada Lovelace",
        "source_ref": "Ada Lovelace",
        "sequence_index": 0,
        "position": 0,
    }
    assert not hasattr(request.candidates[0], "sentence_index")


def test_hotpotqa_graph_projection_outputs_graph_builder_request() -> None:
    request = HotpotQAToGraphBuildRequest().project(_hotpotqa_record())
    assert request.task_id == "hotpot_1"
    assert request.nodes[0].node_id == "m0"
    assert request.nodes[0].source_ref == "Ada Lovelace"
    assert request.nodes[0].group_key == "document:Ada Lovelace"
    assert request.nodes[0].sequence_index == 0


def test_hotpotqa_evaluation_projection_outputs_evaluator_request() -> None:
    ranked_result: RankedResult = {
        "task_id": "hotpot_1",
        "method": "bm25",
        "ranked_nodes": [{"node_id": "m0", "score": 1.0}],
        "retrieved_subgraph": {"nodes": ["m0"], "edges": []},
        "latency_ms": 1.0,
        "input_tokens": 8,
    }
    graph: MemoryGraph = {
        "task_id": "hotpot_1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": "Where?"},
            {
                "id": "m0",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": "Ada Lovelace was born in London.",
                "source_ref": "Ada Lovelace",
                "group_key": "document:Ada Lovelace",
                "sequence_index": 0,
                "metadata": {"title": "Ada Lovelace", "position": 0},
            },
        ],
        "edges": [],
    }
    request = HotpotQAToEvidenceEvaluationRequest().project(
        predictions=[ranked_result], labels=[_hotpotqa_label()], graphs=[graph]
    )
    assert request.labels[0].task_id == "hotpot_1"
    assert request.labels[0].gold_evidence_item_ids == ("m0",)




