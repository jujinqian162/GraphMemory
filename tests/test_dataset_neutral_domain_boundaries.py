from __future__ import annotations

import inspect
from pathlib import Path

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.evaluation.failure_cases import build_failure_cases
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.service import evaluate_results
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.builder import GraphBuilder
from graph_memory.graphs.requests import GraphBuildNode, GraphBuildRequest


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "graph_memory"
REUSABLE_PRODUCTION_ROOTS = (
    PACKAGE_ROOT / "graphs",
    PACKAGE_ROOT / "retrieval" / "execution",
    PACKAGE_ROOT / "retrieval" / "methods",
    PACKAGE_ROOT / "retrieval" / "tuning",
    PACKAGE_ROOT / "training_pairs",
    PACKAGE_ROOT / "models",
    PACKAGE_ROOT / "evaluation",
    PACKAGE_ROOT / "validation",
)
FORBIDDEN_DATASET_IMPORTS = (
    "graph_memory.datasets.hotpotqa.records",
    "graph_memory.datasets.hotpotqa.projectors",
)
FORBIDDEN_DOMAIN_FIELD_TOKENS = (
    "candidate_sentences",
    "gold_evidence_sentence_ids",
    "sentence_id",
    "sentence_index",
    "document_sentence",
)
FIELD_TOKEN_ALLOWLIST = {
    Path("validation/tasks.py"),
}


def _production_files() -> list[Path]:
    files: list[Path] = []
    for root in REUSABLE_PRODUCTION_ROOTS:
        files.extend(path for path in root.rglob("*.py") if path.name != "__pycache__")
    return sorted(set(files))


def test_reusable_domain_packages_do_not_import_hotpotqa_records_or_projectors() -> None:
    offenders: list[str] = []
    for path in _production_files():
        source = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_DATASET_IMPORTS:
            if forbidden in source:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT.parent)} imports {forbidden}")
    assert offenders == []


def test_reusable_domain_packages_do_not_use_hotpotqa_required_field_names() -> None:
    offenders: list[str] = []
    for path in _production_files():
        relative = path.relative_to(PACKAGE_ROOT)
        if relative in FIELD_TOKEN_ALLOWLIST:
            continue
        source = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_DOMAIN_FIELD_TOKENS:
            if forbidden in source:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT.parent)} contains {forbidden}")
    assert offenders == []


def test_graph_build_request_builds_dataset_neutral_graph_items() -> None:
    request = GraphBuildRequest(
        task_id="email-thread-7",
        query_text="What did Mei ask Kai to revise?",
        nodes=(
            GraphBuildNode(
                node_id="turn-1",
                text="Mei asked Kai to revise the methods section.",
                node_kind="dialogue_turn",
                source_ref="thread-7",
                group_key="conversation:thread-7",
                sequence_index=3,
                metadata={"speaker": "Mei", "channel": "email"},
            ),
            GraphBuildNode(
                node_id="turn-2",
                text="Kai said the revision would be ready tomorrow.",
                node_kind="dialogue_turn",
                source_ref="thread-7",
                group_key="conversation:thread-7",
                sequence_index=4,
                metadata={"speaker": "Kai", "channel": "email"},
            ),
        ),
        input_visible_edges=(),
    )

    graph = GraphBuilder(GraphBuildConfig()).build(request)

    graph_items = [node for node in graph["nodes"] if node["id"] != "q"]
    assert graph_items == [
        {
            "id": "turn-1",
            "node_type": "graph_item",
            "node_kind": "dialogue_turn",
            "text": "Mei asked Kai to revise the methods section.",
            "source_ref": "thread-7",
            "group_key": "conversation:thread-7",
            "sequence_index": 3,
            "metadata": {"speaker": "Mei", "channel": "email"},
        },
        {
            "id": "turn-2",
            "node_type": "graph_item",
            "node_kind": "dialogue_turn",
            "text": "Kai said the revision would be ready tomorrow.",
            "source_ref": "thread-7",
            "group_key": "conversation:thread-7",
            "sequence_index": 4,
            "metadata": {"speaker": "Kai", "channel": "email"},
        },
    ]
    for node in graph_items:
        assert "source" not in node
        assert "sentence_id" not in node
        assert "position" not in node


def test_rgcn_inference_does_not_reverse_project_graph_requests_to_hotpotqa_records() -> None:
    import graph_memory.models.graph_retriever.inference as inference_module

    source = inspect.getsource(inference_module)

    assert "HotpotQARankingRecord" not in source
    assert "HotpotQACandidateSentence" not in source
    assert "_ranking_record_from_graph_request" not in source


def test_evaluation_and_failure_cases_consume_request_level_labels() -> None:
    graph: MemoryGraph = {
        "task_id": "task-1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": "Which turn contains the commitment?"},
            {"id": "turn-1", "node_type": "graph_item", "node_kind": "dialogue_turn", "text": "Maybe later."},
            {"id": "turn-2", "node_type": "graph_item", "node_kind": "dialogue_turn", "text": "I will send it tomorrow."},
        ],
        "edges": [],
    }
    prediction: RankedResult = {
        "task_id": "task-1",
        "method": "bm25",
        "ranked_nodes": [{"node_id": "turn-1", "score": 2.0}, {"node_id": "turn-2", "score": 1.0}],
        "retrieved_subgraph": {"nodes": ["turn-1"], "edges": []},
        "latency_ms": 0.0,
        "input_tokens": 7,
    }
    request = EvidenceEvaluationRequest(
        predictions=[prediction],
        labels=[
            EvidenceLabel(
                task_id="task-1",
                gold_answer="tomorrow",
                gold_evidence_item_ids=("turn-2",),
                gold_dependency_edges=(),
            )
        ],
        graphs=[graph],
    )

    rows = evaluate_results(request)
    failure_cases = build_failure_cases(request, top_k=1, limit=1)

    assert rows[0]["Recall@10"] == 1.0
    assert failure_cases == [
        {
            "debug_type": "failure_case",
            "task_id": "task-1",
            "method": "bm25",
            "failure_type": "missing_full_support_at_1",
            "gold_evidence_item_ids": ["turn-2"],
            "retrieved_top_k": ["turn-1"],
            "missing_gold_nodes": ["turn-2"],
            "connected_gold_in_top_k": False,
        }
    ]
