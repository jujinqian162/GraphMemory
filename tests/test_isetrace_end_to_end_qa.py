from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import pytest
from pydantic import ValidationError

import graph_memory.datasets.isetrace.end_to_end_qa.reporting as reporting
import graph_memory.datasets.isetrace.end_to_end_qa.runner as runner
from graph_memory.datasets.isetrace.end_to_end_qa.contracts import (
    AnswerCorrectness,
    AnswerFaithfulness,
    JUDGE_SCHEMA,
    AnswerArtifact,
    AnswerResponse,
    Condition,
    EvidenceItem,
    JudgeResponse,
    JudgmentArtifact,
    PreparedRecord,
    RankedEvidence,
)
from graph_memory.datasets.isetrace.end_to_end_qa.preparation import (
    select_ranked_evidence,
    sha256_file,
)
from graph_memory.datasets.isetrace.end_to_end_qa.prompts import (
    ANSWER_SYSTEM_PROMPT,
    NO_EVIDENCE_SYSTEM_PROMPT,
    answer_system_prompt,
    validate_judgment,
)
from graph_memory.datasets.isetrace.end_to_end_qa.reporting import build_report
from graph_memory.datasets.isetrace.end_to_end_qa.runner import Paths, report
from graph_memory.infrastructure.io import write_json_atomic
from graph_memory.infrastructure.json_stream import iter_json_array


class _ParallelRunner(Protocol):
    def __call__(
        self,
        records: Sequence[PreparedRecord],
        *,
        current: dict[str, AnswerArtifact],
        workers: int,
        output_path: Path,
        request: Callable[[PreparedRecord], AnswerArtifact],
    ) -> None: ...


def test_limit_counts_tasks_not_condition_records() -> None:
    records = [
        _prepared(f"task-{task_index}", condition)
        for task_index in range(3)
        for condition in Condition
    ]

    limit_tasks = cast(
        Callable[[Sequence[PreparedRecord], int | None], list[PreparedRecord]],
        getattr(runner, "_limit_tasks"),
    )
    limited = limit_tasks(records, 2)

    assert len(limited) == 10
    assert {record.task_id for record in limited} == {"task-0", "task-1"}
    assert {record.condition for record in limited} == set(Condition)


def test_run_parallel_persists_successes_when_one_request_fails(
    tmp_path: Path,
) -> None:
    records = [
        _prepared("task-a", Condition.FLAT_DENSE_FT),
        _prepared("task-b", Condition.FLAT_DENSE_FT),
    ]
    output_path = tmp_path / "answers.jsonl"

    def request(record: PreparedRecord) -> AnswerArtifact:
        if record.task_id == "task-a":
            raise RuntimeError("transient failure")
        return AnswerArtifact(
            record_id=record.record_id,
            input_digest="input",
            answer=AnswerResponse(answer="answer", abstained=False),
            request_digest="request",
            response_id=None,
            usage={},
            cached=False,
        )

    run_parallel = cast(_ParallelRunner, getattr(runner, "_run_parallel"))
    with pytest.raises(RuntimeError, match="transient failure"):
        run_parallel(
            records,
            current={},
            workers=2,
            output_path=output_path,
            request=request,
        )

    persisted = [
        AnswerArtifact.model_validate_json(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [artifact.record_id for artifact in persisted] == ["task-b::flat_dense_ft"]


def test_no_evidence_control_uses_distinct_answer_prompt() -> None:
    no_evidence = PreparedRecord(
        record_id="task::no_evidence",
        task_id="task",
        trajectory_id="trajectory",
        memory_mode="direct_recall",
        condition=Condition.NO_EVIDENCE,
        query="query",
        gold_quotes=("gold",),
        evidence=(),
        evidence_token_count=0,
        full_support=False,
    )
    retrieved = PreparedRecord(
        record_id="task::flat_dense_ft",
        task_id="task",
        trajectory_id="trajectory",
        memory_mode="direct_recall",
        condition=Condition.FLAT_DENSE_FT,
        query="query",
        gold_quotes=("gold",),
        evidence=(),
        evidence_token_count=0,
        full_support=False,
    )

    assert answer_system_prompt(no_evidence) == NO_EVIDENCE_SYSTEM_PROMPT
    assert answer_system_prompt(retrieved) == ANSWER_SYSTEM_PROMPT


def test_iter_json_array_streams_across_small_chunks(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    expected: list[object] = [
        {"id": 1, "text": "brackets in string: ][{}"},
        {"id": 2, "nested": [1, {"value": 'escaped \\" quote'}]},
    ]
    _ = path.write_text(json.dumps(expected), encoding="utf-8")

    assert list(iter_json_array(path, chunk_size=7)) == expected


def test_iter_json_array_rejects_non_array(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    _ = path.write_text('{"id": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="top-level JSON array"):
        _ = list(iter_json_array(path, chunk_size=2))


def test_select_ranked_evidence_stops_at_first_over_budget() -> None:
    candidates = {
        item_id: EvidenceItem(
            item_id=item_id,
            text=item_id,
            token_count=token_count,
        )
        for item_id, token_count in (("a", 1200), ("b", 900), ("c", 100))
    }
    ranked = tuple(
        RankedEvidence(
            node_id=item_id,
            score=1.0,
            token_count=candidates[item_id].token_count,
        )
        for item_id in ("a", "b", "c")
    )

    selected, used_tokens = select_ranked_evidence(ranked, candidates)

    assert [item.item_id for item in selected] == ["a"]
    assert used_tokens == 1200


def test_judge_validation_ignores_non_metric_fields() -> None:
    output = validate_judgment(
        {
            "correctness": "not_answered",
            "faithfulness": "not_applicable",
            "abstained": "The answer explicitly abstained.",
        }
    )

    assert output == {
        "correctness": "not_answered",
        "faithfulness": "not_applicable",
    }
    assert set(cast(dict[str, object], JUDGE_SCHEMA["properties"])) == {
        "correctness",
        "faithfulness",
    }


def test_answer_output_enforces_abstention_consistency() -> None:
    with pytest.raises(ValidationError, match="INSUFFICIENT_EVIDENCE"):
        _ = AnswerResponse(answer="unsupported answer", abstained=True)


def test_report_pairs_tasks_by_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reporting, "BOOTSTRAP_SAMPLES", 25)
    prepared: list[PreparedRecord] = []
    judgments: list[JudgmentArtifact] = []
    correctness = {
        Condition.FLAT_DENSE_FT: (
            AnswerCorrectness.INCORRECT,
            AnswerCorrectness.PARTIAL,
        ),
        Condition.PU_DENSE_FT: (
            AnswerCorrectness.PARTIAL,
            AnswerCorrectness.PARTIAL,
        ),
        Condition.RESIDUAL_RGCN: (
            AnswerCorrectness.CORRECT,
            AnswerCorrectness.PARTIAL,
        ),
        Condition.GOLD_ORACLE: (
            AnswerCorrectness.CORRECT,
            AnswerCorrectness.CORRECT,
        ),
        Condition.NO_EVIDENCE: (
            AnswerCorrectness.NOT_ANSWERED,
            AnswerCorrectness.NOT_ANSWERED,
        ),
    }
    for task_index in range(2):
        task_id = f"task-{task_index}"
        for condition, condition_correctness in correctness.items():
            record_id = f"{task_id}::{condition.value}"
            outcome = condition_correctness[task_index]
            abstained = outcome is AnswerCorrectness.NOT_ANSWERED
            prepared.append(
                PreparedRecord(
                    record_id=record_id,
                    task_id=task_id,
                    trajectory_id="trajectory-a",
                    memory_mode="linked_recall",
                    condition=condition,
                    query="query",
                    gold_quotes=("gold",),
                    evidence=(),
                    evidence_token_count=0,
                    full_support=condition is not Condition.NO_EVIDENCE,
                )
            )
            judgments.append(
                JudgmentArtifact(
                    record_id=record_id,
                    input_digest="input",
                    judgment=JudgeResponse(
                        correctness=outcome,
                        faithfulness=(
                            AnswerFaithfulness.NOT_APPLICABLE
                            if abstained
                            else AnswerFaithfulness.FAITHFUL
                        ),
                    ),
                    request_digest="request",
                    response_id=None,
                    usage={},
                    cached=False,
                )
            )

    report = build_report(prepared, judgments)

    assert report["task_count"] == 2
    assert report["record_count"] == 10
    by_condition = _mapping(report["by_condition"])
    flat = _mapping(by_condition[Condition.FLAT_DENSE_FT.value])
    assert flat["strict_accuracy"] == 0.0
    primary = _mapping(report["primary_linked_and_multi_fact"])
    assert primary["task_count"] == 2
    comparisons = _mapping(report["paired_comparisons"])
    comparison = _mapping(comparisons["residual_rgcn_minus_flat_dense_ft"])
    assert comparison["correctness_transitions"] == {
        "improved": 1,
        "unchanged": 1,
    }
    assert comparison["correctness_transition_matrix"] == {
        "incorrect_to_correct": 1,
        "partial_to_partial": 1,
    }
    support_transitions = _mapping(comparison["full_support_transitions"])
    assert _mapping(support_transitions["1_to_1"])["task_count"] == 2
    bootstrap = _mapping(comparison["cluster_bootstrap_delta"])
    strict = _mapping(bootstrap["strict_accuracy"])
    assert strict["mean_delta"] == 0.5
    assert strict["paired_query_count"] == 2
    assert strict["paired_cluster_count"] == 1


def test_report_rejects_judgments_for_stale_prepared_input(tmp_path: Path) -> None:
    paths = Paths(tmp_path)
    paths.output.mkdir(parents=True)
    prepared = PreparedRecord(
        record_id="task::no_evidence",
        task_id="task",
        trajectory_id="trajectory",
        memory_mode="direct_recall",
        condition=Condition.NO_EVIDENCE,
        query="query",
        gold_quotes=("gold",),
        evidence=(),
        evidence_token_count=0,
        full_support=False,
    )
    prepared_path = paths.output / "prepared.jsonl"
    _ = prepared_path.write_text(
        prepared.model_dump_json() + "\n",
        encoding="utf-8",
    )
    write_json_atomic(
        paths.output / "manifest.json",
        {
            "record_count": 1,
            "prepared_sha256": sha256_file(prepared_path),
            "judgments": {
                "prepared_sha256": "stale",
                "judgments_sha256": "stale",
            },
        },
    )

    with pytest.raises(ValueError, match="current prepared inputs"):
        report(paths)


def _prepared(task_id: str, condition: Condition) -> PreparedRecord:
    return PreparedRecord(
        record_id=f"{task_id}::{condition.value}",
        task_id=task_id,
        trajectory_id=f"trajectory-{task_id}",
        memory_mode="direct_recall",
        condition=condition,
        query="query",
        gold_quotes=("gold",),
        evidence=(),
        evidence_token_count=0,
        full_support=False,
    )


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    mapping = cast(Mapping[object, object], value)
    output: dict[str, object] = {}
    for key_value, item_value in mapping.items():
        key = str(key_value)
        output[key] = item_value
    return output
