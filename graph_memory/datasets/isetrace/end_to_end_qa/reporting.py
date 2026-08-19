from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from graph_memory.analysis.paired_bootstrap import paired_cluster_delta_bootstrap_ci
from graph_memory.datasets.isetrace.end_to_end_qa.contracts import (
    AnswerCorrectness,
    AnswerFaithfulness,
    Condition,
    JudgmentArtifact,
    PreparedRecord,
)
from graph_memory.datasets.isetrace.end_to_end_qa.preparation import (
    CONDITIONS,
    RETRIEVAL_CONDITIONS,
)

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 13


class ReportRow(TypedDict):
    record_id: str
    task_id: str
    trajectory_id: str
    memory_mode: str
    condition: Condition
    full_support: bool
    correctness: AnswerCorrectness
    faithfulness: AnswerFaithfulness
    abstained: bool


def build_report(
    prepared_records: Sequence[PreparedRecord],
    judgments: Sequence[JudgmentArtifact],
) -> dict[str, object]:
    prepared = _index_prepared(prepared_records)
    judged = _index_judgments(judgments)
    if set(prepared) != set(judged):
        raise ValueError("report requires one judgment for every prepared record")
    rows = [_report_row(prepared[key], judged[key]) for key in sorted(prepared)]
    by_condition = {
        condition.value: _aggregate(
            [row for row in rows if row["condition"] is condition]
        )
        for condition in CONDITIONS
    }
    by_memory_mode = {
        condition.value: {
            memory_mode: _aggregate(group)
            for memory_mode, group in _groups(
                row for row in rows if row["condition"] is condition
            ).items()
        }
        for condition in CONDITIONS
    }
    primary_rows = [
        row
        for row in rows
        if row["memory_mode"] in {"linked_recall", "multi_fact_recall"}
    ]
    primary_analysis = {
        "task_count": len({row["task_id"] for row in primary_rows}),
        "by_condition": {
            condition.value: _aggregate(
                [row for row in primary_rows if row["condition"] is condition]
            )
            for condition in CONDITIONS
        },
        "paired_comparisons": _comparisons(primary_rows),
    }
    full_support = {
        condition.value: {
            "full_support": _aggregate(
                [
                    row
                    for row in rows
                    if row["condition"] is condition and row["full_support"]
                ]
            ),
            "not_full_support": _aggregate(
                [
                    row
                    for row in rows
                    if row["condition"] is condition and not row["full_support"]
                ]
            ),
        }
        for condition in RETRIEVAL_CONDITIONS
    }
    comparisons = _comparisons(rows)
    return {
        "schema_version": 1,
        "experiment": "ISETrace E10 End-to-End QA",
        "bootstrap": {
            "unit": "trajectory_id",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "record_count": len(rows),
        "task_count": len({row["task_id"] for row in rows}),
        "by_condition": by_condition,
        "by_human_corrected_memory_mode": by_memory_mode,
        "primary_linked_and_multi_fact": primary_analysis,
        "retrieval_full_support_strata": full_support,
        "paired_comparisons": comparisons,
    }


def write_summary_csv(path: Path, report: Mapping[str, object]) -> None:
    by_condition_value = report.get("by_condition")
    if not isinstance(by_condition_value, Mapping):
        raise ValueError("report has no by_condition section")
    by_condition_mapping = cast(Mapping[object, object], by_condition_value)
    by_condition: dict[str, object] = {}
    for key_value, item in by_condition_mapping.items():
        key = str(key_value)
        by_condition[key] = item
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "count",
        "strict_accuracy",
        "partial_or_better_accuracy",
        "abstention_rate",
        "fully_faithful_rate",
        "unsupported_or_mixed_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for condition in CONDITIONS:
            metrics = by_condition.get(condition.value)
            if not isinstance(metrics, Mapping):
                raise ValueError(f"report has no metrics for {condition.value}")
            writer.writerow({"condition": condition.value, **metrics})


def _report_row(
    prepared: PreparedRecord,
    judgment: JudgmentArtifact,
) -> ReportRow:
    return {
        "record_id": prepared.record_id,
        "task_id": prepared.task_id,
        "trajectory_id": prepared.trajectory_id,
        "memory_mode": prepared.memory_mode,
        "condition": prepared.condition,
        "full_support": prepared.full_support,
        "correctness": judgment.judgment.correctness,
        "faithfulness": judgment.judgment.faithfulness,
        "abstained": (judgment.judgment.correctness is AnswerCorrectness.NOT_ANSWERED),
    }


def _aggregate(rows: Sequence[ReportRow]) -> dict[str, object]:
    count = len(rows)
    correctness = Counter(row["correctness"].value for row in rows)
    faithfulness = Counter(row["faithfulness"].value for row in rows)
    if not count:
        return {
            "count": 0,
            "strict_accuracy": None,
            "partial_or_better_accuracy": None,
            "abstention_rate": None,
            "fully_faithful_rate": None,
            "unsupported_or_mixed_rate": None,
            "correctness_counts": {},
            "faithfulness_counts": {},
        }
    return {
        "count": count,
        "strict_accuracy": correctness[AnswerCorrectness.CORRECT.value] / count,
        "partial_or_better_accuracy": (
            correctness[AnswerCorrectness.CORRECT.value]
            + correctness[AnswerCorrectness.PARTIAL.value]
        )
        / count,
        "abstention_rate": sum(row["abstained"] for row in rows) / count,
        "fully_faithful_rate": faithfulness[AnswerFaithfulness.FAITHFUL.value] / count,
        "unsupported_or_mixed_rate": (
            faithfulness[AnswerFaithfulness.MIXED.value]
            + faithfulness[AnswerFaithfulness.UNSUPPORTED.value]
        )
        / count,
        "correctness_counts": dict(sorted(correctness.items())),
        "faithfulness_counts": dict(sorted(faithfulness.items())),
    }


def _comparisons(rows: Sequence[ReportRow]) -> dict[str, object]:
    return {
        f"{right.value}_minus_{left.value}": _comparison(rows, left=left, right=right)
        for left, right in (
            (Condition.FLAT_DENSE_FT, Condition.PU_DENSE_FT),
            (Condition.FLAT_DENSE_FT, Condition.RESIDUAL_RGCN),
            (Condition.PU_DENSE_FT, Condition.RESIDUAL_RGCN),
        )
    }


def _comparison(
    rows: Sequence[ReportRow],
    *,
    left: Condition,
    right: Condition,
) -> dict[str, object]:
    by_condition = {
        condition: {
            row["task_id"]: row for row in rows if row["condition"] is condition
        }
        for condition in (left, right)
    }
    if set(by_condition[left]) != set(by_condition[right]):
        raise ValueError(f"paired comparison {left} vs {right} has mismatched tasks")
    deltas: dict[str, dict[str, list[float]]] = {
        "strict_accuracy": defaultdict(list),
        "partial_or_better_accuracy": defaultdict(list),
        "fully_faithful_rate": defaultdict(list),
    }
    transitions: Counter[str] = Counter()
    correctness_matrix: Counter[str] = Counter()
    full_support_groups: dict[str, list[tuple[ReportRow, ReportRow]]] = defaultdict(
        list
    )
    for task_id in sorted(by_condition[left]):
        left_row = by_condition[left][task_id]
        right_row = by_condition[right][task_id]
        cluster = left_row["trajectory_id"]
        if cluster != right_row["trajectory_id"]:
            raise ValueError(f"trajectory mismatch for paired task={task_id}")
        left_score = _correctness_score(left_row["correctness"])
        right_score = _correctness_score(right_row["correctness"])
        transition = (
            "improved"
            if right_score > left_score
            else "worsened"
            if right_score < left_score
            else "unchanged"
        )
        transitions[transition] += 1
        correctness_matrix[
            f"{left_row['correctness'].value}_to_{right_row['correctness'].value}"
        ] += 1
        support_transition = (
            f"{int(left_row['full_support'])}_to_{int(right_row['full_support'])}"
        )
        full_support_groups[support_transition].append((left_row, right_row))
        deltas["strict_accuracy"][cluster].append(
            float(right_row["correctness"] is AnswerCorrectness.CORRECT)
            - float(left_row["correctness"] is AnswerCorrectness.CORRECT)
        )
        deltas["partial_or_better_accuracy"][cluster].append(
            float(right_score >= 1) - float(left_score >= 1)
        )
        deltas["fully_faithful_rate"][cluster].append(
            float(right_row["faithfulness"] is AnswerFaithfulness.FAITHFUL)
            - float(left_row["faithfulness"] is AnswerFaithfulness.FAITHFUL)
        )
    return {
        "left": left.value,
        "right": right.value,
        "correctness_transitions": dict(sorted(transitions.items())),
        "correctness_transition_matrix": dict(sorted(correctness_matrix.items())),
        "full_support_transitions": {
            transition: _support_transition_summary(pairs)
            for transition, pairs in sorted(full_support_groups.items())
        },
        "cluster_bootstrap_delta": {
            metric: paired_cluster_delta_bootstrap_ci(
                values,
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            )
            for metric, values in deltas.items()
        },
    }


def _support_transition_summary(
    pairs: Sequence[tuple[ReportRow, ReportRow]],
) -> dict[str, object]:
    deltas_by_cluster: dict[str, list[float]] = defaultdict(list)
    for left, right in pairs:
        if left["trajectory_id"] != right["trajectory_id"]:
            raise ValueError("trajectory mismatch inside support transition")
        deltas_by_cluster[left["trajectory_id"]].append(
            float(right["correctness"] is AnswerCorrectness.CORRECT)
            - float(left["correctness"] is AnswerCorrectness.CORRECT)
        )
    return {
        "task_count": len(pairs),
        "left_strict_accuracy": sum(
            left["correctness"] is AnswerCorrectness.CORRECT for left, _ in pairs
        )
        / len(pairs),
        "right_strict_accuracy": sum(
            right["correctness"] is AnswerCorrectness.CORRECT for _, right in pairs
        )
        / len(pairs),
        "strict_accuracy_delta": paired_cluster_delta_bootstrap_ci(
            deltas_by_cluster,
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
    }


def _correctness_score(value: AnswerCorrectness) -> int:
    return {
        AnswerCorrectness.NOT_ANSWERED: 0,
        AnswerCorrectness.INCORRECT: 0,
        AnswerCorrectness.PARTIAL: 1,
        AnswerCorrectness.CORRECT: 2,
    }[value]


def _groups(rows: Iterable[ReportRow]) -> dict[str, list[ReportRow]]:
    groups: dict[str, list[ReportRow]] = defaultdict(list)
    for row in rows:
        groups[row["memory_mode"]].append(row)
    return dict(sorted(groups.items()))


def _index_prepared(
    records: Sequence[PreparedRecord],
) -> dict[str, PreparedRecord]:
    output: dict[str, PreparedRecord] = {}
    for record in records:
        if record.record_id in output:
            raise ValueError(f"duplicate prepared record_id={record.record_id}")
        output[record.record_id] = record
    return output


def _index_judgments(
    records: Sequence[JudgmentArtifact],
) -> dict[str, JudgmentArtifact]:
    output: dict[str, JudgmentArtifact] = {}
    for record in records:
        if record.record_id in output:
            raise ValueError(f"duplicate judgment record_id={record.record_id}")
        output[record.record_id] = record
    return output


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "build_report",
    "write_summary_csv",
]
