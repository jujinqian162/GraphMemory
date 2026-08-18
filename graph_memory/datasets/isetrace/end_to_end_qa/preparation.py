from __future__ import annotations

import gzip
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import TypeAdapter

from graph_memory.datasets.isetrace.end_to_end_qa.contracts import (
    BenchmarkTask,
    CandidateRecord,
    Condition,
    EvidenceItem,
    HumanMetadataRecord,
    LabelRecord,
    PreparedRecord,
    RankedEvidence,
    RankingRecord,
    ReviewPacket,
)
from graph_memory.evaluation.span_metrics import span_metrics_under_token_budget
from graph_memory.infrastructure.json_stream import iter_json_array
from graph_memory.retrieval.results import RankedNodeRecord
from graph_memory.trajectories import SourceSpan

EVIDENCE_TOKEN_BUDGET = 2048
EXPECTED_TASK_COUNT = 469
CONDITIONS = tuple(Condition)
RETRIEVAL_CONDITIONS = (
    Condition.FLAT_DENSE_FT,
    Condition.PU_DENSE_FT,
    Condition.RESIDUAL_RGCN,
)
_STRING_LIST = TypeAdapter(list[str])
_RecordT = TypeVar("_RecordT", HumanMetadataRecord, ReviewPacket)


class Tokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> Mapping[str, object]: ...


def load_frozen_inputs(
    *,
    task_ids_path: Path,
    metadata_path: Path,
    review_packet_path: Path,
) -> tuple[
    list[str],
    dict[str, HumanMetadataRecord],
    dict[str, ReviewPacket],
]:
    task_ids = _STRING_LIST.validate_json(task_ids_path.read_bytes())
    if len(task_ids) != EXPECTED_TASK_COUNT or len(set(task_ids)) != len(task_ids):
        raise ValueError(
            f"expected {EXPECTED_TASK_COUNT} unique human quality-pass task IDs"
        )
    selected = set(task_ids)
    metadata = {
        record.query_id: record
        for record in _iter_jsonl(metadata_path, HumanMetadataRecord)
        if record.query_id in selected
    }
    packets = {
        record.task_id: record
        for record in _iter_jsonl(review_packet_path, ReviewPacket)
        if record.task_id in selected
    }
    if set(metadata) != selected:
        raise ValueError("human-corrected metadata does not cover all selected tasks")
    if set(packets) != selected:
        raise ValueError("review packet does not cover all selected tasks")
    return task_ids, metadata, packets


def load_gold_spans(
    path: Path,
    *,
    task_ids: set[str],
) -> dict[str, tuple[SourceSpan, ...]]:
    labels: dict[str, tuple[SourceSpan, ...]] = {}
    for value in iter_json_array(path):
        record = LabelRecord.model_validate(value)
        if record.task_id not in task_ids:
            continue
        if record.task_id in labels:
            raise ValueError(f"duplicate labels for task_id={record.task_id}")
        labels[record.task_id] = record.gold_evidence_spans
    if set(labels) != task_ids:
        missing = task_ids - set(labels)
        raise ValueError(f"label artifact is missing {len(missing)} selected tasks")
    return labels


def load_rankings(
    path: Path,
    *,
    task_ids: set[str],
    expected_method: str,
) -> dict[str, tuple[RankedEvidence, ...]]:
    rankings: dict[str, tuple[RankedEvidence, ...]] = {}
    with gzip.open(path, "rb") as stream:
        for line in stream:
            record = RankingRecord.model_validate_json(line)
            if record.method != expected_method:
                message = (
                    f"ranking artifact method={record.method!r} does not match "
                    f"expected_method={expected_method!r}"
                )
                raise ValueError(message)
            if record.task_id not in task_ids:
                continue
            if record.task_id in rankings:
                raise ValueError(f"duplicate ranking for task_id={record.task_id}")
            rankings[record.task_id] = record.ranked_nodes
    if set(rankings) != task_ids:
        missing = task_ids - set(rankings)
        raise ValueError(f"ranking artifact is missing {len(missing)} selected tasks")
    return rankings


def iter_selected_tasks(
    path: Path,
    *,
    task_ids: set[str],
) -> Iterator[BenchmarkTask]:
    found: set[str] = set()
    for value in iter_json_array(path):
        record = BenchmarkTask.model_validate(value)
        if record.task_id not in task_ids:
            continue
        if record.task_id in found:
            raise ValueError(f"duplicate task in prepared artifact: {record.task_id}")
        found.add(record.task_id)
        yield record
    if found != task_ids:
        missing = task_ids - found
        raise ValueError(
            f"prepared task artifact is missing {len(missing)} selected tasks"
        )


def select_ranked_evidence(
    ranked: Sequence[RankedEvidence],
    candidates: Mapping[str, EvidenceItem],
    *,
    token_budget: int = EVIDENCE_TOKEN_BUDGET,
) -> tuple[tuple[EvidenceItem, ...], int]:
    selected: list[EvidenceItem] = []
    used_tokens = 0
    for node in ranked:
        if used_tokens + node.token_count > token_budget:
            break
        candidate = candidates.get(node.node_id)
        if candidate is None:
            raise ValueError(f"ranked node={node.node_id} is absent from candidates")
        if candidate.token_count != node.token_count:
            raise ValueError(f"token count disagrees for node={node.node_id}")
        selected.append(candidate)
        used_tokens += node.token_count
    return tuple(selected), used_tokens


def prepare_task_conditions(
    task: BenchmarkTask,
    *,
    metadata: HumanMetadataRecord,
    review_packet: ReviewPacket,
    gold_spans: tuple[SourceSpan, ...],
    rankings: Mapping[Condition, Sequence[RankedEvidence]],
    tokenizer: Tokenizer,
) -> tuple[PreparedRecord, ...]:
    if task.graph_id != metadata.trajectory_id:
        raise ValueError(f"trajectory mismatch for task_id={task.task_id}")
    if review_packet.query != task.query_text:
        raise ValueError(f"query mismatch for task_id={task.task_id}")
    gold_quotes = review_packet.gold_quotes
    if not gold_quotes:
        raise ValueError(f"task_id={task.task_id} has no frozen gold quotes")
    gold_evidence = tuple(
        EvidenceItem(
            item_id=f"gold-oracle:{index}",
            text=quote,
            token_count=_token_count(tokenizer, quote),
        )
        for index, quote in enumerate(gold_quotes, start=1)
    )
    gold_tokens = sum(item.token_count for item in gold_evidence)
    if gold_tokens > EVIDENCE_TOKEN_BUDGET:
        raise ValueError(
            f"gold oracle for task_id={task.task_id} uses {gold_tokens} evidence tokens"
        )

    candidate_sets = {
        Condition.FLAT_DENSE_FT: _candidate_map(task.flat_candidates),
        Condition.PU_DENSE_FT: _candidate_map(task.provenance_candidates),
        Condition.RESIDUAL_RGCN: _candidate_map(task.provenance_candidates),
    }
    outputs: list[PreparedRecord] = []
    for condition in RETRIEVAL_CONDITIONS:
        evidence, used_tokens = select_ranked_evidence(
            rankings[condition], candidate_sets[condition]
        )
        outputs.append(
            PreparedRecord(
                record_id=f"{task.task_id}::{condition.value}",
                task_id=task.task_id,
                trajectory_id=metadata.trajectory_id,
                memory_mode=metadata.memory_mode,
                condition=condition,
                query=task.query_text,
                gold_quotes=gold_quotes,
                evidence=evidence,
                evidence_token_count=used_tokens,
                full_support=_full_support(rankings[condition], gold_spans),
            )
        )
    outputs.extend(
        (
            PreparedRecord(
                record_id=f"{task.task_id}::{Condition.GOLD_ORACLE.value}",
                task_id=task.task_id,
                trajectory_id=metadata.trajectory_id,
                memory_mode=metadata.memory_mode,
                condition=Condition.GOLD_ORACLE,
                query=task.query_text,
                gold_quotes=gold_quotes,
                evidence=gold_evidence,
                evidence_token_count=gold_tokens,
                full_support=True,
            ),
            PreparedRecord(
                record_id=f"{task.task_id}::{Condition.NO_EVIDENCE.value}",
                task_id=task.task_id,
                trajectory_id=metadata.trajectory_id,
                memory_mode=metadata.memory_mode,
                condition=Condition.NO_EVIDENCE,
                query=task.query_text,
                gold_quotes=gold_quotes,
                evidence=(),
                evidence_token_count=0,
                full_support=False,
            ),
        )
    )
    return tuple(outputs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_map(
    candidates: Sequence[CandidateRecord],
) -> dict[str, EvidenceItem]:
    output: dict[str, EvidenceItem] = {}
    for candidate in candidates:
        item = candidate.evidence_item()
        if item.item_id in output:
            raise ValueError(f"duplicate candidate={item.item_id}")
        output[item.item_id] = item
    return output


def _full_support(
    ranked: Sequence[RankedEvidence],
    gold_spans: tuple[SourceSpan, ...],
) -> bool:
    if not gold_spans:
        raise ValueError("gold source spans must be non-empty")
    nodes = tuple(
        RankedNodeRecord(
            node_id=node.node_id,
            score=node.score,
            token_count=node.token_count,
            source_spans=node.source_spans,
        )
        for node in ranked
    )
    return bool(
        span_metrics_under_token_budget(
            nodes, gold_spans, EVIDENCE_TOKEN_BUDGET
        ).full_support
    )


def _token_count(tokenizer: Tokenizer, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        truncation=False,
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, Sequence):
        raise ValueError("tokenizer did not return input_ids")
    return len(input_ids)


def _iter_jsonl(
    path: Path,
    record_type: type[_RecordT],
) -> Iterator[_RecordT]:
    with path.open("rb") as stream:
        for line in stream:
            yield record_type.model_validate_json(line)


__all__ = [
    "CONDITIONS",
    "EVIDENCE_TOKEN_BUDGET",
    "EXPECTED_TASK_COUNT",
    "RETRIEVAL_CONDITIONS",
    "iter_selected_tasks",
    "load_frozen_inputs",
    "load_gold_spans",
    "load_rankings",
    "prepare_task_conditions",
    "select_ranked_evidence",
    "sha256_file",
]
