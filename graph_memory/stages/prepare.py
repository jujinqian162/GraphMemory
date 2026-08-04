from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel, JsonValue

from graph_memory.datasets.hotpotqa import (
    HotpotQAPreparedSplit,
    combined_hotpotqa_records,
    convert_hotpotqa_example,
    convert_hotpotqa_examples,
    parse_hotpotqa_example,
    parse_hotpotqa_examples,
)
from graph_memory.datasets.isetrace import (
    combined_isetrace_records,
    prepare_isetrace_benchmark,
)
from graph_memory.datasets.musique import (
    MuSiQuePreparedSplit,
    combined_musique_records,
    convert_musique_example,
    convert_musique_examples,
    parse_musique_example,
    parse_musique_examples,
)
from graph_memory.datasets.splits import sample_split
from graph_memory.datasets.twowiki import (
    TwoWikiPreparedSplit,
    combined_twowiki_records,
    convert_twowiki_example,
    convert_twowiki_examples,
    parse_twowiki_example,
    parse_twowiki_examples,
)
from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    DatasetArtifactRef,
    DirectorySourceRef,
    FileSourceRef,
    ProcessedAssetStore,
)
from graph_memory.experiment.config import (
    DatasetName,
    ISETraceChunkingConfig,
    ISETraceTrajectorySplitCounts,
    SplitName,
)
from graph_memory.io import read_json, write_json
from graph_memory.stages.results import PreparedSplitResult
from graph_memory.text.chunking import TokenChunkingConfig


@dataclass(frozen=True)
class PreparedSplitData:
    task_inputs: list[object]
    task_labels: list[object]
    combined: list[object]
    counts: dict[str, JsonValue]
    provenance_graphs: list[object] | None = None
    query_metadata: list[object] | None = None
    template_supervision: list[object] | None = None


def prepare_split(
    dataset: DatasetName,
    source: Path,
    *,
    count: int | None,
    seed: int,
    offset: int,
    strict_invalid_examples: bool,
    split: SplitName | None = None,
    trajectory_source: Path | None = None,
    source_revision: str | None = None,
    trajectory_splits: ISETraceTrajectorySplitCounts | None = None,
    chunking: ISETraceChunkingConfig | None = None,
) -> PreparedSplitData:
    if dataset == "hotpotqa":
        return _prepare_hotpotqa(
            source,
            count=count,
            seed=seed,
            offset=offset,
            strict=strict_invalid_examples,
        )
    if dataset == "twowiki":
        return _prepare_twowiki(
            source,
            count=count,
            seed=seed,
            offset=offset,
            strict=strict_invalid_examples,
        )
    if dataset == "musique":
        return _prepare_musique(
            source,
            count=count,
            seed=seed,
            offset=offset,
            strict=strict_invalid_examples,
        )
    if dataset == "isetrace":
        if (
            split is None
            or trajectory_source is None
            or source_revision is None
            or trajectory_splits is None
            or chunking is None
        ):
            raise ValueError(
                "isetrace preparation requires split, trajectory_source, "
                "source_revision, trajectory_splits, and chunking"
            )
        return _prepare_isetrace(
            source,
            split=split,
            trajectory_source=trajectory_source,
            source_revision=source_revision,
            trajectory_splits=trajectory_splits,
            count=count,
            seed=seed,
            offset=offset,
            strict=strict_invalid_examples,
            chunking=chunking,
        )
    raise ValueError(f"unsupported dataset={dataset!r}")


def materialize_prepared_split(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    split: SplitName,
    source: FileSourceRef,
    trajectory_source: FileSourceRef | DirectorySourceRef | None = None,
    count: int | None,
    seed: int,
    offset: int,
    strict_invalid_examples: bool,
    source_revision: str | None = None,
    trajectory_splits: ISETraceTrajectorySplitCounts | None = None,
    chunking: ISETraceChunkingConfig | None = None,
    implementation_version: str,
) -> PreparedSplitResult:
    prepared = prepare_split(
        dataset,
        Path(source.uri),
        count=count,
        seed=seed,
        offset=offset,
        strict_invalid_examples=strict_invalid_examples,
        split=split,
        trajectory_source=(
            None if trajectory_source is None else Path(trajectory_source.uri)
        ),
        source_revision=source_revision,
        trajectory_splits=trajectory_splits,
        chunking=chunking,
    )
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.DATASET,
        namespace=dataset,
        task_identity=f"prepare-{dataset}-{split}",
        origin={
            "stage": "prepare",
            "dataset": dataset,
            "split": split,
            "source_digest": source.digest,
            "trajectory_source_digest": (
                None if trajectory_source is None else trajectory_source.digest
            ),
            "source_revision": source_revision,
            "trajectory_splits": (
                None
                if trajectory_splits is None
                else trajectory_splits.model_dump(mode="json")
            ),
            "chunking": (
                None if chunking is None else chunking.model_dump(mode="json")
            ),
            "implementation_version": implementation_version,
        },
    ) as publisher:
        write_json(
            publisher.workspace / "tasks.json",
            [_json_record(record) for record in prepared.task_inputs],
        )
        write_json(
            publisher.workspace / "labels.json",
            [_json_record(record) for record in prepared.task_labels],
        )
        write_json(
            publisher.workspace / "combined.json",
            [_json_record(record) for record in prepared.combined],
        )
        write_json(publisher.workspace / "counts.json", prepared.counts)
        payloads = {
            "tasks": "tasks.json",
            "labels": "labels.json",
            "combined": "combined.json",
            "counts": "counts.json",
        }
        if prepared.provenance_graphs is not None:
            write_json(
                publisher.workspace / "provenance_graphs.json",
                [_json_record(graph) for graph in prepared.provenance_graphs],
            )
            payloads["provenance_graphs"] = "provenance_graphs.json"
        if prepared.query_metadata is not None:
            write_json(
                publisher.workspace / "query_metadata.json",
                [_json_record(item) for item in prepared.query_metadata],
            )
            payloads["query_metadata"] = "query_metadata.json"
        if prepared.template_supervision is not None:
            write_json(
                publisher.workspace / "template_supervision.json",
                [_json_record(item) for item in prepared.template_supervision],
            )
            payloads["template_supervision"] = "template_supervision.json"
        artifact = publisher.publish(
            payloads,
            shape={
                "tasks": len(prepared.task_inputs),
                "labels": len(prepared.task_labels),
            },
            metadata={
                "split": split,
                "chunking": (
                    None if chunking is None else chunking.model_dump(mode="json")
                ),
            },
        )
    assert isinstance(artifact, DatasetArtifactRef)
    return PreparedSplitResult(
        split=split,
        artifact=artifact,
        counts=prepared.counts,
    )


def _prepare_hotpotqa(
    source: Path, *, count: int | None, seed: int, offset: int, strict: bool
) -> PreparedSplitData:
    raw = read_json(source)
    if not isinstance(raw, list):
        raise ValueError("HotpotQA raw input must be a JSON list.")
    valid, invalid = _valid_records(
        raw,
        strict=strict,
        dataset="HotpotQA",
        validate=lambda value, index: _validate_hotpotqa_raw(value, index),
    )
    selected = sample_split(valid, count=count, seed=seed, offset=offset)
    parsed = parse_hotpotqa_examples(selected)
    conversion = convert_hotpotqa_examples(parsed)
    split = HotpotQAPreparedSplit(
        rankings=tuple(conversion.ranking_records),
        labels=tuple(conversion.label_records),
    )
    tasks = list(split.rankings)
    labels = list(split.labels)
    return _prepared(
        raw=raw,
        valid=valid,
        invalid=invalid,
        selected=selected,
        parsed_count=len(parsed),
        tasks=cast(list[object], tasks),
        labels=cast(list[object], labels),
        combined=cast(list[object], combined_hotpotqa_records(tasks, labels)),
    )


def _validate_hotpotqa_raw(value: object, index: int) -> None:
    # Filter-only: full ranking/label contracts run once after batch convert.
    convert_hotpotqa_example(parse_hotpotqa_example(value, record_index=index))


def _prepare_twowiki(
    source: Path, *, count: int | None, seed: int, offset: int, strict: bool
) -> PreparedSplitData:
    raw = read_json(source)
    if not isinstance(raw, list):
        raise ValueError("2Wiki raw input must be a JSON list.")
    valid, invalid = _valid_records(
        raw,
        strict=strict,
        dataset="2Wiki",
        validate=lambda value, index: _validate_twowiki_raw(value, index),
    )
    selected = sample_split(valid, count=count, seed=seed, offset=offset)
    parsed = parse_twowiki_examples(selected)
    conversion = convert_twowiki_examples(parsed)
    split = TwoWikiPreparedSplit(
        rankings=tuple(conversion.ranking_records),
        labels=tuple(conversion.label_records),
    )
    tasks = list(split.rankings)
    labels = list(split.labels)
    return _prepared(
        raw=raw,
        valid=valid,
        invalid=invalid,
        selected=selected,
        parsed_count=len(parsed),
        tasks=cast(list[object], tasks),
        labels=cast(list[object], labels),
        combined=cast(list[object], combined_twowiki_records(tasks, labels)),
    )


def _validate_twowiki_raw(value: object, index: int) -> None:
    # Filter-only: full ranking/label contracts run once after batch convert.
    convert_twowiki_example(parse_twowiki_example(value, record_index=index))


def _prepare_musique(
    source: Path, *, count: int | None, seed: int, offset: int, strict: bool
) -> PreparedSplitData:
    raw = _read_jsonl(source)
    valid, invalid = _valid_records(
        raw,
        strict=strict,
        dataset="MuSiQue",
        validate=lambda value, index: _validate_musique_raw(value, index),
    )
    selected = sample_split(valid, count=count, seed=seed, offset=offset)
    parsed = parse_musique_examples(selected)
    conversion = convert_musique_examples(parsed)
    split = MuSiQuePreparedSplit(
        rankings=tuple(conversion.ranking_records),
        labels=tuple(conversion.label_records),
    )
    tasks = list(split.rankings)
    labels = list(split.labels)
    return _prepared(
        raw=raw,
        valid=valid,
        invalid=invalid,
        selected=selected,
        parsed_count=len(parsed),
        tasks=cast(list[object], tasks),
        labels=cast(list[object], labels),
        combined=cast(list[object], combined_musique_records(tasks, labels)),
    )


def _validate_musique_raw(value: object, index: int) -> None:
    # Filter-only: full ranking/label contracts run once after batch convert.
    convert_musique_example(parse_musique_example(value, record_index=index))


def _prepare_isetrace(
    source: Path,
    *,
    split: SplitName,
    trajectory_source: Path,
    source_revision: str,
    trajectory_splits: ISETraceTrajectorySplitCounts,
    count: int | None,
    seed: int,
    offset: int,
    strict: bool,
    chunking: ISETraceChunkingConfig,
) -> PreparedSplitData:
    benchmark, summary = prepare_isetrace_benchmark(
        source,
        trajectory_source,
        source_revision=source_revision,
        count=count,
        seed=seed,
        offset=offset,
        strict=strict,
        split=split,
        trajectory_splits=trajectory_splits.model_dump(),
        chunking=TokenChunkingConfig(
            tokenizer_name=chunking.tokenizer_name,
            max_tokens=chunking.max_tokens,
            overlap_tokens=chunking.overlap_tokens,
            reserved_tokens=chunking.reserved_tokens,
        ),
    )
    rankings = list(benchmark.rankings)
    labels = list(benchmark.labels)
    combined = combined_isetrace_records(benchmark.rankings, benchmark.labels)
    counts = cast(dict[str, JsonValue], summary.to_dict())
    counts["parsed_examples"] = len(rankings)
    counts["task_inputs"] = len(rankings)
    counts["task_labels"] = len(labels)
    return PreparedSplitData(
        task_inputs=cast(list[object], rankings),
        task_labels=cast(list[object], labels),
        combined=cast(list[object], combined),
        counts=counts,
        provenance_graphs=cast(list[object], list(benchmark.provenance_graphs)),
        query_metadata=cast(list[object], list(benchmark.query_metadata)),
        template_supervision=cast(
            list[object], list(benchmark.template_supervision)
        ),
    )


def _valid_records(
    raw: Sequence[object],
    *,
    strict: bool,
    dataset: str,
    validate: Callable[[object, int], None],
) -> tuple[list[object], Counter[str]]:
    valid: list[object] = []
    invalid: Counter[str] = Counter()
    for index, value in enumerate(raw):
        try:
            validate(value, index)
        except ValueError as error:
            if strict:
                raise ValueError(
                    f"Invalid {dataset} raw example index={index}: {error}"
                ) from error
            invalid[str(error)] += 1
            continue
        valid.append(value)
    return valid, invalid


def _prepared(
    *,
    raw: Sequence[object],
    valid: Sequence[object],
    invalid: Counter[str],
    selected: Sequence[object],
    parsed_count: int,
    tasks: list[object],
    labels: list[object],
    combined: list[object],
) -> PreparedSplitData:
    counts: dict[str, JsonValue] = {
        "raw_examples": len(raw),
        "valid_examples": len(valid),
        "invalid_examples_dropped": len(raw) - len(valid),
        "invalid_example_reasons": dict(invalid),
        "selected_examples": len(selected),
        "parsed_examples": parsed_count,
        "task_inputs": len(tasks),
        "task_labels": len(labels),
        "path_supported_tasks": sum(
            1 for label in labels if _has_dependency_edges(label)
        ),
    }
    return PreparedSplitData(tasks, labels, combined, counts)


def _json_record(record: object) -> object:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json", exclude_none=True)
    return record


def _has_dependency_edges(label: object) -> bool:
    if isinstance(label, BaseModel):
        return bool(getattr(label, "gold_dependency_edges", ()))
    if isinstance(label, dict):
        return bool(label.get("gold_dependency_edges"))
    return False


def _read_jsonl(path: Path) -> list[object]:
    records: list[object] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL line={line_number}: {path}") from error
    return records


__all__ = ["PreparedSplitData", "materialize_prepared_split", "prepare_split"]
