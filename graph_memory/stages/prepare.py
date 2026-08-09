from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from pydantic import BaseModel, JsonValue

from graph_memory.datasets.hotpotqa import (
    convert_hotpotqa_example,
    parse_hotpotqa_example,
)
from graph_memory.datasets.isetrace import prepare_isetrace_benchmark
from graph_memory.datasets.musique import (
    convert_musique_example,
    parse_musique_example,
)
from graph_memory.datasets.splits import sample_split
from graph_memory.datasets.twowiki import (
    convert_twowiki_example,
    parse_twowiki_example,
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
from graph_memory.text.chunking import TokenChunkingConfig

def prepare_evidence_split(
    dataset: DatasetName,
    source: Path,
    *,
    count: int | None,
    seed: int,
    offset: int,
    strict_invalid_examples: bool,
) -> tuple[list[object], list[object], dict[str, JsonValue]]:
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
    raise ValueError(f"unsupported evidence dataset={dataset!r}")


def materialize_prepared_split(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    split: SplitName,
    source: FileSourceRef,
    trajectory_source: FileSourceRef | DirectorySourceRef | None = None,
    authoring_metadata_source: FileSourceRef | None = None,
    count: int | None,
    seed: int,
    offset: int,
    strict_invalid_examples: bool,
    source_revision: str | None = None,
    trajectory_splits: ISETraceTrajectorySplitCounts | None = None,
    chunking: ISETraceChunkingConfig | None = None,
    implementation_version: str,
) -> DatasetArtifactRef:
    if dataset == "isetrace":
        if (
            trajectory_source is None
            or authoring_metadata_source is None
            or source_revision is None
            or trajectory_splits is None
            or chunking is None
        ):
            raise ValueError(
                "isetrace preparation requires trajectory_source, "
                "authoring_metadata_source, source_revision, trajectory_splits, "
                "and chunking"
            )
        benchmark, summary = prepare_isetrace_benchmark(
            Path(source.uri),
            Path(trajectory_source.uri),
            source_revision=source_revision,
            count=count,
            seed=seed,
            offset=offset,
            strict=strict_invalid_examples,
            split=split,
            trajectory_splits=trajectory_splits.model_dump(),
            chunking=TokenChunkingConfig(
                tokenizer_name=chunking.tokenizer_name,
                max_tokens=chunking.max_tokens,
                overlap_tokens=chunking.overlap_tokens,
                reserved_tokens=chunking.reserved_tokens,
            ),
            authoring_metadata_source=Path(authoring_metadata_source.uri),
        )
        task_inputs: list[object] = list(benchmark.rankings)
        task_labels: list[object] = list(benchmark.labels)
        counts = cast(dict[str, JsonValue], summary.to_dict())
        counts.update(
            parsed_examples=len(task_inputs),
            task_inputs=len(task_inputs),
            task_labels=len(task_labels),
        )
        provenance_graphs: Sequence[object] | None = benchmark.provenance_graphs
        query_metadata: Sequence[object] | None = benchmark.query_metadata
        template_supervision: Sequence[object] | None = benchmark.template_supervision
    else:
        task_inputs, task_labels, counts = prepare_evidence_split(
            dataset,
            Path(source.uri),
            count=count,
            seed=seed,
            offset=offset,
            strict_invalid_examples=strict_invalid_examples,
        )
        provenance_graphs = None
        query_metadata = None
        template_supervision = None
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
            "authoring_metadata_digest": (
                None
                if authoring_metadata_source is None
                else authoring_metadata_source.digest
            ),
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
            [_json_record(record) for record in task_inputs],
        )
        write_json(
            publisher.workspace / "labels.json",
            [_json_record(record) for record in task_labels],
        )
        write_json(publisher.workspace / "counts.json", counts)
        payloads = {
            "tasks": "tasks.json",
            "labels": "labels.json",
            "counts": "counts.json",
        }
        if provenance_graphs is not None:
            write_json(
                publisher.workspace / "provenance_graphs.json",
                [_json_record(graph) for graph in provenance_graphs],
            )
            payloads["provenance_graphs"] = "provenance_graphs.json"
        if query_metadata is not None:
            write_json(
                publisher.workspace / "query_metadata.json",
                [_json_record(item) for item in query_metadata],
            )
            payloads["query_metadata"] = "query_metadata.json"
        if template_supervision is not None:
            write_json(
                publisher.workspace / "template_supervision.json",
                [_json_record(item) for item in template_supervision],
            )
            payloads["template_supervision"] = "template_supervision.json"
        artifact = publisher.publish(
            payloads,
            shape={
                "tasks": len(task_inputs),
                "labels": len(task_labels),
            },
            metadata={
                "split": split,
                "chunking": (
                    None if chunking is None else chunking.model_dump(mode="json")
                ),
            },
        )
    assert isinstance(artifact, DatasetArtifactRef)
    return artifact


def _prepare_hotpotqa(
    source: Path, *, count: int | None, seed: int, offset: int, strict: bool
) -> tuple[list[object], list[object], dict[str, JsonValue]]:
    raw = read_json(source)
    if not isinstance(raw, list):
        raise ValueError("HotpotQA raw input must be a JSON list.")
    valid, invalid = _converted_records(
        raw,
        strict=strict,
        dataset="HotpotQA",
        convert=lambda value, index: convert_hotpotqa_example(
            parse_hotpotqa_example(value, record_index=index)
        ),
    )
    selected = sample_split(valid, count=count, seed=seed, offset=offset)
    tasks = [ranking for ranking, _label in selected]
    labels = [label for _ranking, label in selected]
    return _prepared(
        raw=raw,
        valid=valid,
        invalid=invalid,
        selected=selected,
        parsed_count=len(selected),
        tasks=cast(list[object], tasks),
        labels=cast(list[object], labels),
    )


def _prepare_twowiki(
    source: Path, *, count: int | None, seed: int, offset: int, strict: bool
) -> tuple[list[object], list[object], dict[str, JsonValue]]:
    raw = read_json(source)
    if not isinstance(raw, list):
        raise ValueError("2Wiki raw input must be a JSON list.")
    valid, invalid = _converted_records(
        raw,
        strict=strict,
        dataset="2Wiki",
        convert=lambda value, index: convert_twowiki_example(
            parse_twowiki_example(value, record_index=index)
        ),
    )
    selected = sample_split(valid, count=count, seed=seed, offset=offset)
    tasks = [ranking for ranking, _label in selected]
    labels = [label for _ranking, label in selected]
    return _prepared(
        raw=raw,
        valid=valid,
        invalid=invalid,
        selected=selected,
        parsed_count=len(selected),
        tasks=cast(list[object], tasks),
        labels=cast(list[object], labels),
    )


def _prepare_musique(
    source: Path, *, count: int | None, seed: int, offset: int, strict: bool
) -> tuple[list[object], list[object], dict[str, JsonValue]]:
    raw = _read_jsonl(source)
    valid, invalid = _converted_records(
        raw,
        strict=strict,
        dataset="MuSiQue",
        convert=lambda value, index: convert_musique_example(
            parse_musique_example(value, record_index=index)
        ),
    )
    selected = sample_split(valid, count=count, seed=seed, offset=offset)
    tasks = [ranking for ranking, _label in selected]
    labels = [label for _ranking, label in selected]
    return _prepared(
        raw=raw,
        valid=valid,
        invalid=invalid,
        selected=selected,
        parsed_count=len(selected),
        tasks=cast(list[object], tasks),
        labels=cast(list[object], labels),
    )


def _converted_records(
    raw: Sequence[object],
    *,
    strict: bool,
    dataset: str,
    convert: Callable[[object, int], tuple[object, object]],
) -> tuple[list[tuple[object, object]], Counter[str]]:
    valid: list[tuple[object, object]] = []
    invalid: Counter[str] = Counter()
    for index, value in enumerate(raw):
        try:
            converted = convert(value, index)
        except ValueError as error:
            if strict:
                raise ValueError(
                    f"Invalid {dataset} raw example index={index}: {error}"
                ) from error
            invalid[str(error)] += 1
            continue
        valid.append(converted)
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
) -> tuple[list[object], list[object], dict[str, JsonValue]]:
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
    return tasks, labels, counts


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


__all__ = ["materialize_prepared_split", "prepare_evidence_split"]
