from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.contracts.common import JsonObject
from graph_memory.datasets.splits import sample_split
from graph_memory.datasets.traject_bench import (
    MissingCatalogToolsError,
    PreparedTrajectBenchToolCatalog,
    TrajectBenchExample,
    TrajectBenchLabelRecord,
    TrajectBenchRankingRecord,
    canonicalize_traject_bench_tool_catalog,
    combined_traject_bench_records,
    convert_traject_bench_example,
    convert_traject_bench_examples,
    discover_traject_bench_query_files,
    parse_traject_bench_query,
    parse_traject_bench_tool_catalog,
    partition_for_split,
    prepare_traject_bench_tool_catalog,
    resolve_traject_bench_data_root,
)
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import RawPrepareStageConfig
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.io import read_json, write_json
from graph_memory.validation import (
    validate_traject_bench_label_records,
    validate_traject_bench_ranking_records,
)

LOGGER = logging.getLogger("prepare_traject_bench")


@dataclass(frozen=True)
class LoadedTrajectBenchSource:
    examples: list[TrajectBenchExample]
    catalogs_by_domain: dict[str, PreparedTrajectBenchToolCatalog]
    counts: dict[str, object]


@dataclass(frozen=True)
class ValidTrajectBenchExamples:
    examples: list[TrajectBenchExample]
    invalid_reason_counts: dict[str, int]


@dataclass(frozen=True)
class PreparedTrajectBenchRecords:
    task_inputs: list[TrajectBenchRankingRecord]
    task_labels: list[TrajectBenchLabelRecord]
    counts: dict[str, object]


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        RawPrepareStageConfig,
        description="Prepare a typed TRAJECT-Bench tool-retrieval split.",
        script=Path(__file__),
    )
    config = execution.config
    if config.dataset != "traject_bench":
        raise ValueError(
            "prepare_traject_bench.py requires dataset=traject_bench, "
            f"got {config.dataset}"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    start_time = time.perf_counter()
    with stage_lifecycle(execution.invocation) as observations:
        prepared = prepare_from_raw(config)
        inputs_by_task_id = {
            task_input["task_id"]: task_input for task_input in prepared.task_inputs
        }
        validate_traject_bench_ranking_records(prepared.task_inputs)
        validate_traject_bench_label_records(
            prepared.task_labels,
            inputs_by_task_id,
        )

        write_json(config.outputs.input, prepared.task_inputs)
        write_json(config.outputs.labels, prepared.task_labels)
        write_json(
            config.outputs.combined,
            combined_traject_bench_records(
                prepared.task_inputs,
                prepared.task_labels,
            ),
        )

        counts = cast(
            JsonObject,
            {
                **prepared.counts,
                "task_inputs": len(prepared.task_inputs),
                "task_labels": len(prepared.task_labels),
                "path_supported_tasks": sum(
                    1
                    for label in prepared.task_labels
                    if label["gold_dependency_edges"]
                ),
            },
        )
        for key, value in counts.items():
            observations.count(key, value)
        observations.timing("total_seconds", time.perf_counter() - start_time)
        LOGGER.info("wrote inputs: %s", config.outputs.input)
        LOGGER.info("wrote labels: %s", config.outputs.labels)
        LOGGER.info("wrote combined inspection artifact: %s", config.outputs.combined)
    LOGGER.info("wrote run summary: %s", execution.invocation.summary_path)
    return 0


def prepare_from_raw(config: RawPrepareStageConfig) -> PreparedTrajectBenchRecords:
    loaded = load_traject_bench_source(config.source, split=config.split)
    valid = select_valid_examples(
        loaded.examples,
        catalogs_by_domain=loaded.catalogs_by_domain,
        strict=config.strict_invalid_examples,
    )
    selected = sample_split(
        valid.examples,
        count=config.count,
        seed=config.seed,
        offset=config.offset,
    )
    conversion = convert_traject_bench_examples(
        selected,
        loaded.catalogs_by_domain,
    )
    invalid_examples_dropped = len(loaded.examples) - len(valid.examples)
    LOGGER.info(
        "TRAJECT-Bench split=%s raw=%s valid=%s selected=%s invalid=%s",
        config.split,
        len(loaded.examples),
        len(valid.examples),
        len(selected),
        invalid_examples_dropped,
    )
    return PreparedTrajectBenchRecords(
        task_inputs=conversion.ranking_records,
        task_labels=conversion.label_records,
        counts={
            **loaded.counts,
            "valid_examples": len(valid.examples),
            "invalid_examples_dropped": invalid_examples_dropped,
            "invalid_example_reasons": valid.invalid_reason_counts,
            "selected_examples": len(selected),
        },
    )


def load_traject_bench_source(
    source: Path,
    *,
    split: str,
) -> LoadedTrajectBenchSource:
    root = resolve_traject_bench_data_root(source)
    query_files = discover_traject_bench_query_files(root, split)
    partition = partition_for_split(split)
    domains = sorted({path.parent.name for path in query_files}, key=_text_sort_key)
    catalogs_by_domain: dict[str, PreparedTrajectBenchToolCatalog] = {}
    raw_catalog_entries = 0
    canonical_catalog_entries = 0
    duplicate_catalog_entries = 0
    unresolved_catalog_connections = 0
    for domain_name in domains:
        catalog_path = root / "tools" / f"{domain_name}_tool.json"
        raw_catalog = read_json(catalog_path)
        definitions = parse_traject_bench_tool_catalog(
            raw_catalog,
            domain_name=domain_name,
        )
        canonical = canonicalize_traject_bench_tool_catalog(
            definitions,
            domain_name=domain_name,
        )
        prepared = prepare_traject_bench_tool_catalog(canonical)
        catalogs_by_domain[domain_name] = prepared
        raw_catalog_entries += len(definitions)
        canonical_catalog_entries += len(prepared.candidates)
        duplicate_catalog_entries += prepared.duplicate_entry_count
        unresolved_catalog_connections += prepared.unresolved_connection_count

    examples: list[TrajectBenchExample] = []
    for query_file in query_files:
        raw_records = read_json(query_file)
        if not isinstance(raw_records, list):
            raise ValueError(
                f"TRAJECT-Bench query file must contain a JSON list: {query_file}"
            )
        source_key = query_file.relative_to(root).as_posix()
        for source_index, raw_record in enumerate(raw_records):
            examples.append(
                parse_traject_bench_query(
                    raw_record,
                    source_key=source_key,
                    source_index=source_index,
                    partition=partition,
                    domain_name=query_file.parent.name,
                )
            )

    return LoadedTrajectBenchSource(
        examples=examples,
        catalogs_by_domain=catalogs_by_domain,
        counts={
            "partition": partition,
            "query_files": len(query_files),
            "domains": len(domains),
            "raw_examples": len(examples),
            "raw_catalog_entries": raw_catalog_entries,
            "canonical_catalog_entries": canonical_catalog_entries,
            "duplicate_catalog_entries": duplicate_catalog_entries,
            "unresolved_catalog_connections": unresolved_catalog_connections,
        },
    )


def select_valid_examples(
    examples: Sequence[TrajectBenchExample],
    *,
    catalogs_by_domain: dict[str, PreparedTrajectBenchToolCatalog],
    strict: bool,
) -> ValidTrajectBenchExamples:
    valid: list[TrajectBenchExample] = []
    invalid_reason_counts: Counter[str] = Counter()
    for example in examples:
        catalog = catalogs_by_domain.get(example.domain_name)
        if catalog is None:
            error = ValueError(
                f"Missing TRAJECT-Bench catalog for domain={example.domain_name!r}."
            )
        else:
            try:
                convert_traject_bench_example(example, catalog)
            except ValueError as conversion_error:
                error = conversion_error
            else:
                valid.append(example)
                continue
        reason = (
            "missing_gold_tool_from_public_catalog"
            if isinstance(error, MissingCatalogToolsError)
            else str(error)
        )
        if strict:
            raise ValueError(
                f"Invalid TRAJECT-Bench raw example "
                f"{example.source_key}:{example.source_index}: {error}"
            ) from error
        invalid_reason_counts[reason] += 1
    return ValidTrajectBenchExamples(
        examples=valid,
        invalid_reason_counts=dict(invalid_reason_counts),
    )


def _text_sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


if __name__ == "__main__":
    raise SystemExit(main())
