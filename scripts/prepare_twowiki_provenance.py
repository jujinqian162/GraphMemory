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
from graph_memory.datasets.twowiki_provenance import (
    TwoWikiProvenanceLabelRecord,
    TwoWikiProvenanceRankingRecord,
    TwoWikiProvenanceRawRecord,
    parse_twowiki_provenance_record,
)
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import RawPrepareStageConfig
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.io import read_json, write_json
from graph_memory.validation import (
    validate_twowiki_provenance_label_records,
    validate_twowiki_provenance_ranking_records,
)

LOGGER = logging.getLogger("prepare_twowiki_provenance")


@dataclass(frozen=True)
class PreparedTwoWikiProvenanceRecords:
    task_inputs: list[TwoWikiProvenanceRankingRecord]
    task_labels: list[TwoWikiProvenanceLabelRecord]
    combined: list[TwoWikiProvenanceRawRecord]
    counts: dict[str, object]


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        RawPrepareStageConfig,
        description="Prepare a generated 2Wiki provenance experiment split.",
        script=Path(__file__),
    )
    config = execution.config
    if config.dataset != "twowiki_provenance":
        raise ValueError(
            "prepare_twowiki_provenance.py requires "
            f"dataset=twowiki_provenance, got {config.dataset}"
        )
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )
    started = time.perf_counter()
    with stage_lifecycle(execution.invocation) as observations:
        prepared = prepare_from_raw(config)
        write_json(config.outputs.input, prepared.task_inputs)
        write_json(config.outputs.labels, prepared.task_labels)
        write_json(config.outputs.combined, prepared.combined)
        counts = cast(
            JsonObject,
            {
                **prepared.counts,
                "task_inputs": len(prepared.task_inputs),
                "task_labels": len(prepared.task_labels),
                "path_supported_tasks": len(prepared.task_labels),
            },
        )
        for key, value in counts.items():
            observations.count(key, value)
        observations.timing("total_seconds", time.perf_counter() - started)
    return 0


def prepare_from_raw(
    config: RawPrepareStageConfig,
) -> PreparedTwoWikiProvenanceRecords:
    raw = read_json(config.source)
    if not isinstance(raw, list):
        raise ValueError("2Wiki provenance raw input must be a JSON list.")
    valid: list[TwoWikiProvenanceRawRecord] = []
    invalid: Counter[str] = Counter()
    for index, value in enumerate(raw):
        try:
            record = parse_twowiki_provenance_record(value, record_index=index)
            ranking = record["ranking"]
            label = record["label"]
            validate_twowiki_provenance_ranking_records([ranking])
            validate_twowiki_provenance_label_records(
                [label], {ranking["task_id"]: ranking}
            )
        except ValueError as error:
            if config.strict_invalid_examples:
                raise
            invalid[str(error)] += 1
            continue
        valid.append(record)
    selected = sample_split(
        valid, count=config.count, seed=config.seed, offset=config.offset
    )
    rankings = [record["ranking"] for record in selected]
    labels = [record["label"] for record in selected]
    validate_twowiki_provenance_ranking_records(rankings)
    validate_twowiki_provenance_label_records(
        labels, {record["task_id"]: record for record in rankings}
    )
    return PreparedTwoWikiProvenanceRecords(
        task_inputs=rankings,
        task_labels=labels,
        combined=selected,
        counts={
            "raw_examples": len(raw),
            "valid_examples": len(valid),
            "invalid_examples_dropped": len(raw) - len(valid),
            "invalid_example_reasons": dict(invalid),
            "selected_examples": len(selected),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
