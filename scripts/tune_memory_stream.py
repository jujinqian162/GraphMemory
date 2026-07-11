from __future__ import annotations

import hashlib
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.datasets.hotpotqa.projectors import (
    HotpotQAToTemporalMemoryRankingRequest,
    HotpotQAToTextRankingRequest,
)
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.datasets.hotpotqa.records import (
    HotpotQARankingRecord,
    HotpotQALabelRecord,
)
from graph_memory.io import read_json, write_json
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import MemoryStreamTuneStageConfig
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.methods.memory_stream.contracts import (
    ImportanceArtifact,
)
from graph_memory.retrieval.requests import (
    DenseRuntime,
    TemporalMemoryRankingRequest,
    TextRankingRequest,
)
from graph_memory.retrieval.tuning import (
    memory_stream_grid_from_record,
    tune_memory_stream,
)
from graph_memory.validation import (
    select_importance_records,
    validate_graphs,
    validate_hotpotqa_ranking_records,
    validate_hotpotqa_label_records,
)

LOGGER = logging.getLogger("tune_memory_stream")


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        MemoryStreamTuneStageConfig,
        description="Tune Memory Stream from a resolved stage YAML.",
        script=Path(__file__),
    )
    config = execution.config
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    start_time = time.perf_counter()
    output_config_path = config.selected_config
    candidates_path = config.candidates
    encoder = config.encoder
    dense_config = DenseConfig(
        model_name=encoder.model_name,
        query_prefix=encoder.query_prefix,
        passage_prefix=encoder.passage_prefix,
        batch_size=encoder.batch_size,
    )
    with stage_lifecycle(execution.invocation) as observations:
        task_inputs = cast(list[HotpotQARankingRecord], read_json(config.tasks))
        labels = cast(list[HotpotQALabelRecord], read_json(config.labels))
        graphs = cast(list[MemoryGraph], read_json(config.graphs))
        importance_bytes = config.importance.read_bytes()
        importance_sha256 = hashlib.sha256(importance_bytes).hexdigest()
        importance_artifact = cast(
            ImportanceArtifact,
            read_json(config.importance),
        )
        grid = memory_stream_grid_from_record(config.search_space.model_dump())

        validate_hotpotqa_ranking_records(task_inputs)
        inputs_by_task_id = {
            task_input["task_id"]: task_input for task_input in task_inputs
        }
        validate_hotpotqa_label_records(labels, inputs_by_task_id)
        ranking_requests = _text_requests(task_inputs)
        temporal_requests = _temporal_requests(task_inputs)
        evidence_labels = _evidence_labels(labels)
        validate_graphs(graphs, ranking_requests)
        _ = select_importance_records(importance_artifact, temporal_requests)

        selected_config, candidate_rows = tune_memory_stream(
            temporal_requests=temporal_requests,
            labels=evidence_labels,
            graphs=graphs,
            importance_artifact=importance_artifact,
            grid=grid,
            top_k=config.top_k,
            dense_runtime=DenseRuntime(config=dense_config),
        )
        write_json(output_config_path, selected_config)
        write_json(candidates_path, candidate_rows)
        observations.count("tasks", len(task_inputs))
        observations.count("grid_size", len(grid))
        observations.count("candidate_rows", len(candidate_rows))
        observations.count("importance_sha256", importance_sha256)
        observations.count("selected_scoring_config", selected_config)
        observations.timing("total_seconds", time.perf_counter() - start_time)
        LOGGER.info("selected config: %s", selected_config)
        LOGGER.info("wrote selected config: %s", output_config_path)
    return 0


def _text_requests(
    records: Sequence[HotpotQARankingRecord],
) -> list[TextRankingRequest]:
    projector = HotpotQAToTextRankingRequest()
    return [projector.project(record) for record in records]


def _temporal_requests(
    records: Sequence[HotpotQARankingRecord],
) -> list[TemporalMemoryRankingRequest]:
    projector = HotpotQAToTemporalMemoryRankingRequest()
    return [projector.project(record, {}) for record in records]


def _evidence_labels(labels: Sequence[HotpotQALabelRecord]) -> list[EvidenceLabel]:
    return [
        EvidenceLabel(
            task_id=label["task_id"],
            gold_answer=label["gold_answer"],
            gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
            gold_dependency_edges=tuple(
                _dependency_edge(edge) for edge in label["gold_dependency_edges"]
            ),
        )
        for label in labels
    ]


def _dependency_edge(edge: Sequence[str]) -> tuple[str, str]:
    if len(edge) != 2:
        raise ValueError(
            f"Gold dependency edge must contain exactly two node IDs, got {len(edge)}."
        )
    return edge[0], edge[1]


if __name__ == "__main__":
    raise SystemExit(main())
