from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import graph_memory.stages.prepare as prepare_stage
from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceQueryMetadata,
    ISETraceRankingRecord,
)
from graph_memory.experiment.artifacts import (
    FileSourceRef,
    ProcessedAssetStore,
    artifact_payload_path,
    identify_external_source,
)
from graph_memory.experiment.config import (
    ISETraceChunkingConfig,
    ISETraceTrajectorySplitCounts,
)
from graph_memory.io import read_json
from graph_memory.stages.prepare import materialize_prepared_split
from tests.test_provenance_rgcn_tensorization import _graph_and_request


def test_prepared_isetrace_artifact_publishes_provenance_sidecars(
    monkeypatch,
    tmp_path: Path,
) -> None:
    graph, request = _graph_and_request(
        task_id="artifact-natural-task",
        query_text="Which result completed the chain?",
    )
    positive = next(
        candidate
        for candidate in request.candidates
        if candidate.item_id == "output-content:c4:chunk:0"
    )
    ranking = ISETraceRankingRecord(
        task_id=request.task_id,
        graph_id=graph.graph_id,
        query_text=request.query_text,
        flat_candidates=request.candidates,
        provenance_candidates=request.candidates,
    )
    label = ISETraceLabelRecord(
        task_id=request.task_id,
        graph_id=graph.graph_id,
        gold_evidence_spans=positive.source_spans,
    )
    metadata = ISETraceQueryMetadata(
        task_id=request.task_id,
        graph_id=graph.graph_id,
        memory_mode="direct_recall",
    )
    benchmark = SimpleNamespace(
        rankings=(ranking,),
        labels=(label,),
        provenance_graphs=(graph,),
        query_metadata=(metadata,),
    )
    summary = SimpleNamespace(to_dict=lambda: {"natural_queries_selected": 1})
    monkeypatch.setattr(
        prepare_stage,
        "prepare_isetrace_benchmark",
        lambda *args, **kwargs: (benchmark, summary),
    )
    source_path = tmp_path / "source.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    source = identify_external_source(source_path, repository_root=tmp_path)
    assert isinstance(source, FileSourceRef)

    result = materialize_prepared_split(
        ProcessedAssetStore(tmp_path / "processed"),
        dataset="isetrace",
        split="train",
        source=source,
        trajectory_source=source,
        authoring_metadata_source=source,
        count=None,
        seed=13,
        offset=0,
        strict_invalid_examples=False,
        source_revision="fixture-revision",
        trajectory_splits=ISETraceTrajectorySplitCounts(
            train=1,
            dev=1,
            test=1,
        ),
        chunking=ISETraceChunkingConfig(
            tokenizer_name="fixture-encoder",
            max_tokens=64,
            reserved_tokens=8,
            overlap_tokens=4,
        ),
        implementation_version="provenance-artifact-test-v1",
    )

    roles = {payload.role for payload in result.payloads}
    assert {
        "tasks",
        "labels",
        "counts",
        "provenance_graphs",
        "query_metadata",
    } <= roles
    counts = read_json(artifact_payload_path(result, "counts"))
    assert counts["natural_queries_selected"] == 1
    assert result.origin["authoring_metadata_digest"] == source.digest
    query_metadata = read_json(artifact_payload_path(result, "query_metadata"))
    assert query_metadata[0]["memory_mode"] == "direct_recall"
