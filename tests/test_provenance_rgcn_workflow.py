from __future__ import annotations

from pathlib import Path

import graph_memory.stages.prepare as prepare_stage
from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceQueryMetadata,
    ISETraceRankingRecord,
)
from graph_memory.experiment.artifacts import (
    FileSourceRef,
    ProcessedAssetStore,
    identify_external_source,
)
from graph_memory.experiment.config import (
    ISETraceChunkingConfig,
    ISETraceQueryOriginCounts,
)
from graph_memory.query_synthesis.provenance.contracts import (
    TemplateSupervisionRecord,
)
from graph_memory.stages.prepare import (
    PreparedSplitData,
    materialize_prepared_split,
)
from tests.test_provenance_rgcn_tensorization import _graph_and_request


def test_prepared_isetrace_artifact_publishes_provenance_training_sidecars(
    monkeypatch,
    tmp_path: Path,
) -> None:
    graph, request = _graph_and_request(
        task_id="artifact-template-task",
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
        query_origin="template",
    )
    template = TemplateSupervisionRecord(
        task_id=request.task_id,
        graph_id=graph.graph_id,
        query_text=request.query_text,
        focus_output_ids=("output:c4",),
        participant_output_ids=("output:c2", "output:c3", "output:c4"),
        positive_candidate_ids=(positive.item_id,),
        motif_id="motif:multi_hop_flow:artifact",
        motif_type="multi_hop_flow",
        query_intent="downstream_result",
        graph_fingerprint=graph.fingerprint(),
    )
    prepared = PreparedSplitData(
        task_inputs=[ranking],
        task_labels=[label],
        combined=[],
        counts={
            "natural_queries_selected": 0,
            "template_queries_selected": 1,
        },
        provenance_graphs=[graph],
        query_metadata=[metadata],
        template_supervision=[template],
    )
    monkeypatch.setattr(prepare_stage, "prepare_split", lambda *args, **kwargs: prepared)
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
        count=None,
        seed=13,
        offset=0,
        strict_invalid_examples=False,
        source_revision="fixture-revision",
        query_counts=ISETraceQueryOriginCounts(natural=0, template=1),
        chunking=ISETraceChunkingConfig(
            tokenizer_name="fixture-encoder",
            max_tokens=64,
            reserved_tokens=8,
            overlap_tokens=4,
        ),
        implementation_version="provenance-artifact-test-v1",
    )

    roles = {payload.role for payload in result.artifact.payloads}
    assert {
        "tasks",
        "labels",
        "combined",
        "counts",
        "provenance_graphs",
        "query_metadata",
        "template_supervision",
    } <= roles
    assert result.counts["template_queries_selected"] == 1
