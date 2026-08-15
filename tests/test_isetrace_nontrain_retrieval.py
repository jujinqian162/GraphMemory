from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from graph_memory.datasets.isetrace import (
    adapt_isetrace_record,
    parse_isetrace_record,
    prepare_isetrace_benchmark,
)
from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceQueryMetadata,
    ISETraceRankingRecord,
)
from graph_memory.datasets.isetrace.training import (
    adapt_flat_dense_training_split,
    adapt_provenance_unit_dense_training_split,
)
from graph_memory.datasets.isetrace.retrieval_views import (
    flat_trajectory_candidates,
    provenance_unit_candidates,
)
from graph_memory.evaluation.span_suite import (
    CONTEXT_TOKEN_BUDGETS,
    normalized_budget_auc,
)
from graph_memory.graphs.provenance import (
    ARGUMENT_CHUNK_NODE,
    build_provenance_graph,
    output_content,
    output_source_spans,
)
from graph_memory.query_synthesis.provenance.authoring import (
    AuthoringGold,
    AuthoringQueryRecord,
)
from graph_memory.experiment.config import (
    Bm25MethodConfig,
    DenseEncoderConfig,
    DenseMethodConfig,
    GraphRAGMethodConfig,
    ProvenancePathMethodConfig,
)
from graph_memory.retrieval.requests import (
    TextCandidate,
)
from graph_memory.stages.evaluate import run_evaluate_stage
from graph_memory.stages.retrieve import run_retrieve_stage
from graph_memory.text.chunking import TokenChunkingConfig, token_chunks
from graph_memory.trajectories import SourceSpan
from tests.isetrace_fixtures import (
    isetrace_record,
    set_call_arguments,
    set_tool_output_content,
)


class CharacterOffsetTokenizer:
    is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> dict[str, object]:
        del add_special_tokens, return_offsets_mapping, truncation
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


_TEST_CHUNKING = TokenChunkingConfig(
    tokenizer_name="test-character-tokenizer",
    max_tokens=512,
    overlap_tokens=64,
)


def _test_content_chunks(text: str):
    return token_chunks(
        text,
        tokenizer=CharacterOffsetTokenizer(),
        max_tokens=_TEST_CHUNKING.max_tokens,
        overlap_tokens=_TEST_CHUNKING.overlap_tokens,
    )


class KeywordEncoder:
    vocabulary = ("target", "source", "shared", "other")

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        del batch_size, show_progress_bar
        rows: list[np.ndarray] = []
        for text in texts:
            lowered = text.casefold()
            vector = np.asarray(
                [float(lowered.count(token)) for token in self.vocabulary],
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(vector))
            rows.append(vector / norm if normalize_embeddings and norm else vector)
        return np.asarray(rows, dtype=np.float32)


def _v7_example(
    trajectory, *, query_id: str, sources: tuple[str, ...]
) -> AuthoringQueryRecord:
    call_by_id = {call.call_id: call for call in trajectory.tool_calls}
    output_by_call_id = {output.call_id: output for output in trajectory.tool_outputs}
    sections = [
        f"[I{index} | user_intent]\n{intent.text}"
        for index, intent in enumerate(trajectory.intents, start=1)
    ]
    gold: list[AuthoringGold] = []
    for index, call_id in enumerate(sources, start=1):
        call = call_by_id[call_id]
        output = output_by_call_id[call_id]
        sections.extend(
            (
                f"[A{index} | tool_call | {call.tool_name}]\n{call.raw_arguments}",
                f"[E{index} | tool_output | {output.tool_name}]\n{output.content}",
            )
        )
        gold.append(AuthoringGold(source=f"E{index}", quote=output.content))
    return AuthoringQueryRecord(
        id=query_id,
        text="\n\n".join(sections),
        query="What exact evidence was recorded for this task?",
        gold=tuple(gold),
    )


def _assert_exact_source_coverage(
    candidates: Sequence[TextCandidate],
    *,
    event_id: str,
    json_pointer: str,
    expected_length: int,
) -> None:
    intervals = sorted(
        (span.char_start, span.char_end)
        for candidate in candidates
        for span in candidate.source_spans
        if span.event_id == event_id and span.json_pointer == json_pointer
    )
    assert intervals
    cursor = 0
    for start, end in intervals:
        assert start is not None and end is not None
        assert start <= cursor
        cursor = max(cursor, end)
    assert cursor == expected_length


def test_long_content_is_losslessly_covered_by_both_retrieval_views() -> None:
    raw = isetrace_record()
    raw_arguments = json.dumps({"payload": "A" * 1300})
    output_text = "B" * 1500
    set_call_arguments(raw, message_index=2, call_index=0, value=raw_arguments)
    set_tool_output_content(raw, message_index=3, value=output_text)
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(raw), source_revision="fixture-revision"
    )
    tokenizer = CharacterOffsetTokenizer()

    def chunk_content(text: str):
        return token_chunks(
            text,
            tokenizer=tokenizer,
            max_tokens=64,
            overlap_tokens=8,
        )

    graph = build_provenance_graph(trajectory, content_chunker=chunk_content)
    flat = flat_trajectory_candidates(
        trajectory,
        tokenizer=tokenizer,
        max_tokens=64,
        overlap_tokens=8,
    )
    provenance = provenance_unit_candidates(graph)
    call = trajectory.tool_calls[0]
    output = next(
        item for item in trajectory.tool_outputs if item.call_id == call.call_id
    )

    for candidates in (flat, provenance):
        _assert_exact_source_coverage(
            candidates,
            event_id=call.event_id,
            json_pointer="/raw_arguments",
            expected_length=len(raw_arguments),
        )
        _assert_exact_source_coverage(
            candidates,
            event_id=output.event_id,
            json_pointer="/content",
            expected_length=len(output_text),
        )
    assert output_content(graph, "output:c1") == output_text
    assert output_source_spans(graph, ("output:c1",)) == (
        SourceSpan(
            event_id=output.event_id,
            json_pointer="/content",
            char_start=0,
            char_end=len(output_text),
        ),
    )
    assert len([node for node in graph.nodes if node.kind == ARGUMENT_CHUNK_NODE]) > 1
    assert graph.node_by_id["call:c1"].text == "tool=write"
    assert graph.node_by_id["output:c1"].text == "tool_output=write"


def test_flat_dense_supervision_maps_every_overlapping_chunk() -> None:
    raw = isetrace_record()
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(raw), source_revision="fixture-revision"
    )
    output = trajectory.tool_outputs[0]
    candidates = flat_trajectory_candidates(
        trajectory,
        tokenizer=CharacterOffsetTokenizer(),
        max_tokens=64,
        overlap_tokens=16,
    )
    overlapping = tuple(
        candidate.item_id
        for candidate in candidates
        if any(
            span.event_id == output.event_id
            and span.json_pointer == "/content"
            and span.char_start is not None
            and span.char_end is not None
            and max(span.char_start, 0) < min(span.char_end, len(output.content))
            for span in candidate.source_spans
        )
    )
    ranking = ISETraceRankingRecord(
        task_id="flat-train",
        graph_id=trajectory.trajectory_id,
        query_text="What did the first tool return?",
        flat_candidates=candidates,
        provenance_candidates=(candidates[0],),
    )
    label = ISETraceLabelRecord(
        task_id=ranking.task_id,
        graph_id=ranking.graph_id,
        gold_evidence_spans=(
            SourceSpan(
                event_id=output.event_id,
                json_pointer="/content",
                char_start=0,
                char_end=len(output.content),
            ),
        ),
    )

    requests, labels = adapt_flat_dense_training_split([ranking], [label])

    assert requests[0].candidates == candidates
    assert labels[0].gold_evidence_item_ids == overlapping


def test_provenance_unit_dense_supervision_maps_every_overlapping_unit() -> None:
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(isetrace_record()), source_revision="fixture-revision"
    )
    output = trajectory.tool_outputs[0]
    flat_candidates = flat_trajectory_candidates(
        trajectory,
        tokenizer=CharacterOffsetTokenizer(),
        max_tokens=64,
        overlap_tokens=16,
    )
    candidates = provenance_unit_candidates(build_provenance_graph(trajectory))
    ranking = ISETraceRankingRecord(
        task_id="provenance-unit-train",
        graph_id=trajectory.trajectory_id,
        query_text="What did the first tool return?",
        flat_candidates=flat_candidates,
        provenance_candidates=candidates,
    )
    label = ISETraceLabelRecord(
        task_id=ranking.task_id,
        graph_id=ranking.graph_id,
        gold_evidence_spans=(
            SourceSpan(
                event_id=output.event_id,
                json_pointer="/content",
                char_start=0,
                char_end=len(output.content),
            ),
        ),
    )
    expected = tuple(
        candidate.item_id
        for candidate in candidates
        if any(span.event_id == output.event_id for span in candidate.source_spans)
    )

    requests, labels = adapt_provenance_unit_dense_training_split([ranking], [label])

    assert requests[0].candidates == candidates
    assert labels[0].gold_evidence_item_ids == expected
    assert labels[0].gold_dependency_edges == ()


def test_budget_auc_is_normalized_and_requires_complete_declared_curve() -> None:
    assert normalized_budget_auc(
        {budget: 1.0 for budget in CONTEXT_TOKEN_BUDGETS}
    ) == pytest.approx(1.0)
    assert normalized_budget_auc(
        {
            budget: float(index) / (len(CONTEXT_TOKEN_BUDGETS) - 1)
            for index, budget in enumerate(CONTEXT_TOKEN_BUDGETS)
        }
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="declared token-budget order"):
        normalized_budget_auc({512: 1.0})


def test_v7_benchmark_reuses_graph_and_keeps_one_span_gold(tmp_path: Path) -> None:
    raw = isetrace_record()
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(raw), source_revision="fixture-revision"
    )
    graph = build_provenance_graph(trajectory, content_chunker=_test_content_chunks)
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    examples = (
        _v7_example(trajectory, query_id="query:one", sources=("c1", "c2")),
        _v7_example(trajectory, query_id="query:two", sources=("c2",)),
    )
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in examples), encoding="utf-8"
    )

    benchmark, summary = prepare_isetrace_benchmark(
        query_path,
        trajectory_path,
        source_revision="fixture-revision",
        count=None,
        seed=13,
        offset=0,
        strict=True,
        chunking=_TEST_CHUNKING,
        tokenizer=CharacterOffsetTokenizer(),
    )
    assert len(benchmark.rankings) == 2
    assert len(benchmark.provenance_graphs) == 1
    assert benchmark.provenance_graphs[0].fingerprint() == graph.fingerprint()
    assert (
        benchmark.rankings[0].flat_candidates == benchmark.rankings[1].flat_candidates
    )
    assert (
        benchmark.rankings[0].provenance_candidates
        == benchmark.rankings[1].provenance_candidates
    )
    assert len(benchmark.labels[0].gold_evidence_spans) == 2
    assert len(benchmark.labels[1].gold_evidence_spans) == 1
    assert [item.task_id for item in benchmark.query_metadata] == [
        item.task_id for item in benchmark.rankings
    ]
    assert not hasattr(benchmark.labels[0], "answer_output_ids")
    assert not hasattr(benchmark.labels[0], "support_output_ids")
    assert summary["unique_graphs"] == 1
    assert summary["path_supported_tasks"] == 0


def test_nontrain_stages_run_aligned_isetrace_requests(
    tmp_path: Path,
) -> None:
    raw = isetrace_record()
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(raw), source_revision="fixture-revision"
    )
    trajectory_path = tmp_path / "stage-trajectories.jsonl"
    trajectory_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    query_path = tmp_path / "stage-queries.jsonl"
    query_path.write_text(
        _v7_example(
            trajectory, query_id="query:stage", sources=("c1", "c2")
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    benchmark, _summary = prepare_isetrace_benchmark(
        query_path,
        trajectory_path,
        source_revision="fixture-revision",
        count=None,
        seed=13,
        offset=0,
        strict=True,
        chunking=_TEST_CHUNKING,
        tokenizer=CharacterOffsetTokenizer(),
    )
    encoder_config = DenseEncoderConfig(
        model_name="keyword", query_prefix="", passage_prefix="", batch_size=8
    )
    methods = (
        Bm25MethodConfig(method="bm25"),
        DenseMethodConfig(method="dense", encoder=encoder_config),
        DenseMethodConfig(
            method="dense", variant="provenance_unit", encoder=encoder_config
        ),
        GraphRAGMethodConfig(
            method="graphrag",
            encoder=encoder_config,
            text_unit_size=20,
            text_unit_overlap=4,
            min_node_frequency=1,
            min_edge_weight_percentile=0.0,
            remove_ego_node=False,
            seed_top_s=2,
        ),
        ProvenancePathMethodConfig(
            method="provenance_path",
            encoder=encoder_config,
            preserve_dense_top_n=1,
        ),
    )
    task_inputs = [item.model_dump(mode="json") for item in benchmark.rankings]
    labels: list[object] = [item.model_dump(mode="json") for item in benchmark.labels]
    for method in methods:
        predictions, _provenance = run_retrieve_stage(
            method,
            dataset="isetrace",
            top_k=3,
            task_inputs=task_inputs,
            evidence_graphs=None,
            provenance_graphs=(
                list(benchmark.provenance_graphs)
                if isinstance(method, ProvenancePathMethodConfig)
                else []
            ),
            model=None,
            encoder_source=None,
            device="cpu",
            dense_encoder=(
                None if isinstance(method, Bm25MethodConfig) else KeywordEncoder()
            ),
        )
        assert len(predictions) == 1
        expected_candidates = (
            benchmark.rankings[0].provenance_candidates
            if isinstance(method, ProvenancePathMethodConfig)
            or (
                isinstance(method, DenseMethodConfig)
                and method.variant == "provenance_unit"
            )
            else benchmark.rankings[0].flat_candidates
        )
        assert len(predictions[0].ranked_nodes) == len(expected_candidates)
        metric_rows, _failure_cases, per_task_rows = run_evaluate_stage(
            dataset="isetrace",
            top_k=3,
            failure_case_limit=10,
            predictions=predictions,
            labels=labels,
            graphs=[],
            query_metadata=(
                ISETraceQueryMetadata(
                    task_id=benchmark.rankings[0].task_id,
                    graph_id=benchmark.rankings[0].graph_id,
                    memory_mode="linked_recall",
                ),
            ),
        )
        metric = metric_rows[0]
        assert per_task_rows[0].graph_id == benchmark.rankings[0].graph_id
        assert per_task_rows[0].memory_mode == "linked_recall"
        assert metric.evaluation_schema == "execution_provenance_span_v8"
        assert metric.evidence_density_at_10 != "N/A"
        coverage_curve = (
            metric.coverage_at_256_tokens,
            metric.coverage_at_512_tokens,
            metric.coverage_at_1024_tokens,
            metric.coverage_at_2048_tokens,
            metric.coverage_at_4096_tokens,
            metric.coverage_at_8192_tokens,
        )
        full_support_curve = (
            metric.full_support_at_256_tokens,
            metric.full_support_at_512_tokens,
            metric.full_support_at_1024_tokens,
            metric.full_support_at_2048_tokens,
            metric.full_support_at_4096_tokens,
            metric.full_support_at_8192_tokens,
        )
        assert all(value != "N/A" for value in coverage_curve)
        assert all(value != "N/A" for value in full_support_curve)
        assert list(coverage_curve) == sorted(coverage_curve)
        assert list(full_support_curve) == sorted(full_support_curve)
        assert metric.coverage_budget_auc != "N/A"
        assert metric.full_support_budget_auc != "N/A"
        assert metric.connected_evidence_recall_at_10 != "N/A"
        assert metric.path_recall_at_10 == "N/A"
        assert metric.edge_recall_at_10 == "N/A"
