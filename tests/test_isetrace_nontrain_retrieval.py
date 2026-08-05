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
    ISETraceRankingRecord,
)
from graph_memory.datasets.isetrace.training import adapt_flat_dense_training_split
from graph_memory.datasets.isetrace.retrieval_views import (
    flat_trajectory_candidates,
    provenance_unit_candidates,
)
from graph_memory.evaluation.span_metrics import (
    span_metrics_at,
    span_metrics_under_token_budget,
)
from graph_memory.graphs.provenance import (
    ARGUMENT_CHUNK_NODE,
    FEEDS_EDGE,
    HAS_CONTENT_EDGE,
    OUTPUT_CHUNK_NODE,
    RETURNS_EDGE,
    TOOL_CALL_NODE,
    TOOL_OUTPUT_NODE,
    ProvenanceEdge,
    ProvenanceGraph,
    ProvenanceNode,
    build_provenance_graph,
    logical_output_dependencies,
    output_content,
    output_source_spans,
)
from graph_memory.query_synthesis.provenance.authoring import (
    AuthoringGold,
    AuthoringQueryRecord,
)
from graph_memory.retrieval.contracts import GraphRAGTrace, ProvenancePathTrace
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.methods.graphrag import (
    GraphRAGConfig,
    GraphRAGMethod,
    build_graphrag_request,
)
from graph_memory.retrieval.methods.provenance_path import (
    ProvenancePathConfig,
    ProvenancePathMethod,
)
from graph_memory.experiment.config import (
    Bm25MethodConfig,
    DenseEncoderConfig,
    DenseMethodConfig,
    GraphRAGMethodConfig,
    ProvenancePathMethodConfig,
)
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.retrieval.results import RankedNodeRecord
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


def test_flat_dense_supervision_fails_when_no_chunk_overlaps() -> None:
    raw = isetrace_record()
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(raw), source_revision="fixture-revision"
    )
    candidates = flat_trajectory_candidates(
        trajectory,
        tokenizer=CharacterOffsetTokenizer(),
        max_tokens=64,
        overlap_tokens=8,
    )
    ranking = ISETraceRankingRecord(
        task_id="flat-missing",
        graph_id=trajectory.trajectory_id,
        query_text="Missing evidence",
        flat_candidates=candidates,
        provenance_candidates=(candidates[0],),
    )
    label = ISETraceLabelRecord(
        task_id=ranking.task_id,
        graph_id=ranking.graph_id,
        gold_evidence_spans=(
            SourceSpan(
                event_id="event:missing",
                json_pointer="/content",
                char_start=0,
                char_end=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="flat-missing.*no positive candidates"):
        adapt_flat_dense_training_split([ranking], [label])


def test_v7_rejects_legacy_query_shape(tmp_path: Path) -> None:
    raw = isetrace_record()
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    query_path = tmp_path / "legacy-queries.jsonl"
    query_path.write_text(
        json.dumps({"query": {"query_id": "old"}}) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="invalid v7 query"):
        prepare_isetrace_benchmark(
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


def test_span_density_charges_duplicate_context_and_respects_token_budget() -> None:
    gold = (
        SourceSpan(
            event_id="event:1", json_pointer="/content", char_start=0, char_end=10
        ),
    )
    ranked = (
        RankedNodeRecord(
            node_id="chunk:1", score=1.0, source_spans=gold, token_count=10
        ),
        RankedNodeRecord(
            node_id="chunk:2", score=0.5, source_spans=gold, token_count=10
        ),
    )
    assert span_metrics_at(ranked, gold, 1).density == 1.0
    assert span_metrics_at(ranked, gold, 2).density == 0.5
    assert span_metrics_under_token_budget(ranked, gold, 10).density == 1.0
    assert span_metrics_under_token_budget(ranked, gold, 20).density == 0.5


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
    assert [item.query_origin for item in benchmark.query_metadata] == [
        "natural",
        "natural",
    ]
    assert all(
        "query_origin" not in ranking.model_dump(mode="json")
        for ranking in benchmark.rankings
    )
    assert all(
        "query_origin" not in candidate.model_dump(mode="json")
        for ranking in benchmark.rankings
        for candidate in ranking.provenance_candidates
    )
    assert all(
        "query_origin" not in graph.model_dump(mode="json")
        for graph in benchmark.provenance_graphs
    )
    assert not hasattr(benchmark.labels[0], "answer_output_ids")
    assert not hasattr(benchmark.labels[0], "support_output_ids")
    assert summary["unique_graphs"] == 1
    assert summary["path_supported_tasks"] == 0


def test_v7_raw_directory_drops_uncompilable_queries_when_nonstrict(
    tmp_path: Path,
) -> None:
    raw = isetrace_record()
    duplicate_arguments = json.dumps(
        {"path": "/workspace/report.md", "content": "draft"}
    )
    set_call_arguments(
        raw,
        message_index=7,
        call_index=0,
        value=duplicate_arguments,
    )
    set_tool_output_content(
        raw,
        message_index=8,
        value="Successfully wrote /workspace/report.md",
    )
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(raw), source_revision="fixture-revision"
    )
    raw_root = tmp_path / "raw"
    shards = raw_root / "trajectories"
    shards.mkdir(parents=True)
    (shards / "trajectories-00000.jsonl").write_text(
        json.dumps(raw) + "\n", encoding="utf-8"
    )
    examples = (
        _v7_example(trajectory, query_id="query:ambiguous", sources=("c1",)),
        _v7_example(trajectory, query_id="query:valid", sources=("c2",)),
    )
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in examples),
        encoding="utf-8",
    )

    benchmark, summary = prepare_isetrace_benchmark(
        query_path,
        raw_root,
        source_revision="fixture-revision",
        count=None,
        seed=13,
        offset=0,
        strict=False,
        split="train",
        trajectory_splits={
            "train": {"natural": 1, "template": 0},
            "dev": {"natural": 0, "template": 0},
            "test": {"natural": 0, "template": 0},
        },
        chunking=_TEST_CHUNKING,
        tokenizer=CharacterOffsetTokenizer(),
    )

    assert [item.task_id for item in benchmark.rankings] == ["query:valid"]
    assert summary["queries_seen"] == 2
    assert summary["queries_resolved"] == 1
    assert summary["queries_resolved"] == 1
    assert summary["queries_dropped"] == 1
    assert summary["queries_uncompilable"] == 1
    assert summary["queries_unmatched"] == 0
    assert summary["trajectories_seen"] == 1


def test_v7_argument_quote_is_valid_gold(tmp_path: Path) -> None:
    raw = isetrace_record()
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(raw), source_revision="fixture-revision"
    )
    call = trajectory.tool_calls[0]
    output = next(
        item for item in trajectory.tool_outputs if item.call_id == call.call_id
    )
    text = "\n\n".join(
        (
            *[
                f"[I{i} | user_intent]\n{intent.text}"
                for i, intent in enumerate(trajectory.intents, 1)
            ],
            f"[A1 | tool_call | {call.tool_name}]\n{call.raw_arguments}",
            f"[E1 | tool_output | {output.tool_name}]\n{output.content}",
        )
    )
    example = AuthoringQueryRecord(
        id="query:argument",
        text=text,
        query="What arguments configured the report?",
        gold=(AuthoringGold(source="A1", quote=call.raw_arguments),),
    )
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(example.model_dump_json() + "\n", encoding="utf-8")
    benchmark, _ = prepare_isetrace_benchmark(
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
    assert benchmark.labels[0].gold_evidence_spans == (
        SourceSpan(
            event_id=call.event_id,
            json_pointer="/raw_arguments",
            char_start=0,
            char_end=len(call.raw_arguments),
        ),
    )


def test_logical_dependencies_include_feeds_and_artifact_flow() -> None:
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(isetrace_record()), source_revision="fixture-revision"
    )
    dependencies = logical_output_dependencies(build_provenance_graph(trajectory))
    observed = {
        (item.source_output_id, item.target_output_id, item.relation)
        for item in dependencies
    }

    assert ("output:c1", "output:c2", "resource.flow") in observed
    assert ("output:c2", "output:c3", "data.feeds") in observed


def _execution_graph(
    *,
    graph_id: str,
    texts: dict[str, str],
    feed_pairs: tuple[tuple[str, str], ...],
) -> ProvenanceGraph:
    nodes: list[ProvenanceNode] = []
    edges: list[ProvenanceEdge] = []
    call_ids = tuple(texts)
    for index, call_id in enumerate(call_ids):
        call_event = f"call-event:{call_id}"
        output_event = f"output-event:{call_id}"
        content_id = f"output-content:{call_id}:chunk:0"
        nodes.extend(
            (
                ProvenanceNode(
                    node_id=f"call:{call_id}",
                    kind=TOOL_CALL_NODE,
                    text="tool=exec",
                    source_spans=(SourceSpan(event_id=call_event),),
                    attributes={"tool_name": "exec", "message_index": index * 2},
                ),
                ProvenanceNode(
                    node_id=f"output:{call_id}",
                    kind=TOOL_OUTPUT_NODE,
                    text="tool_output=exec",
                    source_spans=(SourceSpan(event_id=output_event),),
                    attributes={"tool_name": "exec", "message_index": index * 2 + 1},
                ),
                ProvenanceNode(
                    node_id=content_id,
                    kind=OUTPUT_CHUNK_NODE,
                    text=texts[call_id],
                    source_spans=(
                        SourceSpan(
                            event_id=output_event,
                            json_pointer="/content",
                            char_start=0,
                            char_end=len(texts[call_id]),
                        ),
                    ),
                    attributes={"tool_name": "exec", "chunk_index": 0},
                ),
            )
        )
        edges.extend(
            (
                ProvenanceEdge(
                    edge_id=f"return:{call_id}",
                    relation=RETURNS_EDGE,
                    source=f"call:{call_id}",
                    target=f"output:{call_id}",
                    derivation="native",
                    extractor="fixture",
                ),
                ProvenanceEdge(
                    edge_id=f"content:{call_id}",
                    relation=HAS_CONTENT_EDGE,
                    source=f"output:{call_id}",
                    target=content_id,
                    derivation="native",
                    extractor="fixture",
                ),
            )
        )
    for source, target in feed_pairs:
        edges.append(
            ProvenanceEdge(
                edge_id=f"feed:{source}:{target}",
                relation=FEEDS_EDGE,
                source=f"output:{source}",
                target=f"call:{target}",
                derivation="deterministic",
                extractor="fixture",
            )
        )
    return ProvenanceGraph(
        graph_id=graph_id,
        trajectory_fingerprint=("a" if graph_id == "path-graph" else "b") * 64,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def _path_graph(*, include_feed: bool) -> ProvenanceGraph:
    return _execution_graph(
        graph_id="path-graph",
        texts={"c0": "other", "c1": "source", "c2": "target"},
        feed_pairs=(("c1", "c2"),) if include_feed else (),
    )


def _multihop_graph() -> ProvenanceGraph:
    return _execution_graph(
        graph_id="multi-path-graph",
        texts={"c0": "other", "c1": "source", "c2": "middle", "c3": "target"},
        feed_pairs=(("c1", "c2"), ("c2", "c3")),
    )


def _path_request(graph: ProvenanceGraph) -> ExecutionProvenanceRankingRequest:
    return ExecutionProvenanceRankingRequest(
        task_id="q1",
        query_text="target",
        candidates=tuple(
            TextCandidate(
                item_id=node.node_id,
                text=node.text,
                metadata={"graph_id": graph.graph_id},
                source_spans=node.source_spans,
            )
            for node in graph.nodes
            if node.kind == OUTPUT_CHUNK_NODE
        ),
        graph=graph,
    )


def _dense_ranker() -> DenseTaskRetriever:
    return DenseTaskRetriever(
        model_name="keyword",
        query_prefix="",
        passage_prefix="",
        encoder=KeywordEncoder(),
        device="cpu",
    )


def test_provenance_path_promotes_reverse_dependency_and_emits_stored_direction() -> (
    None
):
    method = ProvenancePathMethod(
        dense_ranker=_dense_ranker(),
        config=ProvenancePathConfig(
            seed_top_s=1,
            max_path_hops=4,
            max_partners_per_anchor=1,
            max_expansions=16,
            preserve_dense_top_n=1,
        ),
    )

    result = method.rank_task(_path_request(_path_graph(include_feed=True)), top_k=2)

    assert [node.node_id for node in result.ranked_nodes] == [
        "output-content:c2:chunk:0",
        "output-content:c1:chunk:0",
        "output-content:c0:chunk:0",
    ]
    assert [(edge.source, edge.target) for edge in result.trace.retrieved_edges] == [
        ("output-content:c2:chunk:0", "output-content:c1:chunk:0")
    ]
    trace = result.trace.native_trace
    assert isinstance(trace, ProvenancePathTrace)
    assert trace.exact_dense_fallback is False
    accepted = next(proposal for proposal in trace.proposals if proposal.accepted)
    assert accepted.traversed_reverse == (True, True, True, False)
    ProvenancePathTrace.model_validate(trace.model_dump(mode="json", exclude_none=True))


def test_provenance_path_completes_bounded_multihop_chain() -> None:
    method = ProvenancePathMethod(
        dense_ranker=_dense_ranker(),
        config=ProvenancePathConfig(
            seed_top_s=1,
            max_path_hops=8,
            max_partners_per_anchor=2,
            max_expansions=16,
            preserve_dense_top_n=1,
        ),
    )

    result = method.rank_task(_path_request(_multihop_graph()), top_k=3)

    assert [node.node_id for node in result.ranked_nodes[:3]] == [
        "output-content:c3:chunk:0",
        "output-content:c2:chunk:0",
        "output-content:c1:chunk:0",
    ]
    assert {(edge.source, edge.target) for edge in result.trace.retrieved_edges} == {
        ("output-content:c3:chunk:0", "output-content:c2:chunk:0"),
        ("output-content:c3:chunk:0", "output-content:c1:chunk:0"),
    }


def test_provenance_path_is_exact_dense_without_logical_dependency() -> None:
    graph = _path_graph(include_feed=False)
    request = _path_request(graph)
    dense = _dense_ranker().rank(
        TextRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
        )
    )
    method = ProvenancePathMethod(
        dense_ranker=_dense_ranker(),
        config=ProvenancePathConfig(preserve_dense_top_n=1),
    )

    result = method.rank_task(request, top_k=2)

    assert result.ranked_nodes == tuple(dense)
    assert result.trace.retrieved_edges == ()
    assert isinstance(result.trace.native_trace, ProvenancePathTrace)
    assert result.trace.native_trace.exact_dense_fallback is True


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
            provenance_graphs=list(benchmark.provenance_graphs),
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
            else benchmark.rankings[0].flat_candidates
        )
        assert len(predictions[0].ranked_nodes) == len(expected_candidates)
        metric_rows, _failure_cases, _per_task_rows = run_evaluate_stage(
            dataset="isetrace",
            top_k=3,
            failure_case_limit=10,
            predictions=predictions,
            labels=labels,
            graphs=[],
        )
        metric = metric_rows[0]
        assert metric.evaluation_schema == "execution_provenance_span_v7"
        assert metric.evidence_density_at_10 != "N/A"
        assert metric.coverage_at_512_tokens != "N/A"
        assert metric.coverage_at_1024_tokens != "N/A"
        assert metric.coverage_at_2048_tokens != "N/A"
        assert (
            metric.coverage_at_512_tokens
            <= metric.coverage_at_1024_tokens
            <= metric.coverage_at_2048_tokens
        )
        assert metric.full_support_at_2048_tokens != "N/A"
        assert metric.connected_evidence_recall_at_10 != "N/A"
        assert metric.path_recall_at_10 == "N/A"
        assert metric.edge_recall_at_10 == "N/A"


def test_graphrag_runs_noun_graph_ppr_and_projects_to_candidates() -> None:
    request = TextRankingRequest(
        task_id="g1",
        query_text="target",
        candidates=(
            TextCandidate(
                item_id="a",
                text=(
                    "target report references /workspace/shared/report.md and "
                    "shared report archive"
                ),
                metadata={},
            ),
            TextCandidate(item_id="b", text="other weather summary", metadata={}),
            TextCandidate(
                item_id="z",
                text=(
                    "source report at /workspace/shared/report.md with "
                    "shared report archive"
                ),
                metadata={},
            ),
        ),
    )
    config = GraphRAGConfig(
        text_unit_size=20,
        text_unit_overlap=4,
        min_node_frequency=1,
        min_edge_weight_percentile=0.0,
        remove_ego_node=False,
        seed_top_s=1,
        semantic_weight=0.0,
        graph_weight=1.0,
    )
    method = GraphRAGMethod(dense_ranker=_dense_ranker(), config=config)

    result = method.rank_task(build_graphrag_request(request, config), top_k=2)

    assert [node.node_id for node in result.ranked_nodes[:2]] == ["a", "z"]
    trace = result.trace.native_trace
    assert isinstance(trace, GraphRAGTrace)
    assert trace.trace_kind == "fast_graphrag_ppr"
    assert trace.text_unit_count > 0
    assert trace.entity_count > 0
    assert trace.relation_count > 0
    assert trace.iterations > 0
    assert trace.exact_dense_fallback is False
    GraphRAGTrace.model_validate(trace.model_dump(mode="json", exclude_none=True))


def test_graphrag_is_exact_dense_when_pruning_leaves_no_graph() -> None:
    request = TextRankingRequest(
        task_id="g-empty",
        query_text="target",
        candidates=(
            TextCandidate(item_id="a", text="target unique evidence", metadata={}),
            TextCandidate(item_id="b", text="other isolated material", metadata={}),
        ),
    )
    config = GraphRAGConfig()
    dense_ranker = _dense_ranker()
    dense = dense_ranker.rank(request)
    method = GraphRAGMethod(dense_ranker=dense_ranker, config=config)

    result = method.rank_task(build_graphrag_request(request, config), top_k=2)

    assert result.ranked_nodes == tuple(dense)
    trace = result.trace.native_trace
    assert isinstance(trace, GraphRAGTrace)
    assert trace.exact_dense_fallback is True
    assert trace.entity_count == 0
    assert trace.relation_count == 0
