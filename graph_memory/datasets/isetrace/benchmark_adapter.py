from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from graph_memory.datasets.isetrace.adapter import iter_canonical_trajectories
from graph_memory.datasets.isetrace.retrieval_views import (
    flat_trajectory_candidates,
    provenance_unit_candidates,
)
from graph_memory.datasets.isetrace.benchmark_records import (
    CombinedISETraceBenchmarkRecord,
    ISETraceLabelPolicy,
    ISETraceLabelRecord,
    ISETracePreparedBenchmark,
    ISETraceRankingRecord,
    ISETraceReviewPolicy,
)
from graph_memory.datasets.splits import sample_split
from graph_memory.graphs.provenance import (
    TOOL_OUTPUT_NODE,
    OutputDependency,
    ProvenanceGraph,
    build_provenance_graph,
    logical_output_dependencies,
)
from graph_memory.query_synthesis.provenance.contracts import (
    LlmGenerationProvenance,
    ProvenanceQueryExample,
)
from graph_memory.text.chunking import (
    OffsetTokenizer,
    TokenChunkingConfig,
    load_offset_tokenizer,
    token_chunks,
)
from graph_memory.trajectories import (
    CanonicalTrajectory,
    MessageEvent,
    SourceSpan,
    ToolCallEvent,
    ToolOutputEvent,
)

_QUERY_ADAPTER = TypeAdapter(ProvenanceQueryExample)
_COMPLETE_SUPPORT_INTENTS = frozenset({"complete_chain", "contributing_sources"})


class ISETraceBenchmarkSummary(Counter[str]):
    def to_dict(self) -> dict[str, int]:
        return dict(sorted(self.items()))


def prepare_isetrace_benchmark(
    query_source: Path,
    trajectory_source: Path,
    *,
    source_revision: str,
    count: int | None,
    seed: int,
    offset: int,
    strict: bool,
    review_policy: ISETraceReviewPolicy,
    label_policy: ISETraceLabelPolicy,
    chunking: TokenChunkingConfig = TokenChunkingConfig(
        tokenizer_name="models/intfloat-e5-base-v2",
        max_tokens=512,
        overlap_tokens=64,
    ),
    tokenizer: OffsetTokenizer | None = None,
) -> tuple[ISETracePreparedBenchmark, ISETraceBenchmarkSummary]:
    summary = ISETraceBenchmarkSummary()
    examples = _read_queries(query_source, strict=strict, summary=summary)
    admitted = [
        example
        for example in examples
        if _admitted(example, review_policy=review_policy, summary=summary)
    ]
    if not admitted:
        raise ValueError(
            f"ISETrace review_policy={review_policy!r} admitted no query records"
        )
    selected = sample_split(admitted, count=count, seed=seed, offset=offset)
    summary["queries_selected"] = len(selected)
    graph_ids = {example.query.graph_id for example in selected}
    offset_tokenizer = tokenizer or load_offset_tokenizer(chunking.tokenizer_name)

    def chunk_content(text: str):
        return token_chunks(
            text,
            tokenizer=offset_tokenizer,
            max_tokens=chunking.content_tokens,
            overlap_tokens=chunking.overlap_tokens,
        )

    trajectories, graphs = _load_contexts(
        trajectory_source,
        graph_ids=graph_ids,
        source_revision=source_revision,
        content_chunker=chunk_content,
    )
    flat_by_graph_id = {
        graph_id: flat_trajectory_candidates(
            trajectories[graph_id],
            tokenizer=offset_tokenizer,
            max_tokens=chunking.content_tokens,
            overlap_tokens=chunking.overlap_tokens,
        )
        for graph_id in sorted(graphs)
    }
    provenance_by_graph_id = {
        graph_id: provenance_unit_candidates(graph)
        for graph_id, graph in graphs.items()
    }

    rankings: list[ISETraceRankingRecord] = []
    labels: list[ISETraceLabelRecord] = []
    for example in selected:
        graph = graphs[example.query.graph_id]
        dependencies = logical_output_dependencies(graph)
        ranking = ISETraceRankingRecord(
            task_id=example.query.query_id,
            graph_id=graph.graph_id,
            query_text=example.query.query_text,
            flat_candidates=flat_by_graph_id[graph.graph_id],
            provenance_candidates=provenance_by_graph_id[graph.graph_id],
            logical_dependencies=dependencies,
        )
        label = _label_record(
            example,
            graph=graph,
            trajectory=trajectories[graph.graph_id],
            dependencies=dependencies,
            label_policy=label_policy,
        )
        CombinedISETraceBenchmarkRecord(ranking=ranking, label=label)
        rankings.append(ranking)
        labels.append(label)
        summary[f"query_intent_{label.query_intent}"] += 1
        summary[f"motif_type_{label.motif_type}"] += 1

    benchmark = ISETracePreparedBenchmark(
        rankings=tuple(rankings),
        labels=tuple(labels),
        provenance_graphs=tuple(graphs[graph_id] for graph_id in sorted(graphs)),
    )
    summary["unique_graphs"] = len(graphs)
    summary["ranking_tasks"] = len(rankings)
    summary["label_tasks"] = len(labels)
    summary["flat_candidates"] = sum(
        len(item.flat_candidates) for item in rankings
    )
    summary["provenance_candidates"] = sum(
        len(item.provenance_candidates) for item in rankings
    )
    summary["unique_flat_candidates"] = sum(
        len(flat_by_graph_id[graph_id]) for graph_id in graphs
    )
    summary["unique_provenance_candidates"] = sum(
        len(provenance_by_graph_id[graph_id]) for graph_id in graphs
    )
    summary["path_supported_tasks"] = sum(
        bool(label.gold_dependency_edges) for label in labels
    )
    return benchmark, summary


def combined_isetrace_records(
    rankings: tuple[ISETraceRankingRecord, ...],
    labels: tuple[ISETraceLabelRecord, ...],
) -> list[CombinedISETraceBenchmarkRecord]:
    label_by_id = {label.task_id: label for label in labels}
    return [
        CombinedISETraceBenchmarkRecord(
            ranking=ranking,
            label=label_by_id[ranking.task_id],
        )
        for ranking in rankings
    ]


def _read_queries(
    source: Path,
    *,
    strict: bool,
    summary: ISETraceBenchmarkSummary,
) -> list[ProvenanceQueryExample]:
    result: list[ProvenanceQueryExample] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            summary["queries_seen"] += 1
            try:
                result.append(_QUERY_ADAPTER.validate_json(line))
            except (ValidationError, ValueError) as error:
                summary["invalid_queries"] += 1
                if strict:
                    raise ValueError(
                        f"invalid provenance query at {source}:{line_number}: {error}"
                    ) from error
    task_ids = [example.query.query_id for example in result]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("ISETrace query IDs must be unique")
    return result


def _admitted(
    example: ProvenanceQueryExample,
    *,
    review_policy: ISETraceReviewPolicy,
    summary: ISETraceBenchmarkSummary,
) -> bool:
    generation = example.generation
    if not isinstance(generation, LlmGenerationProvenance):
        summary["template_queries_admitted"] += 1
        return True
    status = generation.human_review_status
    summary[f"review_status_{status}"] += 1
    if status == "rejected":
        summary["queries_excluded_by_review"] += 1
        return False
    if review_policy == "accepted_only" and status not in {"accepted", "edited"}:
        summary["queries_excluded_by_review"] += 1
        return False
    return True


def _load_contexts(
    trajectory_source: Path,
    *,
    graph_ids: set[str],
    source_revision: str,
    content_chunker,
) -> tuple[dict[str, CanonicalTrajectory], dict[str, ProvenanceGraph]]:
    trajectories: dict[str, CanonicalTrajectory] = {}
    graphs: dict[str, ProvenanceGraph] = {}
    for trajectory in iter_canonical_trajectories(
        trajectory_source,
        source_revision=source_revision,
        strict=False,
    ):
        if trajectory.trajectory_id not in graph_ids:
            continue
        graph = build_provenance_graph(
            trajectory,
            content_chunker=content_chunker,
        )
        trajectories[trajectory.trajectory_id] = trajectory
        graphs[graph.graph_id] = graph
        if len(graphs) == len(graph_ids):
            break
    missing = sorted(graph_ids - set(graphs))
    if missing:
        raise ValueError(f"ISETrace queries reference missing graph IDs={missing}")
    return trajectories, graphs


def _label_record(
    example: ProvenanceQueryExample,
    *,
    graph: ProvenanceGraph,
    trajectory: CanonicalTrajectory,
    dependencies: tuple[OutputDependency, ...],
    label_policy: ISETraceLabelPolicy,
) -> ISETraceLabelRecord:
    available = {node.node_id for node in graph.nodes if node.kind == TOOL_OUTPUT_NODE}
    answer = tuple(example.label.answer_output_ids)
    support = tuple(example.label.support_output_ids)
    referenced = {*answer, *support}
    unknown = sorted(referenced - available)
    if unknown:
        raise ValueError(
            f"query_id={example.query.query_id} label references unknown outputs={unknown}"
        )

    derived = {
        (item.source_output_id, item.target_output_id, item.relation)
        for item in dependencies
    }
    for item in example.label.dependencies:
        key = (item.source_output_id, item.target_output_id, item.relation)
        if key not in derived:
            raise ValueError(
                f"query_id={example.query.query_id} dependency is absent from "
                f"query-independent graph: {key}"
            )

    answer_spans = example.label.answer_evidence_spans
    support_spans = example.label.support_evidence_spans
    _validate_exact_spans(
        (*answer_spans, *support_spans),
        trajectory=trajectory,
        query_id=example.query.query_id,
    )
    _spans_by_output_id(
        graph,
        output_ids=answer,
        spans=answer_spans,
        query_id=example.query.query_id,
    )
    _spans_by_output_id(
        graph,
        output_ids=support,
        spans=support_spans,
        query_id=example.query.query_id,
    )

    if label_policy == "answer_only":
        gold = answer
        gold_spans = answer_spans
    elif label_policy == "support":
        gold = support
        gold_spans = support_spans
    elif example.label.query_intent in _COMPLETE_SUPPORT_INTENTS:
        gold = support
        gold_spans = support_spans
    else:
        gold = answer
        gold_spans = answer_spans
    gold_set = set(gold)
    gold_spans_by_output_id = _spans_by_output_id(
        graph,
        output_ids=gold,
        spans=gold_spans,
        query_id=example.query.query_id,
    )
    gold_edges = tuple(
        (item.source_output_id, item.target_output_id)
        for item in example.label.dependencies
        if item.source_output_id in gold_set and item.target_output_id in gold_set
    )
    generation = example.generation
    if isinstance(generation, LlmGenerationProvenance):
        reference_answer = generation.reference_answer
        review_status = generation.human_review_status
    else:
        node_by_id = graph.node_by_id
        reference_answer = "\n\n".join(node_by_id[node_id].text for node_id in gold)
        review_status = "template"
    return ISETraceLabelRecord(
        task_id=example.query.query_id,
        graph_id=graph.graph_id,
        gold_answer=reference_answer,
        gold_evidence_output_ids=gold,
        gold_evidence_spans=gold_spans,
        gold_evidence_spans_by_output_id=gold_spans_by_output_id,
        gold_dependency_edges=gold_edges,
        answer_output_ids=answer,
        answer_evidence_spans=answer_spans,
        support_output_ids=support,
        support_evidence_spans=support_spans,
        motif_id=example.label.motif_id,
        motif_type=example.label.motif_type,
        query_intent=example.label.query_intent,
        authoring_review_status=review_status,
        label_policy=label_policy,
    )


def _spans_by_output_id(
    graph: ProvenanceGraph,
    *,
    output_ids: tuple[str, ...],
    spans: tuple[SourceSpan, ...],
    query_id: str,
) -> dict[str, tuple[SourceSpan, ...]]:
    event_by_output_id = {
        output_id: graph.node_by_id[output_id].source_spans[0].event_id
        for output_id in output_ids
    }
    output_by_event_id = {
        event_id: output_id for output_id, event_id in event_by_output_id.items()
    }
    grouped: dict[str, list[SourceSpan]] = {
        output_id: [] for output_id in output_ids
    }
    for span in spans:
        output_id = output_by_event_id.get(span.event_id)
        if output_id is None:
            raise ValueError(
                f"query_id={query_id} evidence span event={span.event_id} does not "
                "belong to a selected output"
            )
        grouped[output_id].append(span)
    missing = sorted(output_id for output_id, values in grouped.items() if not values)
    if missing:
        raise ValueError(
            f"query_id={query_id} selected outputs lack exact evidence spans: {missing}"
        )
    return {output_id: tuple(values) for output_id, values in grouped.items()}


def _validate_exact_spans(
    spans: tuple[SourceSpan, ...],
    *,
    trajectory: CanonicalTrajectory,
    query_id: str,
) -> None:
    lengths: dict[tuple[str, str], int] = {}
    for event in trajectory.events:
        if isinstance(event, MessageEvent):
            lengths[(event.event_id, "/content")] = len(event.content)
            if event.reasoning_content is not None:
                lengths[(event.event_id, "/reasoning_content")] = len(
                    event.reasoning_content
                )
        elif isinstance(event, ToolCallEvent):
            lengths[(event.event_id, "/raw_arguments")] = len(event.raw_arguments)
        elif isinstance(event, ToolOutputEvent):
            lengths[(event.event_id, "/content")] = len(event.content)
    for span in spans:
        if span.char_start is None or span.char_end is None:
            raise ValueError(f"query_id={query_id} evidence span is not exact: {span}")
        if span.json_pointer is None:
            raise ValueError(
                f"query_id={query_id} evidence span requires json_pointer: {span}"
            )
        key = (span.event_id, span.json_pointer)
        try:
            source_length = lengths[key]
        except KeyError as error:
            raise ValueError(
                f"query_id={query_id} evidence span references unknown source={key}"
            ) from error
        if span.char_end > source_length:
            raise ValueError(
                f"query_id={query_id} evidence span exceeds source length: "
                f"span={span.char_start}:{span.char_end} length={source_length}"
            )


__all__ = [
    "ISETraceBenchmarkSummary",
    "combined_isetrace_records",
    "prepare_isetrace_benchmark",
]
