from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from graph_memory.datasets.isetrace.adapter import iter_canonical_trajectories
from graph_memory.datasets.isetrace.benchmark_records import (
    CombinedISETraceBenchmarkRecord,
    ISETraceLabelRecord,
    ISETracePreparedBenchmark,
    ISETraceRankingRecord,
)
from graph_memory.datasets.isetrace.retrieval_views import (
    flat_trajectory_candidates,
    provenance_unit_candidates,
)
from graph_memory.datasets.splits import sample_split
from graph_memory.graphs.provenance import ProvenanceGraph, build_provenance_graph
from graph_memory.query_synthesis.provenance.authoring import (
    AuthoringQueryRecord,
    AuthoringSource,
    parse_task_intents,
    parse_task_sources,
    resolve_gold_quotes,
)
from graph_memory.text.chunking import (
    OffsetTokenizer,
    TokenChunkingConfig,
    load_offset_tokenizer,
    token_chunks,
)
from graph_memory.trajectories import CanonicalTrajectory, SourceSpan

_QUERY_ADAPTER = TypeAdapter(AuthoringQueryRecord)


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
    chunking: TokenChunkingConfig = TokenChunkingConfig(
        tokenizer_name="models/intfloat-e5-base-v2",
        max_tokens=512,
        overlap_tokens=64,
    ),
    tokenizer: OffsetTokenizer | None = None,
) -> tuple[ISETracePreparedBenchmark, ISETraceBenchmarkSummary]:
    summary = ISETraceBenchmarkSummary()
    examples = _read_queries(query_source, strict=strict, summary=summary)
    selected = sample_split(examples, count=count, seed=seed, offset=offset)
    summary["queries_selected"] = len(selected)
    offset_tokenizer = tokenizer or load_offset_tokenizer(chunking.tokenizer_name)

    def chunk_content(text: str):
        return token_chunks(
            text,
            tokenizer=offset_tokenizer,
            max_tokens=chunking.content_tokens,
            overlap_tokens=chunking.overlap_tokens,
        )

    trajectories, graphs, contexts = _load_contexts(
        trajectory_source,
        examples=selected,
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
        graph_id = contexts[example.id][0]
        spans = _resolve_gold_spans(example, contexts[example.id][1])
        ranking = ISETraceRankingRecord(
            task_id=example.id,
            graph_id=graph_id,
            query_text=example.query,
            flat_candidates=flat_by_graph_id[graph_id],
            provenance_candidates=provenance_by_graph_id[graph_id],
        )
        label = ISETraceLabelRecord(
            task_id=example.id,
            graph_id=graph_id,
            gold_evidence_spans=spans,
        )
        CombinedISETraceBenchmarkRecord(ranking=ranking, label=label)
        rankings.append(ranking)
        labels.append(label)

    benchmark = ISETracePreparedBenchmark(
        rankings=tuple(rankings),
        labels=tuple(labels),
        provenance_graphs=tuple(graphs[graph_id] for graph_id in sorted(graphs)),
    )
    summary["unique_graphs"] = len(graphs)
    summary["ranking_tasks"] = len(rankings)
    summary["label_tasks"] = len(labels)
    summary["flat_candidates"] = sum(len(item.flat_candidates) for item in rankings)
    summary["provenance_candidates"] = sum(
        len(item.provenance_candidates) for item in rankings
    )
    summary["unique_flat_candidates"] = sum(
        len(flat_by_graph_id[graph_id]) for graph_id in graphs
    )
    summary["unique_provenance_candidates"] = sum(
        len(provenance_by_graph_id[graph_id]) for graph_id in graphs
    )
    summary["path_supported_tasks"] = 0
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
) -> list[AuthoringQueryRecord]:
    result: list[AuthoringQueryRecord] = []
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
                        f"invalid v7 query at {source}:{line_number}: {error}"
                    ) from error
    ids = [example.id for example in result]
    if len(ids) != len(set(ids)):
        raise ValueError("ISETrace query IDs must be unique")
    return result


def _load_contexts(
    trajectory_source: Path,
    *,
    examples: list[AuthoringQueryRecord],
    source_revision: str,
    content_chunker,
) -> tuple[
    dict[str, CanonicalTrajectory],
    dict[str, ProvenanceGraph],
    dict[str, tuple[str, dict[str, tuple[str, str, str]]]],
]:
    matches_by_query: dict[
        str, list[tuple[CanonicalTrajectory, dict[str, tuple[str, str, str]]]]
    ] = {example.id: [] for example in examples}
    examples_by_id = {example.id: example for example in examples}
    queries_by_signature: dict[tuple[object, ...], set[str]] = defaultdict(set)
    for example in examples:
        queries_by_signature[_task_signature(example)].add(example.id)

    for trajectory in iter_canonical_trajectories(
        trajectory_source,
        source_revision=source_revision,
        strict=False,
    ):
        intents = tuple(intent.text for intent in trajectory.intents)
        outputs = {output.call_id: output for output in trajectory.tool_outputs}
        candidate_ids: set[str] = set()
        for call in trajectory.tool_calls:
            output = outputs[call.call_id]
            candidate_ids.update(
                queries_by_signature.get(
                    (intents, call.raw_arguments, output.content), ()
                )
            )
        for query_id in candidate_ids:
            mapping = _match_task_text(examples_by_id[query_id], trajectory)
            if mapping is not None:
                matches_by_query[query_id].append((trajectory, mapping))

    missing = sorted(
        query_id for query_id, values in matches_by_query.items() if not values
    )
    ambiguous = {
        query_id: [trajectory.trajectory_id for trajectory, _ in values]
        for query_id, values in matches_by_query.items()
        if len(values) > 1
    }
    if missing:
        raise ValueError(f"ISETrace v7 queries do not match a trajectory: {missing}")
    if ambiguous:
        raise ValueError(
            f"ISETrace v7 queries match multiple trajectories: {ambiguous}"
        )

    trajectories: dict[str, CanonicalTrajectory] = {}
    graphs: dict[str, ProvenanceGraph] = {}
    contexts: dict[str, tuple[str, dict[str, tuple[str, str, str]]]] = {}
    for query_id, values in matches_by_query.items():
        trajectory, mapping = values[0]
        graph = graphs.get(trajectory.trajectory_id)
        if graph is None:
            graph = build_provenance_graph(
                trajectory,
                content_chunker=content_chunker,
            )
            trajectories[trajectory.trajectory_id] = trajectory
            graphs[graph.graph_id] = graph
        contexts[query_id] = (graph.graph_id, mapping)
    return trajectories, graphs, contexts


def _task_signature(example: AuthoringQueryRecord) -> tuple[object, ...]:
    sources = {source.handle: source for source in parse_task_sources(example.text)}
    indices = sorted(
        int(handle[1:])
        for handle in sources
        if handle.startswith("A") and f"E{handle[1:]}" in sources
    )
    if not indices:
        raise ValueError(f"query={example.id} task text has no paired A/E source")
    first = indices[0]
    return (
        parse_task_intents(example.text),
        sources[f"A{first}"].text,
        sources[f"E{first}"].text,
    )


def _match_task_text(
    example: AuthoringQueryRecord,
    trajectory: CanonicalTrajectory,
) -> dict[str, tuple[str, str, str]] | None:
    if parse_task_intents(example.text) != tuple(
        intent.text for intent in trajectory.intents
    ):
        return None
    sources = {source.handle: source for source in parse_task_sources(example.text)}
    call_by_id = {call.call_id: call for call in trajectory.tool_calls}
    output_by_call_id = {output.call_id: output for output in trajectory.tool_outputs}
    mapping: dict[str, tuple[str, str, str]] = {}
    indices = sorted({int(handle[1:]) for handle in sources})
    for index in indices:
        call_source = sources.get(f"A{index}")
        output_source = sources.get(f"E{index}")
        if call_source is None or output_source is None:
            return None
        matches = [
            (call, output_by_call_id.get(call.call_id))
            for call in call_by_id.values()
            if call.raw_arguments == call_source.text
        ]
        matches = [
            (call, output)
            for call, output in matches
            if output is not None and output.content == output_source.text
        ]
        if len(matches) != 1:
            return None
        call, output = matches[0]
        assert output is not None
        mapping[f"A{index}"] = (call.event_id, "/raw_arguments", call.raw_arguments)
        mapping[f"E{index}"] = (output.event_id, "/content", output.content)
    return mapping


def _resolve_gold_spans(
    example: AuthoringQueryRecord,
    sources: dict[str, tuple[str, str, str]],
) -> tuple[SourceSpan, ...]:
    resolved = resolve_gold_quotes(
        example.gold,
        tuple(
            AuthoringSource(
                handle=handle,
                kind="tool_call" if handle.startswith("A") else "tool_output",
                event_id=event_id,
                json_pointer=pointer,
                text=text,
            )
            for handle, (event_id, pointer, text) in sources.items()
        ),
    )
    return tuple(item.span for item in resolved)


__all__ = [
    "ISETraceBenchmarkSummary",
    "combined_isetrace_records",
    "prepare_isetrace_benchmark",
]
