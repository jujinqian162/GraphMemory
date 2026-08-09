from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import TypeAdapter, ValidationError

from graph_memory.datasets.isetrace.adapter import (
    ISETraceIngestionSummary,
    iter_canonical_trajectories,
)
from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETracePreparedBenchmark,
    ISETraceQueryMetadata,
    ISETraceRankingRecord,
)
from graph_memory.datasets.isetrace.retrieval_views import (
    flat_trajectory_candidates,
    provenance_unit_candidates,
)
from graph_memory.datasets.splits import sample_split
from graph_memory.graphs.provenance import ProvenanceGraph, build_provenance_graph
from graph_memory.query_synthesis.provenance import (
    AuthoringQueryMetadataRecord,
    enumerate_template_supervision,
    memory_mode_for_query_intent,
    render_call_result_supervision,
)
from graph_memory.query_synthesis.provenance.contracts import TemplateSupervisionRecord
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
_QUERY_METADATA_ADAPTER = TypeAdapter(AuthoringQueryMetadataRecord)
NaturalSplitName: TypeAlias = Literal["train", "dev", "test"]
_NATURAL_SPLITS: tuple[NaturalSplitName, ...] = ("train", "dev", "test")


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
    split: NaturalSplitName | None = None,
    trajectory_splits: Mapping[str, Mapping[str, int]] | None = None,
    chunking: TokenChunkingConfig = TokenChunkingConfig(
        tokenizer_name="models/intfloat-e5-base-v2",
        max_tokens=512,
        overlap_tokens=64,
    ),
    tokenizer: OffsetTokenizer | None = None,
    authoring_metadata_source: Path | None = None,
) -> tuple[ISETracePreparedBenchmark, ISETraceBenchmarkSummary]:
    summary = ISETraceBenchmarkSummary()
    _validate_authoring_source_identity(
        query_source,
        trajectory_source=trajectory_source,
        source_revision=source_revision,
    )
    examples = _read_queries(query_source, strict=strict, summary=summary)
    authoring_metadata = (
        _read_authoring_query_metadata(
            authoring_metadata_source,
            query_ids={example.id for example in examples},
            summary=summary,
        )
        if authoring_metadata_source is not None
        else {}
    )
    matched_contexts, trajectory_ids = _match_query_contexts(
        trajectory_source,
        examples=examples,
        source_revision=source_revision,
        strict=strict,
        summary=summary,
    )
    valid_examples = [example for example in examples if example.id in matched_contexts]
    summary["queries_resolved"] = len(valid_examples)
    summary["queries_dropped"] = summary["queries_seen"] - len(valid_examples)
    split_arguments = (split, trajectory_splits)
    if any(value is None for value in split_arguments) and any(
        value is not None for value in split_arguments
    ):
        raise ValueError("split and trajectory_splits must be provided together")
    if split is None:
        requested_natural = count
        requested_template = 0
        natural_trajectory_ids = frozenset(
            matched_contexts[example.id][0].trajectory_id for example in valid_examples
        )
        template_trajectory_ids = frozenset()
    else:
        assert trajectory_splits is not None
        if offset != 0:
            raise ValueError("ISETrace trajectory splits do not support offsets")
        natural_by_split, template_by_split = allocate_trajectory_splits(
            [
                matched_contexts[example.id][0].trajectory_id
                for example in valid_examples
            ],
            split_counts=trajectory_splits,
            split_seed=seed,
            template_trajectory_ids=trajectory_ids,
        )
        natural_trajectory_ids = natural_by_split[split]
        template_trajectory_ids = template_by_split[split]
        requested_natural = None
        requested_template = len(template_trajectory_ids)
    split_pool = [
        example
        for example in valid_examples
        if matched_contexts[example.id][0].trajectory_id in natural_trajectory_ids
    ]
    if split is not None and count is not None:
        available = len(split_pool) + len(template_trajectory_ids)
        if count > available:
            raise ValueError(
                "insufficient ISETrace task pool: "
                f"requested={count} available={available} split={split}"
            )
        requested_natural = min(count, len(split_pool))
        requested_template = count - requested_natural
        template_trajectory_ids = frozenset(
            sorted(template_trajectory_ids)[:requested_template]
        )
    if requested_natural is not None and requested_natural > len(split_pool):
        raise ValueError(
            "insufficient ISETrace natural query pool: "
            f"requested={requested_natural} available={len(split_pool)} split={split}"
        )
    selected = sample_split(
        split_pool,
        count=requested_natural,
        seed=seed,
        offset=offset,
    )
    summary["natural_queries_available"] = len(split_pool)
    summary["natural_trajectories_selected"] = len(natural_trajectory_ids)
    summary["template_trajectories_selected"] = len(template_trajectory_ids)
    summary["natural_queries_requested"] = (
        len(split_pool) if requested_natural is None else requested_natural
    )
    summary["natural_queries_selected"] = len(selected)
    offset_tokenizer = tokenizer or load_offset_tokenizer(chunking.tokenizer_name)

    def chunk_content(text: str):
        return token_chunks(
            text,
            tokenizer=offset_tokenizer,
            max_tokens=chunking.content_tokens,
            overlap_tokens=chunking.overlap_tokens,
        )

    graph_examples = selected
    trajectories, graphs, contexts = _build_selected_contexts(
        graph_examples,
        matched_contexts=matched_contexts,
        extra_trajectory_ids=template_trajectory_ids,
        trajectory_source=trajectory_source,
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
    template_supervision = ()
    template_pool = ()
    if split is not None and requested_template > 0:
        template_pool = enumerate_template_supervision(graphs.values())
        first_by_graph: dict[str, TemplateSupervisionRecord] = {}
        for record in sorted(template_pool, key=lambda item: item.task_id):
            first_by_graph.setdefault(record.graph_id, record)
        for graph_id in sorted(template_trajectory_ids):
            first_by_graph.setdefault(
                graph_id,
                render_call_result_supervision(graphs[graph_id]),
            )
        template_supervision = tuple(
            first_by_graph[graph_id] for graph_id in sorted(template_trajectory_ids)
        )
    summary["template_queries_available"] = len(template_pool)
    summary["template_queries_requested"] = requested_template
    summary["template_queries_selected"] = len(template_supervision)
    summary["queries_selected"] = len(selected) + len(template_supervision)

    rankings: list[ISETraceRankingRecord] = []
    labels: list[ISETraceLabelRecord] = []
    query_metadata: list[ISETraceQueryMetadata] = []
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
        rankings.append(ranking)
        labels.append(label)
        authored = authoring_metadata.get(example.id)
        if authored is not None and authored.trajectory_id != graph_id:
            raise ValueError(
                f"query={example.id!r} authoring metadata trajectory changed: "
                f"recorded={authored.trajectory_id!r} resolved={graph_id!r}"
            )
        query_metadata.append(
            ISETraceQueryMetadata(
                task_id=example.id,
                graph_id=graph_id,
                query_origin="natural",
                memory_mode=None if authored is None else authored.memory_mode,
            )
        )

    for template in template_supervision:
        candidates = provenance_by_graph_id[template.graph_id]
        candidate_by_id = {candidate.item_id: candidate for candidate in candidates}
        positive_spans: list[SourceSpan] = []
        seen_span_keys: set[tuple[object, ...]] = set()
        for candidate_id in template.positive_candidate_ids:
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                raise ValueError(
                    f"template={template.task_id!r} references missing "
                    f"candidate={candidate_id!r}"
                )
            for span in candidate.source_spans:
                key = (
                    span.event_id,
                    span.json_pointer,
                    span.char_start,
                    span.char_end,
                )
                if key not in seen_span_keys:
                    seen_span_keys.add(key)
                    positive_spans.append(span)
        ranking = ISETraceRankingRecord(
            task_id=template.task_id,
            graph_id=template.graph_id,
            query_text=template.query_text,
            flat_candidates=flat_by_graph_id[template.graph_id],
            provenance_candidates=candidates,
        )
        label = ISETraceLabelRecord(
            task_id=template.task_id,
            graph_id=template.graph_id,
            gold_evidence_spans=tuple(positive_spans),
        )
        rankings.append(ranking)
        labels.append(label)
        query_metadata.append(
            ISETraceQueryMetadata(
                task_id=template.task_id,
                graph_id=template.graph_id,
                query_origin="template",
                memory_mode=memory_mode_for_query_intent(template.query_intent),
            )
        )

    selected_graph_ids = {ranking.graph_id for ranking in rankings}
    benchmark = ISETracePreparedBenchmark(
        rankings=tuple(rankings),
        labels=tuple(labels),
        query_metadata=tuple(query_metadata),
        template_supervision=template_supervision,
        provenance_graphs=tuple(
            graphs[graph_id] for graph_id in sorted(selected_graph_ids)
        ),
    )
    summary["unique_graphs"] = len(selected_graph_ids)
    summary["ranking_tasks"] = len(rankings)
    summary["label_tasks"] = len(labels)
    summary["flat_candidates"] = sum(len(item.flat_candidates) for item in rankings)
    summary["provenance_candidates"] = sum(
        len(item.provenance_candidates) for item in rankings
    )
    summary["unique_flat_candidates"] = sum(
        len(flat_by_graph_id[graph_id]) for graph_id in selected_graph_ids
    )
    summary["unique_provenance_candidates"] = sum(
        len(provenance_by_graph_id[graph_id]) for graph_id in selected_graph_ids
    )
    summary["path_supported_tasks"] = 0
    return benchmark, summary


def allocate_trajectory_splits(
    natural_trajectory_ids: Sequence[str] | set[str],
    *,
    split_counts: Mapping[str, Mapping[str, int]],
    split_seed: int,
    template_trajectory_ids: Sequence[str] | set[str] = (),
) -> tuple[
    dict[NaturalSplitName, frozenset[str]],
    dict[NaturalSplitName, frozenset[str]],
]:
    """Select test by query order, then deterministic disjoint train/dev sets."""

    if set(split_counts) != set(_NATURAL_SPLITS):
        raise ValueError("ISETrace trajectory splits must define train, dev, and test")
    normalized: dict[NaturalSplitName, tuple[int, int]] = {}
    for split in _NATURAL_SPLITS:
        counts = split_counts[split]
        if set(counts) != {"natural", "template"}:
            raise ValueError(f"ISETrace trajectory split={split} must define origins")
        normalized[split] = (
            _nonnegative_count(counts["natural"], name=f"{split} natural"),
            _nonnegative_count(counts["template"], name=f"{split} template"),
        )

    ordered_natural = list(dict.fromkeys(natural_trajectory_ids))
    natural_required = sum(counts[0] for counts in normalized.values())
    if natural_required > len(ordered_natural):
        raise ValueError(
            "insufficient ISETrace natural trajectory pool: "
            f"requested={natural_required} available={len(ordered_natural)}"
        )
    test_count = normalized["test"][0]
    test_ids = frozenset(ordered_natural[:test_count])
    remaining_natural = sorted(set(ordered_natural) - test_ids)
    random.Random(split_seed).shuffle(remaining_natural)
    train_count = normalized["train"][0]
    dev_count = normalized["dev"][0]
    natural_by_split: dict[NaturalSplitName, frozenset[str]] = {
        "train": frozenset(remaining_natural[:train_count]),
        "dev": frozenset(
            remaining_natural[train_count : train_count + dev_count]
        ),
        "test": test_ids,
    }

    eligible_template = set(template_trajectory_ids)
    template_by_split: dict[NaturalSplitName, frozenset[str]] = {
        split: frozenset() for split in _NATURAL_SPLITS
    }
    assigned = set(test_ids)
    for split in ("train", "dev"):
        target = normalized[split][1]
        overlap = sorted(
            natural_by_split[split] & eligible_template,
            key=lambda item: _seeded_split_key(split_seed, item),
        )[:target]
        template_by_split[split] = frozenset(overlap)
        assigned.update(natural_by_split[split])
    available = sorted(
        eligible_template - assigned,
        key=lambda item: _seeded_split_key(split_seed, item),
    )
    cursor = 0
    for split in ("train", "dev"):
        target = normalized[split][1]
        current = set(template_by_split[split])
        needed = target - len(current)
        current.update(available[cursor : cursor + needed])
        cursor += needed
        template_by_split[split] = frozenset(current)
    requested_templates = sum(normalized[split][1] for split in ("train", "dev"))
    available_templates = len(eligible_template - test_ids)
    if any(
        len(template_by_split[split]) != normalized[split][1]
        for split in ("train", "dev")
    ):
        raise ValueError(
            "insufficient ISETrace template trajectory pool: "
            f"requested={requested_templates} available={available_templates}"
        )
    return natural_by_split, template_by_split


def _seeded_split_key(seed: int, trajectory_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{trajectory_id}".encode()).hexdigest()


def _nonnegative_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"ISETrace {name} must be a nonnegative integer")
    return value


def _validate_authoring_source_identity(
    query_source: Path,
    *,
    trajectory_source: Path,
    source_revision: str,
) -> None:
    manifest_path = query_source.with_suffix(query_source.suffix + ".run.json")
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ISETrace authoring run metadata: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid ISETrace authoring run metadata: {manifest_path}")
    recorded_revision = manifest.get("source_revision")
    if recorded_revision != source_revision:
        raise ValueError(
            "ISETrace authoring source revision conflicts with the registered "
            f"revision: recorded={recorded_revision!r} registered={source_revision!r}"
        )
    recorded_files = manifest.get("source_files")
    if not isinstance(recorded_files, list) or not recorded_files:
        raise ValueError("ISETrace authoring run metadata has no source_files identity")
    recorded_by_name: dict[str, dict[str, object]] = {}
    for item in recorded_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("ISETrace authoring source_files entries are invalid")
        name = Path(item["path"]).name
        if name in recorded_by_name:
            raise ValueError(f"duplicate ISETrace authoring source filename={name!r}")
        recorded_by_name[name] = item

    source_files = _trajectory_identity_files(trajectory_source)
    actual_by_name = {path.name: path for path in source_files}
    if len(actual_by_name) != len(source_files):
        raise ValueError("ISETrace trajectory source filenames must be unique")
    if set(recorded_by_name) != set(actual_by_name):
        raise ValueError(
            "ISETrace authoring source files conflict with the configured trajectory source"
        )
    for name, path in actual_by_name.items():
        recorded = recorded_by_name[name]
        recorded_size = recorded.get("bytes")
        recorded_digest = recorded.get("sha256")
        if recorded_size != path.stat().st_size or not isinstance(recorded_digest, str):
            raise ValueError(f"ISETrace authoring source identity conflicts for {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != recorded_digest:
            raise ValueError(f"ISETrace authoring source digest conflicts for {name}")


def _trajectory_identity_files(source: Path) -> tuple[Path, ...]:
    if source.is_file():
        return (source,)
    root = source / "trajectories"
    if not root.is_dir():
        root = source
    files = tuple(sorted(path for path in root.rglob("*.jsonl") if path.is_file()))
    if not files:
        raise FileNotFoundError(
            f"ISETrace trajectory directory contains no JSONL shards: {source}"
        )
    return files


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
    summary["queries_parsed"] = len(result)
    ids = [example.id for example in result]
    if len(ids) != len(set(ids)):
        raise ValueError("ISETrace query IDs must be unique")
    return result


_SourceMapping = dict[str, tuple[str, str, str]]
_MatchedContext = tuple[CanonicalTrajectory, _SourceMapping]


def _read_authoring_query_metadata(
    source: Path,
    *,
    query_ids: set[str],
    summary: ISETraceBenchmarkSummary,
) -> dict[str, AuthoringQueryMetadataRecord]:
    records: dict[str, AuthoringQueryMetadataRecord] = {}
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = _QUERY_METADATA_ADAPTER.validate_json(line)
            except (ValidationError, ValueError) as error:
                raise ValueError(
                    f"invalid ISETrace authoring metadata at {source}:{line_number}: "
                    f"{error}"
                ) from error
            if record.query_id in records:
                raise ValueError(
                    f"duplicate ISETrace authoring metadata query_id={record.query_id!r}"
                )
            records[record.query_id] = record
    if set(records) != query_ids:
        missing = sorted(query_ids - set(records))
        extra = sorted(set(records) - query_ids)
        raise ValueError(
            "ISETrace authoring queries and metadata must align exactly: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    summary["authoring_metadata_records"] = len(records)
    return records


def _match_query_contexts(
    trajectory_source: Path,
    *,
    examples: list[AuthoringQueryRecord],
    source_revision: str,
    strict: bool,
    summary: ISETraceBenchmarkSummary,
) -> tuple[dict[str, _MatchedContext], set[str]]:
    matches_by_query: dict[str, list[_MatchedContext]] = {
        example.id: [] for example in examples
    }
    examples_by_id = {example.id: example for example in examples}
    queries_by_signature: dict[tuple[object, ...], set[str]] = defaultdict(set)
    invalid_task_ids: set[str] = set()
    uncompilable_candidate_ids: set[str] = set()
    trajectory_ids: set[str] = set()
    for example in examples:
        try:
            signature = _task_signature(example)
        except ValueError:
            if strict:
                raise
            invalid_task_ids.add(example.id)
            summary["queries_invalid_task_text"] += 1
            continue
        queries_by_signature[signature].add(example.id)

    ingestion = ISETraceIngestionSummary()
    for trajectory in iter_canonical_trajectories(
        trajectory_source,
        source_revision=source_revision,
        strict=False,
        summary=ingestion,
    ):
        if any(output.content for output in trajectory.tool_outputs):
            trajectory_ids.add(trajectory.trajectory_id)
        intents = tuple(intent.text for intent in trajectory.intents)
        outputs = {output.call_id: output for output in trajectory.tool_outputs}
        candidate_ids: set[str] = set()
        for call in trajectory.tool_calls:
            output = outputs.get(call.call_id)
            if output is None:
                continue
            candidate_ids.update(
                queries_by_signature.get(
                    (intents, call.raw_arguments, output.content), ()
                )
            )
        for query_id in candidate_ids:
            mapping = _match_task_text(examples_by_id[query_id], trajectory)
            if mapping is None:
                uncompilable_candidate_ids.add(query_id)
            else:
                matches_by_query[query_id].append((trajectory, mapping))

    summary["trajectories_seen"] = ingestion.records_seen
    summary["trajectories_accepted"] = ingestion.records_accepted
    summary["trajectories_rejected"] = ingestion.records_rejected
    for reason, value in ingestion.rejection_reasons.items():
        summary[f"trajectory_rejected_{reason}"] = value

    missing = sorted(
        query_id
        for query_id, values in matches_by_query.items()
        if query_id not in invalid_task_ids and not values
    )
    ambiguous = {
        query_id: [trajectory.trajectory_id for trajectory, _ in values]
        for query_id, values in matches_by_query.items()
        if len(values) > 1
    }
    if strict and missing:
        raise ValueError(f"ISETrace v7 queries do not match a trajectory: {missing}")
    if strict and ambiguous:
        raise ValueError(
            f"ISETrace v7 queries match multiple trajectories: {ambiguous}"
        )
    uncompilable = {
        query_id for query_id in missing if query_id in uncompilable_candidate_ids
    }
    summary["queries_uncompilable"] = len(uncompilable)
    summary["queries_unmatched"] = len(set(missing) - uncompilable)
    summary["queries_ambiguous"] = len(ambiguous)
    return (
        {
            query_id: values[0]
            for query_id, values in matches_by_query.items()
            if len(values) == 1
        },
        trajectory_ids,
    )


def _build_selected_contexts(
    examples: list[AuthoringQueryRecord],
    *,
    matched_contexts: dict[str, _MatchedContext],
    extra_trajectory_ids: frozenset[str],
    trajectory_source: Path,
    source_revision: str,
    content_chunker,
) -> tuple[
    dict[str, CanonicalTrajectory],
    dict[str, ProvenanceGraph],
    dict[str, tuple[str, _SourceMapping]],
]:
    trajectories: dict[str, CanonicalTrajectory] = {}
    graphs: dict[str, ProvenanceGraph] = {}
    contexts: dict[str, tuple[str, _SourceMapping]] = {}
    for example in examples:
        trajectory, mapping = matched_contexts[example.id]
        graph = graphs.get(trajectory.trajectory_id)
        if graph is None:
            graph = build_provenance_graph(
                trajectory,
                content_chunker=content_chunker,
            )
            trajectories[trajectory.trajectory_id] = trajectory
            graphs[graph.graph_id] = graph
        contexts[example.id] = (graph.graph_id, mapping)
    missing = set(extra_trajectory_ids) - set(trajectories)
    if missing:
        for trajectory in iter_canonical_trajectories(
            trajectory_source,
            source_revision=source_revision,
            strict=False,
        ):
            if trajectory.trajectory_id not in missing:
                continue
            graph = build_provenance_graph(
                trajectory,
                content_chunker=content_chunker,
            )
            trajectories[trajectory.trajectory_id] = trajectory
            graphs[graph.graph_id] = graph
            missing.remove(trajectory.trajectory_id)
            if not missing:
                break
    if missing:
        raise ValueError(
            "selected ISETrace trajectories are missing from the source: "
            f"count={len(missing)} first={min(missing)!r}"
        )
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
    "allocate_trajectory_splits",
    "prepare_isetrace_benchmark",
]
