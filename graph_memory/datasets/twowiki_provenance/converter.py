from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from graph_memory.datasets.twowiki import (
    convert_twowiki_example,
    parse_twowiki_example,
)
from graph_memory.contracts.common import JsonValue
from graph_memory.datasets.twowiki.records import TwoWikiCandidateSentence
from graph_memory.datasets.twowiki_provenance.records import (
    TWOWIKI_PROVENANCE_SCHEMA_VERSION,
    ConvertedTwoWikiProvenanceExample,
    ProvenanceBindingRecord,
    ProvenanceCandidateRecord,
    ProvenanceEdgeRecord,
    ProvenanceNodeRecord,
    TwoWikiProvenanceConversionResult,
    TwoWikiProvenanceLabelRecord,
    TwoWikiProvenanceRankingRecord,
    TwoWikiProvenanceRawRecord,
)
from graph_memory.datasets.twowiki_provenance.scoring import (
    ProvenanceGraphConstructionConfig,
    ProvenanceSemanticRanker,
)
from graph_memory.retrieval.contracts import RankedNode, SeedRanker
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest

MINIMUM_CANDIDATES = 4


@dataclass(frozen=True)
class DenseRankerFactory:
    """Picklable description that lets each worker rebuild its own dense ranker.

    Torch-backed rankers cannot cross a process boundary, so the parallel path
    ships this instead of a live ranker and materializes one ranker per worker.
    """

    model_name: str
    query_prefix: str
    passage_prefix: str
    batch_size: int

    def build(self, *, device: str | None) -> SeedRanker:
        from graph_memory.retrieval.methods.flat.dense import (
            DenseConfig,
            DenseTaskRetriever,
        )

        return DenseTaskRetriever(
            config=DenseConfig(
                model_name=self.model_name,
                query_prefix=self.query_prefix,
                passage_prefix=self.passage_prefix,
                batch_size=self.batch_size,
                device=device,
            ),
            device=device,
        )


def convert_twowiki_source_records(
    raw_records: Sequence[object],
    *,
    candidate_cap: int = 32,
    seed: int = 13,
    strict: bool = False,
    graph_config: ProvenanceGraphConstructionConfig | None = None,
    dense_ranker: SeedRanker | None = None,
    workers: int | None = None,
    dense_ranker_factory: DenseRankerFactory | None = None,
    devices: Sequence[str] | None = None,
) -> TwoWikiProvenanceConversionResult:
    if candidate_cap < MINIMUM_CANDIDATES:
        raise ValueError(f"candidate_cap must be at least {MINIMUM_CANDIDATES}.")
    resolved_graph_config = graph_config or ProvenanceGraphConstructionConfig()

    resolved_workers = workers if workers is not None else 1
    if resolved_workers > 1 and len(raw_records) > 1:
        return _convert_in_parallel(
            raw_records,
            candidate_cap=candidate_cap,
            seed=seed,
            strict=strict,
            graph_config=resolved_graph_config,
            dense_ranker_factory=dense_ranker_factory,
            workers=resolved_workers,
            devices=tuple(devices) if devices else (),
        )

    semantic_ranker = ProvenanceSemanticRanker(
        resolved_graph_config, dense_ranker=dense_ranker
    )
    records: list[TwoWikiProvenanceRawRecord] = []
    rejected: Counter[str] = Counter()
    for index, raw_record in enumerate(raw_records):
        try:
            records.append(
                convert_twowiki_source_record(
                    raw_record,
                    record_index=index,
                    candidate_cap=candidate_cap,
                    seed=seed,
                    graph_config=resolved_graph_config,
                    semantic_ranker=semantic_ranker,
                ).raw_record
            )
        except ValueError as error:
            if strict:
                raise
            rejected[_rejection_reason(error)] += 1
    records.sort(key=lambda record: record["ranking"]["task_id"])
    return TwoWikiProvenanceConversionResult(
        records=records,
        rejected_reason_counts=dict(sorted(rejected.items())),
    )


_worker_ranker: ProvenanceSemanticRanker | None = None
_worker_state: _WorkerConfig | None = None


@dataclass(frozen=True)
class _WorkerConfig:
    graph_config: ProvenanceGraphConstructionConfig
    candidate_cap: int
    seed: int
    strict: bool
    dense_ranker_factory: DenseRankerFactory | None
    devices: tuple[str, ...]


def _init_worker(config: _WorkerConfig) -> None:
    # Each worker owns one core so N processes do not oversubscribe the CPU
    # while torch also tries to fan matmuls across every core.
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass

    device: str | None = None
    if config.devices:
        slot = _worker_slot() % len(config.devices)
        device = config.devices[slot]

    dense_ranker: SeedRanker | None = None
    if config.dense_ranker_factory is not None:
        dense_ranker = config.dense_ranker_factory.build(device=device)

    global _worker_ranker, _worker_state
    _worker_state = config
    _worker_ranker = ProvenanceSemanticRanker(
        config.graph_config, dense_ranker=dense_ranker
    )


def _worker_slot() -> int:
    # ProcessPoolExecutor names workers "SpawnProcess-<n>"/"ForkProcess-<n>";
    # the trailing integer gives each worker a stable slot for device binding.
    import multiprocessing

    name = multiprocessing.current_process().name
    tail = name.rsplit("-", 1)[-1]
    return int(tail) - 1 if tail.isdigit() else 0


def _convert_chunk(
    indexed_records: Sequence[tuple[int, object]],
) -> tuple[list[TwoWikiProvenanceRawRecord], dict[str, int]]:
    assert _worker_ranker is not None and _worker_state is not None
    state = _worker_state
    records: list[TwoWikiProvenanceRawRecord] = []
    rejected: Counter[str] = Counter()
    for index, raw_record in indexed_records:
        try:
            records.append(
                convert_twowiki_source_record(
                    raw_record,
                    record_index=index,
                    candidate_cap=state.candidate_cap,
                    seed=state.seed,
                    graph_config=state.graph_config,
                    semantic_ranker=_worker_ranker,
                ).raw_record
            )
        except ValueError as error:
            if state.strict:
                raise
            rejected[_rejection_reason(error)] += 1
    return records, dict(rejected)


def _convert_in_parallel(
    raw_records: Sequence[object],
    *,
    candidate_cap: int,
    seed: int,
    strict: bool,
    graph_config: ProvenanceGraphConstructionConfig,
    dense_ranker_factory: DenseRankerFactory | None,
    workers: int,
    devices: tuple[str, ...],
) -> TwoWikiProvenanceConversionResult:
    if graph_config.strategy != "bm25" and dense_ranker_factory is None:
        raise ValueError(
            "Parallel conversion with a dense/hybrid scorer requires a "
            "dense_ranker_factory; a live ranker cannot cross a process boundary."
        )
    worker_config = _WorkerConfig(
        graph_config=graph_config,
        candidate_cap=candidate_cap,
        seed=seed,
        strict=strict,
        dense_ranker_factory=dense_ranker_factory,
        devices=devices,
    )
    indexed = list(enumerate(raw_records))
    chunk_count = min(workers, len(indexed))
    chunks = [indexed[start::chunk_count] for start in range(chunk_count)]

    records: list[TwoWikiProvenanceRawRecord] = []
    rejected: Counter[str] = Counter()
    with ProcessPoolExecutor(
        max_workers=chunk_count,
        initializer=_init_worker,
        initargs=(worker_config,),
    ) as pool:
        for chunk_records, chunk_rejected in pool.map(_convert_chunk, chunks):
            records.extend(chunk_records)
            rejected.update(chunk_rejected)
    records.sort(key=lambda record: record["ranking"]["task_id"])
    return TwoWikiProvenanceConversionResult(
        records=records,
        rejected_reason_counts=dict(sorted(rejected.items())),
    )


def resolve_worker_count(workers: int | None) -> int:
    # None means "unset" -> serial. Auto-fanning across every core made a
    # `workers: null` config silently spawn os.cpu_count() model-loading
    # processes; opting into parallelism must be explicit.
    if workers is None:
        return 1
    return max(1, workers)


def audit_twowiki_source_records(
    raw_records: Sequence[object],
    *,
    candidate_cap: int = 32,
    seed: int = 13,
    graph_config: ProvenanceGraphConstructionConfig | None = None,
    dense_ranker: SeedRanker | None = None,
) -> tuple[int, dict[str, int]]:
    resolved_graph_config = graph_config or ProvenanceGraphConstructionConfig()
    semantic_ranker = ProvenanceSemanticRanker(
        resolved_graph_config, dense_ranker=dense_ranker
    )
    accepted = 0
    rejected: Counter[str] = Counter()
    for index, raw_record in enumerate(raw_records):
        try:
            convert_twowiki_source_record(
                raw_record,
                record_index=index,
                candidate_cap=candidate_cap,
                seed=seed,
                graph_config=resolved_graph_config,
                semantic_ranker=semantic_ranker,
            )
            accepted += 1
        except ValueError as error:
            rejected[_rejection_reason(error)] += 1
    return accepted, dict(sorted(rejected.items()))


def convert_twowiki_source_record(
    raw_record: object,
    *,
    record_index: int | None = None,
    candidate_cap: int = 32,
    seed: int = 13,
    graph_config: ProvenanceGraphConstructionConfig | None = None,
    semantic_ranker: ProvenanceSemanticRanker | None = None,
) -> ConvertedTwoWikiProvenanceExample:
    resolved_graph_config = graph_config or ProvenanceGraphConstructionConfig()
    resolved_ranker = semantic_ranker or ProvenanceSemanticRanker(
        resolved_graph_config
    )
    example = parse_twowiki_example(raw_record, record_index=record_index)
    support_keys = [
        (support.title, support.sentence_id) for support in example.supporting_facts
    ]
    if len(support_keys) != len(set(support_keys)):
        raise ValueError("duplicated_gold_support")
    converted = convert_twowiki_example(example)
    if converted.label_record["metadata"]["mapping_ambiguity_count"]:
        raise ValueError("ambiguous_gold_chain")
    gold_ids = converted.label_record["gold_evidence_sentence_ids"]
    gold_edges = converted.label_record["gold_dependency_edges"]
    if len(gold_ids) != 2:
        raise ValueError("unsupported_gold_evidence_count")
    if len(gold_edges) != 1:
        raise ValueError("unrecoverable_ordered_gold_chain")
    gold_source, gold_target = gold_edges[0]
    if gold_source == gold_target or {gold_source, gold_target} != set(gold_ids):
        raise ValueError("invalid_ordered_gold_chain")

    candidates_by_source_id = {
        candidate["sentence_id"]: candidate
        for candidate in converted.ranking_record["candidate_sentences"]
    }
    if (
        gold_source not in candidates_by_source_id
        or gold_target not in candidates_by_source_id
    ):
        raise ValueError("gold_candidate_missing")
    selected = _select_candidates(
        converted.ranking_record["candidate_sentences"],
        question=example.question,
        gold_ids={gold_source, gold_target},
        candidate_cap=candidate_cap,
        ranker=resolved_ranker,
    )
    if len(selected) < MINIMUM_CANDIDATES:
        raise ValueError("insufficient_branch_candidates")

    task_id = f"2wiki_provenance_{example.raw_id}"
    id_map = {
        candidate["sentence_id"]: _node_ids(example.raw_id, candidate["sentence_id"])
        for candidate in selected
    }
    gold_output_source = id_map[gold_source][1]
    gold_output_target = id_map[gold_target][1]

    candidate_order = list(selected)
    _rng(seed, example.raw_id, "candidates").shuffle(candidate_order)
    candidate_records: list[ProvenanceCandidateRecord] = []
    for position, candidate in enumerate(candidate_order):
        call_id, output_id = id_map[candidate["sentence_id"]]
        candidate_records.append(
            {
                "output_id": output_id,
                "call_id": call_id,
                "title": candidate["title"],
                "sentence_index": candidate["sentence_index"],
                "position": position,
                "text": candidate["text"],
            }
        )

    nodes = _graph_nodes(task_id, example.question, candidate_records)
    edges, gold_semantic_rank, gold_rank_bucket, gold_branch_role, gold_weight = (
        _graph_edges(
        example.raw_id,
        example.question,
        candidate_records,
        gold_output_source=gold_output_source,
        gold_output_target=gold_output_target,
        seed=seed,
        graph_config=resolved_graph_config,
        ranker=resolved_ranker,
        )
    )
    ranking: TwoWikiProvenanceRankingRecord = {
        "task_id": task_id,
        "question": example.question,
        "question_type": example.question_type,
        "candidates": candidate_records,
        "graph": {"task_id": task_id, "nodes": nodes, "edges": edges},
        "metadata": {
            "dataset": "twowiki_provenance",
            "source_dataset": "2wiki",
            "source_raw_id": example.raw_id,
            "synthetic_execution_graph": True,
            "schema_version": TWOWIKI_PROVENANCE_SCHEMA_VERSION,
            "graph_construction": resolved_graph_config.identity(),
        },
    }
    label: TwoWikiProvenanceLabelRecord = {
        "task_id": task_id,
        "gold_answer": example.answer,
        "gold_evidence_output_ids": [gold_output_source, gold_output_target],
        "gold_dependency_edges": [[gold_output_source, gold_output_target]],
        "metadata": {
            "dataset": "twowiki_provenance",
            "question_type": example.question_type,
            "source_path_label_source": converted.label_record["metadata"][
                "path_label_source"
            ],
            "gold_edge_semantic_rank": gold_semantic_rank,
            "gold_edge_rank_bucket": gold_rank_bucket,
            "gold_edge_branch_role": gold_branch_role,
            "gold_edge_is_head": gold_branch_role == "semantic_head",
            "gold_edge_calibrated_weight": gold_weight,
        },
    }
    return ConvertedTwoWikiProvenanceExample(
        raw_record={
            "schema_version": TWOWIKI_PROVENANCE_SCHEMA_VERSION,
            "ranking": ranking,
            "label": label,
        }
    )


def deterministic_dev_test_partition(
    records: Sequence[TwoWikiProvenanceRawRecord],
    *,
    seed: int,
    dev_fraction: float = 0.5,
) -> tuple[list[TwoWikiProvenanceRawRecord], list[TwoWikiProvenanceRawRecord]]:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1.")
    ordered = sorted(records, key=lambda record: record["ranking"]["task_id"])
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    boundary = max(1, min(len(shuffled) - 1, round(len(shuffled) * dev_fraction)))
    return (
        sorted(shuffled[:boundary], key=lambda record: record["ranking"]["task_id"]),
        sorted(shuffled[boundary:], key=lambda record: record["ranking"]["task_id"]),
    )


def _select_candidates(
    candidates: Sequence[TwoWikiCandidateSentence],
    *,
    question: str,
    gold_ids: set[str],
    candidate_cap: int,
    ranker: ProvenanceSemanticRanker,
) -> list[TwoWikiCandidateSentence]:
    gold = [
        candidate for candidate in candidates if candidate["sentence_id"] in gold_ids
    ]
    negatives = [
        candidate
        for candidate in candidates
        if candidate["sentence_id"] not in gold_ids
    ]
    request = TextRankingRequest(
        task_id="candidate_selection",
        query_text=question,
        candidates=tuple(
            TextCandidate(
                item_id=candidate["sentence_id"],
                text=f"{candidate['title']}. {candidate['text']}",
                metadata={},
            )
            for candidate in negatives
        ),
    )
    ranked_ids = [item.node_id for item in ranker.rank(request)] if negatives else []
    negative_by_id = {candidate["sentence_id"]: candidate for candidate in negatives}
    selected = [
        *gold,
        *[
            negative_by_id[node_id]
            for node_id in ranked_ids[: max(0, candidate_cap - len(gold))]
        ],
    ]
    return selected


def _graph_nodes(
    task_id: str,
    question: str,
    candidates: Sequence[ProvenanceCandidateRecord],
) -> list[ProvenanceNodeRecord]:
    nodes: list[ProvenanceNodeRecord] = [
        {
            "node_id": f"task_{_digest(task_id)}",
            "node_type": "task",
            "text": question,
            "metadata": {"role": "query"},
        },
        {
            "node_id": f"agent_{_digest(task_id)}",
            "node_type": "agent",
            "text": "evidence retrieval agent",
            "metadata": {"role": "retriever"},
        },
    ]
    for candidate in candidates:
        nodes.extend(
            (
                {
                    "node_id": candidate["call_id"],
                    "node_type": "tool_call",
                    "text": f"retrieve evidence from {candidate['title']}",
                    "metadata": {
                        "tool_name": "retrieve_evidence",
                        "source_ref": candidate["title"],
                        "input_parameters": ["context"],
                    },
                },
                {
                    "node_id": candidate["output_id"],
                    "node_type": "tool_output",
                    "text": f"{candidate['title']}. {candidate['text']}",
                    "metadata": {
                        "source_ref": candidate["title"],
                        "sentence_index": candidate["sentence_index"],
                        "output_field_hashes": {
                            "evidence": _candidate_field_hash(candidate)
                        },
                    },
                },
            )
        )
    return nodes


def _graph_edges(
    raw_id: str,
    question: str,
    candidates: Sequence[ProvenanceCandidateRecord],
    *,
    gold_output_source: str,
    gold_output_target: str,
    seed: int,
    graph_config: ProvenanceGraphConstructionConfig,
    ranker: ProvenanceSemanticRanker,
) -> tuple[list[ProvenanceEdgeRecord], int, str, str, float]:
    task_id = f"task_{_digest(f'2wiki_provenance_{raw_id}')}"
    agent_id = f"agent_{_digest(f'2wiki_provenance_{raw_id}')}"
    by_output = {candidate["output_id"]: candidate for candidate in candidates}
    edges: list[ProvenanceEdgeRecord] = [
        _edge(task_id, agent_id, "contains"),
    ]
    for candidate in candidates:
        edges.extend(
            (
                _edge(task_id, candidate["call_id"], "contains"),
                _edge(task_id, candidate["output_id"], "contains"),
                _edge(agent_id, candidate["call_id"], "invokes"),
                _edge(candidate["call_id"], candidate["output_id"], "returns"),
            )
        )

    source_requests: list[tuple[str, TextRankingRequest]] = []
    for source_output in sorted(by_output):
        source = by_output[source_output]
        target_candidates = tuple(
            TextCandidate(
                item_id=target["output_id"],
                text=f"{target['title']}. {target['text']}",
                metadata={},
            )
            for target in candidates
            if target["output_id"] != source_output
        )
        request = TextRankingRequest(
            task_id=f"{raw_id}:{source_output}",
            query_text=_source_query(
                question,
                source,
                query_template_version=graph_config.query_template_version,
            ),
            candidates=target_candidates,
        )
        source_requests.append((source_output, request))
    source_rankings: dict[str, _SourceRanking] = {}
    ranked_requests = ranker.rank_many([request for _, request in source_requests])
    for (source_output, _), ranked in zip(
        source_requests, ranked_requests, strict=True
    ):
        if len(ranked) < graph_config.successors_per_output:
            raise ValueError("insufficient_branch_candidates")
        source_rankings[source_output] = _SourceRanking(
            ranked=tuple(ranked),
            normalized_scores=_normalized_scores(ranked),
        )

    selected_by_source: dict[str, list[_SelectedSuccessor]] = {}
    gold_ranking = source_rankings[gold_output_source]
    gold_semantic_rank, _, _ = gold_ranking.details(gold_output_target)
    gold_bucket = "head" if gold_semantic_rank == 1 else _bucket_for_rank(
        gold_semantic_rank, graph_config
    )
    if gold_bucket is None:
        raise ValueError("unmatched_gold_branch_bucket")

    for source_output in sorted(by_output):
        ranking = source_rankings[source_output]
        head_target = ranking.ranked[0].node_id
        selected = [
            ranking.selected(
                head_target,
                branch_role="semantic_head",
                rank_bucket="head",
            )
        ]
        if source_output == gold_output_source and gold_semantic_rank != 1:
            selected.append(
                ranking.selected(
                    gold_output_target,
                    branch_role="rank_banded_branch",
                    rank_bucket=gold_bucket,
                )
            )
        else:
            selected.append(
                _deterministic_branch(
                    ranking,
                    graph_config=graph_config,
                    seed=seed,
                    raw_id=raw_id,
                    source_output=source_output,
                )
            )
        selected_by_source[source_output] = selected

    if gold_bucket != "head":
        _ensure_matching_non_gold_branch(
            selected_by_source,
            source_rankings=source_rankings,
            gold_output_source=gold_output_source,
            gold_output_target=gold_output_target,
            required_bucket=gold_bucket,
            graph_config=graph_config,
            seed=seed,
            raw_id=raw_id,
        )

    gold_weight = 0.0
    gold_branch_role = ""
    for source_output in sorted(by_output):
        source = by_output[source_output]
        selected = selected_by_source[source_output]
        probabilities = _source_probabilities(
            [item.normalized_score for item in selected],
            temperature=graph_config.semantic_temperature,
        )
        weights = [
            graph_config.weight_floor
            + (1.0 - graph_config.weight_floor) * probability
            for probability in probabilities
        ]
        expected_mass = (
            len(selected) * graph_config.weight_floor
            + (1.0 - graph_config.weight_floor)
        )
        if not math.isclose(sum(weights), expected_mass, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("invalid_source_feed_mass")
        for successor, probability, weight in zip(
            selected, probabilities, weights, strict=True
        ):
            target_output = successor.target_output
            target_call = by_output[target_output]["call_id"]
            edges.append(
                _edge(
                    source_output,
                    target_call,
                    "feeds",
                    binding={
                        "output_field": "evidence",
                        "input_parameter": "context",
                        "binding_value_hash": _candidate_field_hash(source),
                        "binding_kind": "semantic_reference",
                    },
                    weight=weight,
                    metadata={
                        "semantic_scorer": graph_config.strategy,
                        "scorer_identity": graph_config.scorer_identity,
                        "query_template_version": graph_config.query_template_version,
                        "semantic_rank": successor.semantic_rank,
                        "semantic_score": successor.semantic_score,
                        "normalized_score": successor.normalized_score,
                        "source_probability": probability,
                        "calibrated_weight": weight,
                        "branch_role": successor.branch_role,
                        "rank_bucket": successor.rank_bucket,
                    },
                )
            )
            if (
                source_output == gold_output_source
                and target_output == gold_output_target
            ):
                gold_weight = weight
                gold_branch_role = successor.branch_role
    _rng(seed, raw_id, "edges").shuffle(edges)
    if not gold_branch_role:
        raise ValueError("gold_candidate_missing")
    return (
        edges,
        gold_semantic_rank,
        gold_bucket,
        gold_branch_role,
        gold_weight,
    )


@dataclass(frozen=True)
class _SelectedSuccessor:
    target_output: str
    semantic_rank: int
    semantic_score: float
    normalized_score: float
    branch_role: str
    rank_bucket: str


@dataclass(frozen=True)
class _SourceRanking:
    ranked: tuple[RankedNode, ...]
    normalized_scores: dict[str, float]

    def details(self, target_output: str) -> tuple[int, float, float]:
        for rank, item in enumerate(self.ranked, start=1):
            node_id = item.node_id
            if node_id == target_output:
                return rank, item.score, self.normalized_scores[node_id]
        raise ValueError("gold_candidate_missing")

    def selected(
        self,
        target_output: str,
        *,
        branch_role: str,
        rank_bucket: str,
    ) -> _SelectedSuccessor:
        semantic_rank, semantic_score, normalized_score = self.details(target_output)
        return _SelectedSuccessor(
            target_output=target_output,
            semantic_rank=semantic_rank,
            semantic_score=semantic_score,
            normalized_score=normalized_score,
            branch_role=branch_role,
            rank_bucket=rank_bucket,
        )


def _normalized_scores(ranked: Sequence[RankedNode]) -> dict[str, float]:
    scores = [item.score for item in ranked]
    lower = min(scores)
    upper = max(scores)
    if math.isclose(lower, upper):
        denominator = max(1, len(ranked) - 1)
        return {
            item.node_id: 1.0 - index / denominator
            for index, item in enumerate(ranked)
        }
    return {
        item.node_id: (item.score - lower) / (upper - lower)
        for item in ranked
    }


def _bucket_for_rank(
    semantic_rank: int,
    graph_config: ProvenanceGraphConstructionConfig,
) -> str | None:
    for name, (start, end) in graph_config.rank_buckets:
        if semantic_rank >= start and (end is None or semantic_rank <= end):
            return name
    return None


def _bucket_candidates(
    ranking: _SourceRanking,
    *,
    bucket: str,
    graph_config: ProvenanceGraphConstructionConfig,
    excluded_targets: set[str] | None = None,
) -> list[str]:
    bounds = dict(graph_config.rank_buckets)[bucket]
    start, end = bounds
    excluded = excluded_targets or set()
    return [
        item.node_id
        for rank, item in enumerate(ranking.ranked, start=1)
        if rank >= start
        and (end is None or rank <= end)
        and item.node_id not in excluded
    ]


def _deterministic_branch(
    ranking: _SourceRanking,
    *,
    graph_config: ProvenanceGraphConstructionConfig,
    seed: int,
    raw_id: str,
    source_output: str,
    required_bucket: str | None = None,
    excluded_targets: set[str] | None = None,
) -> _SelectedSuccessor:
    feasible = [
        name
        for name, _ in graph_config.rank_buckets
        if _bucket_candidates(
            ranking,
            bucket=name,
            graph_config=graph_config,
            excluded_targets=excluded_targets,
        )
    ]
    if required_bucket is not None:
        feasible = [name for name in feasible if name == required_bucket]
    if not feasible:
        raise ValueError("unmatched_gold_branch_bucket")
    bucket_rng = _rng(seed, raw_id, f"branch-bucket:{source_output}")
    bucket = feasible[bucket_rng.randrange(len(feasible))]
    candidates = _bucket_candidates(
        ranking,
        bucket=bucket,
        graph_config=graph_config,
        excluded_targets=excluded_targets,
    )
    target_rng = _rng(seed, raw_id, f"branch-target:{source_output}:{bucket}")
    target = candidates[target_rng.randrange(len(candidates))]
    return ranking.selected(
        target,
        branch_role="rank_banded_branch",
        rank_bucket=bucket,
    )


def _ensure_matching_non_gold_branch(
    selected_by_source: dict[str, list[_SelectedSuccessor]],
    *,
    source_rankings: Mapping[str, _SourceRanking],
    gold_output_source: str,
    gold_output_target: str,
    required_bucket: str,
    graph_config: ProvenanceGraphConstructionConfig,
    seed: int,
    raw_id: str,
) -> None:
    if any(
        source != gold_output_source
        and successors[1].rank_bucket == required_bucket
        and successors[1].target_output != gold_output_target
        for source, successors in selected_by_source.items()
    ):
        return
    eligible_sources = [
        source
        for source in sorted(source_rankings)
        if source != gold_output_source
        and _bucket_candidates(
            source_rankings[source],
            bucket=required_bucket,
            graph_config=graph_config,
            excluded_targets={gold_output_target},
        )
    ]
    if not eligible_sources:
        raise ValueError("unmatched_gold_branch_bucket")
    source_rng = _rng(seed, raw_id, f"gold-bucket-match:{required_bucket}")
    source = eligible_sources[source_rng.randrange(len(eligible_sources))]
    selected_by_source[source][1] = _deterministic_branch(
        source_rankings[source],
        graph_config=graph_config,
        seed=seed,
        raw_id=raw_id,
        source_output=source,
        required_bucket=required_bucket,
        excluded_targets={gold_output_target},
    )


def _source_probabilities(
    normalized_scores: Sequence[float],
    *,
    temperature: float,
) -> list[float]:
    maximum = max(normalized_scores)
    exponentials = [
        math.exp((score - maximum) / temperature) for score in normalized_scores
    ]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _source_query(
    question: str,
    source: ProvenanceCandidateRecord,
    *,
    query_template_version: str,
) -> str:
    if query_template_version != "question_source_v1":
        raise ValueError(
            f"Unsupported provenance query template={query_template_version!r}."
        )
    return f"{question}\nSource evidence: {source['title']}. {source['text']}"


def _edge(
    source: str,
    target: str,
    edge_type: str,
    *,
    binding: ProvenanceBindingRecord | None = None,
    weight: float = 1.0,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ProvenanceEdgeRecord:
    edge_metadata: dict[str, JsonValue] = {"synthetic": True}
    if metadata is not None:
        edge_metadata.update(metadata)
    return {
        "source": source,
        "target": target,
        "edge_type": edge_type,
        "binding": binding,
        "weight": weight,
        "metadata": edge_metadata,
    }


def _node_ids(raw_id: str, source_id: str) -> tuple[str, str]:
    token = _digest(f"{raw_id}|{source_id}")
    return f"call_{token}", f"output_{token}"


def _candidate_field_hash(candidate: ProvenanceCandidateRecord) -> str:
    return _digest(f"{candidate['title']}\n{candidate['text']}")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _rng(seed: int, raw_id: str, namespace: str) -> random.Random:
    material = f"{seed}|{namespace}|{raw_id}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _rejection_reason(error: ValueError) -> str:
    text = str(error)
    known = {
        "unsupported_gold_evidence_count",
        "unrecoverable_ordered_gold_chain",
        "invalid_ordered_gold_chain",
        "ambiguous_gold_chain",
        "duplicated_gold_support",
        "gold_candidate_missing",
        "insufficient_branch_candidates",
        "unmatched_gold_branch_bucket",
        "invalid_source_feed_mass",
    }
    return text if text in known else "invalid_source_record"


__all__ = [
    "DenseRankerFactory",
    "ProvenanceGraphConstructionConfig",
    "audit_twowiki_source_records",
    "convert_twowiki_source_record",
    "convert_twowiki_source_records",
    "deterministic_dev_test_partition",
    "resolve_worker_count",
]
