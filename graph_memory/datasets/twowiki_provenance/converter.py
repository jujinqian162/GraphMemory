from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict

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
from graph_memory.retrieval.contracts import SeedRanker
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest

MINIMUM_CANDIDATES = 4


def convert_twowiki_source_records(
    raw_records: Sequence[object],
    *,
    candidate_cap: int = 32,
    seed: int = 13,
    strict: bool = False,
    graph_config: ProvenanceGraphConstructionConfig | None = None,
    dense_ranker: SeedRanker | None = None,
) -> TwoWikiProvenanceConversionResult:
    if candidate_cap < MINIMUM_CANDIDATES:
        raise ValueError(f"candidate_cap must be at least {MINIMUM_CANDIDATES}.")
    resolved_graph_config = graph_config or ProvenanceGraphConstructionConfig()
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
    edges, gold_semantic_rank, gold_fallback = _graph_edges(
        example.raw_id,
        example.question,
        candidate_records,
        gold_output_source=gold_output_source,
        gold_output_target=gold_output_target,
        seed=seed,
        graph_config=resolved_graph_config,
        ranker=resolved_ranker,
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
            "graph_construction": asdict(resolved_graph_config),
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
            "gold_edge_fallback": gold_fallback,
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
) -> tuple[list[ProvenanceEdgeRecord], int, bool]:
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

    gold_semantic_rank = 0
    gold_fallback = False
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
            query_text=(
                f"{question}\nSource evidence: {source['title']}. {source['text']}"
            ),
            candidates=target_candidates,
        )
        ranked = ranker.rank(request)
        rank_by_target = {
            item.node_id: (rank_index, item.score)
            for rank_index, item in enumerate(ranked, start=1)
        }
        successor_count = min(graph_config.successors_per_output, len(ranked))
        selected_targets = [item.node_id for item in ranked[:successor_count]]
        if source_output == gold_output_source:
            gold_semantic_rank = rank_by_target[gold_output_target][0]
            if gold_output_target not in selected_targets:
                gold_fallback = True
                selected_targets[-1] = gold_output_target
        for target_output in selected_targets:
            target_call = by_output[target_output]["call_id"]
            semantic_rank, semantic_score = rank_by_target[target_output]
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
                    weight=1.0 / semantic_rank,
                    metadata={
                        "semantic_scorer": graph_config.strategy,
                        "semantic_rank": semantic_rank,
                        "semantic_score": float(semantic_score),
                    },
                )
            )
    _rng(seed, raw_id, "edges").shuffle(edges)
    return edges, gold_semantic_rank, gold_fallback


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
    }
    return text if text in known else "invalid_source_record"


__all__ = [
    "ProvenanceGraphConstructionConfig",
    "audit_twowiki_source_records",
    "convert_twowiki_source_record",
    "convert_twowiki_source_records",
    "deterministic_dev_test_partition",
]
