from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from graph_memory.contracts.ranking import RankedResult
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.datasets.selection import (
    evidence_evaluation_request_for_dataset,
    execution_provenance_requests_for_dataset,
    text_ranking_requests_for_dataset,
)
from graph_memory.evaluation.service import evaluate_results
from graph_memory.datasets.twowiki_provenance import (
    ProvenanceGraphConstructionConfig,
    TWOWIKI_PROVENANCE_SCHEMA_VERSION,
    convert_twowiki_source_records,
    deterministic_dev_test_partition,
    parse_twowiki_provenance_record,
)
from graph_memory.datasets.twowiki_provenance.records import ProvenanceEdgeRecord
from graph_memory.graphs.provenance import ProvenanceEdgeType, ProvenanceNodeType
from graph_memory.validation import (
    ContractValidationError,
    validate_twowiki_provenance_label_records,
    validate_twowiki_provenance_ranking_records,
)
from graph_memory.experiment.artifacts import FileSourceRef, identify_external_source
from graph_memory.experiment.config import TwoWikiProvenanceTransformConfig
from graph_memory.stages.transform import materialize_transform_twowiki

ROOT = Path(__file__).resolve().parents[1]


def test_conversion_is_deterministic_balanced_and_leakage_safe() -> None:
    raw = [_source_example("a"), _source_example("b")]

    first = convert_twowiki_source_records(raw, candidate_cap=6, seed=13)
    second = convert_twowiki_source_records(raw, candidate_cap=6, seed=13)

    assert first == second
    assert first.rejected_reason_counts == {}
    ranking = first.records[0]["ranking"]
    label = first.records[0]["label"]
    validate_twowiki_provenance_ranking_records([ranking])
    validate_twowiki_provenance_label_records([label], {ranking["task_id"]: ranking})
    assert '"gold' not in json.dumps(ranking, sort_keys=True)
    assert label["gold_dependency_edges"] == [label["gold_evidence_output_ids"]]

    request = execution_provenance_requests_for_dataset(
        "twowiki_provenance", [ranking]
    )[0]
    candidate_ids = {candidate.item_id for candidate in request.candidates}
    assert candidate_ids == {
        node.node_id
        for node in request.graph.nodes
        if node.node_type is ProvenanceNodeType.TOOL_OUTPUT
    }
    feeds_out = Counter(
        edge.source
        for edge in request.graph.edges
        if edge.edge_type is ProvenanceEdgeType.FEEDS
    )
    assert set(feeds_out) == candidate_ids
    assert set(feeds_out.values()) == {2}


def test_schema_v3_gold_spine_uses_matched_fixed_degree_branches() -> None:
    result = convert_twowiki_source_records(
        [_source_example("schema-v3")],
        candidate_cap=6,
        seed=13,
    )

    assert result.rejected_reason_counts == {}
    raw_record = result.records[0]
    assert TWOWIKI_PROVENANCE_SCHEMA_VERSION == 3
    assert raw_record["schema_version"] == 3
    ranking = raw_record["ranking"]
    label = raw_record["label"]
    by_output = {
        candidate["output_id"]: candidate for candidate in ranking["candidates"]
    }
    output_by_call = {
        candidate["call_id"]: candidate["output_id"]
        for candidate in ranking["candidates"]
    }
    feed_edges = [
        edge for edge in ranking["graph"]["edges"] if edge["edge_type"] == "feeds"
    ]
    by_source: dict[str, list[ProvenanceEdgeRecord]] = {}
    for edge in feed_edges:
        by_source.setdefault(cast(str, edge["source"]), []).append(edge)

    assert set(by_source) == set(by_output)
    assert all(len(edges) == 2 for edges in by_source.values())
    assert all(
        Counter(cast(str, edge["metadata"]["branch_role"]) for edge in edges)
        == {"semantic_head": 1, "rank_banded_branch": 1}
        for edges in by_source.values()
    )

    gold_source, gold_target = label["gold_dependency_edges"][0]
    gold_target_call = by_output[gold_target]["call_id"]
    gold_edge = next(
        edge
        for edge in by_source[gold_source]
        if edge["target"] == gold_target_call
    )
    gold_bucket = cast(str, gold_edge["metadata"]["rank_bucket"])
    assert label["metadata"]["gold_edge_rank_bucket"] == gold_bucket
    if gold_bucket != "head":
        assert any(
            edge["source"] != gold_source
            and edge["metadata"]["branch_role"] == "rank_banded_branch"
            and edge["metadata"]["rank_bucket"] == gold_bucket
            and output_by_call[cast(str, edge["target"])] != gold_target
            for edge in feed_edges
        )


def test_schema_v3_feed_weights_are_bounded_and_source_mass_preserving() -> None:
    ranking = convert_twowiki_source_records(
        [_source_example("weight-mass")], candidate_cap=6, seed=13
    ).records[0]["ranking"]
    feed_edges = [
        edge for edge in ranking["graph"]["edges"] if edge["edge_type"] == "feeds"
    ]
    by_source: dict[str, list[ProvenanceEdgeRecord]] = {}
    for edge in feed_edges:
        by_source.setdefault(cast(str, edge["source"]), []).append(edge)

    required_metadata = {
        "synthetic",
        "semantic_scorer",
        "scorer_identity",
        "query_template_version",
        "semantic_rank",
        "semantic_score",
        "normalized_score",
        "source_probability",
        "calibrated_weight",
        "branch_role",
        "rank_bucket",
    }
    for edges in by_source.values():
        assert sum(cast(float, edge["weight"]) for edge in edges) == pytest.approx(1.5)
        for edge in edges:
            assert 0.5 <= cast(float, edge["weight"]) <= 1.0
            assert cast(float, edge["metadata"]["calibrated_weight"]) == pytest.approx(
                cast(float, edge["weight"])
            )
            assert set(edge["metadata"]) == required_metadata

    assert all(
        edge["weight"] == 1.0
        for edge in ranking["graph"]["edges"]
        if edge["edge_type"] != "feeds"
    )


def test_schema_v3_rejects_gold_branch_outside_configured_rank_buckets() -> None:
    result = convert_twowiki_source_records(
        [_source_example("unmatched-bucket")],
        candidate_cap=6,
        seed=13,
        graph_config=ProvenanceGraphConstructionConfig(
            strategy="dense",
            successors_per_output=2,
            near_rank_bucket=(2, 2),
            mid_rank_bucket=(3, 3),
            tail_rank_bucket=(4, 4),
        ),
        dense_ranker=_GoldTargetLastRanker(),
    )

    assert result.records == []
    assert result.rejected_reason_counts == {"unmatched_gold_branch_bucket": 1}


def test_schema_v3_parser_rejects_v2_artifact_explicitly() -> None:
    record = convert_twowiki_source_records(
        [_source_example("v2-reject")], candidate_cap=6, seed=13
    ).records[0]
    legacy = deepcopy(record)
    legacy["schema_version"] = 2

    with pytest.raises(ValueError, match="v2 artifacts are not compatible"):
        parse_twowiki_provenance_record(legacy)


def _mutable_ranking_edges(ranking: dict[str, object]) -> list[dict[str, object]]:
    graph = cast(dict[str, object], ranking["graph"])
    return cast(list[dict[str, object]], graph["edges"])


def _mutate_feed_mass(ranking: dict[str, object]) -> dict[str, object]:
    broken = deepcopy(ranking)
    feed = next(
        edge for edge in _mutable_ranking_edges(broken) if edge["edge_type"] == "feeds"
    )
    feed["weight"] = 0.5
    metadata = cast(dict[str, object], feed["metadata"])
    metadata["calibrated_weight"] = 0.5
    return broken


def _mutate_confidence_metadata(ranking: dict[str, object]) -> dict[str, object]:
    broken = deepcopy(ranking)
    feed = next(
        edge for edge in _mutable_ranking_edges(broken) if edge["edge_type"] == "feeds"
    )
    metadata = cast(dict[str, object], feed["metadata"])
    del metadata["source_probability"]
    return broken


def _mutate_binding_hash(ranking: dict[str, object]) -> dict[str, object]:
    broken = deepcopy(ranking)
    feeds = next(
        edge for edge in _mutable_ranking_edges(broken) if edge["edge_type"] == "feeds"
    )
    assert feeds["binding"] is not None
    binding = cast(dict[str, object], feeds["binding"])
    binding["binding_value_hash"] = "wrong-hash"
    return broken


def _mutate_label_leakage(ranking: dict[str, object]) -> dict[str, object]:
    leaked = deepcopy(ranking)
    metadata = cast(dict[str, object], leaked["metadata"])
    metadata["is_gold"] = True
    return leaked


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (_mutate_feed_mass, "feed mass"),
        (_mutate_confidence_metadata, "confidence metadata"),
        (_mutate_binding_hash, "inconsistent bindings"),
        (_mutate_label_leakage, "forbidden"),
    ),
    ids=("feed-mass", "confidence-metadata", "binding-hash", "label-leakage"),
)
def test_ranking_validation_rejects_contract_violations(
    mutator, match: str
) -> None:
    ranking = convert_twowiki_source_records(
        [_source_example("validation")], candidate_cap=6, seed=13
    ).records[0]["ranking"]

    with pytest.raises(ContractValidationError, match=match):
        validate_twowiki_provenance_ranking_records([mutator(ranking)])


def test_bm25_successor_edges_follow_semantic_matches_not_shuffled_order() -> None:
    converted = convert_twowiki_source_records(
        [_source_example("semantic")],
        candidate_cap=6,
        seed=13,
        graph_config=ProvenanceGraphConstructionConfig(
            strategy="bm25",
            successors_per_output=2,
        ),
    ).records[0]
    ranking = converted["ranking"]
    by_text = {candidate["text"]: candidate for candidate in ranking["candidates"]}
    source = by_text["Alpha is an artist."]
    related = by_text["Alpha was born in Bridge City."]
    feeds_targets = {
        edge["target"]
        for edge in ranking["graph"]["edges"]
        if edge["edge_type"] == "feeds" and edge["source"] == source["output_id"]
    }

    assert related["call_id"] in feeds_targets
    feed_edges = [
        edge for edge in ranking["graph"]["edges"] if edge["edge_type"] == "feeds"
    ]
    assert {cast(str, edge["metadata"]["semantic_scorer"]) for edge in feed_edges} == {
        "bm25"
    }
    assert all("semantic_rank" in edge["metadata"] for edge in feed_edges)


def test_dense_successor_uses_injected_ranker_and_batches_queries() -> None:
    ranker = _BatchCountingRanker()
    result = convert_twowiki_source_records(
        [_source_example("dense-semantic")],
        candidate_cap=6,
        seed=13,
        graph_config=ProvenanceGraphConstructionConfig(
            strategy="dense",
            successors_per_output=2,
        ),
        dense_ranker=ranker,
    )

    assert result.rejected_reason_counts == {}
    assert ranker.batch_sizes == [1, 6]
    feed_edges = [
        edge
        for edge in result.records[0]["ranking"]["graph"]["edges"]
        if edge["edge_type"] == "feeds"
    ]
    assert {cast(str, edge["metadata"]["semantic_scorer"]) for edge in feed_edges} == {
        "dense"
    }
    assert all(cast(int, edge["metadata"]["semantic_rank"]) >= 1 for edge in feed_edges)


def test_ambiguous_gold_chain_is_rejected_before_graph_construction() -> None:
    result = convert_twowiki_source_records(
        [_ambiguous_source_example()], candidate_cap=4, seed=13
    )

    assert result.records == []
    assert result.rejected_reason_counts == {"ambiguous_gold_chain": 1}


def test_generated_bindings_match_source_field_hash_and_target_input() -> None:
    ranking = convert_twowiki_source_records(
        [_source_example("binding")], candidate_cap=6, seed=13
    ).records[0]["ranking"]
    node_by_id = {node["node_id"]: node for node in ranking["graph"]["nodes"]}

    for edge in ranking["graph"]["edges"]:
        if edge["edge_type"] != "feeds":
            continue
        binding = edge["binding"]
        assert binding is not None
        source_hashes = cast(
            dict[str, str],
            node_by_id[edge["source"]]["metadata"]["output_field_hashes"],
        )
        target_inputs = cast(
            list[str],
            node_by_id[edge["target"]]["metadata"]["input_parameters"],
        )
        assert source_hashes[binding["output_field"]] == binding["binding_value_hash"]
        assert binding["input_parameter"] in target_inputs


def test_dev_test_partition_is_fixed_by_split_seed() -> None:
    records = convert_twowiki_source_records(
        [_source_example(str(index)) for index in range(8)],
        candidate_cap=6,
        seed=7,
    ).records
    # The dev/test boundary is a pure function of the split seed, independent of
    # the conversion seed. The workflow passes the fixed split_seed here so the
    # provenance test partition stays identical across training seeds.
    dev_a, test_a = deterministic_dev_test_partition(records, seed=13)
    dev_b, test_b = deterministic_dev_test_partition(records, seed=13)
    assert [r["ranking"]["task_id"] for r in test_a] == [
        r["ranking"]["task_id"] for r in test_b
    ]
    assert [r["ranking"]["task_id"] for r in dev_a] == [
        r["ranking"]["task_id"] for r in dev_b
    ]
    # A different split seed is allowed to move the boundary, proving the
    # partition truly derives from the seed argument.
    _, test_other = deterministic_dev_test_partition(records, seed=41)
    assert {r["ranking"]["task_id"] for r in test_a} != {
        r["ranking"]["task_id"] for r in test_other
    }


def test_flat_projection_and_split_partition_preserve_identity() -> None:
    records = convert_twowiki_source_records(
        [_source_example(str(index)) for index in range(6)],
        candidate_cap=6,
        seed=7,
    ).records
    dev, test = deterministic_dev_test_partition(records, seed=7)

    dev_ids = {record["ranking"]["task_id"] for record in dev}
    test_ids = {record["ranking"]["task_id"] for record in test}
    assert dev_ids
    assert test_ids
    assert dev_ids.isdisjoint(test_ids)
    assert dev_ids | test_ids == {record["ranking"]["task_id"] for record in records}
    text_request = text_ranking_requests_for_dataset(
        "twowiki_provenance", [records[0]["ranking"]]
    )[0]
    assert len(text_request.candidates) == 6
    assert all(
        candidate.item_id.startswith("output_") for candidate in text_request.candidates
    )


def test_transform_is_byte_deterministic_and_raw_only(
    tmp_path: Path,
) -> None:
    train_source = tmp_path / "train.json"
    dev_source = tmp_path / "dev.json"
    train_source.write_text(
        json.dumps([_source_example("train-1"), _source_example("train-2")]),
        encoding="utf-8",
    )
    dev_source.write_text(
        json.dumps([_source_example(f"dev-{index}") for index in range(6)]),
        encoding="utf-8",
    )
    train_ref = cast(
        FileSourceRef,
        identify_external_source(train_source, repository_root=tmp_path),
    )
    dev_ref = cast(
        FileSourceRef,
        identify_external_source(dev_source, repository_root=tmp_path),
    )
    config = TwoWikiProvenanceTransformConfig(edge_scorer="bm25", candidate_cap=6, seed=17)
    output_a = tmp_path / "out-a"
    output_b = tmp_path / "out-b"

    result_a = materialize_transform_twowiki(
        train_source=train_ref,
        dev_source=dev_ref,
        config=config,
        schema_version=3,
        output_root=output_a,
        repository_root=tmp_path,
    )
    result_b = materialize_transform_twowiki(
        train_source=train_ref,
        dev_source=dev_ref,
        config=config,
        schema_version=3,
        output_root=output_b,
        repository_root=tmp_path,
    )

    assert result_a.version_tag == result_b.version_tag
    version_dir_a = output_a / result_a.version_tag
    version_dir_b = output_b / result_b.version_tag
    for filename in ("train.json", "dev.json", "test.json"):
        assert (
            (version_dir_a / filename).read_bytes()
            == (version_dir_b / filename).read_bytes()
        )
    assert not (version_dir_a / "manifest.json").exists()
    assert not (version_dir_a / "statistics.json").exists()
    assert result_a.train.digest == result_b.train.digest
    assert result_a.dev.digest == result_b.dev.digest
    assert result_a.test.digest == result_b.test.digest
    assert len(json.loads((version_dir_a / "train.json").read_text("utf-8"))) == 2
    assert len(json.loads((version_dir_a / "dev.json").read_text("utf-8"))) == 3
    assert len(json.loads((version_dir_a / "test.json").read_text("utf-8"))) == 3


def test_committed_smoke_fixture_matches_schema_v3_generator_input() -> None:
    source = json.loads(
        (ROOT / "tests/fixtures/twowiki_provenance_smoke_source.json").read_text(
            encoding="utf-8"
        )
    )
    expected = convert_twowiki_source_records(
        source, candidate_cap=6, seed=13, strict=True
    ).records
    committed = json.loads(
        (ROOT / "tests/fixtures/twowiki_provenance_smoke.json").read_text(
            encoding="utf-8"
        )
    )

    assert committed == expected
    validate_twowiki_provenance_ranking_records(
        [record["ranking"] for record in committed]
    )


def test_provenance_path_metrics_do_not_require_an_evidence_graph() -> None:
    converted = convert_twowiki_source_records(
        [_source_example("metric")], candidate_cap=6, seed=13
    ).records[0]
    ranking = converted["ranking"]
    label = converted["label"]
    candidate_ids = [candidate["output_id"] for candidate in ranking["candidates"]]
    gold_source, gold_target = label["gold_dependency_edges"][0]
    prediction: RankedResult = {
        "task_id": ranking["task_id"],
        "method": "execution_provenance_retriever",
        "ranked_nodes": [
            {"node_id": candidate_id, "score": float(len(candidate_ids) - index)}
            for index, candidate_id in enumerate(candidate_ids)
        ],
        "retrieved_subgraph": {
            "nodes": candidate_ids,
            "edges": [
                {
                    "source": gold_source,
                    "target": gold_target,
                    "edge_type": "sequential",
                    "weight": 1.0,
                    "directed": True,
                }
            ],
        },
        "latency_ms": 1.0,
        "input_tokens": 0,
    }

    rows = evaluate_results(
        evidence_evaluation_request_for_dataset(
            "twowiki_provenance",
            predictions=[prediction],
            labels=[label],
            graphs=[],
        )
    )

    assert rows[0]["Path Recall@10"] == 1.0
    assert rows[0]["Edge Recall@10"] == 1.0
    assert rows[0]["Edge Precision@10"] == 1.0
    assert rows[0]["Edge F1@10"] == 1.0
    assert rows[0]["Abstention Rate"] == 0.0

    no_edges = deepcopy(prediction)
    no_edges["retrieved_subgraph"]["edges"] = []
    zero_rows = evaluate_results(
        evidence_evaluation_request_for_dataset(
            "twowiki_provenance",
            predictions=[no_edges],
            labels=[label],
            graphs=[],
        )
    )
    assert zero_rows[0]["Edge Precision@10"] == 0.0
    assert zero_rows[0]["Edge Recall@10"] == 0.0
    assert zero_rows[0]["Edge F1@10"] == 0.0


def _source_example(raw_id: str) -> dict[str, object]:
    return {
        "_id": raw_id,
        "type": "compositional",
        "question": "In which country is the birthplace of Alpha located?",
        "context": [
            ["Alpha", ["Alpha was born in Bridge City.", "Alpha is an artist."]],
            [
                "Bridge City",
                ["Bridge City is located in Country Z.", "It has a river."],
            ],
            ["Wrong One", ["Wrong One is located elsewhere."]],
            ["Wrong Two", ["Wrong Two mentions Alpha but not the answer."]],
        ],
        "supporting_facts": [["Alpha", 0], ["Bridge City", 0]],
        "evidences": [
            ["Alpha", "birth place", "Bridge City"],
            ["Bridge City", "country", "Country Z"],
        ],
        "evidences_id": [],
        "answer_id": "Country Z",
        "answer": "Country Z",
    }


def _ambiguous_source_example() -> dict[str, object]:
    return {
        "_id": "ambiguous",
        "type": "compositional",
        "question": "Where is Alpha located?",
        "context": [
            [
                "Alpha Bridge City A",
                ["Alpha was born in Bridge City.", "Distractor A."],
            ],
            [
                "Alpha Bridge City B",
                ["Alpha was born in Bridge City and Country Z.", "Distractor B."],
            ],
        ],
        "supporting_facts": [
            ["Alpha Bridge City A", 0],
            ["Alpha Bridge City B", 0],
        ],
        "evidences": [
            ["Alpha", "birth place", "Bridge City"],
            ["Bridge City", "country", "Country Z"],
        ],
        "evidences_id": [],
        "answer_id": "Country Z",
        "answer": "Country Z",
    }


class _GoldTargetLastRanker:
    @property
    def method_name(self) -> str:
        return "gold_target_last_test_ranker"

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        ordered = sorted(
            request.candidates,
            key=lambda candidate: (
                "Bridge City is located in Country Z." in candidate.text,
                candidate.item_id,
            ),
        )
        return [
            RankedNode(candidate.item_id, float(len(ordered) - index))
            for index, candidate in enumerate(ordered)
        ]


class _BatchCountingRanker:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    @property
    def method_name(self) -> str:
        return "batch_counting_test_ranker"

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        raise AssertionError("rank_many must be used")

    def rank_many(
        self, requests: list[TextRankingRequest]
    ) -> list[list[RankedNode]]:
        self.batch_sizes.append(len(requests))
        return [
            [
                RankedNode(candidate.item_id, float(len(request.candidates) - index))
                for index, candidate in enumerate(
                    sorted(request.candidates, key=lambda item: item.item_id)
                )
            ]
            for request in requests
        ]
