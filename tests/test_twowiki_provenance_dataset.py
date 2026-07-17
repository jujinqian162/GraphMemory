from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir

from graph_memory.contracts.ranking import RankedResult
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.datasets.selection import (
    evidence_evaluation_request_for_dataset,
    execution_provenance_requests_for_dataset,
    text_ranking_requests_for_dataset,
)
from graph_memory.evaluation.service import evaluate_results
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.datasets.twowiki_provenance import (
    ProvenanceGraphConstructionConfig,
    convert_twowiki_source_records,
    deterministic_dev_test_partition,
)
from graph_memory.graphs.provenance import ProvenanceEdgeType, ProvenanceNodeType
from graph_memory.experiment.config import (
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.layout import RunLayout
from graph_memory.experiment.inspect import inspect_catalog
from graph_memory.experiment.planning import WorkflowPlanner
from graph_memory.experiment.stage_models import (
    PairStageConfig,
    ProvenanceRgcnRetrieveStageConfig,
    ProvenanceRgcnTrainStageConfig,
)
from graph_memory.validation import (
    ContractValidationError,
    validate_twowiki_provenance_label_records,
    validate_twowiki_provenance_ranking_records,
)
from scripts.data.convert_2wiki_to_execution_provenance import main as convert_main
from scripts.build_train_pairs import _train_pair_tasks

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
        edge
        for edge in ranking["graph"]["edges"]
        if edge["edge_type"] == "feeds"
    ]
    assert {
        cast(str, edge["metadata"]["semantic_scorer"]) for edge in feed_edges
    } == {"bm25"}
    assert all("semantic_rank" in edge["metadata"] for edge in feed_edges)


def test_dense_successor_strategy_uses_injected_dense_ranker() -> None:
    converted = convert_twowiki_source_records(
        [_source_example("dense-semantic")],
        candidate_cap=6,
        seed=13,
        graph_config=ProvenanceGraphConstructionConfig(
            strategy="dense",
            successors_per_output=1,
        ),
        dense_ranker=_ReverseIdRanker(),
    ).records[0]
    feed_edges = [
        edge
        for edge in converted["ranking"]["graph"]["edges"]
        if edge["edge_type"] == "feeds"
    ]

    assert {
        cast(str, edge["metadata"]["semantic_scorer"]) for edge in feed_edges
    } == {
        "dense"
    }
    assert all(
        cast(int, edge["metadata"]["semantic_rank"]) >= 1 for edge in feed_edges
    )


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


def test_ranking_validation_rejects_inconsistent_binding_hash() -> None:
    ranking = convert_twowiki_source_records(
        [_source_example("bad-binding")], candidate_cap=6, seed=13
    ).records[0]["ranking"]
    broken = deepcopy(ranking)
    feeds = next(
        edge
        for edge in broken["graph"]["edges"]
        if edge["edge_type"] == "feeds"
    )
    assert feeds["binding"] is not None
    feeds["binding"]["binding_value_hash"] = "wrong-hash"

    with pytest.raises(ContractValidationError, match="inconsistent bindings"):
        validate_twowiki_provenance_ranking_records([broken])


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


def test_ranking_validation_rejects_label_leakage() -> None:
    converted = convert_twowiki_source_records(
        [_source_example("leak")], candidate_cap=6, seed=13
    ).records[0]
    leaked = deepcopy(converted["ranking"])
    leaked["metadata"]["is_gold"] = True

    with pytest.raises(ContractValidationError, match="forbidden"):
        validate_twowiki_provenance_ranking_records([leaked])


def test_provenance_rgcn_workflow_has_no_evidence_graph_stage(tmp_path: Path) -> None:
    source = tmp_path / "generated.json"
    source.write_text("[]", encoding="utf-8")
    config = _provenance_config(
        f"twowiki-provenance-plan-{tmp_path.name}", source=source
    )

    plan = WorkflowPlanner(config, RunLayout(ROOT, config.name)).build(
        validate_external=False
    )
    by_id = {invocation.identifier: invocation for invocation in plan.invocations}

    assert {invocation.stage for invocation in plan.invocations} == {
        "prepare",
        "pairs",
        "train",
        "retrieve",
        "evaluate",
        "aggregate",
    }
    assert not any(item.stage == "evidence_graphs" for item in plan.invocations)
    pair = by_id["pairs:execution_provenance_rgcn_retriever"]
    train = by_id["train:execution_provenance_rgcn_retriever"]
    retrieve = by_id["retrieve:execution_provenance_rgcn_retriever"]
    assert isinstance(pair.config, PairStageConfig)
    assert pair.config.evidence_graphs is None
    assert isinstance(train.config, ProvenanceRgcnTrainStageConfig)
    assert isinstance(retrieve.config, ProvenanceRgcnRetrieveStageConfig)
    assert pair.dependencies == ("prepare:train",)
    assert train.dependencies == (
        "pairs:execution_provenance_rgcn_retriever",
        "prepare:dev",
    )
    assert retrieve.dependencies == (
        "prepare:test",
        "train:execution_provenance_rgcn_retriever",
    )


def test_dense_ft_workflow_has_text_only_pairs_and_no_evidence_graph_stage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generated.json"
    source.write_text("[]", encoding="utf-8")
    config = _provenance_config(
        f"twowiki-provenance-dense-ft-{tmp_path.name}",
        source=source,
        methods="[dense_ft]",
    )

    plan = WorkflowPlanner(config, RunLayout(ROOT, config.name)).build(
        validate_external=False
    )
    by_id = {invocation.identifier: invocation for invocation in plan.invocations}

    assert {invocation.stage for invocation in plan.invocations} == {
        "prepare",
        "pairs",
        "train",
        "retrieve",
        "evaluate",
        "aggregate",
    }
    assert not any(item.stage == "evidence_graphs" for item in plan.invocations)
    pair = by_id["pairs:dense_ft"]
    assert isinstance(pair.config, PairStageConfig)
    assert pair.config.evidence_graphs is None
    assert pair.config.sampling.hard_graph_neighbor_per_positive == 0
    assert all(input_ref.role != "evidence_graphs" for input_ref in pair.inputs)
    assert pair.dependencies == ("prepare:train",)


def test_graphless_pair_tasks_use_text_candidates_without_method_dispatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generated.json"
    source.write_text("[]", encoding="utf-8")
    config = _provenance_config(
        f"twowiki-provenance-pairs-{tmp_path.name}", source=source
    )
    plan = WorkflowPlanner(config, RunLayout(ROOT, config.name)).build(
        validate_external=False
    )
    pair = next(
        invocation
        for invocation in plan.invocations
        if invocation.identifier == "pairs:execution_provenance_rgcn_retriever"
    )
    assert isinstance(pair.config, PairStageConfig)
    converted = convert_twowiki_source_records(
        [_source_example("graphless-pairs")], candidate_cap=6, seed=13
    ).records[0]

    tasks = _train_pair_tasks(
        pair.config,
        [converted["ranking"]],
        [converted["label"]],
        [],
    )
    assert len(tasks) == 1
    assert tasks[0].graph is None
    result = build_train_pairs(
        tasks,
        NegativeSamplingConfig(
            random_seed=13,
            easy_random_per_positive=1,
            hard_bm25_per_positive=1,
            hard_dense_per_positive=0,
            hard_graph_neighbor_per_positive=0,
            hard_pool_size=10,
        ),
    )
    assert result.summary["positive_count"] == 2
    assert "hard_graph_neighbor" not in result.summary["negative_count_by_type"]


def test_provenance_rgcn_ablation_discovery_and_planning(tmp_path: Path) -> None:
    source = tmp_path / "generated.json"
    source.write_text("[]", encoding="utf-8")
    catalog = cast(dict[object, list[dict[str, object]]], inspect_catalog(
        "ablations", repository_root=ROOT
    ))
    rows = catalog["execution_provenance_rgcn_retriever"]

    assert [row["variant"] for row in rows] == [
        "full_rgcn",
        "wo_graph",
        "wo_edge_type",
        "wo_edge_weight",
        "wo_hard_negatives",
    ]

    config = _provenance_config(
        f"twowiki-provenance-ablation-{tmp_path.name}",
        source=source,
        extra=[
            "ablation.enable=true",
            "ablation.variants=[wo_graph,wo_edge_type]",
        ],
    )
    plan = WorkflowPlanner(config, RunLayout(ROOT, config.name)).build(
        validate_external=False
    )
    by_id = {invocation.identifier: invocation for invocation in plan.invocations}
    aliases = {alias.identifier: alias for alias in plan.aliases}

    assert "pairs:execution_provenance_rgcn_retriever:wo_graph" in aliases
    assert "pairs:execution_provenance_rgcn_retriever:wo_edge_type" in aliases
    wo_graph = by_id["train:execution_provenance_rgcn_retriever:wo_graph"]
    wo_edge_type = by_id[
        "train:execution_provenance_rgcn_retriever:wo_edge_type"
    ]
    assert isinstance(wo_graph.config, ProvenanceRgcnTrainStageConfig)
    assert wo_graph.config.train.model.ablation == "wo_graph"
    assert wo_graph.config.train.model.num_layers == 0
    assert isinstance(wo_edge_type.config, ProvenanceRgcnTrainStageConfig)
    assert wo_edge_type.config.train.model.ablation == "wo_edge_type"
    assert {(item.method.value, item.variant) for item in plan.ablation_selections} == {
        ("execution_provenance_rgcn_retriever", "full_rgcn"),
        ("execution_provenance_rgcn_retriever", "wo_graph"),
        ("execution_provenance_rgcn_retriever", "wo_edge_type"),
    }


def test_provenance_rgcn_hard_negative_ablation_rebuilds_pairs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generated.json"
    source.write_text("[]", encoding="utf-8")
    config = _provenance_config(
        f"twowiki-provenance-hard-negative-{tmp_path.name}",
        source=source,
        extra=[
            "ablation.enable=true",
            "ablation.variants=[wo_hard_negatives]",
        ],
    )
    plan = WorkflowPlanner(config, RunLayout(ROOT, config.name)).build(
        validate_external=False
    )
    by_id = {invocation.identifier: invocation for invocation in plan.invocations}
    pair = by_id[
        "pairs:execution_provenance_rgcn_retriever:wo_hard_negatives"
    ]

    assert isinstance(pair.config, PairStageConfig)
    assert pair.config.sampling.easy_random_per_positive == 2
    assert pair.config.sampling.hard_bm25_per_positive == 0
    assert pair.config.sampling.hard_dense_per_positive == 0
    assert pair.config.sampling.hard_graph_neighbor_per_positive == 0
    assert not any(
        alias.identifier
        == "pairs:execution_provenance_rgcn_retriever:wo_hard_negatives"
        for alias in plan.aliases
    )


def test_converter_cli_is_byte_deterministic_and_writes_manifest(
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
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    common_args = [
        "--train-source",
        str(train_source),
        "--dev-source",
        str(dev_source),
        "--candidate-cap",
        "6",
        "--seed",
        "17",
    ]

    assert convert_main([*common_args, "--output-dir", str(output_a)]) == 0
    assert convert_main([*common_args, "--output-dir", str(output_b)]) == 0

    for filename in (
        "train.json",
        "dev.json",
        "test.json",
        "manifest.json",
        "statistics.json",
    ):
        assert (output_a / filename).read_bytes() == (output_b / filename).read_bytes()
    manifest = json.loads((output_a / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {
        "train": 2,
        "dev": 3,
        "test": 3,
        "train_rejected": {},
        "dev_source_rejected": {},
    }


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


def _provenance_config(
    name: str,
    *,
    source: Path,
    methods: str = "[execution_provenance_rgcn_retriever]",
    extra: list[str] | None = None,
):
    source_value = source.resolve().as_posix()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                f"name={name}",
                "dataset=twowiki_provenance",
                "profile=smoke",
                f"methods={methods}",
                "device=cpu",
                *[
                    f"dataset.splits.{split}.source={source_value}"
                    for split in ("train", "dev", "test")
                ],
                *[
                    override
                    for split in ("train", "dev", "test")
                    for override in (
                        f"dataset.splits.{split}.offset=0",
                        f"dataset.splits.{split}.capacity=1",
                    )
                ],
                *(extra or []),
            ],
        )
    return resolve_experiment_config(
        validate_composed_config(composed), repository_root=ROOT
    )


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


class _ReverseIdRanker:
    @property
    def method_name(self) -> str:
        return "reverse_id_test_ranker"

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        return [
            RankedNode(candidate.item_id, float(index))
            for index, candidate in enumerate(
                sorted(request.candidates, key=lambda item: item.item_id, reverse=True),
                start=1,
            )
        ]
