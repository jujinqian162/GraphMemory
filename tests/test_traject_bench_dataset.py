from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir

from graph_memory.datasets.traject_bench import (
    MissingCatalogToolsError,
    TrajectBenchToEvidenceEvaluationRequest,
    TrajectBenchToEvidenceGraphBuildRequest,
    TrajectBenchToTextRankingRequest,
    canonicalize_traject_bench_tool_catalog,
    convert_traject_bench_example,
    discover_traject_bench_query_files,
    parse_traject_bench_query,
    parse_traject_bench_tool_catalog,
    prepare_traject_bench_tool_catalog,
    stable_traject_bench_tool_id,
)
from graph_memory.evaluation.service import evaluate_results
from graph_memory.experiment.config import (
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.layout import RunLayout
from graph_memory.experiment.planning import WorkflowPlanner
from graph_memory.experiment.stage_models import (
    Bm25RetrieveStageConfig,
    PrepareOutputs,
    RawPrepareStageConfig,
)
from graph_memory.stages.retrieve import run_retrieve_stage
from graph_memory.validation import (
    ContractValidationError,
    validate_traject_bench_label_records,
    validate_traject_bench_ranking_records,
)
from scripts.prepare_traject_bench import prepare_from_raw

ROOT = Path(__file__).resolve().parents[1]


def test_partition_discovery_is_deterministic_and_includes_travel_sequential(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    for relative in (
        "parallel/Travel/simple_ver.json",
        "parallel/Education/simple_ver.json",
        "parallel/Travel/hard_ver.json",
        "parallel/Education/hard_ver.json",
        "sequential/Travel/simple_ver.json",
        "sequential/Travel/traj_query.json",
        "sequential/Education/traj_query.json",
        "sequential/Travel/sequence.json",
    ):
        _write_json(root / relative, [])
    (root / "tools").mkdir(parents=True)

    train = discover_traject_bench_query_files(root, "train")
    dev = discover_traject_bench_query_files(root, "dev")
    test = discover_traject_bench_query_files(root, "test")

    assert [_relative(root, path) for path in train] == [
        "parallel/Education/simple_ver.json",
        "parallel/Travel/simple_ver.json",
    ]
    assert [_relative(root, path) for path in dev] == [
        "parallel/Education/hard_ver.json",
        "parallel/Travel/hard_ver.json",
    ]
    assert [_relative(root, path) for path in test] == [
        "sequential/Education/traj_query.json",
        "sequential/Travel/simple_ver.json",
        "sequential/Travel/traj_query.json",
    ]


def test_catalog_canonicalization_and_repeated_gold_calls_are_lossless() -> None:
    definitions = parse_traject_bench_tool_catalog(
        [
            _catalog_tool("Provider: Alpha", connected=["Provider: Beta"]),
            _catalog_tool(
                "Provider: Alpha",
                description="A longer catalog-owned description for alpha.",
                connected=["Provider: Missing"],
            ),
            _catalog_tool("Provider: Beta"),
        ],
        domain_name="Education",
    )
    canonical = canonicalize_traject_bench_tool_catalog(
        definitions,
        domain_name="Education",
    )
    catalog = prepare_traject_bench_tool_catalog(canonical)
    example = parse_traject_bench_query(
        _sequential_query(
            ["Provider: Alpha", "Provider: Alpha", "Provider: Beta"]
        ),
        source_key="sequential/Education/traj_query.json",
        source_index=7,
        partition="sequential",
        domain_name="Education",
    )

    converted = convert_traject_bench_example(example, catalog)
    alpha_id = stable_traject_bench_tool_id("Provider: Alpha")
    beta_id = stable_traject_bench_tool_id("Provider: Beta")

    assert canonical.duplicate_entry_count == 1
    assert canonical.unresolved_connection_count == 1
    assert len(catalog.candidates) == 2
    assert "A longer catalog-owned description" in catalog.candidates[0]["text"]
    assert converted.label_record["gold_tool_ids"] == [alpha_id, beta_id]
    assert converted.label_record["gold_tool_sequence_ids"] == [
        alpha_id,
        alpha_id,
        beta_id,
    ]
    assert converted.label_record["gold_dependency_edges"] == [[alpha_id, beta_id]]
    assert "gold_tool_ids" not in converted.ranking_record
    assert "final_answer" not in converted.ranking_record


def test_hugging_face_catalog_category_is_parsed_but_not_ranked() -> None:
    raw_tool = _catalog_tool("Provider: Alpha")
    raw_tool["category"] = "Upstream HF category"

    definitions = parse_traject_bench_tool_catalog(
        [raw_tool],
        domain_name="Education",
    )
    catalog = prepare_traject_bench_tool_catalog(
        canonicalize_traject_bench_tool_catalog(
            definitions,
            domain_name="Education",
        )
    )

    assert definitions[0].category == "Upstream HF category"
    assert "Upstream HF category" not in catalog.candidates[0]["text"]


def test_hugging_face_parallel_query_json_tool_list_is_parsed_strictly() -> None:
    raw_query = _parallel_query(["Provider: Alpha", "Provider: Beta"])
    raw_query["tool_list"] = json.dumps(raw_query.pop("tool list"))
    raw_query["tool_count"] = raw_query.pop("tool count")
    raw_query["task_name"] = "HF task"
    raw_query["task_description"] = "HF task description"

    example = parse_traject_bench_query(
        raw_query,
        source_key="parallel/Education/simple_ver.json",
        source_index=0,
        partition="parallel_simple",
        domain_name="Education",
    )

    assert example.tool_names == ("Provider: Alpha", "Provider: Beta")


def test_missing_gold_tool_is_rejected_instead_of_added_to_candidates() -> None:
    definitions = parse_traject_bench_tool_catalog(
        [_catalog_tool("Provider: Alpha")],
        domain_name="Education",
    )
    catalog = prepare_traject_bench_tool_catalog(
        canonicalize_traject_bench_tool_catalog(
            definitions,
            domain_name="Education",
        )
    )
    example = parse_traject_bench_query(
        _parallel_query(["Provider: Missing"]),
        source_key="parallel/Education/simple_ver.json",
        source_index=3,
        partition="parallel_simple",
        domain_name="Education",
    )

    with pytest.raises(MissingCatalogToolsError, match="missing from public catalog"):
        convert_traject_bench_example(example, catalog)

    assert [candidate["tool_name"] for candidate in catalog.candidates] == [
        "Provider: Alpha"
    ]


def test_graph_projection_contains_catalog_edges_but_not_gold_order() -> None:
    definitions = parse_traject_bench_tool_catalog(
        [
            _catalog_tool("Provider: Alpha", connected=["Provider: Beta"]),
            _catalog_tool("Provider: Beta"),
        ],
        domain_name="Education",
    )
    catalog = prepare_traject_bench_tool_catalog(
        canonicalize_traject_bench_tool_catalog(
            definitions,
            domain_name="Education",
        )
    )
    example = parse_traject_bench_query(
        _sequential_query(["Provider: Beta", "Provider: Alpha"]),
        source_key="sequential/Education/traj_query.json",
        source_index=1,
        partition="sequential",
        domain_name="Education",
    )
    converted = convert_traject_bench_example(example, catalog)

    text_request = TrajectBenchToTextRankingRequest().project(
        converted.ranking_record
    )
    graph_request = TrajectBenchToEvidenceGraphBuildRequest().project(
        converted.ranking_record
    )
    alpha_id = stable_traject_bench_tool_id("Provider: Alpha")
    beta_id = stable_traject_bench_tool_id("Provider: Beta")

    assert {candidate.item_id for candidate in text_request.candidates} == {
        alpha_id,
        beta_id,
    }
    assert [(edge.source, edge.target) for edge in graph_request.input_visible_edges] == [
        (alpha_id, beta_id)
    ]
    assert converted.label_record["gold_dependency_edges"] == [[beta_id, alpha_id]]
    assert all(node.sequence_index is None for node in graph_request.nodes)


def test_validation_dispatch_and_bm25_evaluation_use_distinct_tool_labels() -> None:
    definitions = parse_traject_bench_tool_catalog(
        [
            _catalog_tool(
                "Provider: Alpha",
                description="Find alpha education records.",
            ),
            _catalog_tool(
                "Provider: Beta",
                description="Look up beta education records.",
            ),
        ],
        domain_name="Education",
    )
    catalog = prepare_traject_bench_tool_catalog(
        canonicalize_traject_bench_tool_catalog(
            definitions,
            domain_name="Education",
        )
    )
    example = parse_traject_bench_query(
        _parallel_query(["Provider: Alpha"]),
        source_key="parallel/Education/simple_ver.json",
        source_index=0,
        partition="parallel_simple",
        domain_name="Education",
    )
    converted = convert_traject_bench_example(example, catalog)
    ranking_records = [converted.ranking_record]
    label_records = [converted.label_record]
    ranking_by_task_id = {
        converted.ranking_record["task_id"]: converted.ranking_record
    }

    validate_traject_bench_ranking_records(ranking_records)
    validate_traject_bench_label_records(label_records, ranking_by_task_id)
    retrieval = run_retrieve_stage(
        Bm25RetrieveStageConfig(
            stage="retrieve",
            method="bm25",
            variant=None,
            dataset="traject_bench",
            tasks=Path("unused.json"),
            output=Path("unused.ranked.json"),
            top_k=2,
        ),
        task_inputs=ranking_records,
        evidence_graphs=None,
    )
    request = TrajectBenchToEvidenceEvaluationRequest().project(
        predictions=retrieval.predictions,
        labels=label_records,
        graphs=[],
    )
    rows = evaluate_results(request)

    assert rows[0]["Method"] == "bm25"
    assert rows[0]["Recall@2"] == 1.0
    assert rows[0]["Full Support@5"] == 1.0

    leaked = cast(
        dict[str, object],
        cast(object, deepcopy(converted.ranking_record)),
    )
    leaked["gold_tool_ids"] = converted.label_record["gold_tool_ids"]
    with pytest.raises(ContractValidationError, match="unknown fields|forbidden label"):
        validate_traject_bench_ranking_records([leaked])


def test_prepare_filters_missing_catalog_tools_before_seeded_sampling(
    tmp_path: Path,
) -> None:
    raw = _minimal_official_layout(tmp_path / "raw")
    _write_json(
        raw / "parallel/Education/simple_ver.json",
        [
            _parallel_query(["Provider: Alpha"]),
            _parallel_query(["Provider: Missing"]),
            _parallel_query(["Provider: Beta"]),
        ],
    )
    _write_json(
        raw / "tools/Education_tool.json",
        [
            _catalog_tool("Provider: Alpha"),
            _catalog_tool("Provider: Beta"),
        ],
    )
    config = RawPrepareStageConfig(
        stage="prepare",
        kind="raw",
        dataset="traject_bench",
        split="train",
        source=raw,
        outputs=PrepareOutputs(
            input=tmp_path / "input.json",
            labels=tmp_path / "labels.json",
            combined=tmp_path / "combined.json",
        ),
        count=2,
        seed=13,
        offset=0,
        strict_invalid_examples=False,
    )

    prepared = prepare_from_raw(config)

    assert prepared.counts["raw_examples"] == 3
    assert prepared.counts["valid_examples"] == 2
    assert prepared.counts["invalid_examples_dropped"] == 1
    assert prepared.counts["invalid_example_reasons"] == {
        "missing_gold_tool_from_public_catalog": 1
    }
    assert len(prepared.task_inputs) == len(prepared.task_labels) == 2
    assert all("gold_tool_ids" not in record for record in prepared.task_inputs)
    assert all(
        "gold-value" not in candidate["text"]
        and "gold-output" not in candidate["text"]
        for record in prepared.task_inputs
        for candidate in record["candidate_tools"]
    )


def test_traject_bench_plan_uses_directory_source_and_frozen_baseline_stages(
    tmp_path: Path,
) -> None:
    raw = _minimal_official_layout(tmp_path / "raw")
    config = _traject_config(
        f"traject-plan-{tmp_path.name}",
        source=raw,
        methods="[bm25,dense]",
    )
    plan = WorkflowPlanner(config, RunLayout(ROOT, config.name)).build(
        validate_external=False
    )

    prepare = next(
        invocation
        for invocation in plan.invocations
        if invocation.identifier == "prepare:train"
    )
    stages = {invocation.stage for invocation in plan.invocations}
    assert config.dataset.source_kind == "directory"
    assert prepare.inputs[0].kind == "directory"
    assert stages == {"prepare", "retrieve", "evaluate", "aggregate"}


def _catalog_tool(
    name: str,
    *,
    description: str = "Catalog-owned description.",
    connected: list[str] | None = None,
) -> dict[str, object]:
    return {
        "parent tool name": name.split(":", 1)[0],
        "parent tool description": "Provider description.",
        "tool name": name,
        "tool description": description,
        "required_parameters": [
            {
                "name": "query",
                "type": "STRING",
                "description": "Search query.",
                "default": "",
            }
        ],
        "optional_parameters": [],
        "code": "",
        "API name": name.split(":", 1)[-1].strip(),
        "domain name": "Education",
        "output_info": {},
        "connected tools": [
            {"tool name": target, "connect params": []}
            for target in (connected or [])
        ],
    }


def _parallel_query(tool_names: list[str]) -> dict[str, object]:
    return {
        "query": "Find the right tools for this task.",
        "tool list": [_query_tool(name) for name in tool_names],
        "trajectory_type": "parallel",
        "tool count": len(tool_names),
        "final_answer": "Done.",
    }


def _sequential_query(tool_names: list[str]) -> dict[str, object]:
    return {
        "query": "Use the tools in dependency order.",
        "tool_list": [_query_tool(name) for name in tool_names],
        "final_answer": {"answer": "Done.", "reason": "Completed."},
        "num_tools_used": len(tool_names),
        "num_successful_tools": len(tool_names),
        "sequence_name": "Test sequence",
        "sequence_description": "A test dependency chain.",
        "domain": "Education",
        "executable": True,
        "generation_info": {},
    }


def _query_tool(name: str) -> dict[str, object]:
    return {
        "tool name": name,
        "tool description": "Gold-local description that must not become a candidate.",
        "required parameters": [{"name": "secret", "value": "gold-value"}],
        "optional parameters": [],
        "executed_output": "gold-output",
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _minimal_official_layout(root: Path) -> Path:
    for directory in ("parallel", "sequential", "tools"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    return root


def _traject_config(name: str, *, source: Path, methods: str):
    source_value = source.resolve().as_posix()
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.3",
    ):
        composed = compose(
            config_name="config",
            overrides=[
                f"name={name}",
                "dataset=traject_bench",
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
            ],
        )
    return resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=ROOT,
    )
