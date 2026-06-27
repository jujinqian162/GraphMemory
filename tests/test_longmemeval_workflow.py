from __future__ import annotations

from pathlib import Path
from typing import Any

from graph_memory.io import read_json, write_json
from scripts import build_graphs
from scripts.workflow.manifest import initialize_experiment, list_recipe_specs, load_experiment_config
from scripts.workflow.planner import build_stage_plan
from scripts.workflow.status import inspect_experiment_status
from scripts.workflow.types import StageId


def _longmemeval_workflow_config() -> dict[str, Any]:
    return {
        "dataset": "longmemeval",
        "enable_ablation": False,
        "defaults": {
            "dense_encoder": "models/intfloat-e5-base-v2",
            "passage_prefix": "passage: ",
            "query_prefix": "query: ",
            "seed": 13,
            "top_k": 10,
        },
        "graph": {
            "max_bridge_edges": 50,
            "max_entity_neighbors": 10,
            "max_query_overlap": 20,
            "use_spacy": False,
        },
        "methods": ["bm25", "dense"],
        "profiles": {
            "smoke": {
                "dev_examples": 1,
                "test_examples": 1,
                "train_examples": 1,
            }
        },
        "raw": {
            "cleaned_s": "data/longmemeval/raw/longmemeval_s_cleaned.json",
        },
        "recipe": "longmemeval_v1_memory_retrieval",
        "split_offsets": {
            "dev": 1,
            "test": 2,
            "train": 0,
        },
        "split_sources": {
            "dev": "cleaned_s",
            "test": "cleaned_s",
            "train": "cleaned_s",
        },
        "task": "long_memory_retrieval",
    }


def _longmemeval_task() -> dict[str, object]:
    return {
        "task_id": "longmem_q1",
        "question": "Where did I say I planned to meet Alex?",
        "question_datetime": "2024-01-10T12:00:00",
        "candidate_items": [
            {
                "item_id": "m0",
                "session_id": "s1",
                "session_order": 0,
                "turn_index": 0,
                "global_position": 0,
                "role": "user",
                "datetime": "2024-01-01T09:00:00",
                "text": "Let's meet Alex at the library tomorrow.",
            }
        ],
        "metadata": {
            "dataset": "longmemeval_v1",
            "raw_id": "q1",
            "question_type": "single-session-user",
            "candidate_granularity": "turn",
        },
    }


def test_longmemeval_workflow_uses_dataset_specific_prepare_and_stage_configs(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "longmemeval-workflow",
        config=_longmemeval_workflow_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=["bm25", "dense"],
        force=True,
    )

    commands = build_stage_plan(manifest, from_stage="prepare", to_stage="graphs", methods=["bm25"])
    prepare_command = next(command for command in commands if command.stage is StageId.PREPARE and command.split == "dev")
    graph_command = next(command for command in commands if command.stage is StageId.GRAPHS and command.split == "dev")

    assert manifest["effective_config"]["dataset"] == "longmemeval"
    assert prepare_command.argv[1] == "scripts/prepare_longmemeval.py"
    assert prepare_command.argv[prepare_command.argv.index("--input") + 1] == "data/longmemeval/raw/longmemeval_s_cleaned.json"
    assert graph_command.argv[1] == "scripts/build_graphs.py"
    assert graph_command.argv[graph_command.argv.index("--dataset") + 1] == "longmemeval"

    for method in ("bm25", "dense"):
        retrieve_config = read_json(manifest["stage_configs"]["retrieve"][method])
        evaluate_config = read_json(manifest["stage_configs"]["evaluate"][method])
        assert retrieve_config["dataset"] == "longmemeval"
        assert evaluate_config["dataset"] == "longmemeval"


def test_build_graphs_script_accepts_longmemeval_dataset_selector(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    graphs_path = tmp_path / "graphs.json"
    write_json(tasks_path, [_longmemeval_task()])

    assert build_graphs.main(["--dataset", "longmemeval", "--input", str(tasks_path), "--output", str(graphs_path)]) == 0

    graphs = read_json(graphs_path)
    assert graphs[0]["task_id"] == "longmem_q1"
    assert graphs[0]["nodes"][1]["node_kind"] == "conversation_turn"
    assert graphs[0]["nodes"][1]["metadata"]["global_position"] == 0


def test_longmemeval_memory_stream_stage_config_does_not_default_to_hotpotqa_importance(tmp_path: Path) -> None:
    config = _longmemeval_workflow_config()
    config["methods"] = ["memory_stream"]
    config["memory_stream_relevance_weight"] = 1.0
    config["memory_stream_recency_weight"] = 0.1
    config["memory_stream_importance_weight"] = 0.0
    config["memory_stream_recency_decay"] = 0.99

    manifest = initialize_experiment(
        "longmemeval-memory-stream-workflow",
        config=config,
        run_root=tmp_path,
        profile="smoke",
        methods=["memory_stream"],
        force=True,
    )

    retrieve_config = read_json(manifest["stage_configs"]["retrieve"]["memory_stream"])
    assert retrieve_config["dataset"] == "longmemeval"
    assert retrieve_config["io"]["importance"] is None
    assert "data/hotpotqa/processed/memory_stream" not in str(retrieve_config)
    assert retrieve_config["job"]["scoring"] == {
        "relevance_weight": 1.0,
        "recency_weight": 0.1,
        "importance_weight": 0.0,
        "recency_decay": 0.99,
    }

def test_longmemeval_memory_stream_tune_command_uses_dataset_without_hotpotqa_importance(tmp_path: Path) -> None:
    config = _longmemeval_workflow_config()
    config["methods"] = ["memory_stream"]
    config["search_spaces"] = {
        "memory_stream": "configs/search_spaces/memory_stream.json",
    }
    config["memory_stream_importance_weight"] = 0.0

    manifest = initialize_experiment(
        "longmemeval-memory-stream-tune",
        config=config,
        run_root=tmp_path,
        profile="smoke",
        methods=["memory_stream"],
        force=True,
    )

    commands = build_stage_plan(
        manifest,
        from_stage="tune",
        to_stage="tune",
        methods=["memory_stream"],
    )
    tune_command = commands[0]

    assert tune_command.argv[tune_command.argv.index("--dataset") + 1] == "longmemeval"
    assert "--importance" not in tune_command.argv
    assert "data/hotpotqa/processed/memory_stream" not in str(tune_command.argv)

def test_active_experiment_configs_route_memory_stream_to_longmemeval_only() -> None:
    recipes = {row["name"]: row for row in list_recipe_specs()}

    assert "hotpotqa_memory_stream" not in recipes
    assert "longmemeval_v1_memory_retrieval" in recipes
    assert "longmemeval_v1_graph_retrieval" in recipes
    for row in recipes.values():
        if row["dataset"] in {"hotpotqa", "twowiki"}:
            assert "memory_stream" not in row["methods"]
    assert "memory_stream" in recipes["longmemeval_v1_memory_retrieval"]["methods"]


def test_longmemeval_memory_retrieval_config_initializes_without_hotpotqa_importance(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "longmemeval-active-memory-config",
        config=load_experiment_config("longmemeval_v1_memory_retrieval"),
        run_root=tmp_path,
        profile="smoke",
        methods=["memory_stream"],
        force=True,
    )

    commands = build_stage_plan(
        manifest,
        from_stage="tune",
        to_stage="retrieve",
        methods=["memory_stream"],
    )
    tune_command = next(command for command in commands if command.stage is StageId.TUNE)
    retrieve_config = read_json(manifest["stage_configs"]["retrieve"]["memory_stream"])
    rendered = f"{manifest} {commands} {retrieve_config}"

    assert manifest["effective_config"]["dataset"] == "longmemeval"
    assert manifest["effective_config"]["splits"]["train"]["source"] == "cleaned_s"
    assert tune_command.argv[tune_command.argv.index("--dataset") + 1] == "longmemeval"
    assert "--importance" not in tune_command.argv
    assert retrieve_config["io"]["importance"] is None
    status_rows = inspect_experiment_status(manifest)
    tune_status = next(row for row in status_rows if row["stage"] == "tune")
    assert tune_status["state"] == "missing"
    assert "data/hotpotqa/processed/memory_stream" not in rendered
