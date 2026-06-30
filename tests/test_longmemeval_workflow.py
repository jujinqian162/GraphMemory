from __future__ import annotations

from pathlib import Path
from typing import Any

from graph_memory.io import read_json, write_json
from scripts import build_graphs
from scripts.workflow.manifest import initialize_experiment, list_recipe_specs, load_experiment_config
from scripts.workflow.planner import build_stage_plan
from scripts.workflow.status import inspect_experiment_status
from scripts.workflow.types import StageId

DENSE_FT = "dense_ft"
RGCN = "dense_rgcn_graph_retriever"
DENSE_FT_SEEDED_RGCN = "dense_ft_rgcn_graph_retriever"
TRAINABLE_METHODS = (DENSE_FT, RGCN, DENSE_FT_SEEDED_RGCN)
LONGMEMEVAL_DENSE_FT_CONFIG = "configs/methods/longmemeval_dense_ft.json"
LONGMEMEVAL_RGCN_CONFIG = "configs/methods/longmemeval_dense_rgcn_graph_retriever.json"
LONGMEMEVAL_DENSE_FT_RGCN_CONFIG = "configs/methods/longmemeval_dense_ft_rgcn_graph_retriever.json"


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
        "recipe": "longmemeval_v1_retrieval",
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
    assert "longmemeval_v1_memory_retrieval" not in recipes
    assert "longmemeval_v1_graph_retrieval" not in recipes
    assert "longmemeval_v1_retrieval" in recipes
    for row in recipes.values():
        if row["dataset"] in {"hotpotqa", "twowiki"}:
            assert "memory_stream" not in row["methods"]
    assert recipes["longmemeval_v1_retrieval"]["methods"] == (
        "bm25, dense, memory_stream, bm25_graph_rerank, dense_graph_rerank, "
        "dense_ft, dense_rgcn_graph_retriever, dense_ft_rgcn_graph_retriever"
    )


def test_longmemeval_retrieval_config_initializes_without_hotpotqa_importance(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "longmemeval-active-memory-config",
        config=load_experiment_config("longmemeval_v1_retrieval"),
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


def test_longmemeval_memory_stream_search_space_tunes_recency_not_importance() -> None:
    config = load_experiment_config("longmemeval_v1_retrieval")
    search_space_path = Path(str(config["search_spaces"]["memory_stream"]))
    search_space = read_json(search_space_path)

    assert search_space["importance_weight"] == [0.0]
    recency_weights = [float(value) for value in search_space["recency_weight"]]
    assert 0.0 in recency_weights
    assert float(config["memory_stream_recency_weight"]) in recency_weights
    assert max(recency_weights) >= 0.5


def test_longmemeval_retrieval_config_supports_method_subset_override(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "longmemeval-method-subset",
        config=load_experiment_config("longmemeval_v1_retrieval"),
        run_root=tmp_path,
        profile="smoke",
        methods=["bm25", "dense", "memory_stream"],
        force=True,
    )

    assert manifest["selected_methods"] == ["bm25", "dense", "memory_stream"]
    assert set(manifest["artifacts"]["predictions"]) == {"bm25", "dense", "memory_stream"}


def test_longmemeval_retrieval_config_exposes_trainable_methods() -> None:
    config = load_experiment_config("longmemeval_v1_retrieval")

    assert set(TRAINABLE_METHODS).issubset(set(config["methods"]))
    assert config["method_configs"] == {
        RGCN: LONGMEMEVAL_RGCN_CONFIG,
        DENSE_FT: LONGMEMEVAL_DENSE_FT_CONFIG,
        DENSE_FT_SEEDED_RGCN: LONGMEMEVAL_DENSE_FT_RGCN_CONFIG,
    }


def test_longmemeval_cloud_full_dense_ft_uses_long_context_batch_size(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "longmemeval-dense-ft-cloud-full",
        config=load_experiment_config("longmemeval_v1_retrieval"),
        run_root=tmp_path,
        profile="cloud-full",
        methods=[DENSE_FT],
        force=True,
    )

    resolved_dense_ft = manifest["effective_config"]["resolved_method_configs"][DENSE_FT]
    train_config = read_json(manifest["stage_configs"]["train"][DENSE_FT])

    assert resolved_dense_ft["train"]["trainer"]["train_batch_size"] == 8
    assert resolved_dense_ft["train"]["trainer"]["eval_batch_size"] == 32
    assert train_config["job"]["trainer"]["train_batch_size"] == 8
    assert train_config["job"]["trainer"]["eval_batch_size"] == 32


def test_longmemeval_cloud_full_rgcn_configs_use_smaller_batch_and_more_epochs(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "longmemeval-rgcn-cloud-full",
        config=load_experiment_config("longmemeval_v1_retrieval"),
        run_root=tmp_path,
        profile="cloud-full",
        methods=[RGCN, DENSE_FT_SEEDED_RGCN],
        force=True,
    )

    for method in (RGCN, DENSE_FT_SEEDED_RGCN):
        resolved_method = manifest["effective_config"]["resolved_method_configs"][method]
        train_config = read_json(manifest["stage_configs"]["train"][method])

        assert resolved_method["train"]["trainer"]["batch_size"] == 8
        assert resolved_method["train"]["trainer"]["epochs"] == 15
        assert train_config["job"]["trainer"]["batch_size"] == 8
        assert train_config["job"]["trainer"]["epochs"] == 15


def test_longmemeval_trainable_stage_configs_use_dataset_and_dependencies(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "longmemeval-trainable",
        config=load_experiment_config("longmemeval_v1_retrieval"),
        run_root=tmp_path,
        profile="smoke",
        methods=list(TRAINABLE_METHODS),
        force=True,
    )

    assert manifest["selected_methods"] == list(TRAINABLE_METHODS)
    for method in TRAINABLE_METHODS:
        for stage in ("pairs", "train", "retrieve", "evaluate"):
            stage_config = read_json(manifest["stage_configs"][stage][method])
            assert stage_config["dataset"] == "longmemeval"

    dense_ft_train = read_json(manifest["stage_configs"]["train"][DENSE_FT])
    dense_ft_retrieve = read_json(manifest["stage_configs"]["retrieve"][DENSE_FT])
    rgcn_train = read_json(manifest["stage_configs"]["train"][RGCN])
    rgcn_retrieve = read_json(manifest["stage_configs"]["retrieve"][RGCN])
    seeded_rgcn_train = read_json(manifest["stage_configs"]["train"][DENSE_FT_SEEDED_RGCN])
    seeded_rgcn_retrieve = read_json(manifest["stage_configs"]["retrieve"][DENSE_FT_SEEDED_RGCN])

    assert dense_ft_train["job"]["trainer"]["device"] == "cuda"
    assert dense_ft_retrieve["job"]["device"] == "cuda"
    assert dense_ft_retrieve["io"]["graphs"] is None
    assert "train_graphs" not in dense_ft_train["io"]
    assert "dev_graphs" not in dense_ft_train["io"]

    assert rgcn_train["job"]["trainer"]["device"] == "cuda"
    assert rgcn_retrieve["job"]["device"] == "cuda"
    assert isinstance(rgcn_train["io"]["train_graphs"], str)
    assert isinstance(rgcn_train["io"]["dev_graphs"], str)
    assert isinstance(rgcn_retrieve["io"]["graphs"], str)

    assert seeded_rgcn_train["job"]["trainer"]["device"] == "cuda"
    assert seeded_rgcn_retrieve["job"]["device"] == "cuda"
    assert Path(seeded_rgcn_train["io"]["seed_checkpoint"]) == Path(dense_ft_train["io"]["model_dir"])
    assert isinstance(seeded_rgcn_retrieve["io"]["graphs"], str)


def test_longmemeval_retrieval_config_has_cloud_full_profile_for_all_valid_cleaned_s_examples() -> None:
    config = load_experiment_config("longmemeval_v1_retrieval")

    assert config["profiles"]["cloud-full"] == {
        "train_examples": 300,
        "dev_examples": 100,
        "test_examples": 40,
    }
    assert config["split_offsets"] == {
        "train": 0,
        "dev": 300,
        "test": 400,
    }
