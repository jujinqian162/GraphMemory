from __future__ import annotations

from pathlib import Path
from typing import Any

from graph_memory.io import read_json
from scripts.workflow.manifest import initialize_experiment, load_experiment_config
from scripts.workflow.planner import build_stage_plan
from scripts.workflow.types import StageId


DENSE_FT = "dense_ft"
RGCN = "dense_rgcn_graph_retriever"
TRAINABLE_METHODS = (DENSE_FT, RGCN)


def _twowiki_workflow_config() -> dict[str, Any]:
    return {
        "dataset": "twowiki",
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
        "method_configs": {
            "dense_rgcn_graph_retriever": "configs/methods/dense_rgcn_graph_retriever.json",
        },
        "methods": ["bm25", "dense_rgcn_graph_retriever"],
        "profiles": {
            "smoke": {
                "dev_examples": 1,
                "test_examples": 1,
                "train_examples": 1,
            }
        },
        "raw": {
            "dev": "data/2wiki/raw/dev.json",
            "train": "data/2wiki/raw/train.json",
        },
        "recipe": "2wiki_evidence_retrieval",
        "split_offsets": {
            "dev": 0,
            "test": 500,
            "train": 0,
        },
        "split_sources": {
            "dev": "dev",
            "test": "dev",
            "train": "train",
        },
        "task": "evidence_retrieval",
    }


def test_twowiki_workflow_uses_dataset_specific_prepare_and_stage_configs(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "twowiki-workflow",
        config=_twowiki_workflow_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=["bm25", "dense_rgcn_graph_retriever"],
        force=True,
    )

    commands = build_stage_plan(manifest, from_stage="prepare", to_stage="graphs", methods=["bm25"])
    prepare_command = next(command for command in commands if command.stage is StageId.PREPARE and command.split == "dev")
    graph_command = next(command for command in commands if command.stage is StageId.GRAPHS and command.split == "dev")

    assert manifest["effective_config"]["dataset"] == "twowiki"
    assert prepare_command.argv[1] == "scripts/prepare_2wiki.py"
    assert prepare_command.argv[prepare_command.argv.index("--input") + 1] == "data/2wiki/raw/dev.json"
    assert graph_command.argv[1] == "scripts/build_graphs.py"
    assert graph_command.argv[graph_command.argv.index("--dataset") + 1] == "twowiki"

    for stage in ("pairs", "train", "retrieve", "evaluate"):
        config = read_json(manifest["stage_configs"][stage]["dense_rgcn_graph_retriever"])
        assert config["dataset"] == "twowiki"

    bm25_retrieve = read_json(manifest["stage_configs"]["retrieve"]["bm25"])
    bm25_evaluate = read_json(manifest["stage_configs"]["evaluate"]["bm25"])
    assert bm25_retrieve["dataset"] == "twowiki"
    assert bm25_evaluate["dataset"] == "twowiki"


def test_named_twowiki_tiny_config_initializes_dataset_workflow(tmp_path: Path) -> None:
    config = load_experiment_config("2wiki_tiny")

    manifest = initialize_experiment(
        "2wiki-tiny",
        config=config,
        run_root=tmp_path,
        methods=["bm25", "dense_graph_rerank"],
        force=True,
    )
    commands = build_stage_plan(
        manifest,
        from_stage="prepare",
        to_stage="evaluate",
        methods=["bm25", "dense_graph_rerank"],
    )

    assert manifest["effective_config"]["dataset"] == "twowiki"
    assert manifest["effective_config"]["profile"] == "smoke"
    assert manifest["effective_config"]["splits"]["test"]["source"] == "dev"
    assert manifest["effective_config"]["raw"]["dev"] == "data/2wiki/raw/dev.json"
    assert manifest["effective_config"]["raw"]["train"] == "data/2wiki/raw/train.json"
    assert any(command.argv[1] == "scripts/prepare_2wiki.py" for command in commands)
    prepare_command = next(command for command in commands if command.stage is StageId.PREPARE and command.split == "test")
    assert prepare_command.argv[prepare_command.argv.index("--input") + 1] == "data/2wiki/raw/dev.json"
    assert prepare_command.argv[prepare_command.argv.index("--max_examples") + 1] == "5"
    assert any(
        command.stage is StageId.GRAPHS
        and command.argv[command.argv.index("--dataset") + 1] == "twowiki"
        for command in commands
    )


def test_named_twowiki_tiny_config_exposes_trainable_methods() -> None:
    config = load_experiment_config("2wiki_tiny")

    assert set(TRAINABLE_METHODS).issubset(set(config["methods"]))
    assert config["method_configs"] == {
        RGCN: "configs/methods/dense_rgcn_graph_retriever.json",
        DENSE_FT: "configs/methods/dense_ft.json",
    }


def test_named_twowiki_tiny_trainable_stage_configs_use_dataset_cuda_and_graph_boundaries(
    tmp_path: Path,
) -> None:
    config = load_experiment_config("2wiki_tiny")

    manifest = initialize_experiment(
        "2wiki-tiny-trainable",
        config=config,
        run_root=tmp_path,
        profile="smoke",
        methods=list(TRAINABLE_METHODS),
        force=True,
    )

    assert manifest["selected_methods"] == list(TRAINABLE_METHODS)
    for method in TRAINABLE_METHODS:
        for stage in ("pairs", "train", "retrieve", "evaluate"):
            stage_config = read_json(manifest["stage_configs"][stage][method])
            assert stage_config["dataset"] == "twowiki"

    dense_ft_train = read_json(manifest["stage_configs"]["train"][DENSE_FT])
    dense_ft_retrieve = read_json(manifest["stage_configs"]["retrieve"][DENSE_FT])
    rgcn_train = read_json(manifest["stage_configs"]["train"][RGCN])
    rgcn_retrieve = read_json(manifest["stage_configs"]["retrieve"][RGCN])

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


def test_named_twowiki_evidence_retrieval_config_matches_full_method_workflow(tmp_path: Path) -> None:
    config = load_experiment_config("2wiki_evidence_retrieval")

    manifest = initialize_experiment(
        "2wiki-evidence",
        config=config,
        run_root=tmp_path,
        force=True,
    )

    assert manifest["effective_config"]["dataset"] == "twowiki"
    assert manifest["effective_config"]["profile"] == "quick"
    assert manifest["selected_methods"] == [
        "bm25",
        "dense",
        "bm25_graph_rerank",
        "dense_graph_rerank",
        "fast_graphrag",
        "dense_rgcn_graph_retriever",
        "dense_ft",
    ]
    assert manifest["effective_config"]["splits"]["test"] == {
        "source": "dev",
        "max_examples": 100,
        "seed": 13,
        "offset": 500,
    }
    assert manifest["effective_config"]["raw"] == {
        "dev": "data/2wiki/raw/dev.json",
        "train": "data/2wiki/raw/train.json",
    }
    assert config["profiles"]["cloud-full"] == {
        "dev_examples": 500,
        "test_examples": 12076,
        "train_examples": 167454,
    }
    assert manifest["effective_config"]["resolved_method_configs"]["dense_rgcn_graph_retriever"]["method"] == (
        "dense_rgcn_graph_retriever"
    )
    assert manifest["effective_config"]["resolved_method_configs"]["dense_ft"]["method"] == "dense_ft"
