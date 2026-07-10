from __future__ import annotations

from pathlib import Path

from graph_memory.io import read_json
from scripts.workflow.manifest import initialize_experiment, load_experiment_config
from scripts.workflow.planner import build_stage_plan
from scripts.workflow.types import StageId


DENSE_FT = "dense_ft"
RGCN = "dense_rgcn_graph_retriever"
DENSE_FT_SEEDED_RGCN = "dense_ft_rgcn_graph_retriever"
TRAINABLE_METHODS = (DENSE_FT, RGCN, DENSE_FT_SEEDED_RGCN)


def test_named_musique_evidence_retrieval_config_matches_full_method_workflow(tmp_path: Path) -> None:
    config = load_experiment_config("musique_evidence_retrieval")

    manifest = initialize_experiment(
        "musique-evidence",
        config=config,
        run_root=tmp_path,
        force=True,
    )
    commands = build_stage_plan(
        manifest,
        from_stage="prepare",
        to_stage="evaluate",
        methods=["bm25", "dense_graph_rerank"],
    )

    assert manifest["effective_config"]["dataset"] == "musique"
    assert manifest["effective_config"]["profile"] == "quick"
    assert manifest["selected_methods"] == [
        "bm25",
        "dense",
        "bm25_graph_rerank",
        "dense_graph_rerank",
        RGCN,
        DENSE_FT,
        DENSE_FT_SEEDED_RGCN,
    ]
    assert manifest["effective_config"]["splits"]["test"] == {
        "source": "dev",
        "max_examples": 100,
        "seed": 13,
        "offset": 500,
    }
    assert manifest["effective_config"]["raw"] == {
        "dev": "data/musique/raw/musique_ans_v1.0_dev.jsonl",
        "train": "data/musique/raw/musique_ans_v1.0_train.jsonl",
    }
    assert any(command.argv[1] == "scripts/prepare_musique.py" for command in commands)
    assert any(
        command.stage is StageId.GRAPHS
        and command.argv[command.argv.index("--dataset") + 1] == "musique"
        for command in commands
    )
    assert config["profiles"]["cloud-full"] == {
        "dev_examples": 500,
        "test_examples": 1917,
        "train_examples": 19938,
    }


def test_named_musique_config_exposes_trainable_methods_and_stage_configs(tmp_path: Path) -> None:
    config = load_experiment_config("musique_evidence_retrieval")

    assert set(TRAINABLE_METHODS).issubset(set(config["methods"]))
    assert config["method_configs"] == {
        RGCN: "configs/methods/dense_rgcn_graph_retriever.json",
        DENSE_FT: "configs/methods/dense_ft.json",
        DENSE_FT_SEEDED_RGCN: "configs/methods/dense_ft_rgcn_graph_retriever.json",
    }

    manifest = initialize_experiment(
        "musique-trainable",
        config=config,
        run_root=tmp_path,
        profile="smoke",
        methods=list(TRAINABLE_METHODS),
        force=True,
    )

    for method in TRAINABLE_METHODS:
        for stage in ("pairs", "train", "retrieve", "evaluate"):
            stage_config = read_json(manifest["stage_configs"][stage][method])
            assert stage_config["dataset"] == "musique"
