from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.workflow.manifest import initialize_experiment
from scripts.workflow.planner import build_stage_plan
from scripts.workflow.registry import METHOD_WORKFLOW_REGISTRY, validate_workflow_registry
from scripts.workflow.types import StageId, WorkflowId
from scripts.workflow.workflows import DENSE_FT_WORKFLOW
from tests.test_experiment_runner import _write_experiment_config

DENSE_FT_METHOD = "dense_ft"


def _write_dense_ft_training_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": DENSE_FT_METHOD,
                "default_profile": "smoke",
                "defaults": {
                    "encoder": {
                        "model": "fake-e5",
                        "query_prefix": "query: ",
                        "passage_prefix": "passage: ",
                        "batch_size": 64,
                    },
                    "pair_sampling": {
                        "random_seed": 13,
                        "easy_random_per_positive": 1,
                        "hard_bm25_per_positive": 1,
                        "hard_dense_per_positive": 0,
                        "hard_graph_neighbor_per_positive": 1,
                        "hard_pool_size": 30,
                    },
                    "data": {"hard_negatives_per_positive": 1},
                    "trainer": {
                        "learning_rate": 0.00002,
                        "train_batch_size": 16,
                        "eval_batch_size": 64,
                        "epochs": 1,
                        "warmup_steps": 0,
                        "max_grad_norm": 1.0,
                        "random_seed": 13,
                        "device": "cuda",
                        "use_amp": False,
                    },
                    "selection": {
                        "best_metric": "eval_dev_cos_sim_map@100",
                        "higher_is_better": True,
                    },
                },
                "profiles": {
                    "smoke": {
                        "trainer": {
                            "train_batch_size": 1,
                            "eval_batch_size": 4,
                            "device": "cpu",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _dense_ft_manifest(tmp_path: Path) -> dict[str, Any]:
    raw_path = Path("tests/fixtures/hotpotqa_smoke.json")
    experiment_config_path = tmp_path / "configs" / "experiments" / "dense_ft.json"
    training_config_path = tmp_path / "configs" / "training" / DENSE_FT_METHOD / "base.json"
    _write_experiment_config(experiment_config_path, raw_path)
    _write_dense_ft_training_config(training_config_path)
    experiment_config = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    experiment_config["methods"] = [DENSE_FT_METHOD]
    experiment_config["training_configs"] = {DENSE_FT_METHOD: str(training_config_path)}
    experiment_config["profiles"]["smoke"] = {
        "train_examples": 1,
        "dev_examples": 1,
        "test_examples": 1,
    }
    return initialize_experiment(
        "dense_ft_smoke",
        config=experiment_config,
        run_root=tmp_path / "runs",
        profile="smoke",
        methods=[DENSE_FT_METHOD],
    )


def test_dense_ft_workflow_is_registered_with_full_train_lifecycle() -> None:
    validate_workflow_registry()

    assert WorkflowId.DENSE_FINETUNE_RETRIEVAL.value == "dense_finetune_retrieval"
    assert METHOD_WORKFLOW_REGISTRY[DENSE_FT_METHOD] is DENSE_FT_WORKFLOW
    assert [step.stage for step in DENSE_FT_WORKFLOW.steps] == [
        StageId.PREPARE,
        StageId.GRAPHS,
        StageId.PAIRS,
        StageId.TRAIN,
        StageId.RETRIEVE,
        StageId.EVALUATE,
        StageId.AGGREGATE,
    ]


def test_dense_ft_manifest_projects_model_directory_train_and_retrieve_configs(tmp_path: Path) -> None:
    manifest = _dense_ft_manifest(tmp_path)
    learned = manifest["artifacts"]["learned"][DENSE_FT_METHOD]
    train_projection = manifest["stage_configs"]["train"][DENSE_FT_METHOD]
    retrieve_projection = manifest["stage_configs"]["retrieve"][DENSE_FT_METHOD]

    assert Path(learned["best_checkpoint"]) == (
        tmp_path / "runs" / "dense_ft_smoke" / "learned" / DENSE_FT_METHOD / "checkpoints" / "best_model"
    )
    assert train_projection["method"] == DENSE_FT_METHOD
    assert train_projection["io"]["model_dir"] == learned["best_checkpoint"]
    assert "train_graphs" not in train_projection["io"]
    assert "dev_graphs" not in train_projection["io"]
    assert retrieve_projection["job"]["checkpoint"] == learned["best_checkpoint"]
    assert retrieve_projection["io"]["graphs"] is None


def test_dense_ft_plan_uses_unified_train_entry_and_checkpoint_only_retrieval(tmp_path: Path) -> None:
    manifest = _dense_ft_manifest(tmp_path)
    commands = build_stage_plan(
        manifest,
        stages=["pairs", "train", "retrieve", "evaluate", "aggregate"],
        methods=[DENSE_FT_METHOD],
    )
    train = next(command for command in commands if command.stage is StageId.TRAIN)
    retrieve = next(command for command in commands if command.stage is StageId.RETRIEVE)

    assert train.argv[1].endswith("scripts/train_method.py")
    assert train.argv[train.argv.index("--method") + 1] == DENSE_FT_METHOD
    assert "--model_dir" in train.argv
    assert "--train_graphs" not in train.argv
    assert "--dev_graphs" not in train.argv
    assert retrieve.argv[retrieve.argv.index("--checkpoint") + 1].endswith(
        "learned/dense_ft/checkpoints/best_model"
    )
    assert "--graphs" not in retrieve.argv
    assert "--encoder_model" not in retrieve.argv
    assert "--query_prefix" not in retrieve.argv
    assert "--passage_prefix" not in retrieve.argv
