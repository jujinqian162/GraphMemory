from __future__ import annotations

import json
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from mlflow import MlflowClient
from omegaconf import OmegaConf


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "refactor_7_9"
HYDRA_CONFIG_DIR = FIXTURE_ROOT / "hydra_spike"


def test_hydra_spike_composes_dataset_group_and_rejects_unknown_override() -> None:
    with initialize_config_dir(
        version_base="1.3", config_dir=str(HYDRA_CONFIG_DIR.resolve())
    ):
        config = compose(
            config_name="config", overrides=["name=spike", "dataset=twowiki"]
        )
        resolved = OmegaConf.to_container(config, resolve=True)
        assert isinstance(resolved, dict)
        assert resolved["dataset"] == {
            "name": "twowiki",
            "source": "data/2wiki/raw/dev.json",
        }
        assert resolved["dataset_name"] == "twowiki"
        assert resolved["dataset_source"] == "data/2wiki/raw/dev.json"

        with pytest.raises(ConfigCompositionException):
            compose(config_name="config", overrides=["name=spike", "unknown_key=1"])


def test_hydra_spike_uses_basic_launcher_without_chdir_and_sequential_multirun(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "multirun.jsonl"
    output_root = tmp_path / "outputs"
    environment = {
        **os.environ,
        "HYDRA_SPIKE_LOG": str(log_path),
        "HYDRA_SPIKE_ROOT": str(output_root),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(FIXTURE_ROOT / "hydra_spike_app.py"),
            "-m",
            "name=spike",
            "dataset=twowiki",
            "seed=1,2",
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["job_num"] for row in rows] == [0, 1]
    assert [row["seed"] for row in rows] == [1, 2]
    assert [row["dataset"] for row in rows] == ["twowiki", "twowiki"]
    assert all(
        row["cwd"] == Path(__file__).parents[1].resolve().as_posix() for row in rows
    )
    assert all(row["launcher"].endswith("BasicLauncher") for row in rows)
    assert rows[0]["output_dir"].endswith("/spike/0_seed=1")
    assert rows[1]["output_dir"].endswith("/spike/1_seed=2")


def test_hydra_134_concise_num_layers_multirun_matches_run_layout(
    tmp_path: Path,
) -> None:
    assert importlib.metadata.version("hydra-core") == "1.3.4"
    log_path = tmp_path / "multirun.jsonl"
    output_root = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            str(FIXTURE_ROOT / "hydra_spike_app.py"),
            "-m",
            "name=layers",
            "dataset=twowiki",
            "method_configs.dense_rgcn_graph_retriever.train.model.num_layers=2,3,4",
        ],
        cwd=Path(__file__).parents[1],
        env={
            **os.environ,
            "HYDRA_SPIKE_LOG": str(log_path),
            "HYDRA_SPIKE_ROOT": str(output_root),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    expected = ["0_num_layers=2", "1_num_layers=3", "2_num_layers=4"]
    assert [Path(row["output_dir"]).name for row in rows] == expected
    assert [Path(row["layout_dir"]).name for row in rows] == expected
    assert [row["suffix"] for row in rows] == [
        "num_layers=2",
        "num_layers=3",
        "num_layers=4",
    ]


def test_mlflow_sqlite_spike_persists_parent_children_and_curated_artifact(
    tmp_path: Path,
) -> None:
    database = (tmp_path / "mlflow.db").resolve()
    artifact_root = (tmp_path / "artifacts").resolve()
    tracking_uri = f"sqlite:///{database.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        "refactor-runtime-spike",
        artifact_location=artifact_root.as_uri(),
    )
    curated = tmp_path / "resolved_config.yaml"
    curated.write_text("seed: 13\n", encoding="utf-8")
    parent_id = ""
    first_child_id = ""
    resumed_child_id = ""

    with mlflow.start_run(experiment_id=experiment_id, run_name="parent") as parent:
        parent_id = parent.info.run_id
        mlflow.log_param("dataset", "hotpotqa")
        mlflow.log_artifact(str(curated), artifact_path="curated")
        with mlflow.start_run(run_name="prepare", nested=True) as first_child:
            first_child_id = first_child.info.run_id
            mlflow.log_metric("selected_examples", 1.0)

    with mlflow.start_run(run_id=parent_id):
        with mlflow.start_run(run_name="retrieve", nested=True) as resumed_child:
            resumed_child_id = resumed_child.info.run_id
            mlflow.log_metric("recall_at_5", 1.0)

    client = MlflowClient(tracking_uri=tracking_uri)
    parent_run = client.get_run(parent_id)
    first_run = client.get_run(first_child_id)
    resumed_run = client.get_run(resumed_child_id)
    assert parent_run.data.params["dataset"] == "hotpotqa"
    assert first_run.data.tags["mlflow.parentRunId"] == parent_id
    assert resumed_run.data.tags["mlflow.parentRunId"] == parent_id
    assert first_run.data.metrics["selected_examples"] == 1.0
    assert resumed_run.data.metrics["recall_at_5"] == 1.0
    assert [item.path for item in client.list_artifacts(parent_id, "curated")] == [
        "curated/resolved_config.yaml"
    ]


def test_mlflow_314_note_content_accepts_plain_text_markdown_table_and_csv_fallback(
    tmp_path: Path,
) -> None:
    assert mlflow.__version__ == "3.14.0"
    database = (tmp_path / "mlflow.db").resolve()
    artifact_root = (tmp_path / "artifacts").resolve()
    tracking_uri = f"sqlite:///{database.as_posix()}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        "mlflow-note-spike",
        artifact_location=artifact_root.as_uri(),
    )
    run_id = client.create_run(experiment_id).info.run_id
    table = "Final baseline results\n\n| Method | Recall@10 |\n| --- | --- |\n| bm25 | 0.5 |"
    result_csv = tmp_path / "main_results.csv"
    result_csv.write_text("Method,Recall@10\nbm25,0.5\n", encoding="utf-8")

    client.set_tag(run_id, "mlflow.note.content", table)
    client.log_artifact(run_id, str(result_csv), artifact_path="results")

    assert client.get_run(run_id).data.tags["mlflow.note.content"] == table
    assert [item.path for item in client.list_artifacts(run_id, "results")] == [
        "results/main_results.csv"
    ]
