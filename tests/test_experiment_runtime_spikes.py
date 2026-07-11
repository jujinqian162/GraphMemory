from __future__ import annotations

import json
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
    assert rows[0]["output_dir"].endswith("/spike/0_dataset=twowiki,seed=1")
    assert rows[1]["output_dir"].endswith("/spike/1_dataset=twowiki,seed=2")


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
