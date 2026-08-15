from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from hydra import compose, initialize_config_dir

import graph_memory.experiment.tracking as tracking
from graph_memory.experiment.config import (
    ResolvedExperimentConfig,
    parse_composed_config,
    resolve_experiment_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> ResolvedExperimentConfig:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                f"name={name}",
                "dataset=hotpotqa",
                "profile=smoke",
                "method=bm25",
            ],
        )
    return resolve_experiment_config(
        parse_composed_config(composed), repository_root=ROOT
    )


def test_tracking_rejects_logging_without_the_single_active_run(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tracking.mlflow, "active_run", lambda: None)
    with pytest.raises(RuntimeError, match="one active MLflow run"):
        tracking.log_experiment_result(
            _config("tracking-no-active"),
            cast(Any, SimpleNamespace()),
            run_output=tmp_path,
            prefect_flow_run_id="prefect-flow-456",
        )
