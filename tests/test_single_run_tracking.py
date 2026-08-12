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


def test_tracking_configuration_uses_fluent_experiment_api(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        tracking.mlflow,
        "set_tracking_uri",
        lambda value: calls.append(("set_tracking_uri", value)),
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "get_experiment_by_name",
        lambda value: SimpleNamespace(name=value),
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "create_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("existing experiment must not be recreated")
        ),
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "set_experiment",
        lambda value: calls.append(("set_experiment", value)),
    )

    config = _config("tracking-config")
    tracking.configure_tracking(config)

    assert calls == [
        ("set_tracking_uri", config.tracking.tracking_uri),
        ("set_experiment", config.tracking.experiment_name),
    ]


def test_one_active_run_receives_final_metrics_tags_and_assets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(tracking.mlflow, "active_run", lambda: object())
    for name in ("log_params", "set_tags", "log_metrics"):
        monkeypatch.setattr(
            tracking.mlflow,
            name,
            lambda value, _name=name, **kwargs: captured.setdefault(_name, []).append(
                (value, kwargs)
            ),
        )
    monkeypatch.setattr(
        tracking.mlflow,
        "log_dict",
        lambda value, path: captured.setdefault("log_dict", []).append((value, path)),
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "log_artifacts",
        lambda path: captured.setdefault("log_artifacts", []).append(path),
    )
    monkeypatch.setattr(
        tracking,
        "artifact_csv_rows",
        lambda *_args: [{
            "Recall@10": "0.75",
            "Evidence Density@5": "0.25",
            "Evidence Density@10": "0.2",
            "Coverage@256 Tokens": "0.2",
            "Coverage@512 Tokens": "0.4",
            "Coverage@1024 Tokens": "0.6",
            "Coverage@2048 Tokens": "0.8",
            "Coverage@4096 Tokens": "0.9",
            "Coverage@8192 Tokens": "1.0",
            "Full Support@256 Tokens": "0.1",
            "Full Support@512 Tokens": "0.2",
            "Full Support@1024 Tokens": "0.4",
            "Full Support@2048 Tokens": "0.6",
            "Full Support@4096 Tokens": "0.8",
            "Full Support@8192 Tokens": "1.0",
            "Coverage Budget-AUC": "0.85",
            "Full Support Budget-AUC": "0.75",
            "Span F1@2048 Tokens": "0.4",
            "Evidence Density@2048 Tokens": "0.3",
            "MRR": "0.5",
        }],
    )
    monkeypatch.setattr(tracking, "prediction_production_seconds", lambda _ref: 0.0)
    monkeypatch.setattr(tracking, "artifact_shape_count", lambda *_args: 0)
    result = SimpleNamespace(
        method="bm25",
        variant=None,
        ranking=object(),
        evaluation=object(),
        assets=(),
        model=None,
    )
    run_output = tmp_path / "run"
    run_output.mkdir()

    tracking.log_experiment_result(
        _config("tracking-result"),
        cast(Any, result),
        run_output=run_output,
        prefect_flow_run_id="prefect-flow-123",
    )

    tags = captured["set_tags"][0][0]
    metrics = captured["log_metrics"][0][0]
    assert "graph_memory.cache_used" not in tags
    assert tags["graph_memory.variant"] == "none"
    assert tags["graph_memory.prefect_flow_run_id"] == "prefect-flow-123"
    assert metrics["final.recall_at_10"] == 0.75
    assert metrics["final.mrr"] == 0.5
    assert metrics["final.evidence_density_at_5"] == 0.25
    assert metrics["final.evidence_density_at_10"] == 0.2
    assert metrics["final.coverage_at_256_tokens"] == 0.2
    assert metrics["final.coverage_at_512_tokens"] == 0.4
    assert metrics["final.coverage_at_1024_tokens"] == 0.6
    assert metrics["final.coverage_at_2048_tokens"] == 0.8
    assert metrics["final.coverage_at_4096_tokens"] == 0.9
    assert metrics["final.coverage_at_8192_tokens"] == 1.0
    assert metrics["final.full_support_at_256_tokens"] == 0.1
    assert metrics["final.full_support_at_512_tokens"] == 0.2
    assert metrics["final.full_support_at_1024_tokens"] == 0.4
    assert metrics["final.full_support_at_2048_tokens"] == 0.6
    assert metrics["final.full_support_at_4096_tokens"] == 0.8
    assert metrics["final.full_support_at_8192_tokens"] == 1.0
    assert metrics["final.coverage_budget_auc"] == 0.85
    assert metrics["final.full_support_budget_auc"] == 0.75
    assert metrics["final.span_f1_at_2048_tokens"] == 0.4
    assert metrics["final.evidence_density_at_2048_tokens"] == 0.3
    assert metrics["final.retrieval_latency_ms_per_query"] == 0.0
    assert {path for _, path in captured["log_dict"]} == {"assets/manifest.json"}
    assert captured["log_artifacts"] == [str(run_output)]


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
