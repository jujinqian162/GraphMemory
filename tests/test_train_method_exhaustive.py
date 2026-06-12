from __future__ import annotations

from typing import cast

import pytest

from graph_memory.registry.method_configs import TrainableMethodConfig
from graph_memory.registry.stage_configs import TrainStageConfig
from graph_memory.stages.train import TrainingResult
from scripts.train_method import _metric_records, _output_paths
from scripts.workflow.stage_configs import _train_stage_config


def test_train_config_union_does_not_default_to_dense_ft() -> None:
    unsupported = cast(TrainStageConfig, object())

    with pytest.raises(AssertionError):
        _output_paths(unsupported)


def test_train_result_union_rejects_unknown_result_type() -> None:
    unsupported = cast(TrainingResult, object())

    with pytest.raises(AssertionError):
        _metric_records(unsupported)


def test_workflow_train_config_union_rejects_unknown_config_type() -> None:
    unsupported = cast(TrainableMethodConfig, object())

    with pytest.raises(AssertionError):
        _train_stage_config(
            {"artifacts": {"learned": {"unsupported": {}}}},
            "unsupported",
            unsupported,
        )
