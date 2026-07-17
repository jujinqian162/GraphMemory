from __future__ import annotations

# ruff: noqa: E402 -- experiment/inspect.py otherwise shadows stdlib inspect

import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
sys.path = [entry for entry in sys.path if Path(entry).resolve() != SCRIPT_DIRECTORY]
sys.path.insert(0, str(REPOSITORY_ROOT))

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from graph_memory.experiment.config import (
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.experiment.output import resolved_overrides
from graph_memory.experiment.tracking import configure_tracking
from graph_memory.experiment.workflow import run_experiment


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(composed: DictConfig) -> None:
    config = resolve_experiment_config(
        parse_composed_config(composed),
        repository_root=REPOSITORY_ROOT,
    )
    run_output = Path(HydraConfig.get().runtime.output_dir).resolve()
    configure_tracking(config)
    if mlflow.active_run() is not None:
        raise RuntimeError("experiment entrypoint requires no pre-existing MLflow run")

    mlflow.start_run(run_name=config.name)
    try:
        result = run_experiment.with_options(flow_run_name=config.name)(
            config,
            run_output=run_output,
            overrides=resolved_overrides(),
        )
    except BaseException:
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")

    metric_row = result.evaluation.metric_rows[0]
    print(
        f"run={config.name} method={result.method} "
        f"variant={result.variant or 'none'} "
        f"recall@10={metric_row.get('Recall@10', 'NA')}"
    )


if __name__ == "__main__":
    main()
