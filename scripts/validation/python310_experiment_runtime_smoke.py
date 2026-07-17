from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
import mlflow
import prefect
import pydantic
from omegaconf import OmegaConf

import graph_memory
from graph_memory.experiment.artifacts import ArtifactKind
from graph_memory.experiment.workflow import run_experiment


def main() -> int:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Expected Python 3.10, got {sys.version}")
    if not hydra.__version__.startswith("1.3."):
        raise RuntimeError(f"Expected Hydra 1.3.x, got {hydra.__version__}")
    if not pydantic.__version__.startswith("2."):
        raise RuntimeError(f"Expected Pydantic V2, got {pydantic.__version__}")
    if not mlflow.__version__.startswith("3."):
        raise RuntimeError(f"Expected MLflow 3.x, got {mlflow.__version__}")
    if not prefect.__version__.startswith("3."):
        raise RuntimeError(f"Expected Prefect 3.x, got {prefect.__version__}")
    if (
        ArtifactKind.MODEL.value != "model"
        or run_experiment.name != "graph-memory-experiment"
    ):
        raise RuntimeError(
            "Prefect workflow and processed artifact imports are inconsistent"
        )

    resolved = OmegaConf.to_container(
        OmegaConf.create({"root": 13, "copy": "${root}"}),
        resolve=True,
    )
    if resolved != {"root": 13, "copy": 13}:
        raise RuntimeError(f"OmegaConf interpolation failed: {resolved!r}")

    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "hydra-core": version("hydra-core"),
                "pydantic": version("pydantic"),
                "mlflow": version("mlflow"),
                "prefect": version("prefect"),
                "graph_memory": str(graph_memory.__path__),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
