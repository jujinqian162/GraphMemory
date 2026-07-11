from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
import mlflow
import pydantic
from omegaconf import OmegaConf

import graph_memory


def main() -> int:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Expected Python 3.10, got {sys.version}")
    if not hydra.__version__.startswith("1.3."):
        raise RuntimeError(f"Expected Hydra 1.3.x, got {hydra.__version__}")
    if not pydantic.__version__.startswith("2."):
        raise RuntimeError(f"Expected Pydantic V2, got {pydantic.__version__}")
    if not mlflow.__version__.startswith("3."):
        raise RuntimeError(f"Expected MLflow 3.x, got {mlflow.__version__}")

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
                "graph_memory": str(graph_memory.__path__),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
