from __future__ import annotations

# ruff: noqa: E402 -- the public filename inspect.py otherwise shadows stdlib inspect

import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry).resolve() != SCRIPT_DIRECTORY]
sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))

import hydra
from omegaconf import DictConfig

from graph_memory.experiment.planning import format_plan
from graph_memory.experiment.service import initialize_from_hydra


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    print(format_plan(initialize_from_hydra(config).plan))


if __name__ == "__main__":
    main()
