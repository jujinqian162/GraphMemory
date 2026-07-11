from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="hydra_spike", config_name="config")
def main(config: DictConfig) -> None:
    runtime = HydraConfig.get()
    record = {
        "cwd": Path.cwd().resolve().as_posix(),
        "dataset": config.dataset.name,
        "job_num": int(runtime.job.num),
        "launcher": runtime.launcher._target_,
        "output_dir": Path(runtime.runtime.output_dir).resolve().as_posix(),
        "override_dirname": runtime.job.override_dirname,
        "seed": int(config.seed),
    }
    output = Path(os.environ["HYDRA_SPIKE_LOG"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
