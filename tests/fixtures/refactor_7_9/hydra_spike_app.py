from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from graph_memory.experiment.layout import (
    MultirunIdentity,
    RunLayout,
    concise_override_dirname,
)


@hydra.main(version_base="1.3", config_path="hydra_spike", config_name="config")
def main(config: DictConfig) -> None:
    runtime = HydraConfig.get()
    suffix = concise_override_dirname(str(runtime.job.override_dirname))
    layout = RunLayout(
        Path(os.environ["HYDRA_SPIKE_ROOT"]),
        str(config.name),
        identity=MultirunIdentity(job_num=int(runtime.job.num), suffix=suffix),
    )
    record = {
        "cwd": Path.cwd().resolve().as_posix(),
        "dataset": config.dataset.name,
        "job_num": int(runtime.job.num),
        "launcher": runtime.launcher._target_,
        "output_dir": Path(runtime.runtime.output_dir).resolve().as_posix(),
        "layout_dir": layout.run_dir.resolve().as_posix(),
        "override_dirname": runtime.job.override_dirname,
        "suffix": suffix,
        "seed": int(config.seed),
    }
    output = Path(os.environ["HYDRA_SPIKE_LOG"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
