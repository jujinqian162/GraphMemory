from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from mlflow import MlflowClient

from graph_memory.experiment.state import read_run_state

ROOT = Path(__file__).resolve().parents[1]


def test_shared_only_run_and_status_resume_without_baseline_children(tmp_path: Path) -> None:
    name = f"entrypoint-{tmp_path.name}"
    run_root = ROOT / "runs" / name
    command = _run_command(name, tmp_path=tmp_path)
    try:
        first = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert first.returncode == 0, first.stderr
        state = read_run_state(run_root / "run_state.yaml")
        assert state.mlflow_parent_run_id is not None

        second = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert second.returncode == 0, second.stderr
        assert "skipped=3" in second.stdout

        status = subprocess.run(
            [
                sys.executable,
                "experiment/status.py",
                f"name={name}",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert status.returncode == 0, status.stderr
        assert status.stdout.count(" complete") == 3

        client = MlflowClient(
            tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
        )
        experiment = client.get_experiment_by_name("graph-memory")
        assert experiment is not None
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        assert runs[0].data.tags["graph_memory.run_kind"] == "parent"
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def test_run_entrypoint_executes_sequential_hydra_multirun(tmp_path: Path) -> None:
    name = f"multirun-{tmp_path.name}"
    named_root = ROOT / "runs" / name
    command = _run_command(name, tmp_path=tmp_path, multirun=True)
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        states = sorted(named_root.glob("*/run_state.yaml"))
        assert len(states) == 2
        assert [path.parent.name for path in states] == ["0_seed=13", "1_seed=17"]
        loaded = [read_run_state(path) for path in states]
        assert [state.config.seed for state in loaded] == [13, 17]
        assert all(state.mode == "multirun" for state in loaded)
        assert all(state.mlflow_parent_run_id is not None for state in loaded)
        assert completed.stdout.find("seed=13") < completed.stdout.find("seed=17")
        assert (named_root / "multirun.yaml").is_file()
        assert all(
            (path.parent / "config/resolved.yaml").is_file()
            and (path.parent / "config/overrides.yaml").is_file()
            for path in states
        )

        ambiguous = subprocess.run(
            [sys.executable, "experiment/status.py", f"name={name}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert ambiguous.returncode != 0
        assert "0_seed=13" in ambiguous.stderr
        assert "1_seed=17" in ambiguous.stderr

        selected = subprocess.run(
            [
                sys.executable,
                "experiment/status.py",
                f"name={name}",
                "job=0_seed=13",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert selected.returncode == 0, selected.stderr
        assert "prepare train complete" in selected.stdout

        inspected = subprocess.run(
            [
                sys.executable,
                "experiment/inspect.py",
                "kind=jobs",
                f"name={name}",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert inspected.returncode == 0, inspected.stderr
        assert json.loads(inspected.stdout) == ["0_seed=13", "1_seed=17"]

        reset = subprocess.run(
            [
                sys.executable,
                "experiment/reset.py",
                f"name={name}",
                "job=0_seed=13",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert reset.returncode == 0, reset.stderr
        assert not (named_root / "0_seed=13").exists()
        assert (named_root / "1_seed=17").is_dir()
    finally:
        shutil.rmtree(named_root, ignore_errors=True)


def test_plan_inspect_status_and_reset_top_level_files(tmp_path: Path) -> None:
    name = f"public-files-{tmp_path.name}"
    run_root = ROOT / "runs" / name
    plan_command = _run_command(name, tmp_path=tmp_path)
    plan_command[1] = "experiment/plan.py"
    try:
        planned = subprocess.run(
            plan_command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert planned.returncode == 0, planned.stderr
        assert "[1] stage=prepare split=train" in planned.stdout

        status = subprocess.run(
            [sys.executable, "experiment/status.py", f"name={name}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert status.returncode == 0, status.stderr
        assert "prepare train missing" in status.stdout

        inspected = subprocess.run(
            [sys.executable, "experiment/inspect.py", "kind=methods"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert inspected.returncode == 0, inspected.stderr
        assert '"method": "bm25"' in inspected.stdout

        reset = subprocess.run(
            [sys.executable, "experiment/reset.py", f"name={name}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert reset.returncode == 0, reset.stderr
        assert not run_root.exists()
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def test_public_commands_reject_missing_required_values() -> None:
    for script in ("plan.py", "run.py", "status.py", "inspect.py", "reset.py"):
        completed = subprocess.run(
            [sys.executable, f"experiment/{script}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0, script


def _run_command(
    name: str,
    *,
    tmp_path: Path,
    multirun: bool = False,
) -> list[str]:
    source = (ROOT / "tests/fixtures/hotpotqa_smoke.json").as_posix()
    database = (tmp_path / "mlflow.db").as_posix()
    artifacts = (tmp_path / "mlflow-artifacts").as_posix()
    command = [sys.executable, "experiment/run.py"]
    if multirun:
        command.append("-m")
    command.extend(
        [
            f"name={name}",
            "dataset=hotpotqa",
            "profile=smoke",
            "methods=[bm25]",
            "device=cpu",
            "stages.to=prepare",
            f"tracking.database={database}",
            f"tracking.artifact_root={artifacts}",
            *[
                f"dataset.splits.{split}.source={source}"
                for split in ("train", "dev", "test")
            ],
            *[
                override
                for split in ("train", "dev", "test")
                for override in (
                    f"dataset.splits.{split}.offset=0",
                    f"dataset.splits.{split}.capacity=1",
                )
            ],
        ]
    )
    if multirun:
        command.extend(
            [
                "seed=13,17",
                "hydra.job.config.override_dirname.exclude_keys="
                "[name,dataset,profile,methods,device,stages.to,tracking.database,"
                "tracking.artifact_root,dataset.splits.train.source,"
                "dataset.splits.dev.source,dataset.splits.test.source,"
                "dataset.splits.train.offset,dataset.splits.dev.offset,"
                "dataset.splits.test.offset,dataset.splits.train.capacity,"
                "dataset.splits.dev.capacity,dataset.splits.test.capacity]",
            ]
        )
    return command
