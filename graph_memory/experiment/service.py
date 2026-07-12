from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from graph_memory.experiment.config import (
    ResolvedExperimentConfig,
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.layout import MultirunIdentity, RunLayout, RunMode
from graph_memory.experiment.persistence import (
    write_yaml_atomic,
)
from graph_memory.experiment.planning import WorkflowPlan, WorkflowPlanner
from graph_memory.experiment.state import (
    RunState,
    assert_named_root_mode,
    create_run_state,
    read_run_state,
    validate_run_identity,
    write_run_state,
)


@dataclass(frozen=True)
class InitializedExperiment:
    config: ResolvedExperimentConfig
    layout: RunLayout
    plan: WorkflowPlan
    state: RunState
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class ExistingExperiment:
    layout: RunLayout
    state: RunState
    plan: WorkflowPlan


def initialize_experiment(
    config: ResolvedExperimentConfig,
    *,
    layout: RunLayout,
    overrides: tuple[str, ...] = (),
) -> InitializedExperiment:
    assert_named_root_mode(layout)
    if layout.run_state.is_file():
        state = read_run_state(layout.run_state)
        validate_run_identity(layout=layout, config=config, existing=state)
    else:
        _reject_unowned_outputs(layout)
        state = None

    plan = WorkflowPlanner(config, layout).build()
    if state is None:
        state = create_run_state(layout=layout, config=config, plan=plan)

    write_yaml_atomic(layout.resolved_config, config)
    write_yaml_atomic(layout.overrides, list(overrides))
    for invocation in plan.invocations:
        write_yaml_atomic(invocation.config_path, invocation)

    if not layout.run_state.is_file():
        write_run_state(layout.run_state, state)
    return InitializedExperiment(
        config=config,
        layout=layout,
        plan=plan,
        state=state,
        overrides=overrides,
    )


def initialize_from_hydra(
    composed: DictConfig,
) -> InitializedExperiment:
    root = Path(__file__).resolve().parents[2]
    config = resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=root,
    )
    runtime = HydraConfig.get()
    mode: RunMode = (
        "multirun" if str(runtime.mode).lower().endswith("multirun") else "single"
    )
    layout = (
        RunLayout(root, config.name)
        if mode == "single"
        else RunLayout(
            root,
            config.name,
            identity=MultirunIdentity(
                job_num=int(runtime.job.num),
                override_dirname=str(runtime.job.override_dirname),
            ),
        )
    )
    expected_output = layout.run_dir.resolve()
    actual_output = Path(str(runtime.runtime.output_dir)).resolve()
    if actual_output != expected_output:
        raise ValueError(
            f"Hydra output directory mismatch: expected={expected_output} actual={actual_output}"
        )
    overrides = tuple(str(value) for value in runtime.overrides.task)
    return initialize_experiment(config, layout=layout, overrides=overrides)


def load_existing_experiment(
    name: str,
    *,
    repository_root: Path | None = None,
    job: str | None = None,
) -> ExistingExperiment:
    root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
    single = RunLayout(root, name)
    if single.run_state.is_file():
        if job is not None:
            raise ValueError(
                f"single run name={name} does not accept a multirun job selector"
            )
        layout = single
    else:
        if job is not None and (Path(job).name != job or job in {".", ".."}):
            raise ValueError(f"invalid multirun job selector: {job!r}")
        candidates = (
            (single.named_root / job / "run_state.yaml",)
            if job is not None
            else tuple(single.named_root.glob("*/run_state.yaml"))
        )
        candidates = tuple(path for path in candidates if path.is_file())
        if not candidates:
            raise FileNotFoundError(f"no run state found for name={name}")
        if len(candidates) != 1:
            jobs = ", ".join(path.parent.name for path in candidates)
            raise ValueError(
                f"multirun name={name} requires job=<job-dir>; available: {jobs}"
            )
        state_path = candidates[0]
        leaf = state_path.parent.name
        job_number, separator, override_dirname = leaf.partition("_")
        if not separator or not job_number.isdigit() or not override_dirname:
            raise ValueError(f"invalid multirun job directory: {leaf}")
        layout = RunLayout(
            root,
            name,
            identity=MultirunIdentity(
                job_num=int(job_number),
                override_dirname=override_dirname,
            ),
        )
        if layout.run_state.resolve() != state_path.resolve():
            raise ValueError(f"multirun job layout mismatch: {state_path}")
    state = read_run_state(layout.run_state)
    validate_run_identity(layout=layout, config=state.config, existing=state)
    plan = WorkflowPlanner(state.config, layout).build(validate_external=False)
    return ExistingExperiment(layout=layout, state=state, plan=plan)


def _reject_unowned_outputs(layout: RunLayout) -> None:
    if not layout.run_dir.is_dir():
        return
    allowed = {layout.run_dir / ".hydra", layout.run_dir / "config"}
    unexpected = [
        path
        for path in layout.run_dir.iterdir()
        if path not in allowed and not (path.is_file() and path.suffix == ".log")
    ]
    if unexpected:
        raise ValueError(
            f"run directory has outputs but no run state: {layout.run_dir}; reset it before reuse"
        )


__all__ = [
    "ExistingExperiment",
    "InitializedExperiment",
    "initialize_experiment",
    "initialize_from_hydra",
    "load_existing_experiment",
]
