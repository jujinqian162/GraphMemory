## Why

The experiment runner currently exposes the workflow as a low-level stage list, so starting a run still requires users to remember which stages are necessary for each method. Discovery is also weak: supported stages, methods, configs, profiles, and training configs are embedded in code or config files instead of being visible from the CLI.

## What Changes

- Add method-first experiment startup: `scripts/experiment.py run <name> --method <method>` initializes or reuses the run and plans the complete required workflow for that method by default.
- Replace ad hoc stage subsets with ordered stage ranges: users may provide `--from <stage>` and optional `--to <stage>` to select a contiguous slice of the workflow. The old explicit stage-list selector is retired from the public runner contract.
- Make trainable methods include their required upstream stages automatically, including graph construction, train-pair construction, training, retrieval, evaluation, and aggregation.
- Add discovery subcommands for public runner contracts: stages, methods, experiment configs, search-space configs, training configs, profiles, and complete recipe summaries.
- Resolve configs by contract names rather than requiring paths at the top-level CLI: experiment configs come from `configs/experiments/`, search spaces from `configs/search_spaces/`, and training configs from `configs/training/<method>/`.
- Improve `plan` output readability with per-command spacing, file labels, and colored option names when stdout is a terminal.
- Preserve the existing manifest/run directory contract, low-level script contracts, and `CLI > config` precedence.

## Capabilities

### New Capabilities
- `experiment-runner`: User-facing experiment orchestration, discovery, method-first defaults, stage-range planning, and readable plan output for named runs.

### Modified Capabilities
- None.

## Impact

- Affected code: `scripts/experiment.py`, `graph_memory/experiment.py`, `tests/test_experiment_runner.py`, and OpenSpec artifacts under `openspec/changes/improve-experiment-cli-ux/`.
- User-facing API: `--method`, `--methods`, `--from`, `--to`, `stages list`, `methods list`, `configs list`, `profiles list`, `recipes list`, `init`, `plan`, `run`, and `status` remain supported. Explicit stage-list selection is no longer part of the public runner API.
- Dependencies: no new runtime dependency; formatting uses standard-library terminal capability checks.
