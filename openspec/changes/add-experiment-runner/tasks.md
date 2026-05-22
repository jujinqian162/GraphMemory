## 1. Tests First

- [x] 1.1 Add tests for experiment config loading, profile selection, CLI override precedence, and `effective_config.json` output.
- [x] 1.2 Add tests for deterministic run directory layout and manifest path generation under `runs/<experiment_name>/`.
- [x] 1.3 Add tests for dry-run planning that verify generated low-level commands include explicit input and output paths and do not execute.
- [x] 1.4 Add tests for method selection, including accepted current methods and fail-fast rejection of unknown methods.
- [x] 1.5 Add tests for status reporting across missing, complete, and stale or mismatched artifacts.

## 2. Config And Manifest Model

- [x] 2.1 Create clear experiment config files under `configs/experiments/` and graph-rerank search-space config under `configs/search_spaces/`.
- [x] 2.2 Add small typed structures or helper functions for experiment config, profiles, generated artifact paths, and manifest records.
- [x] 2.3 Implement config merge behavior with precedence `CLI overrides > experiment config/profile > code defaults`.
- [x] 2.4 Implement manifest creation, loading, updating, and deterministic artifact path generation.
- [x] 2.5 Ensure run-local tuned configs are generated under `runs/<experiment_name>/tuned/` and global configs are not modified during ordinary runs.

## 3. Stage Planning

- [x] 3.1 Implement a small evidence-retrieval recipe for `prepare`, `graphs`, `tune`, `retrieve`, `evaluate`, and `aggregate`.
- [x] 3.2 Generate low-level commands for each stage using manifest paths and existing script arguments.
- [x] 3.3 Implement `plan` behavior that prints or returns commands without creating stage outputs.
- [x] 3.4 Implement stage and method filtering for selected workflows such as retrieve/evaluate only for dense methods.
- [x] 3.5 Implement dependency handling for resume behavior, including `--from` style stage selection.

## 4. Runner CLI

- [x] 4.1 Add `scripts/experiment.py` with subcommands for `init`, `plan`, `run`, and `status`.
- [x] 4.2 Wire `init` to create the run directory, effective config, and manifest without running stage commands.
- [x] 4.3 Wire `plan` to render the concrete low-level command sequence for selected stages and methods.
- [x] 4.4 Wire `run` to execute selected stage commands and update manifest status metadata.
- [x] 4.5 Wire `status` to inspect expected outputs, run summaries or manifest metadata, and report missing, complete, and stale stages.

## 5. Integration And Safety

- [x] 5.1 Add a small smoke integration test using the existing HotpotQA fixture and a graph-free or BM25-only path to avoid dense model downloads.
- [x] 5.2 Verify low-level scripts remain runnable without manifest files and without changed argument names.
- [x] 5.3 Add clear errors for config-changing attempts against an existing manifest unless the user explicitly resets or reinitializes.
- [x] 5.4 Add stale artifact detection for at least method, split, input path, and effective config mismatches when metadata is available.

## 6. Documentation

- [x] 6.1 Update `docs/40-operations/commands.md` so the experiment runner is the recommended command path and low-level commands remain the contract/debug path.
- [x] 6.2 Update `docs/40-operations/reproducibility.md` with run manifest, run directory layout, config precedence, and config directory roles.
- [x] 6.3 Update `docs/40-operations/implementation-handoff.md` with runner review entry points and extension guidance.
- [x] 6.4 Update `README.md` with the short recommended runner quick start.
- [x] 6.5 Remove or replace ambiguous config names once all references point to the new config layout.

## 7. Verification

- [x] 7.1 Run focused experiment-runner tests.
- [x] 7.2 Run existing Phase 1 CLI smoke tests to confirm low-level compatibility.
- [x] 7.3 Run OpenSpec validation for `add-experiment-runner`.
- [x] 7.4 Run the full test suite or the repository-local verified fallback command.
- [x] 7.5 Review the diff for scope and confirm no Phase 2/3 methods or dataset implementations were added.
