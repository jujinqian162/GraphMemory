## ADDED Requirements

### Requirement: Default cache-aware resume
The experiment runner SHALL resume repeated `run` invocations by skipping the completed prefix of the selected command plan when live artifact evidence shows those commands are reusable.

#### Scenario: Full run resumes from first missing stage
- **WHEN** a user runs all selected stages for an experiment whose planned prefix is complete and whose next planned command is missing
- **THEN** the runner SHALL skip the completed prefix
- **THEN** the runner SHALL execute the first missing command and all following commands in the selected plan

#### Scenario: Stale output stops prefix skipping
- **WHEN** a planned command has an existing output whose run summary does not match the manifest
- **THEN** the runner SHALL treat that command as stale
- **THEN** the runner SHALL execute that stale command and all following commands in the selected plan

#### Scenario: Completed run executes nothing
- **WHEN** every command in the selected plan has complete or aliased live status
- **THEN** the runner SHALL execute no low-level commands
- **THEN** the runner SHALL still update manifest status metadata

### Requirement: Explicit no-cache control
The experiment runner SHALL expose a `--no-cache` flag on `run` and `plan` so users can bypass cache-aware prefix pruning.

#### Scenario: Run no-cache executes full selected plan
- **WHEN** a user invokes `scripts/experiment.py run <name> --no-cache`
- **THEN** the runner SHALL execute the full selected command plan even if planned commands are already complete

#### Scenario: Plan no-cache renders full selected plan
- **WHEN** a user invokes `scripts/experiment.py plan <name> --no-cache`
- **THEN** the rendered plan SHALL include the full selected command plan without cache-aware prefix pruning

#### Scenario: Help output documents cache behavior
- **WHEN** a user requests help for `scripts/experiment.py run` or `scripts/experiment.py plan`
- **THEN** the help text SHALL show `--no-cache`
- **THEN** the help text SHALL describe that the default behavior uses cached completed work or prunes completed commands

### Requirement: Run-summary-backed status evidence
The experiment runner SHALL validate completed stage status using run summaries when the stage's low-level script writes a run summary.

#### Scenario: Matching run summary reports complete
- **WHEN** an expected output exists and its run summary records success with matching inputs, outputs, and effective config
- **THEN** status inspection SHALL report the command's artifact as complete

#### Scenario: Mismatched run summary reports stale
- **WHEN** an expected output exists but its run summary records a different input path, output path, method, or effective config value
- **THEN** status inspection SHALL report the command's artifact as stale

#### Scenario: Status evidence is read live
- **WHEN** files or run summaries change after `manifest.json` was last updated
- **THEN** cache-aware resume SHALL use live status inspection instead of trusting the manifest's saved `stage_status` snapshot
