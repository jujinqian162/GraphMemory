## ADDED Requirements

### Requirement: Collect compact run delivery artifacts
The system SHALL provide a name-based CLI that copies compact, report-relevant artifacts from `runs/<name>/` into a delivery directory while preserving paths relative to the run root.

#### Scenario: Preserve selected artifact paths
- **WHEN** the collector is run with `--name rgcn_full_train` and the default output root
- **THEN** copied files SHALL be written under `results/rgcn_full_train/` using the same relative paths they had under `runs/rgcn_full_train/`

#### Scenario: Help documents the convention
- **WHEN** help is requested for the collector CLI
- **THEN** the help text SHALL document the `--name <train_id>` contract and the default `runs/<name>` to `results/<name>` path convention

#### Scenario: Include report evidence files
- **WHEN** the run contains manifest, config, tables, metrics CSVs, run summaries, graph stats, training metrics, training summaries, train-pair summaries, selected tuning configs, and failure-case debug files
- **THEN** the collector SHALL include those files unless they exceed the configured maximum file size

### Requirement: Exclude large intermediate artifacts by default
The system SHALL exclude large reproducible or transfer-heavy intermediate artifacts by default.

#### Scenario: Skip known large artifact classes
- **WHEN** the run contains full prepared inputs, full graph JSON files, full ranked predictions, checkpoint binaries, model binaries, embedding files, or raw train-pair JSON
- **THEN** the collector SHALL skip those files and record a skip reason in the delivery manifest

#### Scenario: Apply size guard
- **WHEN** a candidate file exceeds the configured maximum file size
- **THEN** the collector SHALL skip the file and record `too_large` as the skip reason

### Requirement: Write an auditable delivery manifest
The system SHALL write a machine-readable manifest describing copied and skipped files.

#### Scenario: Manifest records transfer decisions
- **WHEN** collection completes
- **THEN** `delivery_manifest.json` SHALL contain source run path, output path, copied file entries, skipped file entries, total copied bytes, and the effective maximum file size

#### Scenario: Missing run fails fast
- **WHEN** the requested run directory does not exist
- **THEN** the collector SHALL fail without creating a misleading delivery manifest
