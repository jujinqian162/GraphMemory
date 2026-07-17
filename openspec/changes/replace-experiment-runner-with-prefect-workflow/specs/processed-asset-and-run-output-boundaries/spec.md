## ADDED Requirements

### Requirement: Reusable computation assets live under data processed
Every file or directory that may be consumed by a later experiment SHALL live below `data/processed/`. This includes prepared inputs and labels, evidence graphs, training pairs, reusable indexes or features, predictions reused by evaluation, checkpoints, and trained model directories. Newly produced reusable computation assets MUST NOT live below `runs/`.

#### Scenario: Publish a trained model
- **WHEN** a training Task completes successfully
- **THEN** its final model is published below `data/processed/models/<method>/...` and downstream retrieval receives a typed reference to that location

#### Scenario: Publish reusable predictions
- **WHEN** a ranking Task produces predictions that evaluation or a later experiment may consume
- **THEN** the predictions are published below `data/processed/` rather than a named run directory

### Requirement: Prefect persists small typed references
Prefect result storage SHALL persist small stable result objects rather than serializing complete large datasets, graphs, predictions, checkpoints, or model directories. An artifact reference SHALL include at least URI, artifact kind, content digest or immutable revision, manifest URI, declared payload metadata, origin identity, and relevant size or shape metadata.

#### Scenario: Cached graph result
- **WHEN** a graph Task hits cache
- **THEN** Prefect returns a typed graph artifact reference without deserializing the graph payload through the Prefect result record

### Requirement: Asset publication is validated and atomic
A Task that produces a reusable artifact SHALL write to a task-specific staging directory below `data/processed/`, validate all declared payloads, write a manifest, and atomically publish the completed artifact on the same filesystem before returning its reference. Incomplete staging contents MUST NOT be returned or treated as cache-valid assets.

#### Scenario: Successful publication
- **WHEN** model files and their manifest pass validation
- **THEN** the staging directory is atomically moved to an immutable processed asset location before the Task completes

#### Scenario: Failure before validation
- **WHEN** a Task fails while writing an artifact
- **THEN** no processed artifact reference is returned and no incomplete destination is visible as a valid asset

### Requirement: Processed asset identity is independent of run name
Processed asset paths SHALL use opaque artifact identity and semantic kind, not run name, study name, Hydra job number, MLflow run ID, or the path of a `runs/` directory. Changing presentation identity MUST NOT create a distinct processed path when a Task hits cache.

#### Scenario: Same computation under a new name
- **WHEN** an experiment is repeated with a different run name and all scientific inputs are equal
- **THEN** cached Tasks return the same processed artifact references

### Requirement: Run directories are output only
New experiment directories below `runs/<run-name>/` SHALL contain only files written for human inspection, tracking projection, debug review, reporting, or delivery. Scientific Tasks and stage services MUST NOT read any path below `runs/` as computation input.

#### Scenario: Write final run output
- **WHEN** the Flow has obtained its final Task results
- **THEN** uncached Flow-level output code writes resolved config, summaries, metrics, small debug/report files, and an asset manifest below the run directory

#### Scenario: Reject run-local input
- **WHEN** a new scientific Task is configured with an input path below `runs/`
- **THEN** validation rejects the path instead of consuming run-local output

### Requirement: Run output contains a complete small-file delivery surface
A completed run output SHALL contain its resolved configuration, applied overrides, workflow summary, final metrics, applicable small training/debug/report files, and an asset manifest. Large reusable artifacts MUST be represented by typed reference metadata rather than copied into the run directory.

#### Scenario: Inspect without Prefect or MLflow
- **WHEN** a user opens a collected run directory offline
- **THEN** the user can identify the dataset, method, variant, seed, final metrics, and referenced reusable asset URIs and digests from the delivered small files

### Requirement: Single and multirun output layouts remain collectable
A single Hydra job SHALL write its output below `runs/<run-name>/`; a Hydra multirun SHALL write each independent job below a deterministic child directory of `runs/<run-name>/`. `scripts/deliver/collect_run_artifacts.py --name <run-name>` SHALL continue to mirror the complete output-only tree into `results/<run-name>` without traversing `data/processed/`.

#### Scenario: Collect one single-method run
- **WHEN** the collector receives the name of a completed single run
- **THEN** it copies the run's small output files and asset manifest into `results/<run-name>`

#### Scenario: Collect a baseline sweep
- **WHEN** the collector receives the name of a completed Hydra multirun
- **THEN** it copies every single-method job output and preserves their deterministic job selectors

### Requirement: Historical runs remain historical
The change MUST NOT migrate, rewrite, or delete existing run directories, results deliveries, processed data, Prefect records, or MLflow rows. New code MUST NOT add a compatibility reader that treats old run-local computation artifacts as new processed inputs.

#### Scenario: Old run remains on disk
- **WHEN** the new workflow is adopted in a repository containing historical runs
- **THEN** those runs remain available for manual inspection and existing delivery use but are not candidates for the new Task cache

### Requirement: Asset garbage collection is outside the first implementation
The first implementation MUST NOT automatically delete or garbage-collect processed assets. Any future cleanup capability SHALL be specified separately and MUST account for Prefect cache references, run manifests, deliveries, and MLflow references.

#### Scenario: Orphan candidate exists
- **WHEN** a processed artifact no longer appears in a current run manifest
- **THEN** the workflow leaves it unchanged rather than assuming it is safe to delete
