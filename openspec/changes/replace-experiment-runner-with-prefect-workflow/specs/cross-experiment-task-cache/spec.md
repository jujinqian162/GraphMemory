## ADDED Requirements

### Requirement: Expensive reusable stages are Prefect Tasks
Preparation by split, graph construction, training-pair construction, model training, ranking generation, and evaluation SHALL be Prefect Tasks when their outputs are expensive, reusable, or independently observable. Batch, epoch, optimizer step, configuration conversion, run-output projection, and MLflow formatting MUST NOT become Prefect Tasks solely to make the DAG more granular.

#### Scenario: Trainable workflow DAG
- **WHEN** an R-GCN workflow is observed in Prefect
- **THEN** preparation, graph, pairs, training, retrieval, and evaluation appear as stage-level Tasks while epochs and batches remain inside training

### Requirement: Cache keys use only effective Task inputs
Every reusable Task SHALL use persistent cross-flow caching based on its effective typed inputs, upstream artifact identities, Task source, and a narrow stage implementation version. Run name, study name, Hydra job identity, run output path, Prefect IDs, MLflow IDs, tracking configuration, and unrelated downstream configuration MUST NOT affect the cache key.

#### Scenario: Rename an experiment
- **WHEN** two experiments differ only in run or study display name
- **THEN** all scientific Tasks resolve to the same cache identities

#### Scenario: Change evaluation only
- **WHEN** only evaluation configuration changes
- **THEN** preparation, graph, pairs, training, and ranking remain cache-compatible while evaluation recomputes

#### Scenario: Change training only
- **WHEN** only trainer configuration changes
- **THEN** compatible preparation, graph, and pair Tasks remain cached while training and dependent downstream Tasks recompute

### Requirement: External files and models have content identity
A raw-data source, local model directory, checkpoint, or other external path SHALL be converted to a typed reference with an immutable revision or content digest before it is supplied to a cached scientific Task. A bare path string MUST NOT be the sole cache identity for mutable external content.

#### Scenario: Raw data replaced in place
- **WHEN** raw dataset content changes while retaining the same filesystem path
- **THEN** its source reference changes and preparation does not reuse the old cached result

#### Scenario: Local encoder replaced in place
- **WHEN** local encoder contents change under the same configured directory
- **THEN** encoder-dependent pair, training, or retrieval Tasks receive a different model identity

### Requirement: Helper implementation changes have explicit invalidation
Each important cached Task SHALL include a short stage-specific implementation version in addition to Task source. The repository MUST NOT use the whole Git commit as a universal Task input.

#### Scenario: Semantic helper change
- **WHEN** a helper change alters graph construction semantics without changing graph inputs
- **THEN** the graph implementation version is advanced and graph plus downstream Tasks recompute without invalidating unrelated preparation Tasks

### Requirement: Dense-FT dependencies are reusable across final methods
Dense-FT training invoked as a final method and Dense-FT training invoked as the seed dependency of a composite method SHALL share a cache entry when their effective prepared data, pair artifact, model identity, trainer configuration, selection configuration, seed, and device are equal.

#### Scenario: Reuse a prior Dense-FT model
- **WHEN** a Dense-FT-seeded R-GCN experiment requests the same Dense-FT effective inputs as an earlier Dense-FT experiment
- **THEN** the Dense-FT Task returns the prior processed model reference from cache and R-GCN training continues from it

#### Scenario: Change only R-GCN configuration
- **WHEN** only the R-GCN model or trainer configuration changes
- **THEN** Dense-FT remains cached while R-GCN training and its downstream Tasks recompute

### Requirement: Cache refresh is explicit and operational
The root experiment interface SHALL expose an explicit cache refresh option that causes reusable Tasks in the current Flow to recompute and replace their Prefect cache records. Refresh configuration MUST NOT become a scientific Task input and the first implementation MUST NOT add arbitrary per-stage selection syntax.

#### Scenario: Force complete scientific recomputation
- **WHEN** a user runs an experiment with cache refresh enabled
- **THEN** reusable Tasks execute rather than loading prior cached results while retaining the same scientific input values

### Requirement: Cached references are validated before use
Pydantic SHALL validate cached processed-reference structure, including kind and declared payload metadata. Ordinary consumers SHALL resolve the requested role from the typed reference and check that the declared path exists; they MUST NOT reread and rehash the complete artifact on every payload access. Missing declared payloads MUST fail with the exact path and an explicit refresh recovery instruction, and the system MUST NOT read a same-named run-local substitute.

#### Scenario: Processed model was deleted
- **WHEN** a cached training result references a missing processed model directory
- **THEN** downstream execution fails before model loading and tells the user to refresh the cache

### Requirement: Failed Tasks are retried by rerunning the Flow
Scientific Tasks SHALL NOT configure automatic retries in the first implementation. Repeating the full experiment command SHALL reuse compatible completed Task results and execute the failed or invalidated Task again. Attempt state MUST NOT become a final processed asset.

#### Scenario: Training configuration failure
- **WHEN** training fails because its configuration or accelerator state is invalid
- **THEN** the Task fails once and a later full Flow invocation is the explicit recovery action

### Requirement: Concurrent first computation needs no repository lock layer
The repository SHALL NOT implement a custom cache lock manager or future registry. Processed artifacts SHALL use opaque destinations and atomic publication so concurrent jobs cannot overwrite one another, while Prefect remains responsible for cache records.

#### Scenario: Concurrent first request
- **WHEN** two Hydra jobs concurrently request the same uncached prepared split
- **THEN** they may duplicate first computation, but neither overwrites the other's processed artifact or run output and later invocations can reuse Prefect cache
