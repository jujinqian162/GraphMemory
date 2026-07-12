## MODIFIED Requirements

### Requirement: Current R-GCN checkpoint contract
R-GCN training SHALL write and read a strict R-GCN-specific checkpoint contract without a checkpoint version field. The checkpoint SHALL require typed R-GCN model, trainer, encoder, decoder, beam-search, beam-loss, optimizer-phase, and learned-state records needed to reproduce beam-aware training and inference.

#### Scenario: Load a current beam R-GCN checkpoint
- **WHEN** retrieval loads a checkpoint written by the updated R-GCN trainer
- **THEN** the model factory SHALL restore R-GCN encoding, base scorer, first-hop scorer, subsequent-hop scorer, stop head, effective beam configuration, and learned state

#### Scenario: Reject a pre-change R-GCN checkpoint
- **WHEN** retrieval loads an R-GCN checkpoint without required decoder state or beam configuration
- **THEN** loading SHALL fail with an error that identifies the incompatible checkpoint and requires retraining

#### Scenario: Reject a versioned checkpoint
- **WHEN** a checkpoint contains the retired `checkpoint_version` field
- **THEN** checkpoint loading SHALL fail

## ADDED Requirements

### Requirement: Beam provenance is artifact-backed
R-GCN training and retrieval run summaries SHALL serialize effective beam configuration and decoder provenance from typed runtime state rather than infer them from method names or script defaults.

#### Scenario: Training summary records effective beam settings
- **WHEN** an R-GCN train stage succeeds
- **THEN** its run summary SHALL record effective beam size, maximum steps, length penalty, loss weights, decoder settings, optimizer-phase settings, and checkpoint output

#### Scenario: Retrieval summary records checkpoint beam settings
- **WHEN** checkpoint-backed R-GCN retrieval succeeds
- **THEN** its run summary SHALL record the beam and decoder settings restored from the checkpoint and the effective retrieval device
