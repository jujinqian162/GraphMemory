## ADDED Requirements

### Requirement: Test split uses a fixed split seed decoupled from the training seed
The test split SHALL be selected using a dedicated fixed `split_seed` that is independent of the top-level experiment `seed`. Changing the experiment `seed` (which drives training randomness and train/dev sampling) SHALL NOT change which records or which record order appear in the test split. The default `split_seed` SHALL be `13`.

#### Scenario: Two runs differ only by training seed
- **WHEN** two experiments run the same dataset and profile with different top-level `seed` values
- **THEN** their prepared test splits contain identical records in identical order, and the test prepare artifact digest is equal

#### Scenario: Deterministic method run once
- **WHEN** a deterministic (non-trainable) retrieval method is run a single time on the fixed test split
- **THEN** its evaluated test population is the same test split shared by trainable methods across all their seeds

### Requirement: Train and dev sampling keep using the experiment seed
Train and dev split sampling SHALL keep using the top-level experiment `seed` so that trainable methods still receive seed-dependent training and validation data across the five seeds.

#### Scenario: Trainable method across five seeds
- **WHEN** a trainable method is run with five different top-level `seed` values
- **THEN** train and dev split selections may differ per seed while the test split remains identical across all five runs

### Requirement: Provenance transform test partition uses the same fixed split seed
The `twowiki_provenance` transform dev/test partition SHALL use the same fixed `split_seed` semantics as the general datasets, so the transformed test split is constant across any experiment `seed` and any method.

#### Scenario: Provenance transform under varying training seed
- **WHEN** the `twowiki_provenance` experiment runs with different top-level `seed` values
- **THEN** `deterministic_dev_test_partition` produces an identical test partition and the transform test output is byte-identical across those runs

#### Scenario: Fixed split seed is the single source of test identity
- **WHEN** the fixed `split_seed` value is configured
- **THEN** both the general `sample_split` test path and the transform dev/test partition derive test membership from that same fixed value rather than from the training seed
