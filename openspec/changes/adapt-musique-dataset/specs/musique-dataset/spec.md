## ADDED Requirements

### Requirement: Parse official MuSiQue-Ans records
The system SHALL parse official MuSiQue JSONL records with paragraph-level support labels and decomposition metadata.

#### Scenario: Valid official example
- **WHEN** a raw MuSiQue-Ans record contains `id`, `question`, `answer`, `answer_aliases`, `answerable`, `paragraphs`, and `question_decomposition`
- **THEN** the parser returns a typed MuSiQue example preserving paragraph indices, titles, text, support flags, and decomposition support indices

#### Scenario: Invalid official example
- **WHEN** a raw MuSiQue record is missing required fields or contains malformed paragraphs/decomposition steps
- **THEN** parsing fails with a `ValueError` identifying the invalid example or field

### Requirement: Convert MuSiQue examples into leakage-safe artifacts
The system SHALL convert MuSiQue-Ans examples into separate ranking input and label artifacts without exposing gold support labels to retrieval inputs.

#### Scenario: Ranking and label conversion
- **WHEN** a valid MuSiQue-Ans example is converted
- **THEN** the ranking record contains the question and candidate paragraphs only
- **AND** the label record contains the gold answer, gold paragraph item IDs, and derived dependency edges

#### Scenario: Label-only fields stay out of graph inputs
- **WHEN** a MuSiQue ranking record is projected into graph build requests
- **THEN** graph nodes do not contain `is_supporting`, gold answer, answer aliases, decomposition answers, or dependency labels

### Requirement: Project MuSiQue records into request-first consumers
The system SHALL project MuSiQue ranking and label records into the existing dataset-neutral retrieval, temporal memory, graph build, and evidence evaluation request contracts.

#### Scenario: Text retrieval projection
- **WHEN** the text retrieval stage consumes MuSiQue ranking records
- **THEN** it receives `TextRankingRequest` objects with paragraph candidates and stable item IDs

#### Scenario: Evidence evaluation projection
- **WHEN** the evaluation stage consumes MuSiQue labels and graph artifacts
- **THEN** it receives an `EvidenceEvaluationRequest` whose labels expose paragraph-level gold evidence IDs and dependency edges

### Requirement: Prepare MuSiQue raw files for the workflow
The system SHALL provide a prepare script that reads official MuSiQue JSONL files and writes workflow-compatible input and label JSON artifacts.

#### Scenario: Prepare script outputs artifacts
- **WHEN** `scripts/prepare_musique.py` is run on an official MuSiQue-Ans JSONL file
- **THEN** it writes `*.input.json`, `*.labels.json`, and a run summary

#### Scenario: Non-strict invalid examples
- **WHEN** the prepare script runs without strict invalid-example mode and encounters malformed records
- **THEN** it drops invalid records, records reason counts in the run summary, and continues with valid examples

### Requirement: Expose a usable MuSiQue experiment config
The system SHALL expose a named experiment config for MuSiQue evidence retrieval that can initialize the existing workflow.

#### Scenario: Named config loads
- **WHEN** a user loads config `musique_evidence_retrieval`
- **THEN** the config resolves with `dataset` set to `musique`, raw train/dev paths under `data/musique/raw/`, and supported retrieval/trainable methods

#### Scenario: Workflow routes dataset-specific prepare stage
- **WHEN** the experiment workflow runs with the MuSiQue config
- **THEN** the prepare stage invokes `scripts/prepare_musique.py` and downstream dataset-aware stages receive `--dataset musique`
