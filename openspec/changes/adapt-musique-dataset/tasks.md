## 1. Dataset core

- [x] 1.1 Add `graph_memory/datasets/musique/records.py` with raw example, ranking record, label record, and conversion result types.
- [x] 1.2 Add strict official JSONL parser for MuSiQue-Ans fields and validation-oriented error messages.
- [x] 1.3 Add converter that emits paragraph-level ranking records, gold evidence IDs, and dependency edges from decomposition references.
- [x] 1.4 Add projectors for text retrieval, temporal memory retrieval, graph build, graph ranking, and evidence evaluation requests.
- [x] 1.5 Export the MuSiQue dataset package API.

## 2. Validation and dataset dispatch

- [x] 2.1 Add MuSiQue ranking and label validators.
- [x] 2.2 Extend dataset selector dispatch to support `DatasetId = "musique"`.
- [x] 2.3 Extend direct dataset-aware scripts to accept `--dataset musique`.

## 3. Prepare script and workflow routing

- [x] 3.1 Add `scripts/prepare_musique.py` for official MuSiQue JSONL files with split sampling, invalid-record handling, separated artifacts, optional combined inspection output, and run summary.
- [x] 3.2 Extend workflow prepare-stage routing and status reporting to use `scripts/prepare_musique.py`.
- [x] 3.3 Extend registry/workflow config validation surfaces to accept `musique`.

## 4. Experiment config

- [x] 4.1 Add `configs/experiments/musique_evidence_retrieval.json` with smoke/quick/full profiles, raw train/dev paths, split sources/offsets, methods, and method configs.
- [x] 4.2 Ensure the named config can initialize the workflow and exposes trainable methods consistently with 2Wiki.
- [x] 4.3 Update active project documentation with the implemented MuSiQue boundary, official download command, expected raw paths, and run instructions.

## 5. Tests and verification

- [x] 5.1 Add parser/converter/projector/leakage tests for MuSiQue.
- [x] 5.2 Add selector/script/workflow/config tests for MuSiQue.
- [x] 5.3 Run focused pytest coverage for MuSiQue and touched workflow tests.
- [x] 5.4 Run OpenSpec status and report remaining archive state.
