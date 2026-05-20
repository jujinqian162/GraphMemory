# Reproducibility Strategy

Date: 2026-05-20

Status: Working reference.

## Goal

A Phase 1 result should be reproducible from the repository, raw data paths, config files, command sequence, run summaries, and output artifacts. A reader should be able to answer what was run, with which parameters, on which data, and where the outputs came from.

## Principles

- Config files define reproducible defaults.
- CLI arguments may override config for temporary runs.
- The effective config must be recorded.
- Every script must write a run summary.
- Dataset splits must be deterministic and documented.
- Dev tuning and test evaluation must remain separate.
- Commands must be documented after implementation so users know exactly how to run the system.

## Reproducibility Chain

```text
raw data
  -> split seed/count/offset
  -> input + label artifacts
  -> graph config
  -> graph artifacts
  -> retrieval config
  -> ranked results
  -> dev-selected graph rerank config
  -> test predictions
  -> evaluation metrics
  -> aggregate tables
```

Every arrow should be recoverable from either command documentation or run summaries.

## Required Stable Inputs

Record these for every experiment:

- Dataset name and setting: `hotpotqa_distractor`.
- Raw data file paths.
- Split names.
- Split counts.
- Split seed.
- Split offset.
- Dense encoder model.
- Dense query and passage prefixes.
- Graph construction config.
- Graph rerank config.
- Top-k cutoffs.
- Method names.

## Config Policy

Priority:

```text
CLI overrides > config file > code defaults
```

Rules:

- Config files are for repeatable runs.
- CLI overrides are for temporary changes.
- The final `effective_config` must be written into run summaries.
- If a CLI override changes a scientific setting, the run summary must make that visible.

The default Phase 1 config should remain stable:

```text
configs/phase1_default.json
```

The dev-selected graph rerank config should be a separate artifact:

```text
configs/phase1_graph_rerank_dev_selected.json
```

## Split Reproducibility

Required Phase 1 split policy:

```text
train: 5,000 examples from labeled HotpotQA train, seed=13, offset=0
dev:     500 examples from labeled HotpotQA dev,   seed=13, offset=0
test:  1,000 examples from labeled HotpotQA dev,   seed=13, offset=500
```

Rules:

- Dev and test must be disjoint.
- If the labeled dev file has too few examples, fail.
- Do not use unlabeled official test data for metrics.
- Split metadata must be recorded in run summaries.

## Run Summaries

Every runnable script must write a run summary near its output.

A run summary should include:

- script name
- start and finish timestamps
- status
- effective config
- input paths
- output paths
- counts
- timings
- environment notes

This is mandatory for Phase 1.

## Command Documentation

CLI parameter details are not designed here. The implementation must produce command usage documentation after the scripts exist.

Required command documentation location:

```text
docs/40-operations/commands.md
```

It should show:

- prepare train/dev/test input and label artifacts
- build graphs
- run BM25
- run dense
- tune BM25 graph rerank on dev
- tune dense graph rerank on dev
- run fixed graph rerank configs on test
- evaluate all methods
- aggregate tables
- run leakage checks
- run tests

The command documentation should include both:

- leakage-safe commands using `.input.json` and `.labels.json`
- compatibility notes for the original project command surface where relevant

## Environment Recording

Run summaries should record, when available:

- Python version
- operating system/platform
- project version or git commit if available
- relevant dependency versions
- dense encoder model name
- whether optional spaCy was enabled
- hardware notes for dense retrieval if known

Do not make environment recording brittle. Missing optional environment details should be recorded in `notes`, not silently ignored.

## Artifact Naming

Use stable method and split names:

```text
data/hotpotqa/processed/train_memory_tasks.input.json
data/hotpotqa/processed/train_memory_tasks.labels.json
data/hotpotqa/processed/train_graphs.json
results/ranked_results_bm25.json
results/ranked_results_dense.json
results/ranked_results_bm25_graph_rerank.json
results/ranked_results_dense_graph_rerank.json
results/main_results.csv
```

For exploratory runs, prefer an explicit output directory or suffix rather than overwriting canonical outputs.

## Dev/Test Separation

Rules:

- Dev labels may be used for graph rerank parameter tuning.
- Test labels may only be used for final evaluation.
- Test retrieval must use fixed configs selected before test evaluation.
- The selected config path must be recorded in test run summaries.

## Reproducibility Checklist

Before sharing a result, confirm:

- The run used labeled evaluation data only.
- The split seed/count/offset are recorded.
- The effective config is recorded.
- The dense encoder and prefixes are recorded.
- The graph config is recorded.
- The graph rerank config is recorded.
- Each script produced a run summary.
- The command sequence is documented.
- Tests passed or skipped only for documented local-model reasons.
- Final result tables can be traced back to ranked result files.

## Deferred Topics

Not covered in this document yet:

- Validation error taxonomy.
- Debug artifact format.
- Caching policy.
- CI policy.
