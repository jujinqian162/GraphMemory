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
- Implementation handoff must be documented after implementation so reviewers know how to read and review the system.

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

For experiment-runner runs, the chain also includes:

```text
experiment config + profile + CLI overrides
  -> runs/<experiment_name>/manifest.json
  -> runs/<experiment_name>/config/effective_config.json
  -> run-local artifacts
```

The manifest records generated paths, selected stages, selected methods, and status metadata. It is the first file to inspect when reproducing or debugging a named run.

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
CLI overrides > experiment config/profile > code defaults
```

Rules:

- Config files are for repeatable runs.
- CLI overrides are for temporary changes.
- The final `effective_config` must be written into run summaries and, for experiment-runner runs, into `runs/<experiment_name>/config/effective_config.json`.
- If a CLI override changes a scientific setting, the run summary must make that visible.

Stable experiment defaults live under:

```text
configs/experiments/
```

Tuning search spaces live under:

```text
configs/search_spaces/
```

Published, curated selected configs live under:

```text
configs/published/
```

Ordinary runs should write selected tuning outputs under:

```text
runs/<experiment_name>/tuned/
```

The current HotpotQA evidence-retrieval defaults are:

```text
configs/experiments/hotpotqa_evidence_retrieval.json
configs/search_spaces/graph_rerank.json
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

The root `README.md` should stay short: project purpose, setup, the fastest safe command path, and a link to `docs/40-operations/commands.md`. Avoid duplicating the full command runbook in both places.

## Tuning Simplicity

Phase 1 intentionally does not require a persistent initial-score cache.

Rules:

- Dev tuning may recompute BM25 or dense initial rankings while the pipeline is being debugged.
- The rerank function still receives explicit initial scores so future score reuse remains easy to add.
- Any quick tuning on a reduced dev artifact is a debug aid only; official Phase 1 graph rerank configs must be selected on the documented dev split.
- If a score cache is added later, it must become a named artifact with validation and run summary fields.

## Implementation Handoff

The implementation must produce a code review handoff after Phase 1 scripts and core modules exist.

Required handoff location:

```text
docs/40-operations/implementation-handoff.md
```

It must explain:

- where to start reading the code
- the main control flow from raw data to final tables
- where core abstractions live
- which files are CLI adapters versus core logic
- which tests cover each major boundary
- what a reviewer should check first
- what Phase 1 intentionally excludes

This handoff is required before the implementation is considered review-ready.

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

For normal exploratory and repeatable runs, prefer the experiment-runner layout:

```text
runs/<experiment_name>/manifest.json
runs/<experiment_name>/config/effective_config.json
runs/<experiment_name>/inputs/test.input.json
runs/<experiment_name>/graphs/test.graphs.json
runs/<experiment_name>/tuned/dense_graph_rerank.dev_selected.json
runs/<experiment_name>/predictions/test.dense_graph_rerank.ranked.json
runs/<experiment_name>/metrics/test.dense_graph_rerank.metrics.csv
runs/<experiment_name>/tables/main_results.csv
```

The older `data/hotpotqa/processed` and `results` examples remain useful for low-level contract debugging, but named runs should avoid overwriting shared canonical paths.

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
