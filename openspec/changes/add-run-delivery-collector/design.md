## Context

The current experiment runner writes complete run artifacts under `runs/<experiment_name>/`. A full Phase 2 training run includes small audit files and very large reproducible intermediates in the same tree. The report in `report/phase2_rgcn_full_train_analysis.md` depends on compact artifacts such as aggregate tables, per-method metrics, training curves/history, graph stats, run summaries, and capped failure cases. It does not require full prepared inputs, full graph JSON, complete ranked predictions, checkpoint binaries, or raw train-pair JSON for normal analysis.

The collector must be usable on the HPC server before copying data back to this workspace. It should therefore be a thin standard-library CLI under `scripts/deliver/` and should preserve the source directory layout to make later analysis paths obvious.

## Goals / Non-Goals

**Goals:**

- Copy compact report-relevant files from `runs/<train_id>/` to `results/<train_id>/` by default.
- Preserve relative paths under the run directory.
- Exclude known large intermediates by directory, filename, extension, and size guard.
- Write a delivery manifest that records copied and skipped files with byte counts and reasons.
- Keep the implementation independent from training/evaluation code and free of new dependencies.

**Non-Goals:**

- Do not compress files or invoke `scp`.
- Do not regenerate missing artifacts.
- Do not copy checkpoints, full graphs, full predictions, raw prepared inputs, or raw train pairs by default.
- Do not change report content or metric definitions.

## Decisions

1. Use inclusion rules before size fallback.

   The collector will explicitly include small evidence classes: `manifest.json`, `config/**/*.json`, `tables/**/*`, `metrics/**/*.csv`, all `*.run_summary.json`, graph `*.stats.json`, learned `effective_training_config.json`, `train_metrics.jsonl`, `train_run_summary.json`, `train.pairs.summary.json`, tuning `*.dev_selected.json`, capped `debug/failure_cases_*.jsonl`, and optional report files. A max-size guard then protects against unexpected large files that happen to match a broad pattern.

   Alternative considered: copy everything below a size threshold. That risks missing semantically important small files only if thresholds are mis-set less often, but it also makes the delivery contract opaque and may include noisy intermediate fragments.

2. Exclude high-volume artifact classes by default.

   The default exclusion list covers `inputs/`, full `graphs/*.graphs.json`, full `predictions/*.ranked.json`, learned `checkpoints/`, raw `train.pairs.json`, dense embeddings, model binaries, and common checkpoint extensions. Summary sidecars under those directories remain eligible when they match inclusion rules.

   Alternative considered: exclude whole directories such as `graphs/` or `predictions/`. That would drop useful `run_summary` and graph stats files needed to audit the report.

3. Keep the public CLI convention-based.

   The CLI takes `--name <train_id>`, optional `--output-root`, optional `--include-report`, and optional `--max-file-size-mb`. By convention, `--name rgcn_full_train` reads `runs/rgcn_full_train/` and writes `results/rgcn_full_train/` by default. Existing destinations are merged/overwritten only for copied files, so rerunning after more artifacts appear is safe.

   Alternative considered: expose `--run-dir` as the primary input. That is more flexible, but it makes the common path noisier and hides the repo's run-directory contract from the help text.

4. Make the manifest the audit surface.

   `delivery_manifest.json` records source run path, output directory, copied file entries, skipped file entries, total bytes, and rules metadata. This lets the local analysis verify exactly what was transferred without needing the original large tree.

## Risks / Trade-offs

- A future report may need a currently excluded full prediction sample -> add an explicit CLI option or fixture-specific include later rather than silently copying all predictions now.
- Size guard may skip a legitimate unusually large debug file -> manifest records `too_large`, making the omission visible.
- Preserving layout under `results/<run_id>` can coexist with existing result files -> default output is run-scoped, avoiding flat namespace collisions.
