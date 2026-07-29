## 1. Freeze failing contracts

- [x] 1.1 Add failing test: two experiments differing only by `config.seed` produce identical prepared test records, order, and test prepare artifact digest (general datasets).
- [x] 1.2 Add failing test: train/dev sampling still varies with `config.seed` while test stays fixed.
- [x] 1.3 Add failing test: `twowiki_provenance` transform dev/test partition is identical across differing `config.seed` values and derives from the fixed `split_seed`.
- [x] 1.4 Add failing test: evaluate stage emits a per-task artifact keyed by `task_id` whose averaged metrics reproduce the aggregate row and whose `task_id` set equals the prediction `task_id` set.
- [x] 1.5 Add failing test: paired bootstrap main-results aggregation reports trainable mean±std, deterministic single value, and baseline-minus-method 95% CI with paired query count.
- [x] 1.6 Add failing test: paired analysis raises an explicit error when compared per-task `task_id` sets differ.

## 2. Block A — fixed test split decoupled from training seed

- [x] 2.1 Add top-level `split_seed: ScientificInt = 13` to `ExperimentConfig` and `ResolvedExperimentConfig` (`graph_memory/experiment/config.py`) and thread it through `resolve_experiment_config`; add default to `configs/config.yaml`.
- [x] 2.2 In `graph_memory/experiment/workflow.py:_prepare_config`, pass `seed=config.split_seed` for `split == "test"` and keep `config.seed` for train/dev.
- [x] 2.3 In `graph_memory/stages/transform.py`, make `deterministic_dev_test_partition` use the fixed `split_seed` for the dev/test boundary while keeping `transform.seed` for conversion-internal shuffles; thread `split_seed` into `transform_twowiki_task`.
- [x] 2.4 Confirm test prepare / transform artifact origin digests depend only on the effective seed value (13), so no gratuitous cache invalidation occurs when the seed value is unchanged.
- [x] 2.5 Update `configs/dataset/twowiki_provenance.yaml` if needed so the transform dev/test seed aligns with the fixed `split_seed` semantics.

## 3. Block B — persist per-task metrics

- [x] 3.1 In `graph_memory/evaluation/suites.py`, return `per_task_rows` (each carrying `task_id`) alongside the aggregate row instead of discarding them.
- [x] 3.2 Extend `graph_memory/stages/evaluate.py` (`EvaluateStageResult`, `run_evaluate_stage`, `materialize_evaluation`) to write `per_task.jsonl` and add it to the published artifact payload map.
- [x] 3.3 Project per-task metrics into the run output directory in `graph_memory/experiment/output.py`.
- [x] 3.4 Add/extend validation for the per-task payload contract (task_id uniqueness, metric coverage) near existing metric validation.

## 4. Block D — generalize paired bootstrap CI

- [x] 4.1 Extract a reusable `paired_bootstrap_ci` (and per-query delta pairing) from `graph_memory/analysis/provenance_ablation.py` into a shared analysis module; refactor provenance ablation to call it with unchanged behavior.
- [x] 4.2 Implement main-results aggregation logic: trainable mean±std, deterministic single value, baseline-minus-method paired 95% CI, reading per-task metrics by `task_id`.
- [x] 4.3 Enforce shared-test alignment: raise an explicit error when compared per-task `task_id` sets differ; never silently intersect.
- [x] 4.4 Add a `scripts/` entry point that consumes collected per-task metrics for a chosen method set and emits the aggregated main-results table + CIs.

## 5. Verification

- [x] 5.1 Run the focused new tests (fixed test split, per-task persistence, paired bootstrap, alignment error) and make them pass.
- [x] 5.2 Run `uv run ruff` and `uv run basedpyright` on touched modules.
- [x] 5.3 Run the existing provenance ablation test suite to confirm the extracted bootstrap keeps identical results.
- [x] 5.4 Smoke-run two experiments differing only by `seed` on a quick profile and diff the test prepare artifacts to confirm identity.
- [x] 5.5 `openspec validate add-fixed-testset-and-paired-bootstrap --strict`.
