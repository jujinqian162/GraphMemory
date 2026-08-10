# ISETrace provisioning and trajectory split

The benchmark uses the revision-pinned ISETrace release, not the 32-trajectory sample.

## Configuration

`configs/dataset/isetrace.yaml` configures trajectory counts directly:

```yaml
trajectories:
  splits:
    train: {natural: 2410, template: 7788}
    dev: {natural: 352, template: 1160}
    test: {natural: 1207, template: 0}
```

The numbers count trajectories, not queries. For `profile=full`, they define the complete scientific split:

- Authored natural records are resolved back to source trajectories in deterministic query-authoring order.
- The first `test.natural` distinct trajectories in that order own the natural-only test split; every valid authored query from those trajectories stays in test.
- The remaining authored-natural trajectory IDs are sorted and shuffled with `split_seed`. Train takes the first configured number while dev takes the last configured number, leaving any middle trajectories unused; changing only the train count therefore preserves dev and test.
- `natural: N` selects N trajectories and keeps every valid authored natural query resolved to those trajectories. Smaller train counts form deterministic nested prefixes of larger train counts.
- `template: N` selects N trajectories and creates one template query per trajectory.
- Natural and template selections may overlap inside one split.
- The union of selected trajectories is disjoint across train, dev, and test.

For bounded execution profiles, the resolved profile count caps the total materialized tasks after this trajectory-level ownership is established. In particular, `profile=smoke` materializes one task per required split while preserving disjoint split ownership; it does not evaluate the full test corpus.

Test is natural-only and requires `natural > 0, template = 0`.

## Provisioning

The registered trajectory revision is owned by `graph_memory/datasets/isetrace/registration.py`:

```bash
uv run python scripts/prepare_dataset.py \
  --dataset isetrace \
  --name isetrace \
  --mirror
```

Preparation validates source identity, resolves authored queries, selects trajectories, then builds retrieval views only for the selected trajectories. `query_metadata.json` records natural/template origin for reporting; origin is not a model feature.

## Leakage rule

- Training consumes only train trajectories.
- Checkpoint selection consumes only dev trajectories.
- Test consumes only natural queries from test trajectories.
- No trajectory occurs in more than one split.

This rule controls exact trajectory overlap only. The active RQ2 workflow does **not** group trajectories by shared `source_intent_id` or exact normalized intent text, and therefore does not claim an unseen-intent, unseen-task-family, or component-safe split. `scripts/build_isetrace_split.py` and `data/isetrace/splits/v1` implement a separate legacy intent-component allocation, but `graph_memory/datasets/isetrace/benchmark_adapter.py` does not consume that manifest.

Any future component-safe or unseen-family evaluation must use a separately named protocol and must not be presented as the split used by the current 2,000-query main test set.

## Reproducible split identity

A formal run must retain:

- the prepared test dataset artifact digest from `assets/manifest.yaml`;
- the unique `task_id` set in `metrics/per_task.jsonl`;
- the `graph_id`/trajectory cluster for every test query;
- the query-authoring metadata sidecar digest;
- fixed authoring and split seeds.

The main-results aggregator rejects different test artifact digests or task sets. Legacy per-task files without `graph_id` can be joined to the frozen query-authoring metadata sidecar with `--query-metadata`; new evaluations persist `graph_id` directly.
