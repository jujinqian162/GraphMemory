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

The numbers count trajectories, not queries:

- `natural: N` selects N trajectories and keeps every valid authored natural query resolved to those trajectories.
- `template: N` selects N trajectories and creates one template query per trajectory.
- Natural and template selections may overlap inside one split.
- The union of selected trajectories is disjoint across train, dev, and test.

Selection is deterministic from `split_seed`. If the configured counts require more trajectories than the valid natural-query source covers, preparation fails with requested and available counts; adjust the config for that corpus.

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
