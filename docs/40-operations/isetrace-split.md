# ISETrace provisioning and query split

The benchmark uses the revision-pinned ISETrace release, not the 32-trajectory sample. The sample under `data/isetrace/sample/` remains for smoke tests and authoring inspection.

## Provisioning

The registered trajectory revision is owned by `graph_memory/datasets/isetrace/registration.py`. Users do not repeat it in Hydra configuration.

```bash
uv run python scripts/prepare_dataset.py \
  --dataset isetrace \
  --name isetrace \
  --mirror
```

Preparation validates registered source sizes and digests. When natural-query authoring run metadata is present, its source revision, file size, and digest must agree with the registered trajectory source.

## Query allocation

`configs/dataset/isetrace.yaml` names one `natural_query_source` and a normalized `queries.split_ratio`. Preparation:

1. parses and resolves natural query records against pinned trajectories;
2. excludes and counts malformed or unresolvable records;
3. groups every valid query by trajectory;
4. deterministically assigns whole trajectory groups using `split_seed`;
5. applies the requested split's count/offset only after allocation.

No hand-filtered corpus or user-supplied split manifest is required. Every query for one trajectory stays in one split. Model/training seeds do not change split ownership.

The committed ratio is:

```yaml
queries:
  split_ratio:
    train: 0.5333333333333333
    dev: 0.13333333333333333
    test: 0.3333333333333333
```

The values are normalized and must cover exactly `train`, `dev`, and `test` with positive weights.

## Train/dev mixtures

Train and dev retain all selected natural queries and may add deterministic templates according to their own `queries.mix_ratio`. Template records are rendered only from provenance graphs whose trajectories already belong to that split. An insufficient template pool fails instead of silently changing the configured ratio.

Test is always natural-only. Template records are never generated for test.

Prepared artifacts include deterministic counts for resolved, malformed, unresolvable, split-target, natural-selected, template-selected, and origin totals. `query_metadata.json` records origin for reporting, while model-facing requests and graphs remain origin-free.

## Full profile

For `dataset=isetrace method=provenance_rgcn profile=full`, train/dev/test consume all queries assigned to their configured mixed splits. The evidence workflow's fixed dev cap is not applied.

## Leakage rule

- Training consumes only train trajectories.
- Checkpoint selection consumes only dev trajectories and uses natural dev metrics as primary.
- Test retrieval consumes only natural queries from test trajectories.

Generated natural queries remain unreviewed until separately reviewed and frozen. Split correctness does not make them formal paper gold.
