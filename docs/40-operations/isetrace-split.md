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

`configs/dataset/isetrace.yaml` names one `natural_query_source` and explicit per-origin query counts. Preparation:

1. parses and resolves natural query records against pinned trajectories;
2. excludes and counts malformed or unresolvable records;
3. groups every valid query by trajectory;
4. deterministically assigns whole trajectory groups using `split_seed` and the registered v7 ownership weights;
5. selects exactly the configured natural and template task counts from each frozen split.

No hand-filtered corpus or user-supplied split manifest is required. Every query for one trajectory stays in one split. Model/training seeds do not change split ownership. For the 4,066-query server corpus, the registered ownership plan resolves to 2,692 train, 393 dev, and 981 test natural queries; these targets are data identity, not experiment mixture controls.

The user-facing configuration contains only exact task counts:

```yaml
queries:
  splits:
    train: {natural: 2692, template: 5384}
    dev: {natural: 393, template: 393}
    test: {natural: 981, template: 0}
```

Train/dev `natural` may be zero, which enables template-only supervision while preserving the same trajectory ownership. Template records are rendered only from provenance graphs belonging to that split. Requested counts are checked exactly; an insufficient natural or template pool fails with requested and available counts instead of silently truncating.

Test is strictly natural-only and requires `natural > 0, template = 0`.

Prepared artifacts include deterministic counts for resolved, malformed, unresolvable, split-target, natural-selected, template-selected, and origin totals. `query_metadata.json` records origin for reporting, while model-facing requests and graphs remain origin-free.

## Full profile

For ISETrace, `dataset.queries.splits` is the sole query-count authority. Evidence-dataset profile count caps are not applied. Smoke/quick jobs that need fewer ISETrace tasks must override the explicit natural/template counts.

## Leakage rule

- Training consumes only train trajectories.
- Checkpoint selection consumes only dev trajectories: natural Recall@5 when natural dev tasks exist, otherwise template Recall@5 for template-only dev.
- Test retrieval consumes only natural queries from test trajectories.

Generated natural queries remain unreviewed until separately reviewed and frozen. Split correctness does not make them formal paper gold.
