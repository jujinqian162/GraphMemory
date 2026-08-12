# ISETrace Cross-Encoder baselines

`cross_encoder` is the ISETrace-only trainable interaction scorer used for the strongest flat baseline. It exposes `variant=flat|provenance_unit`; only the candidate view changes.

## Scientific contract

- Backbone: pretrained `BAAI/bge-reranker-base` at fixed revision `2cfc18c9415c912f9d8155881c133215df768a70`, including its pretrained one-logit reranking head.
- Input: the raw query and candidate form one tokenizer pair; no graph, relation, graph identifier, query type, or Dense score is supplied.
- Supervision: every persisted exact-span positive and graph-free sampled negative becomes one binary example.
- Loss/selection: `BCEWithLogitsLoss`; one epoch; checkpoint selected by complete task-local development Recall@5 over the pretrained epoch-0 reranker and the completed epoch.
- Candidate pool: **all candidates in the selected task-local view**. There is no Dense first stage or tunable top-N cutoff.
- Variants share the backbone, tokenizer, max length 512, optimizer, sampling, epoch count, seed, and selection rule.
- Checkpoint metadata records the variant and retrieval rejects cross-view reuse.
- Evaluation uses the fixed ISETrace v7 test artifact and schema-v8 six-budget Coverage/Full Support metrics plus Budget-AUC.

The flat variant is the strongest flat comparison. The provenance-unit variant is not a flat baseline; it tests the same stronger scorer over source-backed provenance units without graph access.

## Smoke

```bash
uv run python experiment/run.py \
  name=isetrace_cross_encoder_flat_smoke \
  dataset=isetrace profile=smoke method=cross_encoder \
  method.variant=flat seed=13 split_seed=13 device=cuda:0

uv run python experiment/run.py \
  name=isetrace_cross_encoder_pu_smoke \
  dataset=isetrace profile=smoke method=cross_encoder \
  method.variant=provenance_unit seed=13 split_seed=13 device=cuda:0
```

## Formal runs

Run six independent jobs over seeds 13, 17, and 29:

```bash
uv run python experiment/run.py \
  name=isetrace_v7_cross_encoder_flat_s13_evalv8 \
  dataset=isetrace profile=full method=cross_encoder method.variant=flat \
  seed=13 split_seed=13 device=cuda:0

uv run python experiment/run.py \
  name=isetrace_v7_cross_encoder_provenance_unit_s13_evalv8 \
  dataset=isetrace profile=full method=cross_encoder method.variant=provenance_unit \
  seed=13 split_seed=13 device=cuda:1
```

Repeat with seeds 17 and 29 without changing other scientific settings. Separate jobs may run concurrently on separate GPUs. Do not set `cache.refresh=true` unless intentionally invalidating reusable scientific tasks.

Report mean ± sample standard deviation and trajectory-cluster paired bootstrap intervals for:

1. Cross-Encoder Flat minus Flat Dense-FT;
2. Cross-Encoder PU minus Cross-Encoder Flat;
3. residual R-GCN minus Cross-Encoder Flat;
4. residual R-GCN minus Cross-Encoder PU.

Also report measured full-candidate online latency and explicitly state that no first-stage candidate truncation was used.
