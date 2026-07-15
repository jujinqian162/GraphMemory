## Context

Both current R-GCN methods encode a complete task graph and then apply an independent evidence-node scorer. Inference sorts those node logits once, so the score of a later-hop paragraph cannot change after an earlier supporting paragraph has been selected. This is a poor fit for MuSiQue, where two-to-four-hop evidence dependencies must be completed inside a top-five budget and the formal Dense-FT-seeded R-GCN result has a much larger Full Support@5 gap than Full Support@10 gap.

The repository already provides the required label-side supervision through dataset-neutral `EvidenceLabel.gold_evidence_item_ids` and `gold_dependency_edges`. It also already has request-authoritative graph inputs, frozen text-embedding providers, typed R-GCN batching, strict checkpoints, `RankedResult`, retrieved-subgraph construction, and experiment-owned Dense-FT dependencies. This change should reuse those contracts rather than create a parallel workflow or a new public method.

The current branch uses composed YAML configuration under `configs/method_configs/` and typed experiment/stage models. The change must fit that live configuration path and must not restore retired registry/config compatibility layers.

## Goals / Non-Goals

**Goals:**

- Make both existing R-GCN methods decode evidence conditionally on the evidence selected so far.
- Preserve graph-aware behavior by exposing R-GCN node states and model-visible frontier signals to the decoder.
- Support chain and branching dependency structures without imposing one arbitrary gold order.
- Train on model-produced beam states so the decoder can recover after an early distractor.
- Decode at most five unique evidence nodes, matching the Full Support@5 budget while allowing one distractor in a four-hop question.
- Keep existing public method ids, graph inputs, Dense-FT dependency ownership, ranked output, metrics, and workflow artifact locations.
- Emit enough beam diagnostics to distinguish representation, search, stopping, and final-beam selection failures.

**Non-Goals:**

- Add a new retriever method id or a post-retrieval wrapper.
- Jointly fine-tune the Dense-FT text encoder in the first implementation.
- Add learned proposal graphs, semantic candidate edges, new graph relation types, or a query-conditioned edge gate.
- Change Full Support, path, connectivity, or latency metric formulas.
- Use decomposition questions, answers, supporting flags, or dependency labels at inference time.
- Preserve loading of checkpoints created before this change.
- Start with beam sizes larger than four or perform a broad hyperparameter sweep.

## Decisions

### 1. Modify the existing R-GCN methods in place

`dense_rgcn_graph_retriever` and `dense_ft_rgcn_graph_retriever` will both use the same beam-decoder implementation. Their only existing difference remains the source of frozen text embeddings. The workflow will not register another method, add another result-table row, or create another dependency chain.

This matches the requested ownership model and avoids duplicating graph tensorization, checkpoint loading, retrieval adapters, and experiment planning. The cost is an intentional checkpoint migration: the meaning and state of the existing R-GCN model changes.

**Alternative considered:** add `dense_ft_beam_rgcn_graph_retriever` as a separate method. This would isolate comparisons but was rejected because the requested change is an in-place R-GCN capability upgrade.

### 2. Encode the graph once and decode from cached node states

For each task, the existing input projection and typed R-GCN produce node states `H` and a question state `h_q` once. The existing base node scorer produces fallback logits from the same states. Beam expansion reuses `H`; it does not rerun Dense-FT or the R-GCN for every hypothesis.

The model boundary will expose an internal encoded-graph result containing node states, question indices, task offsets, node ids, and base logits. This result remains owned by `graph_memory.models.graph_retriever` and is not added to public retrieval requests or result contracts.

This design makes beam decoding cheap for approximately twenty candidates per MuSiQue task. It also isolates whether selection-conditioned decoding adds value before considering selection-conditioned message passing.

**Alternative considered:** rerun R-GCN after adding a selected-node feature at every step. This is more expressive but multiplies graph computation by beam size and step count and is deferred until the cached-state decoder is measured.

### 3. Use a hybrid ordered/set beam state

Each hypothesis contains:

- an ordered tuple of selected node indices;
- a set key used for hypothesis deduplication;
- the last selected node state;
- an attention-pooled selected-set context conditioned on `h_q`;
- a cumulative normalized score;
- a stopped flag and stop score.

The selected-set context handles branching structures where multiple topological orders are valid. The last selected node preserves a useful chain-local signal. Two hypotheses with the same selected set compete for one beam slot; the higher-scoring order is retained.

**Alternative considered:** a GRU state alone. It was rejected because it makes equivalent branching orders unnecessarily different and can waste capacity memorizing order artifacts.

### 4. Use separate first-hop and subsequent-hop heads

The first-hop head scores each memory node from its R-GCN state, the question state, their interaction, scorer features, and the base logit. The subsequent-hop head additionally receives the selected-set context, last-node context, step index, and graph-frontier features.

Graph-frontier features are derived only from the model-visible request graph and include deterministic relation-wise connectivity/weight summaries between the selected set and a candidate. They do not use gold dependency edges.

A separate stop head scores a synthetic `STOP` action from the question, selected-set context, last-node context, and step index. Decoder heads share the R-GCN hidden dimension rather than introducing a second text encoder.

### 5. Derive a dependency-aware dynamic oracle from generic labels

For gold evidence set `G`, gold dependency DAG `E`, and a selected set `S`, the valid next nodes are:

```text
Ready(S) = {v in G - S | every gold predecessor of v is in S}
```

At a training state:

- nodes in `Ready(S)` are valid next actions;
- distractor nodes are negative actions;
- unselected gold nodes that are not ready are masked, not labeled negative;
- `STOP` is positive exactly when `G` is a subset of `S` and negative otherwise.

When dependency edges are absent, `Ready(S)` falls back to all unselected gold nodes. Multiple ready nodes are trained as alternative valid actions, so no dataset-specific fixed sequence is required.

The oracle consumes only `EvidenceLabel`. MuSiQue parsing and conversion remain responsible for deriving the label DAG; model code does not import MuSiQue records.

### 6. Train on the same beam regime used at inference

The default beam size is `2`. At each training step, the decoder expands current hypotheses, accumulates action losses, retains model-preferred hypotheses, and guarantees at least one oracle-reachable hypothesis while gold completion remains possible. This exposes the decoder to its own mistakes without allowing early training collapse to remove all useful supervision.

The step loss is the negative log probability mass assigned to all valid next actions. The stop loss is binary cross entropy for the `STOP` action. The existing sampled node-ranking BCE remains as an auxiliary loss and continues to use existing train pairs.

Initial weights are configuration-owned:

```text
next_action_loss_weight = 1.0
stop_loss_weight = 1.0
aux_node_loss_weight = 0.2
```

Dense text embeddings stay frozen. A short decoder-only warmup is supported by freezing the existing R-GCN parameters, followed by joint R-GCN/decoder optimization with separate learning rates. The standard full profile will use the joint phase; decoder-only training is a diagnostic, not the final default.

### 7. Decode a maximum of five unique evidence nodes

Inference starts from an empty hypothesis, expands every non-stopped hypothesis with remaining candidates plus `STOP`, deduplicates equivalent selected sets, and retains the configured beam width. Decoding ends when every beam is stopped or `max_steps=5` is reached.

The initial configuration is:

```text
beam_size = 2
max_steps = 5
length_penalty_alpha = 1.0
deduplicate_selected_sets = true
```

Five steps are required even though MuSiQue has at most four gold paragraphs: a four-hop question must still be recoverable after one distractor enters the selected set.

Beam sizes `1`, `2`, and `4` are the bounded evaluation matrix. Training and inference beam sizes must match for a comparable run.

### 8. Preserve complete ranking and retrieved-subgraph semantics

The best stopped or maximum-length hypothesis supplies the selected prefix in its decoder order. If it stops before five nodes, base R-GCN logits fill the remaining top-five positions. Every remaining candidate follows exactly once in descending base-logit order. Selected nodes are never duplicated.

The existing graph view and induced-subgraph logic consume the final top-k node ids unchanged. `RankedResult.metadata` records the selected sequence, selected count, beam size, stop state, stop score, and best beam score. Debug output may include alternative final beams and oracle-coverage diagnostics during dev evaluation, but public ranking metrics continue to read `ranked_nodes` and `retrieved_subgraph` normally.

### 9. Make the checkpoint migration strict

The R-GCN checkpoint stores required decoder configuration, decoder weights, beam-training loss settings, and the existing model/trainer/encoder state. Loading a checkpoint without decoder fields fails with an actionable retraining error. No checkpoint version, optional fallback decoder, state-dict key patch, or legacy model path is introduced.

Dense-FT model directories are unaffected.

### 10. Select checkpoints and report diagnostics around evidence completion

The primary dev selection signal remains centered on Full Support@5. Training metrics additionally record next-action loss, stop loss, auxiliary node loss, hypothesis counts, oracle-reachable beam rate, premature-stop rate, average selected length, and beam-size settings.

Evaluation must report existing overall metrics plus diagnostic breakdowns needed to interpret the new decoder. A formal MuSiQue run should compare beam sizes `1`, `2`, and `4`, with Full Support@5 as the primary decision metric and Full Support@10, path metrics, and latency as guardrails.

## Risks / Trade-offs

- **[Exposure bias remains if model hypotheses are excluded]** → Train with model-produced beam states and keep training/inference beam sizes equal.
- **[Early model collapse removes every gold-reachable beam]** → Retain one oracle-reachable hypothesis during training and report the oracle-reachable beam rate.
- **[Branching orders consume duplicate beam slots]** → Deduplicate by selected-set key and keep the highest-scoring sequence for that set.
- **[A future gold node is incorrectly trained as a negative]** → Mask gold nodes whose dependency predecessors are not yet selected.
- **[The decoder ignores graph structure and behaves like a sequential dense reranker]** → Include relation-wise frontier features and require a no-frontier ablation.
- **[Beam search improves oracle coverage but final scoring chooses the wrong beam]** → Report both any-beam gold coverage and final-beam gold coverage.
- **[STOP shortens four-hop retrieval too aggressively]** → Train explicit stop labels, retain auxiliary node scoring, cap at five rather than four steps, and report premature-stop rate by hop count.
- **[Training cost rises because full candidate expansions replace sampled ranking only]** → Cache graph states per task, vectorize expansion across hypotheses, use beam size two by default, and keep the text encoder frozen.
- **[Existing checkpoints stop loading]** → Fail explicitly and retrain; do not hide the model change behind compatibility behavior.
- **[Existing HotpotQA/2Wiki behavior regresses]** → Preserve generic unordered-label fallback and run the existing R-GCN consuming workflows before completion.

## Migration Plan

1. Add beam config and internal contracts with failing tests while the old scorer remains executable.
2. Expose encoded graph states and implement first/subsequent/stop heads plus deterministic beam expansion.
3. Add dynamic-oracle and beam-aware losses, then switch the existing R-GCN trainer to the new objective.
4. Extend strict checkpoint save/load and switch existing R-GCN inference to beam decoding.
5. Update both canonical R-GCN method configs and composed experiment/stage config models.
6. Update training metrics, prediction metadata, run summaries, and operational documentation.
7. Run focused tests, static checks, the existing graph-retriever suite, and fresh smoke/quick consuming workflows.
8. Run a formal MuSiQue comparison for beam sizes `1`, `2`, and `4` before choosing the production beam size.

Rollback requires reverting the code/config change and using a pre-change code revision with its matching checkpoint. Checkpoints are not interchangeable across the boundary.

## Open Questions

None block implementation. The production beam size remains an empirical choice bounded to `1`, `2`, or `4`; configuration defaults to `2` until the formal MuSiQue comparison is available.
