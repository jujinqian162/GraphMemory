## Context

graphrag currently mixes Dense similarity with global PPR, so a noisy entity hub can perturb every candidate. execution_provenance_retriever currently adds several mostly structural terms to many paths and globally reorders path nodes, which can improve chain coverage while lowering MRR.

The implementation boundary is now explicit: improve only these two non-training methods. Existing datasets and caches are inputs, not migration targets. Any concern about label-conditioned provenance data belongs to a separate future dataset change with a distinct dataset identity.

## Goals / Non-Goals

**Goals:**

- Make both methods abstain to exact Dense output unless a bounded local proposal survives method-owned gates.
- Preserve a configurable Dense prefix and the relative order of all non-promoted candidates.
- Bound proposals per original Dense seed and resolve conflicts deterministically.
- Keep the current method IDs, workflow, dataset inputs, prediction schema, and artifact roles.
- Expose method-native trace evidence for accepted/rejected proposals without changing evaluation output contracts.

**Non-Goals:**

- Any change under graph_memory/datasets/, dataset YAML, converter scripts, fixtures, schema versions, manifests, or prepare cache identity.
- Any change to trainable methods, training pairs, checkpoints, or model caches.
- New shared evaluation metrics, paired-bootstrap tooling, counterfactual workflow entrypoints, or output artifact roles.
- LLMs, external knowledge, model training, a new public method ID, planner/workflow changes, or a generic graph wrapper.

## Decisions

### 1. Treat existing requests as immutable scientific inputs

Both methods consume the current projected request. Execution provenance uses the request graph's existing edge weights exactly once as method-local proposal evidence; it does not reinterpret or regenerate the dataset. GraphRAG derives its private entity/title evidence from the existing text-ranking request.

Alternative: clean or replace the shared provenance dataset in this change. Rejected because it invalidates unrelated experiment caches and exceeds the requested method-only scope.

### 2. Use bounded validity-gated execution paths

A path proposal must begin at an original Dense top-S seed, end at a distinct candidate, satisfy endpoint binding and required FEEDS/RETURNS structure, and avoid invalidation. Structural checks are boolean gates rather than additive bonuses. Confidence consumes existing semantic edge weights once and applies only a hop penalty. Search is bounded by seed count, beam width, hop count, expansion count, and one selected partner per seed.

### 3. Use pair-local GraphRAG bridges

GraphRAG separates title and body entity evidence, suppresses ambiguous aliases and high-frequency hubs, and resolves a title group to at most one sentence candidate with the same frozen encoder instance used by Dense ranking. It produces at most one partner proposal per anchor and never runs global PPR/RRF fusion.

### 4. Apply protected-prefix stable insertion inside each method

Proposals originate from the original Dense ranking. Protected-prefix members retain membership and order. Partners already above their anchor, inside the protected prefix, below threshold, or producing no movement are rejected. Conflicts use confidence, original anchor rank, original partner rank, and node ID. A real promotion inserts the partner directly after its anchor subject to the prefix; all non-promoted candidates retain relative order.

If no effective promotion exists, the original RankedNode objects, order, and scores are returned unchanged. After a real promotion, the original descending Dense score multiset is assigned to the final order so scores remain deterministic and monotonic.

### 5. Keep diagnostics method-local

Native traces record original/final ranks, confidence, gate outcomes, rejection/conflict reasons, displacement, exact fallback, and emitted edges. Trace serialization/validation may be extended for these two native trace variants, but aggregate metric columns and evaluation artifact payloads remain unchanged.

## Risks / Trade-offs

- Existing provenance graph weights may reflect historical dataset construction. This implementation improves method behavior over that existing contract but makes no claim that the dataset itself is leakage-free.
- Heuristic GraphRAG evidence may abstain frequently. Exact Dense fallback makes low intervention safe and observable.
- Local insertion may still hurt individual tasks. The method thresholds remain explicit config inputs; broader statistical selection is deferred.

## Migration Plan

1. Restore every dataset, converter, fixture, cache, training-method, and shared-evaluation file to the branch baseline.
2. Adapt execution provenance to the existing request graph and retain only local gated promotion behavior.
3. Retain GraphRAG typed local bridge behavior over the existing text-ranking request.
4. Reduce native traces and validators to the fields directly needed by the two methods.
5. Run focused method/config/workflow tests and the repository quality gates.
