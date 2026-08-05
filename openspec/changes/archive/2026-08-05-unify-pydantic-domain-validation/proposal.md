## Why

Scientific artifacts and runtime results currently split their contract across `TypedDict` declarations, dataclasses, copied field-name sets, hand-written `validate_*` functions, serializers, and unchecked `cast()` calls after JSON reads. This has already caused an expensive late failure: `dense_ft_rgcn_graph_retriever` was added to the Registry but omitted from the ranked-result validator's independent method allowlist, so training and ranking could complete before validation rejected a method that the rest of the system accepted.

The same failure mode still exists for ranked results and native traces, dataset ranking/label records, evidence and provenance graphs, train pairs, metric rows, R-GCN configuration, and checkpoint envelopes. The repository needs one authoritative, domain-owned Pydantic contract per scientific value and validation must run before expensive training or at the moment each result is produced, not at the end of a long workflow.

## What Changes

- **BREAKING**: Replace `graph_memory/contracts/ranking.py` and the complete `graph_memory/validation/` package with domain-owned Pydantic V2 models; no compatibility re-export or legacy validation facade will remain.
- Replace persisted/scientific `TypedDict` contracts and validation-bearing dataclasses with closed, frozen Pydantic models whose fields, constraints, cross-field invariants, serialization, and deserialization are co-located.
- Make retrieval results and every supported native trace a closed Pydantic discriminated union. Remove copied method allowlists, trace-kind dispatch validators, and hand-written trace serializers.
- Validate each retrieval result against its request immediately after the method returns, and make graph-retriever dev evaluation use the same result construction path as production retrieval.
- Model cross-record scientific invariants explicitly with aggregate Pydantic contracts for dataset ranking/label joins, graph/request joins, train pairs, retrieval batches, evaluation inputs, and metric tables.
- Validate artifact payloads with `TypeAdapter` immediately after JSON/JSONL/CSV reads and dump Pydantic models explicitly at publication. Remove type-only casts that currently assert unvalidated JSON has a scientific contract type.
- Replace duplicated experiment/domain R-GCN, negative-sampling, trace, metric, and checkpoint schemas with one low-level authoritative model reused by outer configuration and stage adapters.
- Move current evidence/provenance checkpoint metadata validation, including the parallel provenance-RGCN payload validator, into Pydantic checkpoint envelope models while leaving tensor/state-dict internals opaque to Pydantic.
- Preserve current scientific behavior, field names, JSON shapes, method IDs, trace kinds, metric formulas, graph semantics, ranking semantics, active schema versions, and deterministic artifact content for valid inputs.
- Preserve custom leakage, graph, ranking, probability-mass, connectivity, label, pair-supervision, checkpoint, and metric invariants as model validators; this change removes duplicate validation surfaces, not scientific safeguards.
- Keep runtime tensor shape/device/index assertions and algorithmic precondition errors at their computation boundaries; they are not persisted data-schema validation and will not be forced into Pydantic models.
- Update architecture documentation, OpenSpec-facing guidance, and owning-boundary tests so future methods and fields extend one model rather than a type, serializer, and validator in parallel.

## Capabilities

### New Capabilities

- `domain-owned-pydantic-contracts`: Closed Pydantic models are the single source of truth for scientific records, requests, graphs, results, traces, training records, model configuration, checkpoints, and metric rows.
- `pydantic-retrieval-result-boundary`: Retrieval and training-time dev evaluation construct, serialize, and validate ranked results and native traces through one immediate Pydantic boundary.
- `validated-scientific-artifact-boundary`: Prepared datasets, graphs, train pairs, predictions, checkpoints, and evaluation artifacts are parsed and joined through typed Pydantic boundaries instead of unchecked casts or detached validators.
- `fail-fast-training-contracts`: Every schema, configuration, join, and checkpoint invariant needed by an expensive training run is established before the epoch loop, while dev results use the production result contract.
- `legacy-validation-retirement`: The central validation package, ranking TypedDict module, copied field sets, compatibility exceptions/facades, and residual imports are removed with architecture guards against their return.

### Modified Capabilities

None. The repository has no promoted baseline specifications under `openspec/specs/`; this change establishes the replacement contract while superseding older change-local decisions that centralized validators under `graph_memory/validation/`.

## Impact

- Primary removals: `graph_memory/contracts/ranking.py`, `graph_memory/validation/`, and obsolete `ContractValidationError` compatibility surfaces once all callers use Pydantic validation errors or owning-domain errors.
- Primary migrations: `graph_memory/contracts/{graphs,metrics,training_pairs}.py`, dataset `records.py` modules, `graph_memory/graphs/provenance/`, `graph_memory/retrieval/{contracts,requests,execution}/`, `graph_memory/training_pairs/`, `graph_memory/evaluation/`, `graph_memory/models/{dense_finetune,graph_retriever,provenance_rgcn}/`, Registry payload/settings models, and stage artifact readers/writers.
- Workflow/stage impact: prepare, graph, pair, train, retrieve, evaluate, benchmark, cache/publication, and test fixture construction paths will consume model instances and explicitly dump JSON values.
- Internal Python construction changes from dictionary indexing to model attributes. Existing serialized scientific artifacts remain valid when they conform to the current active schema; no retired-schema fallback is added.
- Existing OpenSpec changes that mention `graph_memory/validation/` remain historical context. Durable architecture documentation and future implementation targets will use domain-owned Pydantic contracts.
- Pydantic is already a locked dependency (`>=2.13.4,<3`); no new runtime package is introduced.
