## Context

The repository currently has two different contract architectures.

The experiment, processed-artifact reference, and stage-result layers use closed Pydantic V2 models. Most scientific payloads still use a legacy split architecture:

1. a `TypedDict` or dataclass declares the static shape;
2. producers manually assemble dictionaries;
3. serializers manually enumerate fields or trace variants;
4. `graph_memory/validation/` independently repeats allowed fields, enum membership, scalar checks, and cross-record rules;
5. stage consumers decode JSON and use `typing.cast` instead of runtime parsing; and
6. validation often runs after an entire batch or training run.

`graph_memory/validation/` is approximately 3,010 lines and contains about 251 explicit `ContractValidationError` raises. The largest files are ranked-result/native-trace validation (942 lines), 2Wiki-provenance validation (582), dataset task validation (410), and train-pair validation (293).

This is a correctness problem, not only a size problem. Commit `3942114` introduced `dense_ft_rgcn_graph_retriever` in `RetrievalMethodId` but did not add it to the ranked-result validator's copied allowlist. Commit `1b5b98d` later repaired the validator to derive its values from the Registry. In the affected workflow shape, the method could prepare data and train before final ranked-result validation reported `unsupported method`. The copied allowlist is gone, but equivalent duplication remains across trace variants, serializers, dataset records, graph records, pair records, metrics, model configuration, and checkpoints.

Current late boundaries include:

- `retrieval/execution/service.py`: validates the complete prediction list only after every query has run;
- `models/graph_retriever/dev_evaluation.py`: constructs unchecked ranked-result dictionaries during training;
- `models/graph_retriever/checkpoint.py`: applies manual checkpoint/config validation at save/load in addition to dataclass `TypeAdapter` parsing;
- stage readers: cast decoded payloads to `EvidenceGraph`, `TrainPairRecord`, `RankedResult`, and dataset record types without parsing them;
- `models/provenance_rgcn/checkpoint.py`: maintains a separate manual checkpoint payload validator outside `graph_memory/validation/`.

This change is written against the current checkout on `feature/provenance-rgcn-ablation-validity`. All active schema-v3 provenance calibration, pair, inference, evaluation, and checkpoint invariants in that checkout are part of the preservation inventory even when their current implementation task is not yet fully exercised by long-running experiments.

Older change-local OpenSpec documents intentionally centralized validators in `graph_memory/validation/` and sometimes stated that cross-record checks should remain outside Pydantic. There are no promoted baseline specs under `openspec/specs/`; this change explicitly supersedes that implementation direction while preserving the scientific checks themselves.

## Goals / Non-Goals

**Goals:**

- Make one domain-owned Pydantic model the source of truth for every persisted scientific contract and every runtime value that currently owns validation or serialization logic.
- Delete `graph_memory/contracts/ranking.py` and the complete `graph_memory/validation/` package without compatibility facades.
- Preserve valid active artifact JSON shapes and all current scientific/leakage invariants.
- Detect configuration and input-contract failures before expensive model/encoder/tensor work.
- Detect task-local ranking and trace failures immediately after each method call.
- Make training dev evaluation and production retrieval use one ranked-result construction path.
- Replace unchecked artifact casts with Pydantic parsing at stage boundaries.
- Remove duplicated fields/defaults/allowlists between experiment configuration, domain configuration, serializers, and checkpoint metadata.
- Keep the test suite compact while retaining one owning-boundary behavior test for every custom invariant family.

**Non-Goals:**

- Do not change retrieval formulas, graph construction formulas, pair sampling semantics, loss formulas, checkpoint-selection formulas, metric formulas, method IDs, trace meanings, or active schema versions.
- Do not add a compatibility layer for old Python imports or retired artifact/checkpoint schemas.
- Do not convert every dataclass in the repository. Internal immutable computation-only records may remain dataclasses when they have no duplicate serialization or validation surface.
- Do not force PyTorch tensors, optimizer internals, DataLoader batches, or numerical device/index assertions into Pydantic.
- Do not replace raw benchmark-source parsing errors with Pydantic when the parser is decoding a third-party source format rather than a project artifact contract.
- Do not change artifact hashing, Prefect cache identity policy, MLflow organization, or workflow scheduling except where typed payload construction removes redundant validation inputs.
- Do not use this refactor to retire a retrieval trace or method variant. Such retirement requires its own approved scientific/method change.

## Validation Migration Inventory

| Current surface | Current duplication/problem | Owning replacement | Invariants that must move |
|---|---|---|---|
| `contracts/ranking.py` + `validation/ranking.py` + `retrieval/execution/results.py` | TypedDict, allowed fields, method set, trace dispatch, hand serializer | `retrieval/results` and retrieval trace models | exact candidate coverage, score order, finite values, subgraph references, method enum, metadata policy, all trace invariants |
| `contracts/graphs.py` + `validation/graphs.py` | TypedDict plus repeated node/edge fields | `graphs/contracts.py` | node union, one query node, IDs, endpoint/type/weight checks, request coverage, leakage |
| `graphs/provenance/contracts.py` + record conversion + provenance validator | dataclass `__post_init__`, TypedDict record graph, reconstruction, second validator | one provenance graph model used by records and runtime requests | legal typed transitions, endpoint/edge uniqueness, binding legality, finite weight, task/candidate identity |
| dataset `records.py` + parsers/converters + `validation/tasks.py` | TypedDict fields repeated in validators and dict literals | dataset-owned record and split models | prefixes, candidate IDs/positions, metadata, gold membership, dependency edges, task joins, leakage |
| 2Wiki-provenance records + `validation/twowiki_provenance.py` | record TypedDict, reconstructed graph, large policy validator | dataset Pydantic records embedding the authoritative provenance graph model | v3 identity, candidate/call pairing, branch degree, confidence fields/ranges/mass, buckets, binding, gold path, joins |
| `contracts/training_pairs.py` + config dataclasses + `validation/training_pairs.py` | TypedDict/config/summary fields repeated | `training_pairs/contracts.py` and config models | sample/label relation, membership, uniqueness, exact positives, negative exclusion, summary/config/count/overlap rules |
| `contracts/metrics.py` + `validation/metrics.py` + `evaluation/suites.py` + `evaluation/tables.py` | metric columns copied in three places | suite-owned Pydantic rows with aliases | schema discriminator, finite/range/`N/A` rules, table projections, per-task/failure-case records |
| graph-retriever config dataclasses + experiment config + `validation/model.py` + `to_json_dict` | four representations of model/training config | low-level graph-retriever Pydantic config composed by experiment models | feature/relation sets, ranges, optimizer, batch semantics, ablation/policies |
| evidence R-GCN checkpoint manual validator | field set plus nested manual validators plus TypeAdapter | evidence checkpoint envelope | schema/method/config identity, counters, finite metric, timestamps, required state maps |
| provenance R-GCN `_validate_payload` | parallel validator outside central package | provenance checkpoint envelope | schema/variant/construction/loss/inference/selection identity and required state maps |
| Dense-FT metadata dataclass + TypeAdapter | runtime parsing but no authoritative BaseModel | Dense-FT metadata model | model/prefix/batch/device/selection metadata |
| `validation/common.py` | generic dictionary readers and recursive policies | constraints, annotated types, and aggregate validators near owners | strict scalar behavior, task alignment, label-free JSON, finite numbers, unique IDs |

A migration matrix created in the first implementation batch will expand this table to every public `validate_*` function, helper-only scientific invariant, current caller, replacement model, and focused test. Deletion is blocked until every row is accounted for.

## Decisions

### 1. Use a minimal low-level model base, not a new central contract package

A small `DomainModel` base will live below all scientific domains, likely under `graph_memory/contracts/model.py` or an equivalently low-level primitive module. It will provide only common Pydantic behavior:

- `extra="forbid"`;
- `frozen=True`;
- `validate_default=True`;
- stable JSON-mode dumping conventions; and
- reusable strict/finite annotated scalar types.

It will not contain dataset, graph, retrieval, pair, model, checkpoint, or metric field schemas. Those models stay with their domain owners. `graph_memory/contracts/` will shrink toward primitives/model behavior rather than remain an artifact-schema warehouse.

Scientific booleans, integers, enums/literals, and floats will use field-specific strict aliases. Global `strict=True` is not selected because current JSON arrays must parse into immutable tuples and JSON-compatible enum/string values must round trip without forcing every container to arrive as the exact internal Python type.

Alternative: define all replacement models in `graph_memory/contracts/`. Rejected because it recreates a central schema package, causes upward dependency pressure, and separates models from the projectors/algorithms that own their invariants.

Alternative: use Pydantic `TypeAdapter` directly on the existing TypedDicts. Rejected because it retains dictionary construction/indexing, does not co-locate serializers or behavior, and leaves cross-field validators in a second surface.

### 2. Move authoritative enums down to avoid import cycles

A retrieval result cannot import `RetrievalMethodId` upward from a Registry module that already imports retrieval contracts. The method ID enum will move to a retrieval-owned low-level module and the Registry will import/re-export it. Equivalent ownership cleanup applies to graph edge/node enums, pair sample types, and model variants where a higher layer currently owns an enum required by a lower-layer model.

Alternative: keep method as `str` and validate it through a context set. Rejected because that recreates the exact copied/late allowlist failure that motivated the change.

### 3. Use explicit aggregate models for contextual validation

Local models validate local shape and invariants. Cross-object validation uses explicit, typed aggregates such as:

- dataset-specific prepared split (`rankings`, `labels`);
- evidence graph batch (`requests`, `graphs`);
- train-pair dataset (`requests`, `labels`, optional graphs, `pairs`, `summary`);
- per-task retrieval envelope (`request`, `result`);
- retrieval batch (`tasks`, `results`);
- evidence evaluation input (`predictions`, `labels`, `graphs`); and
- suite-specific metric table.

These aggregates may be transient and need not be serialized. Their purpose is to make every validation dependency visible and typed. Frozen child models can be accepted without reparsing their full content repeatedly.

Alternative: use `model_validate(..., context={...})` everywhere. Rejected as the default because context keys are untyped, invisible in the model field graph, and easy to omit at a call site. Pydantic context remains available only for narrowly justified framework integration, not core scientific joins.

Alternative: retain free `validate_*` functions for joins. Rejected because the central callable remains a second contract entrypoint and callers can again forget it.

### 4. Construct and validate ranked results per task

`assemble_ranked_result` will become a factory that returns a Pydantic `RankedResult`, not a dictionary. Retrieval execution will:

1. receive a typed execution task;
2. call the method;
3. convert method output/trace to typed result models;
4. construct a per-task request/result aggregate;
5. fail immediately on a local or contextual defect; and
6. append the validated result.

After the loop, a cheap retrieval-batch aggregate checks unique task IDs and exact task coverage. It does not repeat trace and node parsing.

The graph-retriever dev path will call the same factory/aggregate. Method latency/token accounting can be supplied explicitly so dev uses valid zero/audit values without a separate result shape.

Alternative: retain one list validator after retrieval. Rejected because it runs all later queries after the first invalid result and preserves the late-failure mechanism.

### 5. Convert all accepted native traces to one Pydantic discriminated union

Every trace branch currently accepted by `validation/ranking.py` will be inventoried. Current emitted and intentionally accepted surfaces include entity-search/GraphRAG history, `typed_local_bridge`, `execution_provenance`, `execution_provenance_local`, and `execution_provenance_subgraph`. Each retained trace becomes a closed BaseModel member with a literal `trace_kind`.

Nested trace records also become models. Existing dataclass trace serializers in `retrieval/execution/results.py` disappear; `model_dump(mode="json", exclude_none=True)` produces the artifact form. Scientific graph/path/probability/connectivity/fallback validators live on the member whose fields they relate.

If another active change replaces or removes an algorithm/trace before implementation, this change will follow the resulting active trace inventory, but the retirement must be explicit and cannot occur merely to simplify this refactor.

Alternative: keep dataclass traces and wrap them with `TypeAdapter`. Rejected because the hand-written serializer and variant dispatch would remain.

### 6. Make persisted graph models equal runtime graph models

Evidence and execution-provenance graphs will have one model representation from artifact parse through projector/request, training, retrieval, and evaluation. The 2Wiki-provenance record will embed the same execution-provenance graph model used by retrieval requests; `provenance_graph_from_record`-style reconstruction and duplicate graph validation will disappear.

Question/evidence nodes use a discriminator. Provenance node and edge enums remain typed. Immutable tuple containers are preferred internally while JSON dump preserves arrays.

Alternative: keep record graph models separate from runtime graphs and translate between them. Rejected because translation is another field-level truth source and currently forces a second validation pass.

### 7. Parse once at each artifact consumption boundary and dump once at publication

Generic `read_json` remains a raw infrastructure primitive. Every scientific stage immediately passes its decoded value to a module-level cached `TypeAdapter` or owning model. Raw dictionaries do not flow deeper into projectors or trainers.

Producers build models during conversion/construction, validate aggregate joins before publication, and dump with a standardized JSON-mode policy. `sort_keys=True` in the existing writer preserves deterministic key order. Optional fields that are absent in current artifacts will remain absent through an explicit model dump policy such as `exclude_none=True`, not a hand-written field enumeration.

Architecture tests will use AST inspection to reject direct `cast(read_json(...))` into scientific types. This is an architectural behavior guard, not a spelling snapshot.

Alternative: trust publication-time validation and cast cached payloads on consumption. Rejected because malformed/stale/manual artifacts and schema drift then fail later, and static casts provide no runtime guarantee.

### 8. Reuse low-level configuration models from experiment composition

Domain configuration models own scientific fields and defaults. The experiment layer may compose them directly, subclass only to add outer discriminators, or use the same strict field aliases. It must not define a structurally independent validation-bearing twin.

Stage adapters will stop converting a validated experiment model to an equivalent dataclass via `**model_dump()`. Trainers and builders receive the domain Pydantic model directly. Registry settings/build payloads that act as typed boundaries will also become Pydantic models or contain authoritative Pydantic child models; pure runtime dependencies such as encoder objects remain explicit arbitrary/opaque fields only where necessary.

Alternative: import experiment config into the domain. Rejected because it reverses the architecture dependency direction.

### 9. Split checkpoint metadata validation from opaque framework state

Each checkpoint family gets a closed Pydantic envelope. JSON-compatible metadata and nested config are fully typed. State dicts use an explicit opaque alias such as `SkipValidation[dict[str, object]]`; Pydantic verifies required slots and outer mapping presence but does not recurse into tensors.

Save flow:

1. construct the envelope from already validated config/results and state maps;
2. dump in Python mode while preserving opaque values; and
3. call `torch.save`.

Load flow:

1. `torch.load` the Python mapping;
2. validate the same envelope;
3. verify expected method/variant through an explicit typed load request/envelope aggregate; and
4. let PyTorch enforce model-state key and tensor-shape compatibility.

Evidence and provenance R-GCN use separate discriminated envelope members if their schemas differ. Dense-FT metadata becomes a normal BaseModel.

Alternative: ask Pydantic to validate every optimizer/tensor value. Rejected because those are framework-owned arbitrary structures and recursive coercion adds cost without a meaningful project schema.

### 10. Let metric row aliases own report columns

Evidence metric rows use valid Python field names with current human-readable CSV names as aliases. Serialization uses `by_alias=True`. The evidence suite owns its aggregate/per-task row adapters. Main/path/efficiency table projections reference model field aliases or suite-declared projections; there is no global `METRIC_COLUMNS` validator copy.

Other future metric suites provide their own row model. The common protocol can expose an adapter/model type rather than a free validation callback.

Alternative: keep metric rows as arbitrary dictionaries because columns contain spaces. Rejected because Pydantic aliases support the existing serialized schema and remove three copied column lists.

### 11. Replace the compatibility error with owning errors

Pydantic schema and model-validator failures will propagate as `pydantic.ValidationError`, preserving locations and nested context. The migration will not format only the first error or wrap all errors in `ContractValidationError`.

Algorithmic preconditions (`top_k > 0`), tensor runtime assertions, and operation-level errors remain direct errors owned by those APIs. Where an operation currently throws `ContractValidationError` for data that should be a model (for example empty gold evidence or mixed metric schemas), the invariant moves to the relevant model/aggregate.

Tests will match model locations and scientific reason fragments where custom logic matters, not the complete old sentence.

Alternative: keep `ContractValidationError` as a wrapper. Rejected because it loses multi-error/location information and preserves a validation compatibility surface the user explicitly wants removed.

### 12. Preserve behavior through an invariant migration matrix, not dual runtime paths

The migration is incremental in commits but not in final runtime architecture. Temporary branches may introduce Pydantic models alongside old validators only long enough to compare behavior in tests. Before a batch is complete, production callers switch to the model and the corresponding old validator code is removed. No permanent model-plus-validator dual execution remains.

The final deletion task runs residual searches for:

- `graph_memory.validation` imports;
- `graph_memory.contracts.ranking` imports;
- retired `validate_*` APIs;
- direct scientific casts after IO;
- hand trace serializer/validator dispatch;
- duplicate field-set constants used for schema validation; and
- `ContractValidationError` usage.

Alternative: retain deprecated aliases for downstream callers. Rejected by the repository's compatibility-free research-code policy and because aliases perpetuate multiple entrypoints.

### 13. Keep validation cost bounded and move it earlier

Pydantic parsing adds model-object construction to artifact reads. The implementation will:

- cache `TypeAdapter` objects at module scope;
- validate each decoded payload once per stage consumption;
- avoid model-to-dict-to-model cycles inside a stage;
- pass frozen model instances through projectors/builders;
- validate 2Wiki-provenance graph records once rather than reconstructing and validating the graph twice;
- use per-task immediate result validation plus a cheap aggregate coverage pass; and
- benchmark representative full prepared, pair, prediction, and provenance payload parsing against the removed validators.

The acceptance criterion is not that Pydantic be faster than every hand-written loop. It must keep preflight practical and avoid repeated validation inside epoch/DataLoader hot paths. No artifact validation runs in `__getitem__`, `collate_fn`, or each epoch.

## Risks / Trade-offs

- **[BaseModel is not dictionary-compatible, so call-site churn is broad]** → Update domain consumers to attribute access and dump only at IO/report boundaries; explicitly reject mapping facades and staged compatibility aliases.
- **[A custom invariant can be lost while deleting 3,010 lines]** → Build the validator-to-model migration matrix first, add behavior-freeze cases for each custom family, and block each module deletion on mapped tests.
- **[Pydantic coercion can silently alter scientific values]** → Use strict scientific scalar aliases and finite constraints; test booleans, strings, NaN, and infinity at representative boundaries.
- **[Cross-record model validators can become new god models]** → Keep aggregate models narrow and use one aggregate per join/use case; local invariants remain on child models.
- **[Import cycles can appear when low-level result models use Registry enums]** → Move authoritative enums down and re-export upward rather than importing Registry from retrieval.
- **[Existing JSON shape can change through `None`, enum, tuple, or alias dumping]** → Freeze representative normalized JSON fixtures and exact model-dump equality before switching producers.
- **[Pydantic validation of large payloads can add startup cost]** → Cache adapters, parse once before expensive work, avoid epoch hot paths, and record representative parsing timings.
- **[Current active changes continue editing validator files]** → Rebase/merge their scientific invariants into the migration matrix before implementation; do not delete an invariant because its old file conflicts.
- **[Checkpoint state cannot be fully described as JSON]** → Type only the envelope and metadata, preserve state maps opaquely, and retain framework loading checks.
- **[Error text changes can cause noisy tests]** → Preserve scientific rejection behavior and field locations, not exact legacy prose; update only owning-boundary tests.
- **[Removing validation after publication could trust bad in-memory models]** → Require producers to construct models, validate aggregate joins before publication, and parse cached content again at consumption.
- **[Changing class representations can affect Prefect serialization/cache keys]** → Keep task inputs/results Pydantic-serializable, compare task input projections and artifact identities, and make explicit JSON dumps before content hashing.
- **[Old OpenSpec documents point to retired paths]** → Update maintained architecture/operations docs and state that this change supersedes the old central-validator decision; historical change records remain historical.

## Migration Plan

1. **Freeze and inventory**: create the complete old-validator/caller/invariant/fixture matrix; add compact behavior-freeze tests and normalized JSON round trips, including the historical method-ID regression.
2. **Add primitives and move enums**: introduce the minimal domain model base and strict aliases; move low-level enums out of Registry/generic contracts where needed; verify dependency direction.
3. **Migrate retrieval vertically**: convert ranked result, subgraph, metadata, every accepted native trace, result factory, per-task envelope, retrieval batch, and dev inference; remove `validation/ranking.py` and `contracts/ranking.py` after parity.
4. **Migrate graphs and dataset records**: convert evidence/provenance graphs and dataset ranking/label/split models; eliminate graph reconstruction and move all leakage/join/provenance-v3 checks; remove `validation/graphs.py`, `validation/tasks.py`, and `validation/twowiki_provenance.py`.
5. **Migrate training pairs and configuration**: convert pair/config/summary aggregates, reuse domain config from experiment/stages, and validate all training input joins before expensive work; remove `validation/training_pairs.py`.
6. **Migrate model/checkpoint contracts**: replace graph-retriever config dataclasses/manual validation, evidence/provenance checkpoint validators, and Dense-FT metadata adapters; remove `validation/model.py`.
7. **Migrate evaluation**: introduce suite-owned row models/aliases and typed evaluation aggregates; derive table projections and remove `validation/metrics.py` plus applicable common errors.
8. **Convert every artifact boundary**: replace stage casts with cached adapters, pass models internally, dump explicitly at publishers, and validate cached payloads on read.
9. **Delete common/compatibility surfaces**: remove `validation/common.py`, `validation/__init__.py`, obsolete `ContractValidationError`, dead TypedDicts/serializers/converters, and all retired imports.
10. **Update docs and guards**: update the maintained architecture map and extension guidance; add AST/import guards; reconcile current active-change references at their maintained targets.
11. **Verify**: run focused model/join/checkpoint/result tests, representative parse timing, compact full pytest, Ruff, basedpyright error-level checking, compileall, Git diff checks, strict OpenSpec validation, and retired-surface scans.

Rollback is batch-local before the final deletion: each vertical migration can be reverted while its frozen behavior tests remain. After final cutover there is no runtime rollback/compatibility switch; reverting means reverting the relevant commits and regenerated artifacts together. Current-schema artifact files remain readable by the pre-change code because their serialized shapes are preserved.

## Open Questions

None blocking. The implementation must inventory the active trace/checkpoint variants after rebasing any concurrent method changes, but this is a reconciliation step, not an unresolved architecture decision.
