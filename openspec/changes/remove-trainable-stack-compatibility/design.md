## Context

The trainable stack grew through successive R-GCN and Dense-FT additions. The current implementation represents the same experiment through legacy method configs, compatibility normalizers, typed stage configs, registry projections, workflow capability booleans, and fallback command assembly. Artifact semantics are split between registry data and method checks, while retrieval summaries reconstruct provenance from CLI defaults instead of the built runtime object.

This is an experimental repository. Old runs, configs, checkpoints, and model directories can be regenerated, so preserving old readers has negative value. Structural validation remains required, but it applies to exactly one current contract.

## Goals / Non-Goals

**Goals:**

- Establish one strict method config union covering R-GCN and Dense-FT.
- Establish one typed method registry that completely describes lifecycle, dependencies, and artifact shape.
- Compile complete typed stage configs before execution and make them the only low-level script input.
- Establish one strict workflow manifest with required stage config references.
- Establish current-only R-GCN checkpoint and Dense-FT metadata contracts.
- Record runtime-produced retrieval provenance in summaries.
- Delete all trainable migrations, aliases, compatibility facades, projections, and fallback paths.
- Make train config branching statically and dynamically exhaustive.

**Non-Goals:**

- Preserving or migrating old trainable configs, manifests, checkpoints, model directories, or run artifacts.
- Changing model architecture, loss functions, sampling algorithms, training semantics, evaluation metrics, or cache invalidation behavior.
- Removing compatibility behavior unrelated to the trainable stack.
- Replacing the bounded SentenceTransformers 2.7 loss-observation hook.

## Decisions

### One current method config per method

Canonical configs live under `configs/methods/` and share the top-level concepts `method`, `default_profile`, `encoder`, `pairs`, `train`, and `profiles`. Experiment configs reference them through `method_configs`.

The loader uses the existing `ConfigLoader.load(spec, argv)` entry point with a typed `TrainableMethodConfig` union. Unknown, missing, and old fields fail. There is no format discriminator because only one format exists.

Alternative considered: retain `schema_version` while supporting only one version. Rejected because it implies a future migration mechanism and adds no validation value.

### Method definitions replace projections and capability booleans

`graph_memory.registry.methods` owns method identity and exact semantics. A `MethodDefinition` declares lifecycle, graph input source, graph config source, model source, encoder source, and train artifact descriptor including basename and file-or-directory shape.

Alternative considered: add more booleans to the existing retrieval registry. Rejected because combinations remain ambiguous and continue to force method checks.

### Stage configs are the workflow/runtime boundary

The workflow compiler loads a method config once, resolves its selected profile, applies ablation patches to typed data, constructs complete stage config dataclasses, and writes each config under the run directory. Commands are always `script --config <stage-config>`.

Low-level pair, train, retrieve, and evaluate scripts never load method configs or reconstruct method-specific argv. The workflow planner never falls back to artifact-derived argv.

Alternative considered: retain generated argv as the execution contract. Rejected because it duplicates config interpretation and cannot be validated as one typed object.

### The manifest is strict current state, not a migration boundary

A typed manifest contract validates the complete current structure, including required stage config references. Resume accepts only this contract. `--force` deletes/rebuilds stale experiment state; it does not convert old manifests.

Alternative considered: keep the v1 reader only for resume. Rejected because that preserves a second planner input and command path indefinitely.

### Train artifacts use method-specific current contracts

R-GCN checkpoint types and functions are renamed to R-GCN-specific names and omit `checkpoint_version`. Dense-FT writes and reads a typed metadata dataclass without `schema_version`. Both reject unknown fields, so old versioned artifacts fail explicitly.

Alternative considered: silently ignore version fields. Rejected because it weakens structural validation and accidentally accepts stale artifacts.

### Runtime construction returns provenance

Retrieval construction returns a built method together with typed `RetrievalProvenance`. The provenance contains only values actually used by the builder: model/checkpoint reference, effective device, and effective encoder settings where applicable. Scripts serialize this object rather than infer values from config classes or CLI defaults.

Alternative considered: teach `run_retrieval.py` more method-specific extraction rules. Rejected because the script does not own runtime resolution.

### Exhaustive unions fail on unsupported methods

Every branch over `TrainStageConfig` and train results explicitly handles R-GCN and Dense-FT and ends with `assert_never`. Unsupported runtime values raise rather than falling into a default Dense-FT path.

## Risks / Trade-offs

- [Large cross-cutting diff] -> Implement in dependency order, keep targeted tests green after each contract slice, then run the full suite.
- [Existing tests encode compatibility behavior] -> Replace those tests with strict current-contract and module-absence tests rather than weakening new contracts.
- [Old local runs become unreadable] -> Treat deletion and rerun as the documented migration procedure.
- [Workflow and scripts can temporarily disagree during implementation] -> Complete stage config compilation and generic command assembly as one integrated slice before marking either task complete.
- [Artifact renames can leave hidden imports] -> Use repository-wide symbol scans plus basedpyright and full pytest.
- [Method registry redesign can affect non-trainable methods] -> Keep definitions generic enough for current static retrieval methods while changing only trainable lifecycle and artifact semantics.

## Migration Plan

1. Add failing architecture and current-contract tests.
2. Introduce the typed method config union and migrate canonical config files.
3. Replace registry projections and capability booleans with method definitions.
4. Compile stage configs and remove legacy workflow command assembly.
5. Enforce the strict manifest.
6. Replace checkpoint, metadata, train union, and provenance contracts.
7. Migrate ablations and documentation.
8. Delete old config/run artifacts used by local smoke tests and regenerate them.

Rollback is source-control rollback only. No runtime downgrade or old artifact conversion is provided.

## Open Questions

None. The reviewed implementation plan resolves the breaking-change decisions.
