## 1. Freeze the zero-compatibility boundary

- [x] 1.1 Add failing architecture tests for retired trainable modules, version fields, legacy keys, aliases, builder identifiers, and workflow fallbacks.
- [x] 1.2 Update existing architecture tests so the intended final boundary is explicit.

## 2. Establish current method configuration

- [x] 2.1 Add strict R-GCN and Dense-FT method config tests, including rejection of old fields.
- [x] 2.2 Implement the typed `TrainableMethodConfig` union and register `Registry.configs.TRAINABLE_METHOD`.
- [x] 2.3 Move canonical R-GCN and Dense-FT configs to `configs/methods/` without version fields.
- [x] 2.4 Rename experiment `training_configs` references to `method_configs` and delete `configs/training/`.
- [x] 2.5 Delete trainable config normalizers, aliases, and compatibility facade modules after migrating callers.

## 3. Replace method registry compatibility views

- [x] 3.1 Add tests for complete method definitions and artifact shapes.
- [x] 3.2 Implement the current method registry with lifecycle, source, and train artifact enums.
- [x] 3.3 Migrate workflow, tuning, validation, and method-listing callers to the current registry.
- [x] 3.4 Delete projection, root registry facade, retrieval catalog facade, builder identifiers, and capability booleans.
- [x] 3.5 Make workflow status validation distinguish train artifact files and directories.

## 4. Rebuild workflow configuration and manifest

- [x] 4.1 Add tests for precompiled stage configs, generic commands, strict manifests, and rejection of missing stage configs.
- [x] 4.2 Compile complete typed pair, train, retrieve, and evaluate stage configs directly from typed method configs.
- [x] 4.3 Persist main and variant stage configs and make low-level `--config` accept only complete stage configs.
- [x] 4.4 Replace workflow command assembly with `script --config <stage-config>` and delete every legacy argv fallback.
- [x] 4.5 Implement the strict typed current manifest without a version field and migrate resume/status handling.
- [x] 4.6 Rename effective training config artifacts to effective method config artifacts across workflow and delivery.

## 5. Tighten training and artifact contracts

- [x] 5.1 Add exhaustive train config/result dispatch tests and remove default-to-Dense-FT branches.
- [x] 5.2 Rename R-GCN checkpoint types/functions, remove checkpoint versioning, and reject old checkpoints.
- [x] 5.3 Add typed Dense-FT metadata, remove metadata versioning, and share the contract between writer and reader.

## 6. Produce runtime provenance

- [x] 6.1 Add tests for R-GCN, Dense-FT, and BM25 retrieval provenance and script-level run summaries.
- [x] 6.2 Return built retrieval methods with typed runtime provenance from registry builders.
- [x] 6.3 Serialize builder-produced provenance in `run_retrieval.py` and delete config-subclass inference helpers.

## 7. Migrate ablations and documentation

- [x] 7.1 Patch ablations against current typed method config fields and remove ablation index versioning.
- [x] 7.2 Compile variant-specific stage configs while preserving baseline alias and invalidation behavior.
- [x] 7.3 Replace old training config documentation with current method config documentation and document rerun-only migration.
- [x] 7.4 Add repository scans that prevent retired trainable compatibility surfaces from returning.

## 8. Verification

- [x] 8.1 Run targeted config, registry, workflow, training, artifact, provenance, and ablation tests.
- [x] 8.2 Run the full pytest suite, Ruff, basedpyright, strict OpenSpec validation, and `git diff --check`.
- [x] 8.3 Run method listing plus R-GCN and Dense-FT workflow planning smoke commands.
