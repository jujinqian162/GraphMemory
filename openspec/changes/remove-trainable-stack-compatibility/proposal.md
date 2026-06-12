## Why

The trainable retrieval stack currently carries multiple compatibility layers for old configuration, registry, workflow manifest, checkpoint, and model metadata formats. These layers obscure ownership, duplicate command assembly, and make new methods depend on legacy schemas even though experiments and artifacts can be regenerated.

## What Changes

- **BREAKING** Replace all trainable method configuration inputs with one strict current format for R-GCN and Dense-FT under `configs/methods/`.
- **BREAKING** Remove trainable schema and artifact version fields, migrations, aliases, normalizers, compatibility facades, projections, and legacy workflow fallbacks.
- Introduce one typed trainable method configuration union loaded through the repository's existing `ConfigLoader`.
- Replace capability booleans and method-specific workflow checks with a typed method definition that declares lifecycle, input sources, model sources, encoder sources, and artifact shape.
- Compile complete typed stage configurations before execution; low-level scripts consume only stage configuration files.
- Make the workflow manifest a strict current-only contract with required stage configuration references.
- Make R-GCN checkpoints and Dense-FT model metadata strict current-only artifacts.
- Return runtime provenance from retrieval construction so summaries record the actual model/checkpoint, device, and encoder settings.
- Add architecture tests that prevent the removed compatibility surfaces from being reintroduced.

## Capabilities

### New Capabilities

- `current-trainable-method-config`: Strict current-only method configuration for every trainable retrieval method.
- `current-method-registry`: Typed method definitions that fully describe lifecycle, dependencies, and artifact semantics.
- `current-workflow-manifest`: Strict workflow manifests and precompiled stage configurations with no legacy command path.
- `current-trainable-artifacts`: Strict current R-GCN checkpoint and Dense-FT metadata contracts without version migration.
- `trainable-runtime-provenance`: Runtime-produced retrieval provenance used by experiment summaries.

### Modified Capabilities

None.

## Impact

- Affects trainable configuration loading, experiment configuration files, registry APIs, workflow planning and resume behavior, stage script CLIs, artifact validation, checkpoint and metadata readers/writers, ablation compilation, and retrieval summaries.
- Deletes `configs/training/`, `graph_memory/config/training_compat.py`, `graph_memory/training_config.py`, `graph_memory/registry/projections.py`, `graph_memory/retrieval_registry.py`, and `graph_memory/retrieval/catalog.py`.
- Existing trainable configs, manifests, checkpoints, Dense-FT model directories, and derived run artifacts are intentionally unsupported and must be deleted and regenerated.
- R-GCN and Dense-FT training and retrieval behavior remains semantically equivalent for current experiments, while their configuration and artifact interfaces become breaking current-only contracts.
