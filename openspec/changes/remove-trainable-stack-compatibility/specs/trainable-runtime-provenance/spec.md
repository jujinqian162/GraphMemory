## ADDED Requirements

### Requirement: Builder-produced provenance
Retrieval construction SHALL return typed provenance alongside the built retrieval method.

#### Scenario: Build Dense-FT retrieval
- **WHEN** the registry builds a Dense-FT retriever
- **THEN** provenance contains the actual model directory, effective device, and encoder settings loaded from current metadata and config

#### Scenario: Build R-GCN retrieval
- **WHEN** the registry builds an R-GCN retriever
- **THEN** provenance contains the actual checkpoint path, effective device, and checkpoint encoder settings

#### Scenario: Build BM25 retrieval
- **WHEN** the registry builds BM25
- **THEN** provenance omits encoder and checkpoint values rather than copying unrelated CLI defaults

### Requirement: Summary serialization
Retrieval run summaries SHALL serialize builder-produced provenance and MUST NOT infer checkpoint, device, or encoder values by inspecting config subclasses in the script layer.

#### Scenario: Write a Dense-FT run summary
- **WHEN** Dense-FT retrieval completes
- **THEN** the summary records the model directory and actual runtime device instead of `checkpoint=null` and `device=cpu`
