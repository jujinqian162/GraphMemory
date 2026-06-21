## ADDED Requirements

### Requirement: 2Wiki experiment exposes trainable methods
The system SHALL make `dense_ft` and `dense_rgcn_graph_retriever` selectable from the named 2Wiki experiment configuration with the method configs required for trainable workflows.

#### Scenario: Named 2Wiki config includes trainable methods
- **WHEN** `configs/experiments/2wiki_tiny.json` is loaded
- **THEN** its configured methods include `dense_ft` and `dense_rgcn_graph_retriever`
- **THEN** its `method_configs` map includes config paths for both trainable methods

#### Scenario: 2Wiki trainable manifest creates stage configs
- **WHEN** an experiment manifest is initialized for dataset `twowiki` with methods `dense_ft` and `dense_rgcn_graph_retriever`
- **THEN** the manifest includes `pairs`, `train`, `retrieve`, and `evaluate` stage configs for both trainable methods
- **THEN** every generated stage config for those methods records `dataset` as `twowiki`

### Requirement: 2Wiki trainable devices use CUDA
The system SHALL resolve every trainable 2Wiki train and retrieve stage device to `cuda`.

#### Scenario: Dense-FT 2Wiki train and retrieve use CUDA
- **WHEN** a 2Wiki smoke manifest is initialized with method `dense_ft`
- **THEN** the generated Dense-FT train stage config has trainer device `cuda`
- **THEN** the generated Dense-FT retrieve stage config has retrieval device `cuda`

#### Scenario: R-GCN 2Wiki train and retrieve use CUDA
- **WHEN** a 2Wiki smoke manifest is initialized with method `dense_rgcn_graph_retriever`
- **THEN** the generated R-GCN train stage config has trainer device `cuda`
- **THEN** the generated R-GCN retrieve stage config has retrieval device `cuda`

#### Scenario: No 2Wiki CPU fallback is introduced
- **WHEN** 2Wiki trainable method configs or generated stage configs are inspected
- **THEN** no trainable 2Wiki smoke path resolves `dense_ft` or `dense_rgcn_graph_retriever` device to `cpu`

### Requirement: 2Wiki trainable supervision remains node-level
The system SHALL train Dense-FT and R-GCN on 2Wiki gold evidence nodes derived from `supporting_facts`.

#### Scenario: Train pairs use gold evidence sentence ids
- **WHEN** train pairs are built for dataset `twowiki`
- **THEN** positive train pairs are created from `EvidenceLabel.gold_evidence_item_ids`
- **THEN** the train-pair builder does not read `gold_dependency_edges` to create positive or negative samples

#### Scenario: Dense-FT training uses text requests and train pairs
- **WHEN** `train_method.py` loads a Dense-FT train stage config for dataset `twowiki`
- **THEN** it projects 2Wiki records to `TextRankingRequest`
- **THEN** it projects 2Wiki labels to `EvidenceLabel`
- **THEN** it builds `DenseFinetuneTrainPayload` without graph artifacts

#### Scenario: R-GCN training uses text requests, graphs, labels, and train pairs
- **WHEN** `train_method.py` loads an R-GCN train stage config for dataset `twowiki`
- **THEN** it projects 2Wiki records to `TextRankingRequest`
- **THEN** it projects 2Wiki labels to `EvidenceLabel`
- **THEN** it builds `RgcnTrainPayload` with train/dev graph artifacts and train pairs

### Requirement: 2Wiki dependency edges remain label-only
The system SHALL keep 2Wiki dependency edges out of retrieval-visible inputs and graph artifacts.

#### Scenario: 2Wiki graph build does not receive gold dependency edges
- **WHEN** 2Wiki ranking records are projected to `GraphBuildRequest`
- **THEN** `GraphBuildRequest.input_visible_edges` does not contain edges derived from `supporting_facts`, `evidences`, `evidences_id`, `answer`, or `gold_dependency_edges`

#### Scenario: R-GCN training graph excludes label-only edges
- **WHEN** R-GCN training loads 2Wiki graph artifacts
- **THEN** the graph artifacts contain only visible graph-construction edges
- **THEN** `gold_dependency_edges` are not inserted into train-time or test-time graph tensors

### Requirement: 2Wiki path metric support follows method semantics
The system SHALL report 2Wiki path metrics according to method graph awareness rather than forcing every method to produce numeric path values.

#### Scenario: Dense-FT path metrics remain unsupported
- **WHEN** `dense_ft` predictions are evaluated for dataset `twowiki`
- **THEN** `Path Recall@10` is `N/A`
- **THEN** `Edge Recall@10` is `N/A`

#### Scenario: R-GCN path metrics are numeric when dependency labels exist
- **WHEN** `dense_rgcn_graph_retriever` predictions are evaluated for dataset `twowiki`
- **WHEN** at least one evaluated label has non-empty `gold_dependency_edges`
- **THEN** `Path Recall@10` is a numeric value
- **THEN** `Edge Recall@10` is a numeric value

#### Scenario: R-GCN missed path is zero rather than unsupported
- **WHEN** `dense_rgcn_graph_retriever` is evaluated on a 2Wiki task with non-empty `gold_dependency_edges`
- **WHEN** the returned `retrieved_subgraph` does not cover the gold dependency path
- **THEN** the task contributes `0.0` to path recall instead of `N/A`

### Requirement: 2Wiki trainable smoke runs through experiment workflow
The system SHALL verify the 2Wiki trainable methods through the named experiment runner rather than only unit-level APIs.

#### Scenario: 2Wiki trainable smoke plan is concrete
- **WHEN** `scripts/experiment.py plan` is run for `2wiki_tiny` with methods `dense_ft` and `dense_rgcn_graph_retriever`
- **THEN** the plan shows concrete `prepare`, `graphs`, `pairs`, `train`, `retrieve`, `evaluate`, and `aggregate` commands as required by the selected method lifecycles

#### Scenario: 2Wiki trainable smoke produces result tables
- **WHEN** `scripts/experiment.py run` succeeds for `2wiki_tiny` with methods `dense_ft` and `dense_rgcn_graph_retriever`
- **THEN** the run writes prediction artifacts for both trainable methods
- **THEN** the run writes `main_results.csv`, `path_results.csv`, and `efficiency_results.csv`
