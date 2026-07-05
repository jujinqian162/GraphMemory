## ADDED Requirements

### Requirement: Learned graph R-GCN uses Dense-FT seed encoder
The system SHALL train `learned_graph_rgcn_retriever` with the run-local `dense_ft` model directory as its R-GCN seed/text encoder.

#### Scenario: Learned graph train stage receives Dense-FT checkpoint
- **WHEN** an experiment manifest is initialized with selected method `learned_graph_rgcn_retriever`
- **THEN** the learned graph train stage config MUST set `io.seed_checkpoint` to the manifest artifact path for `learned/dense_ft/checkpoints/best_model`

#### Scenario: Learned graph method keeps public identity
- **WHEN** an experiment manifest is initialized with selected method `learned_graph_rgcn_retriever`
- **THEN** the selected retrieval methods MUST include `learned_graph_rgcn_retriever` and MUST NOT introduce a separate learned-graph Dense-FT method id

### Requirement: Dense-FT dependency is execution-only for learned graph
The system SHALL include `dense_ft` only as a training dependency when it is needed to seed `learned_graph_rgcn_retriever`.

#### Scenario: Learned-only plan trains Dense-FT dependency
- **WHEN** a stage plan is built for selected method `learned_graph_rgcn_retriever`
- **THEN** the plan MUST include `dense_ft` pair and train commands before the learned graph train command

#### Scenario: Learned-only plan does not evaluate dependency method
- **WHEN** a stage plan is built for selected method `learned_graph_rgcn_retriever`
- **THEN** the plan MUST NOT add `dense_ft` retrieve or evaluate commands unless `dense_ft` was explicitly selected

### Requirement: Proposal graph ownership remains method-local
The system SHALL keep `learned_graph_rgcn_retriever` proposal graph artifacts independent from the shared graph artifact chain.

#### Scenario: Learned graph still reads proposal graphs
- **WHEN** an experiment manifest is initialized with selected method `learned_graph_rgcn_retriever`
- **THEN** pair, train, retrieve, and evaluate stage configs for that method MUST read `graphs/<split>.learned_graph_rgcn_retriever.graphs.json` artifacts

#### Scenario: Dense-FT seed does not replace learned graph proposal stage
- **WHEN** a stage plan is built for selected method `learned_graph_rgcn_retriever`
- **THEN** the learned graph method MUST continue to use the `proposal_graphs` stage for its own graph inputs even if the Dense-FT dependency requires shared graph artifacts for its training pairs
