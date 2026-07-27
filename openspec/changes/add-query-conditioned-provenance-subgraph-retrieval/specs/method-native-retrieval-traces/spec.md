## ADDED Requirements

### Requirement: Query-conditioned subgraph traces use a closed typed record
The result contract SHALL represent the new non-trained EPGM trace with `trace_kind=execution_provenance_subgraph` and declared typed records for relation affinity, normalized transitions, PPR state, connected selection, selected native edges, emitted candidate edges, convergence, objective, active variant, scorer identity, and fallback state.

#### Scenario: Successful connected selection serializes audit evidence
- **WHEN** the retriever selects candidate evidence through connector nodes
- **THEN** serialization records selected candidates, connector ids, stored native edges, added and displaced candidate ids, new-edge and displacement costs, marginal gains, total objective, and emitted candidate edges with no arbitrary fields

### Requirement: Query-conditioned subgraph traces are structurally validated
Ranked-result validation SHALL reject unknown trace fields, non-finite values, duplicate records, unknown native/candidate references, non-normalized source transitions, disconnected selected native subgraphs, selected candidate counts above the retrieval budget, invalid stored-edge orientation, and fallback states inconsistent with emitted intervention.

#### Scenario: Reject non-normalized transition row
- **WHEN** positive transition probabilities for one source do not sum to one within the declared tolerance
- **THEN** ranked-result validation fails with a trace-specific contract error

#### Scenario: Reject connector as ranked evidence
- **WHEN** a trace identifies a connector id that was not a request candidate but the ranked result also introduces it as a candidate
- **THEN** ranked-result validation fails

#### Scenario: Reject inconsistent fallback
- **WHEN** exact Dense fallback is true but the trace contains a connected selection step
- **THEN** ranked-result validation fails

### Requirement: Trace graph context is explicit
The subgraph trace SHALL declare the native graph node ids needed to validate connector and edge references while the shared retrieved-subgraph surface SHALL remain candidate-only.

#### Scenario: Validate non-candidate connector reference
- **WHEN** a selected path contains a tool-call connector outside request candidates
- **THEN** validation accepts it only if the connector appears in the trace's declared native graph context and selected native edges
