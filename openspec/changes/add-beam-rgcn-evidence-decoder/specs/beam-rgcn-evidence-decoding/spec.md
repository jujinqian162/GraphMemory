## ADDED Requirements

### Requirement: R-GCN decoding is conditioned on selected evidence
The existing R-GCN methods SHALL encode each task graph once and SHALL score subsequent evidence actions from the encoded question and node states, the selected evidence set, the last selected node, base node scores, and model-visible graph-frontier features.

#### Scenario: First-hop scoring
- **WHEN** decoding starts from an empty selected set
- **THEN** the decoder SHALL score every memory-node candidate with the first-hop head and SHALL exclude the question node from selectable evidence

#### Scenario: Subsequent-hop scoring
- **WHEN** a hypothesis already contains selected evidence
- **THEN** the subsequent-hop score SHALL change as a function of that hypothesis's selected-set context, last selected node, and graph frontier

#### Scenario: Graph encoding is reused
- **WHEN** one task expands multiple hypotheses across multiple decoding steps
- **THEN** its frozen text embeddings and base R-GCN node states SHALL be computed once and reused by the decoder

### Requirement: Dynamic oracle respects evidence dependencies
Beam training SHALL derive valid next actions from dataset-neutral gold evidence ids and dependency edges without requiring a single fixed gold sequence.

#### Scenario: Dependency-ready gold actions
- **WHEN** an unselected gold node has every gold predecessor in the selected set
- **THEN** that node SHALL be a valid next action

#### Scenario: Future gold is masked
- **WHEN** an unselected gold node still has an unselected gold predecessor
- **THEN** that node SHALL be excluded from the current action loss rather than labeled as a negative

#### Scenario: Branching dependency has multiple valid actions
- **WHEN** multiple unselected gold nodes have all predecessors satisfied
- **THEN** the action loss SHALL accept all of them as alternative valid next actions

#### Scenario: Missing dependency labels fall back to an unordered set
- **WHEN** an evidence label has gold evidence ids and no dependency edges
- **THEN** every unselected gold node SHALL be a valid next action

#### Scenario: Stop becomes valid only after completion
- **WHEN** every gold evidence node is present in the selected set
- **THEN** `STOP` SHALL be positive
- **AND** while any gold evidence node is missing, `STOP` SHALL be negative

### Requirement: Training uses beam-produced states
R-GCN training SHALL accumulate next-action, stop, and auxiliary node-ranking losses over beam states and SHALL expose the decoder to model-produced partial hypotheses.

#### Scenario: Model error remains trainable
- **WHEN** a retained hypothesis contains a distractor and still has capacity to complete the gold set
- **THEN** the dynamic oracle SHALL continue to supervise valid recovery actions from that hypothesis

#### Scenario: Oracle-reachable hypothesis is retained
- **WHEN** model-preferred hypotheses would remove every gold-reachable path during training
- **THEN** training SHALL retain at least one oracle-reachable hypothesis while completion remains possible

#### Scenario: Train and inference beam sizes match
- **WHEN** a configured R-GCN run trains and retrieves with beam decoding
- **THEN** the effective training and inference beam sizes SHALL be equal and recorded in artifacts

#### Scenario: Auxiliary node ranking remains active
- **WHEN** beam-aware loss is computed
- **THEN** the existing sampled node-ranking objective SHALL contribute according to its configured auxiliary weight

### Requirement: Beam inference is bounded and deterministic
R-GCN inference SHALL decode a bounded set of unique evidence nodes with deterministic tie-breaking, set-level hypothesis deduplication, and a learned stop action.

#### Scenario: Equivalent selected sets are deduplicated
- **WHEN** two hypotheses contain the same selected node set in different orders
- **THEN** only the higher-scoring sequence SHALL occupy a beam slot

#### Scenario: Decode at most five evidence nodes
- **WHEN** no hypothesis stops early
- **THEN** decoding SHALL terminate after five unique selected evidence nodes

#### Scenario: All beams stop early
- **WHEN** every retained hypothesis selects `STOP`
- **THEN** decoding SHALL terminate before the maximum step count

#### Scenario: Default beam is two
- **WHEN** an R-GCN config uses repository defaults
- **THEN** training and inference SHALL use beam size `2` and maximum step count `5`

### Requirement: Beam decoding preserves the complete ranking contract
The decoder SHALL return every candidate exactly once through the existing ranked-node contract and SHALL preserve existing retrieved-subgraph semantics.

#### Scenario: Selected evidence forms the ranking prefix
- **WHEN** the winning hypothesis contains selected evidence nodes
- **THEN** those nodes SHALL form the ranking prefix in decoder order

#### Scenario: Early stop fills remaining ranks
- **WHEN** the winning hypothesis stops before selecting five nodes
- **THEN** unselected nodes with the highest base R-GCN scores SHALL fill the remaining top-five positions

#### Scenario: Remaining candidates use base ranking
- **WHEN** the selected prefix and any top-five fill are complete
- **THEN** every other candidate SHALL appear once in descending base-score order with deterministic tie-breaking

#### Scenario: Retrieved subgraph uses final top-k
- **WHEN** a beam-decoded ranking is converted to a retrieval result
- **THEN** the retrieved subgraph SHALL be induced from the same model-visible request graph and final top-k node ids used by existing R-GCN inference

### Requirement: Beam behavior is observable
Training, dev evaluation, and retrieval artifacts SHALL expose beam diagnostics without changing evidence metric formulas.

#### Scenario: Training metrics include decoder signals
- **WHEN** a beam-aware training epoch completes
- **THEN** metrics SHALL record next-action loss, stop loss, auxiliary node loss, average retained hypotheses, oracle-reachable beam rate, premature-stop rate, average selected length, beam size, and maximum steps

#### Scenario: Prediction metadata includes winning beam
- **WHEN** R-GCN inference writes a ranked prediction
- **THEN** metadata SHALL include the selected sequence, selected count, stopped flag, stop score, winning beam score, beam size, and maximum steps

#### Scenario: Existing metrics remain authoritative
- **WHEN** beam-decoded predictions are evaluated
- **THEN** existing Full Support, recall, evidence F1, MRR, connectivity, path, edge, and latency formulas SHALL remain unchanged

### Requirement: Label-only supervision never enters inference
Gold evidence ids, gold dependency edges, decomposition text, answers, and supporting flags SHALL be used only for training or evaluation and MUST NOT alter graph construction or retrieval-time decoding.

#### Scenario: Retrieval has no labels
- **WHEN** checkpoint-backed R-GCN inference receives a graph ranking request without evidence labels
- **THEN** it SHALL decode from request-visible question, candidates, graph, initial scores, checkpoint state, and configured beam settings only
