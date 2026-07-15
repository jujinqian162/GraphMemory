## MODIFIED Requirements

### Requirement: Training uses beam-produced states
R-GCN training SHALL accumulate a jointly normalized next-action loss over eligible candidates and `STOP`, a stop-completion loss derived from the same joint action probability, and an auxiliary node-ranking loss over beam states, and SHALL expose the decoder to model-produced partial hypotheses.

#### Scenario: Candidate and stop actions share calibration
- **WHEN** beam-aware loss is computed for a non-stopped hypothesis
- **THEN** eligible candidate logits and the `STOP` logit SHALL be normalized in one action distribution
- **AND** shifting all candidate logits relative to `STOP` SHALL change the supervised loss

#### Scenario: Incomplete evidence rejects stop in the joint action loss
- **WHEN** at least one gold evidence node remains unselected
- **THEN** dependency-ready gold nodes SHALL form the valid action set
- **AND** `STOP` SHALL remain an invalid action in the same normalization denominator

#### Scenario: Completed evidence selects stop in the joint action loss
- **WHEN** every gold evidence node is selected
- **THEN** `STOP` SHALL be the valid action
- **AND** remaining distractor candidates SHALL remain competing invalid actions

#### Scenario: Future gold remains masked
- **WHEN** an unselected gold node still has an unselected dependency predecessor
- **THEN** that future gold node SHALL be excluded from both the valid-action numerator and supervised action denominator

#### Scenario: Stop component uses joint probability
- **WHEN** the configured stop-loss component is computed
- **THEN** it SHALL use the `STOP` probability from the joint candidate-and-stop distribution rather than interpret the raw stop logit independently

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

#### Scenario: Training honors disabled deduplication
- **WHEN** `deduplicate_selected_sets` is false
- **THEN** training beam pruning SHALL retain distinct ordered hypotheses according to score and beam width without applying a state-key deduplication filter

### Requirement: Beam inference is bounded and deterministic
R-GCN inference SHALL decode a bounded set of unique evidence nodes with deterministic tie-breaking, optional hypothesis deduplication keyed by selected set and last selected node, and a learned stop action.

#### Scenario: Equivalent decoder states are deduplicated
- **WHEN** two hypotheses contain the same selected node set and have the same last selected node
- **THEN** only the higher-scoring sequence SHALL occupy a beam slot when deduplication is enabled

#### Scenario: Different last nodes remain distinct
- **WHEN** two hypotheses contain the same selected node set but have different last selected nodes
- **THEN** both hypotheses SHALL remain eligible for separate beam slots because their subsequent-hop decoder states differ

#### Scenario: Deduplication can be disabled
- **WHEN** `deduplicate_selected_sets` is false
- **THEN** inference SHALL skip state-key deduplication and SHALL prune only by deterministic score ordering and beam width

#### Scenario: Decode at most five evidence nodes
- **WHEN** no hypothesis stops early
- **THEN** decoding SHALL terminate after five unique selected evidence nodes

#### Scenario: All beams stop early
- **WHEN** every retained hypothesis selects `STOP`
- **THEN** decoding SHALL terminate before the maximum step count

#### Scenario: Default beam is two
- **WHEN** an R-GCN config uses repository defaults
- **THEN** training and inference SHALL use beam size `2` and maximum step count `5`

### Requirement: Beam behavior is observable
Training, dev evaluation, and retrieval artifacts SHALL expose beam diagnostics and the complete beam-aware development objective without changing evidence metric formulas.

#### Scenario: Training metrics include decoder signals
- **WHEN** a beam-aware training epoch completes
- **THEN** metrics SHALL record next-action loss, stop loss, auxiliary node loss, average retained hypotheses, oracle-reachable beam rate, premature-stop rate, average selected length, beam size, and maximum steps

#### Scenario: Development loss uses the complete beam objective
- **WHEN** R-GCN development evaluation completes
- **THEN** `dev_loss` SHALL equal the configured weighted total of development next-action, stop, and auxiliary-node losses
- **AND** it SHALL NOT represent only the base node-scorer BCE

#### Scenario: Development loss components are reported
- **WHEN** a training metric record contains `dev_loss`
- **THEN** the same record SHALL contain `dev_next_action_loss`, `dev_stop_loss`, and `dev_aux_node_loss`

#### Scenario: Prediction metadata includes winning beam
- **WHEN** R-GCN inference writes a ranked prediction
- **THEN** metadata SHALL include the selected sequence, selected count, stopped flag, stop score, winning beam score, beam size, and maximum steps

#### Scenario: Existing metrics remain authoritative
- **WHEN** beam-decoded predictions are evaluated
- **THEN** existing Full Support, recall, evidence F1, MRR, connectivity, path, edge, and latency formulas SHALL remain unchanged

## ADDED Requirements

### Requirement: Beam score normalization counts actions consistently
Beam training and inference SHALL normalize cumulative hypothesis scores by the number of actions whose log probabilities contribute to the raw score.

#### Scenario: Candidate expansion increments action count
- **WHEN** a hypothesis selects one additional evidence candidate
- **THEN** its action count SHALL increase by one

#### Scenario: Stop increments action count
- **WHEN** a hypothesis selects `STOP`
- **THEN** its action count SHALL increase by one even though its selected evidence set does not change

#### Scenario: Empty-prefix stop has one action
- **WHEN** the initial empty hypothesis immediately selects `STOP`
- **THEN** its normalized score SHALL use an action count of one

#### Scenario: Training and inference share normalization
- **WHEN** the same raw hypothesis is pruned during training and inference with the same beam settings
- **THEN** both paths SHALL compute the same normalized score and deterministic ordering
