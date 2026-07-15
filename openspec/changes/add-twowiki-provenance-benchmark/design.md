## Context

The repository now separates dataset-owned EvidenceGraph, method-owned GraphRAG entity search, and native ExecutionProvenanceGraph. The current TRAJECT-Bench adapter violates that boundary by projecting catalog-declared tool links into prospective execution dependencies even though the source contains no executed call/output graph. The current stateless provenance method also takes only three seeds, enumerates paths before scoring, traverses every edge in both directions, and cannot expose logical candidate-to-candidate edges to path metrics.

2Wiki provides question/evidence supervision rather than executions, but its labeled support chain can be converted once into a separately named synthetic benchmark. This is an explicit exception to the normal rule that standard evidence datasets are not projected into execution provenance: the converter creates a new raw dataset with a documented simulation contract, while the normal `twowiki` adapter remains unchanged.

The existing EvidenceGraph R-GCN stack is candidate-aligned and node-wise. It cannot faithfully represent Task, Agent, ToolCall, and ToolOutput connector nodes. The new trainable method therefore owns a provenance-specific tensorizer and logical-edge scorer while reusing the established dense encoding, pair sampling, trainer, stage, and result contracts where their input semantics match. It intentionally has no beam decoder or dynamic-oracle contract.

## Goals / Non-Goals

**Goals:**

- Generate deterministic labeled `twowiki_provenance` raw artifacts with complete gold paths, structurally matched hard-negative branches, manifests, and leakage statistics.
- Keep dataset-specific parsing and projection isolated under `graph_memory/datasets/twowiki_provenance`.
- Remove TRAJECT-Bench from the supported dataset surface because tool catalogs do not provide the graph semantics under study.
- Make stateless provenance retrieval a real directed typed beam search.
- Add a trainable provenance-native R-GCN that scores output candidates and logical dependency edges without beam decoding.
- Reuse existing workflow stage types and the request-first domain boundary.

**Non-Goals:**

- Changing standard `twowiki`, HotpotQA, or MuSiQue semantics or published result comparability.
- Constructing TRAJECT execution graphs from gold trajectories, live agents, LLM calls, or executed tool outputs.
- Reintroducing beam behavior into `dense_rgcn_graph_retriever` or `dense_ft_rgcn_graph_retriever`.
- Guaranteeing a preselected leaderboard ordering through dataset construction.
- Adding an answer-generation or answer-accuracy task in the first benchmark version.

## Decisions

### 1. Convert once into a distinct raw dataset

`scripts/data/convert_2wiki_to_execution_provenance.py` reads labeled 2Wiki source files and writes JSON raw splits plus `manifest.json` and `statistics.json`. Source train remains train; source dev is deterministically partitioned into dev/test because the official test labels are unavailable. The manifest records source paths and hashes, seed, split policy, converter schema version, filtering counts, and graph construction parameters.

Only records with an unambiguous ordered two-evidence chain are retained. The converter resolves support nodes by stable title/sentence identity and refuses missing, duplicated, self-loop, or disconnected gold chains instead of repairing them from answer text.

Alternative considered: project standard prepared 2Wiki records at runtime. Rejected because it would blur standard benchmark semantics, make topology dependent on runtime configuration, and weaken reproducibility and leakage auditing.

### 2. Use native typed execution structure with candidate-only outputs

Each task contains one Task node, one Agent node, and a ToolCall/ToolOutput pair per candidate evidence. Retrieval candidate IDs correspond only to ToolOutput nodes. Every call has a `returns` edge. Dependency branches are expressed as `ToolOutput --feeds(FieldBinding)--> ToolCall --returns--> ToolOutput`; Task/Agent containment and invocation edges provide context but are not ranked candidates.

The gold ordered evidence pair becomes one complete output-to-output dependency transition. Every source output builds a successor query from the task question plus that source title and text. A versioned graph-construction strategy ranks all non-self targets with BM25 by default, or with an explicitly configured dense/hybrid ranker, and retains a fixed number of highest-scoring successors. The gold transition is added only as a recall fallback when semantic top-k misses it; it receives the same relation, binding schema, semantic weight, and metadata fields as every other proposal edge. Negative branches therefore come from explainable semantic matches rather than shuffled topology. Candidate order and IDs remain deterministic but shuffled. No answer node, gold flag, final answer, gold-only metadata, or gold-only degree/path-length pattern enters ranking inputs.

Each ToolOutput exposes a deterministic hash for its `evidence` output field and each ToolCall declares its accepted `context` input parameter. A `feeds` binding reuses the source field hash rather than inventing an edge-specific hash. Search validates the hash and parameter against endpoint metadata, while R-GCN tensorization maps the binding field/parameter/kind schema into the message relation identity. The manifest and statistics record scorer strategy, successor count, semantic ranks, gold fallback count, and edge-source distribution.

The converter caps candidate outputs with a versioned parameter and guarantees that every emitted candidate belongs to at least one complete call/output branch. Validation checks complete gold inclusion, legal transitions, non-isolation, path recoverability, candidate identity, and absence of forbidden fields.

Alternative considered: collapse each evidence directly into one ToolCall node. Rejected because it cannot represent the data-flow boundary that the provenance retriever and R-GCN are intended to learn.

### 3. Remove TRAJECT rather than maintain an unrelated benchmark

TRAJECT parsing, projection, validation, preparation, configuration, tests, runbook, and local raw artifacts are removed. The source remains a tool-catalog/linear-trajectory benchmark and cannot validate branching execution-provenance retrieval; retaining flat-only support would add maintenance surface without contributing to this experiment.

Alternative considered: retain the current projection as a `ToolSchemaGraph`. Rejected for this change because no active retriever owns that graph contract and it would expand scope without validating execution reasoning.

### 4. Implement directed incremental beam search for the stateless method

The stateless retriever computes semantic scores once for eligible candidate nodes, selects a configurable `seed_top_s` no smaller than the requested result budget, and expands directed legal transitions. At every hop it scores partial hypotheses and retains `beam_width`; it does not enumerate a whole BFS tree before pruning and does not add unconditional reverse adjacency.

Partial scores combine seed/endpoint relevance, path semantic mean and bottleneck, legal transition completeness, endpoint-validated binding consistency, grounding, invalidation, and normalized length. Binding consistency requires the edge hash to match the source output field and the target call to declare the bound input parameter; mere binding presence receives no credit. Ranking merges semantic and path scores so a disconnected high-quality seed remains visible while coherent paths can improve connected candidates. The method returns exactly the requested ranking budget when enough candidates exist.

Native trace records preserve traversed typed nodes/edges. A second logical trace contracts `ToolOutput -> feeds -> ToolCall -> returns -> ToolOutput` into output dependency edges used by retrieved-subgraph path metrics.

### 5. Add a separate provenance-native R-GCN without beam decoding

`execution_provenance_rgcn_retriever` accepts ExecutionProvenanceRankingRequest and a provenance checkpoint. Its tensorizer embeds every graph node with the shared dense encoding service, adds learned node-type features, creates forward/reverse message relations from the provenance relation plus binding-schema vocabulary, and maintains a candidate mask over ToolOutput nodes. A Task node is the query anchor; only candidate outputs receive ranking logits.

The encoder uses relation-specific graph convolution. A separate edge head scores each contracted output dependency transition independently. Training consumes the materialized positive/easy/BM25/dense pair artifact for candidate supervision and uses class-balanced logical-edge supervision for graph structure. There is no path-decoding loss, beam width, maximum-step field, or dynamic oracle. Evidence-R-GCN checkpoints and stale pre-removal provenance checkpoints are rejected by the provenance-specific schema.

Inference returns a full candidate ranking plus learned logical edges by selecting the highest-scoring legal successor per selected source. Each selected logical edge maps back to its exact `feeds` plus `returns` native edges for path metrics; it does not run multi-step beam search.

Alternative considered: project the graph into EvidenceGraph and reuse the existing R-GCN. Rejected because connector nodes would be lost, relation semantics would collapse into `sequential`, and the implementation would not actually test provenance message passing.

### 6. Reuse workflow stages, extend routing only

No stage type or DAG phase is added. Prepare uses the new adapter; pair and train stages dispatch to the new provenance model; retrieve constructs the native request; evaluate consumes the existing EvidenceEvaluationRequest and logical retrieved edges; aggregate remains unchanged. Registry metadata controls compatibility and lifecycle dependencies.

The execution-provenance family method matrix becomes BM25, Dense, GraphRAG, stateless Execution-Provenance Retriever, and Execution-Provenance R-GCN. Existing evidence profiles and their six-method matrix remain unchanged.

## Risks / Trade-offs

- [Synthetic topology leaks the label] → Use fixed semantic out-degree, identical gold/non-gold edge fields, audit gold fallback counts, shuffle IDs/order, and require topology-only/shuffled-edge controls before full experiments.
- [Negative branches are too easy] → Build edges from BM25/dense/hybrid successor matching, consume BM25/dense pair samples in training, and report scorer/rank/fallback distributions and structural statistics.
- [Two-hop graphs underuse long-horizon search] → Treat version one as a controlled one-transition benchmark and do not invent unsupported longer gold chains or an R-GCN maximum-step control.
- [New trainable stack duplicates evidence-R-GCN utilities] → Reuse dense encoding, generic R-GCN layers, batching patterns, trainer infrastructure, and result contracts while keeping tensorization/checkpoint ownership separate.
- [Workflow wiring grows hard-coded method sets] → Put compatibility in Registry metadata and limit planner edits to existing lifecycle dependency queries.
- [Old TRAJECT runs become non-reproducible on this branch] → Keep the rationale in this change record and rely on Git history for the retired adapter; do not keep runtime compatibility code.
- [Generated raw files are large] → Commit the converter/schema/tests, not full generated data; make manifests sufficient to reproduce artifacts.

## Migration Plan

1. Add failing contract tests and OpenSpec delta requirements.
2. Add the converter, generated-raw parser, validators, dataset selector, and prepare entry point.
3. Remove the TRAJECT adapter and all active runtime/configuration/documentation surfaces.
4. Replace stateless provenance traversal and trace behavior while preserving its public method ID.
5. Add provenance tensorization, pair-supervised candidate/edge training, non-beam inference, checkpointing, and Registry builder.
6. Wire the new method through existing pair/train/retrieve/evaluate stages and add a smoke experiment config.
7. Generate a small pilot dataset, verify deterministic manifests and leakage/graph invariants, then run focused and workflow tests.

Rollback is code-only: remove the new dataset/method registration and restore the prior stateless search implementation. Generated raw artifacts and provenance checkpoints are versioned and can be deleted independently; existing evidence data and checkpoints are unaffected.

## Open Questions

None blocking. Pilot statistics will determine the default candidate cap, semantic successor count, and whether BM25 or hybrid construction becomes the full-run default; all remain explicit configuration rather than hidden converter behavior.
