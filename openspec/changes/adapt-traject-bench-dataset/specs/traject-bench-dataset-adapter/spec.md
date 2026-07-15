## ADDED Requirements

### Requirement: Official TRAJECT-Bench layouts are parsed deterministically
The system SHALL discover the pinned public TRAJECT-Bench layout from a raw directory, SHALL map `train` to parallel/simple, `dev` to parallel/hard, and `test` to all published sequential query files, and SHALL order source records identically across supported operating systems.

#### Scenario: Parallel simple workflow partition
- **WHEN** the prepare stage receives dataset `traject_bench`, split `train`, and a valid official raw directory
- **THEN** it reads every `parallel/<Domain>/simple_ver.json` query in normalized domain/path/index order before applying the seeded selection policy

#### Scenario: Sequential workflow partition
- **WHEN** the prepare stage receives dataset `traject_bench` and split `test`
- **THEN** it includes each published `sequential/<Domain>/traj_query.json` file and the published `sequential/Travel/simple_ver.json` file exactly once

### Requirement: Candidate tools are catalog-owned and leakage-safe
The system SHALL construct each query's candidate pool only from its public domain tool catalog, SHALL canonicalize duplicate exact tool names deterministically, and MUST NOT use gold call descriptions, parameter values, execution outputs, or final answers to construct candidates.

#### Scenario: Duplicate public catalog entries
- **WHEN** a domain catalog contains multiple entries with the same exact tool name
- **THEN** the adapter emits one stable tool candidate, merges catalog-declared connections deterministically, and reports the duplicate count

#### Scenario: Gold tool missing from catalog
- **WHEN** a query references a tool name absent from its canonical public domain catalog
- **THEN** the adapter excludes that query before seeded sampling and reports a missing-gold invalid reason instead of adding the gold definition to candidates

### Requirement: Prepared artifacts separate ranking inputs from trajectory labels
The system SHALL emit ranking records containing query text and catalog-derived candidates, and separate label records containing the gold distinct tool IDs, repeated call sequence IDs, dependency edges, final answer, and label metadata.

#### Scenario: Repeated tool calls
- **WHEN** a sequential query calls the same tool API more than once
- **THEN** `gold_tool_ids` contains that API once, `gold_tool_sequence_ids` preserves every call occurrence, and dependency edges contain no self-loop

#### Scenario: No labels in ranking input
- **WHEN** a prepared TRAJECT-Bench ranking record is validated
- **THEN** gold tool IDs, gold sequences, final answers, execution outputs, and required parameter values are absent from the ranking artifact

### Requirement: TRAJECT-Bench projects to repository-owned retrieval requests
The system SHALL project TRAJECT-Bench ranking records to existing text ranking, EvidenceGraph build, and execution-provenance ranking requests and SHALL project label records to the existing evidence evaluation request.

#### Scenario: Text retrieval projection
- **WHEN** BM25, Dense, or GraphRAG consumes a TRAJECT-Bench task
- **THEN** it receives a `TextRankingRequest` whose candidate IDs and text come from the canonical public tool catalog

#### Scenario: Graph projection without gold leakage
- **WHEN** an EvidenceGraph is built for a TRAJECT-Bench task
- **THEN** candidate tools become `tool_api` nodes, resolved catalog connections may become directed input-visible edges, and query-specific gold trajectory edges are not present in the graph-build request

#### Scenario: Execution-provenance retrieval projection
- **WHEN** the execution-provenance retriever consumes a TRAJECT-Bench task
- **THEN** every catalog candidate becomes a prospective `tool_call` node, resolved catalog connections become input-visible `depends_on` edges, and gold calls, arguments, outputs, answers, and trajectory order remain absent from the request

#### Scenario: Execution-provenance workflow planning
- **WHEN** `execution_provenance_retriever` is selected for TRAJECT-Bench
- **THEN** the planner emits a typed retrieve stage with no train, pair, or EvidenceGraph dependency and the standard runner completes retrieval and evaluation instead of raising an unsupported-config error

### Requirement: Baseline evaluation is scoped as offline tool retrieval
The system SHALL evaluate selected tool IDs with repository retrieval metrics and SHALL document that these values do not represent end-to-end tool execution, parameter correctness, LLM-judged trajectory satisfaction, or final-answer accuracy.

#### Scenario: Frozen baseline workflow
- **WHEN** a user runs the documented BM25, Dense, or execution-provenance TRAJECT-Bench baseline command
- **THEN** the normal experiment runner prepares, retrieves, evaluates, and aggregates the selected workflow partition without requiring an LLM API key or live tool endpoint

### Requirement: Official download and quick-run commands are reproducible
The repository SHALL document a pinned Hugging Face download command, an official GitHub alternative, and smoke/quick baseline commands whose paths match the TRAJECT-Bench dataset configuration.

#### Scenario: Server bootstrap
- **WHEN** a user follows the documented Hugging Face command from the repository root
- **THEN** the expected `data/traject_bench/raw/{parallel,sequential,tools}` layout is available to the experiment runner
