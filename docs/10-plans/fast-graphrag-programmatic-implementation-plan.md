# Programmatic FastGraphRAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `fast_graphrag` retrieval method that consumes a typed `FastGraphRAGRequest`, uses only programmatic NLP plus graph propagation, and returns the existing `RankedResult` shape with full ranking plus a top-k retrieved subgraph.

**Architecture:** Dataset-specific records still project into consumer-owned requests; FastGraphRAG receives a method-specific request assembled from `TextRankingRequest`, visible `MemoryGraph`, and a deterministic entity-relation index built from retrieval-visible text only. The retriever never sees labels, answers, gold dependency edges, prompts, LLM clients, or LLM-style future extension points.

**Tech Stack:** Python dataclasses, existing `TextCandidate` / `MemoryGraph` contracts, pure-Python NLP normalization and entity linking, existing sentence-transformers encoder for dense entity/query scoring, weighted Personalized PageRank implemented locally, pytest, ruff, basedpyright.

---

Date: 2026-06-22

Status: Draft implementation plan. This document records the intended implementation order and verification gates. It does not mean code has been changed.

## Non-Negotiable Boundary

This implementation always avoids LLMs.

That means:

- No OpenAI, local LLM, chat model, completion model, prompt template, or LLM provider.
- No "future LLM adapter" or optional interface kept around for later.
- No fields such as `domain`, `example_queries`, `entity_types` as prompt inputs.
- No generated answers. This method is evidence retrieval only.
- No use of `answer`, `supporting_facts`, `evidences`, `evidences_id`, `gold_dependency_edges`, or any label-derived field during request construction, indexing, retrieval, or graph propagation.

The FastGraphRAG part we keep is the retrieval algorithm shape:

```text
visible query + visible candidate text
  -> deterministic entity mentions
  -> deterministic entity/relation graph
  -> dense and lexical query-to-entity seeds
  -> Personalized PageRank over entity graph
  -> entity/relation/chunk-to-candidate aggregation
  -> full ranked candidate list
  -> top-k candidate graph
```

## Current Repo Fit

The current runtime already has the right high-level pieces:

- `TextRankingRequest` owns task id, query text, and candidate text.
- `GraphRankingRequest` demonstrates method-specific graph requests.
- `RetrievalExecutionTask` separates the evaluation-visible `text_request` from the method-specific `method_request`.
- `RetrievalMethodResult.trace.retrieved_edges` is already assembled into `RankedResult.retrieved_subgraph.edges`.
- Evaluation path metrics are method-capability gated through the registry, not guessed from each prediction.

FastGraphRAG should follow the same shape:

```text
dataset-specific record
  -> dataset-specific projector
  -> TextRankingRequest + MemoryGraph
  -> registry builder assembles FastGraphRAGRequest
  -> FastGraphRAGMethod.rank_task()
  -> RetrievalMethodResult
  -> RankedResult
```

Do not introduce a cross-dataset `View` layer or a broad `dict[str, object]` bag.

## Request Contract

### Required request dataclasses

Add these contracts in `graph_memory/retrieval/requests.py` because they are consumer request objects, not dataset records.

```python
@dataclass(frozen=True)
class FastGraphRAGEntity:
    entity_id: str
    name: str
    normalized_name: str
    entity_type: str
    description: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class FastGraphRAGRelation:
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    description: str
    candidate_ids: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class FastGraphRAGKnowledgeGraph:
    entities: tuple[FastGraphRAGEntity, ...]
    relations: tuple[FastGraphRAGRelation, ...]


@dataclass(frozen=True)
class FastGraphRAGRequest:
    task_id: TaskId
    query_text: str
    candidates: Sequence[TextCandidate]
    candidate_graph: MemoryGraph
    knowledge_graph: FastGraphRAGKnowledgeGraph
```

Extend the type alias:

```python
RankingMethodRequest: TypeAlias = (
    TextRankingRequest
    | GraphRankingRequest
    | TemporalMemoryRankingRequest
    | FastGraphRAGRequest
)
```

### Field semantics

| Field | Required | Owner | Reason |
|---|---:|---|---|
| `task_id` | Yes | existing request boundary | Joins predictions, labels, and graphs. |
| `query_text` | Yes | existing request boundary | Used by programmatic query entity linking and dense seed scoring. |
| `candidates` | Yes | existing request boundary | Defines the full ranked universe. Ranking must include every candidate once. |
| `candidate_graph` | Yes | graph artifact | Source of evaluation-visible `retrieved_subgraph.edges`. |
| `knowledge_graph.entities` | Yes | FastGraphRAG index builder | PPR nodes and entity-to-candidate evidence mapping. |
| `knowledge_graph.relations` | Yes | FastGraphRAG index builder | PPR edges and relation-to-candidate evidence mapping. |

### Fields that must not be added

Do not add these to `FastGraphRAGRequest`:

| Field | Reason |
|---|---|
| `top_k` | Execution setting, not request data. |
| `encoder_model` / `device` | Method settings and runtime provenance. |
| `domain` / `example_queries` / prompt fields | LLM extraction concepts; out of scope permanently. |
| `query_entities` | Computed inside retriever from `query_text` to keep request minimal. |
| `initial_scores` | GraphRAG computes entity seeds internally; candidate initial scores are a different method family. |
| `gold_dependency_edges` | Label-only supervision. |
| `answer` / `supporting_facts` / `evidences` | Label-only data. |

## Programmatic NLP Design

### Entity normalization

Create `graph_memory/retrieval/methods/fast_graphrag/nlp.py`.

Default entity normalization is deterministic and dependency-free:

```python
def normalize_entity_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())
```

Use this for:

- title canonicalization,
- alias lookup,
- query mention matching,
- relation id generation,
- duplicate entity merging.

### Entity catalog construction

Build a per-task entity catalog from candidate metadata and text.

Sources:

1. `candidate.metadata["title"]`
2. `candidate.metadata["source_ref"]`
3. visible title prefix inside `candidate.text`
4. capitalized phrase spans from candidate text
5. parenthetical aliases from titles, such as `Andrew Allen (singer)` -> `Andrew Allen`

The catalog must never inspect labels.

Recommended types:

```python
@dataclass(frozen=True)
class EntityMention:
    entity_id: str
    name: str
    normalized_name: str
    candidate_id: str
    mention_text: str
    source: Literal["title", "source_ref", "title_prefix", "capitalized_phrase", "alias"]
```

First-version extraction rules:

- A candidate title creates one `document_title` entity.
- Parenthetical title base creates an alias mention for the same entity.
- Capitalized spans of two or more tokens create candidate mentions only if they are not stopwords and not already equal to the full title.
- Query mentions are linked by normalized exact match first, then substring match against known title aliases.

### Relation construction

Relations are deterministic co-occurrence edges over visible candidates.

First-version rules:

1. If a candidate has a title entity and one or more mentioned entities, create directed relations from title entity to each mentioned entity.
2. If a candidate has multiple non-title mentions but no title entity, create undirected canonical pair relations between mentions.
3. If a query-linked entity and a candidate title entity co-occur through lexical overlap, create no extra relation; query information is only used at retrieval time.

Relation id:

```text
relation:{min(source_entity_id,target_entity_id)}:{max(source_entity_id,target_entity_id)}:{normalized_candidate_id_hash}
```

Description:

```text
<source name> -- co-occurs with -- <target name> in candidate <candidate_id>: <candidate text>
```

The relation's `candidate_ids` contains every candidate that supports that relation.

### Dense seed scoring

Use the existing sentence-transformers stack through the existing encoder abstraction.

Seed text:

```text
query seed text = query_text
entity seed text = entity.name + "\n" + entity.description
```

Score sources:

- exact linked query entity match: strong lexical seed,
- normalized substring match: medium lexical seed,
- dense cosine similarity: continuous seed for all entities.

Combine:

```python
seed_score = max(
    exact_match_score,
    substring_match_score,
    dense_similarity_score,
)
```

Settings own the constants, not the request.

### Personalized PageRank

Implement local weighted power iteration in `graph_memory/retrieval/methods/fast_graphrag/pagerank.py`.

No new graph dependency is required.

Algorithm:

```python
def personalized_pagerank(
    adjacency: Mapping[str, Mapping[str, float]],
    personalization: Mapping[str, float],
    *,
    damping: float,
    max_iterations: int,
    tolerance: float,
) -> dict[str, float]:
    ...
```

Semantics:

- If personalization is empty, use a uniform distribution over entities.
- Normalize outgoing weights per source entity.
- Treat dangling nodes as redistributing to the personalization vector.
- Stop when L1 delta is below tolerance.
- Return one score per entity id.

### Candidate scoring

Create `graph_memory/retrieval/methods/fast_graphrag/scoring.py`.

Score candidates through entity and relation support:

```python
candidate_score =
    lambda_entity * sum(entity_score[e] for e linked to candidate)
  + lambda_relation * sum(relation_score[r] for r linked to candidate)
  + lambda_dense_fallback * dense_candidate_score[candidate]
```

Defaults:

- `lambda_entity = 1.0`
- `lambda_relation = 1.0`
- `lambda_dense_fallback = 0.05`

The dense fallback prevents candidates with no extracted entity from becoming unrankable. It must use only `query_text` and candidate text.

Tie-breaks:

1. higher score,
2. lower `candidate.metadata["position"]` when present,
3. lexicographic `candidate.item_id`.

## File Responsibility Map

### New production files

| File | Responsibility |
|---|---|
| `graph_memory/retrieval/methods/fast_graphrag/__init__.py` | Export method and settings helpers. |
| `graph_memory/retrieval/methods/fast_graphrag/nlp.py` | Deterministic normalization, entity mention extraction, query linking. |
| `graph_memory/retrieval/methods/fast_graphrag/index.py` | Build `FastGraphRAGKnowledgeGraph` from `TextRankingRequest` and visible `MemoryGraph`. |
| `graph_memory/retrieval/methods/fast_graphrag/pagerank.py` | Dependency-free weighted Personalized PageRank. |
| `graph_memory/retrieval/methods/fast_graphrag/scoring.py` | Entity/relation/candidate score aggregation. |
| `graph_memory/retrieval/methods/fast_graphrag/method.py` | `FastGraphRAGMethod.rank_task()` adapter returning `RetrievalMethodResult`. |

### Modified production files

| File | Responsibility |
|---|---|
| `graph_memory/retrieval/requests.py` | Add FastGraphRAG request and KG dataclasses; extend `RankingMethodRequest`. |
| `graph_memory/registry/retrieval.py` | Add method id, settings, build payload, provenance fields if needed. |
| `graph_memory/registry/retrieval_builders.py` | Build FastGraphRAG method, graph index, KG requests, and execution tasks. |
| `graph_memory/registry/methods.py` | Register `fast_graphrag` as graph-aware and path-metric-capable. |
| `graph_memory/stages/retrieve.py` | Route retrieve stage build payload for `FastGraphRAGRetrievalSettings`. |
| `graph_memory/validation/ranking.py` | Accept `fast_graphrag` method id in ranked results. |
| `scripts/workflow/stage_configs.py` | Add method config construction for workflow-generated retrieve stages. |
| `scripts/workflow/workflows.py` | Map FastGraphRAG to graph-backed retrieve workflow without tuning. |

### Tests

| File | Responsibility |
|---|---|
| `tests/test_fast_graphrag_requests.py` | Request dataclass shape and type alias integration. |
| `tests/test_fast_graphrag_nlp.py` | Normalization, title aliasing, mention extraction, query linking. |
| `tests/test_fast_graphrag_index.py` | KG construction from visible candidates and graph only. |
| `tests/test_fast_graphrag_pagerank.py` | PPR convergence, dangling nodes, personalization behavior. |
| `tests/test_fast_graphrag_scoring.py` | Entity/relation/candidate score aggregation and tie-breaks. |
| `tests/test_fast_graphrag_method.py` | End-to-end method ranking and retrieved subgraph edges. |
| `tests/test_fast_graphrag_registry.py` | Registry builder, method id, path metric support. |
| `tests/test_fast_graphrag_no_llm_boundary.py` | Static scan proving no LLM/prompt/provider concepts in FastGraphRAG code. |
| `tests/test_config_run_retrieval.py` | CLI/stage config can run `fast_graphrag` with graph inputs. |

## Implementation Tasks

### Task 1: Define FastGraphRAG request contracts

**Files:**

- Modify: `graph_memory/retrieval/requests.py`
- Test: `tests/test_fast_graphrag_requests.py`

- [x] **Step 1: Write request contract tests**

Add:

```python
from graph_memory.retrieval.requests import (
    FastGraphRAGEntity,
    FastGraphRAGKnowledgeGraph,
    FastGraphRAGRelation,
    FastGraphRAGRequest,
    RankingMethodRequest,
    TextCandidate,
)


def test_fast_graphrag_request_carries_visible_candidate_graph_and_kg() -> None:
    candidate = TextCandidate(item_id="m0", text="Paris is in France.", metadata={"title": "Paris"})
    entity = FastGraphRAGEntity(
        entity_id="e:paris",
        name="Paris",
        normalized_name="paris",
        entity_type="document_title",
        description="Paris",
        candidate_ids=("m0",),
    )
    relation = FastGraphRAGRelation(
        relation_id="r:e:paris:e:france:m0",
        source_entity_id="e:paris",
        target_entity_id="e:france",
        description="Paris co-occurs with France in m0.",
        candidate_ids=("m0",),
        weight=1.0,
    )
    request = FastGraphRAGRequest(
        task_id="task-1",
        query_text="Where is Paris?",
        candidates=(candidate,),
        candidate_graph={
            "task_id": "task-1",
            "nodes": [{"id": "q", "node_type": "question", "text": "Where is Paris?"}],
            "edges": [],
        },
        knowledge_graph=FastGraphRAGKnowledgeGraph(entities=(entity,), relations=(relation,)),
    )

    typed_request: RankingMethodRequest = request
    assert typed_request.task_id == "task-1"
    assert request.knowledge_graph.entities[0].candidate_ids == ("m0",)
```

- [x] **Step 2: Run the failing test**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_requests.py -q
```

Expected: fail because FastGraphRAG request classes do not exist.

- [x] **Step 3: Add request dataclasses**

Add the dataclasses exactly as defined in "Request Contract" and update `__all__`.

- [x] **Step 4: Run request tests**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_requests.py -q
```

Expected: pass.

### Task 2: Implement deterministic NLP extraction

**Files:**

- Create: `graph_memory/retrieval/methods/fast_graphrag/__init__.py`
- Create: `graph_memory/retrieval/methods/fast_graphrag/nlp.py`
- Test: `tests/test_fast_graphrag_nlp.py`

- [x] **Step 1: Write normalization and linking tests**

Cover:

```python
def test_normalize_entity_text_collapses_case_punctuation_and_parentheses() -> None:
    assert normalize_entity_text("Andrew Allen (Singer)") == "andrew allen singer"


def test_extract_candidate_mentions_uses_title_alias_and_visible_text_only() -> None:
    candidate = TextCandidate(
        item_id="m0",
        text="Andrew Allen is a Canadian singer from Vernon.",
        metadata={"title": "Andrew Allen (singer)", "position": 0},
    )
    mentions = extract_candidate_mentions((candidate,))
    by_name = {mention.name for mention in mentions}
    assert "Andrew Allen (singer)" in by_name
    assert "Andrew Allen" in by_name
    assert "Vernon" in by_name


def test_link_query_entities_matches_known_title_aliases() -> None:
    candidate = TextCandidate(
        item_id="m0",
        text="Changed It is a song by Nicki Minaj.",
        metadata={"title": "Changed It"},
    )
    catalog = build_entity_catalog((candidate,))
    linked = link_query_entities("Who performed Changed It?", catalog)
    assert [entity.name for entity in linked] == ["Changed It"]
```

- [x] **Step 2: Run NLP tests to verify failure**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_nlp.py -q
```

Expected: fail because `fast_graphrag.nlp` does not exist.

- [x] **Step 3: Implement dependency-free NLP helpers**

Implement:

```python
def normalize_entity_text(text: str) -> str: ...
def title_aliases(title: str) -> tuple[str, ...]: ...
def extract_candidate_mentions(candidates: Sequence[TextCandidate]) -> tuple[EntityMention, ...]: ...
def build_entity_catalog(candidates: Sequence[TextCandidate]) -> EntityCatalog: ...
def link_query_entities(query_text: str, catalog: EntityCatalog) -> tuple[CatalogEntity, ...]: ...
```

Use only stdlib modules: `re`, `unicodedata`, `dataclasses`, `hashlib`, and typing collections.

- [x] **Step 4: Run NLP tests**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_nlp.py -q
```

Expected: pass.

### Task 3: Build the visible entity-relation knowledge graph

**Files:**

- Create: `graph_memory/retrieval/methods/fast_graphrag/index.py`
- Test: `tests/test_fast_graphrag_index.py`

- [x] **Step 1: Write KG builder tests**

Cover:

```python
def test_build_fast_graphrag_kg_uses_candidate_titles_mentions_and_visible_relations() -> None:
    request = TextRankingRequest(
        task_id="task-1",
        query_text="Who performed Changed It?",
        candidates=(
            TextCandidate(
                item_id="m0",
                text="Changed It is a song by Nicki Minaj.",
                metadata={"title": "Changed It", "position": 0},
            ),
            TextCandidate(
                item_id="m1",
                text="Nicki Minaj was born in Trinidad and Tobago.",
                metadata={"title": "Nicki Minaj", "position": 1},
            ),
        ),
    )
    graph = {
        "task_id": "task-1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": request.query_text},
            {"id": "m0", "node_type": "graph_item", "node_kind": "document_sentence", "text": request.candidates[0].text},
            {"id": "m1", "node_type": "graph_item", "node_kind": "document_sentence", "text": request.candidates[1].text},
        ],
        "edges": [{"source": "m0", "target": "m1", "edge_type": "entity_overlap", "weight": 1.0, "directed": False}],
    }

    kg = build_fast_graphrag_knowledge_graph(request, graph)

    names = {entity.name for entity in kg.entities}
    assert "Changed It" in names
    assert "Nicki Minaj" in names
    assert any(relation.candidate_ids == ("m0",) for relation in kg.relations)
```

- [x] **Step 2: Run KG tests to verify failure**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_index.py -q
```

Expected: fail because KG builder does not exist.

- [x] **Step 3: Implement KG builder**

Implement:

```python
def build_fast_graphrag_knowledge_graph(
    request: TextRankingRequest,
    graph: MemoryGraph,
) -> FastGraphRAGKnowledgeGraph:
    ...
```

Rules:

- Validate `request.task_id == graph["task_id"]`.
- Validate every candidate id exists in the graph node set.
- Use `build_entity_catalog()` from `nlp.py`.
- Create one `FastGraphRAGEntity` per canonical entity.
- Create deterministic co-occurrence `FastGraphRAGRelation` records.
- Sort entities and relations by id before returning.

- [x] **Step 4: Run KG tests**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_index.py -q
```

Expected: pass.

### Task 4: Implement local Personalized PageRank

**Files:**

- Create: `graph_memory/retrieval/methods/fast_graphrag/pagerank.py`
- Test: `tests/test_fast_graphrag_pagerank.py`

- [x] **Step 1: Write PPR tests**

Cover:

```python
def test_personalized_pagerank_prefers_seed_connected_nodes() -> None:
    adjacency = {
        "a": {"b": 1.0},
        "b": {"a": 1.0, "c": 1.0},
        "c": {"b": 1.0},
    }
    scores = personalized_pagerank(
        adjacency,
        {"a": 1.0},
        damping=0.85,
        max_iterations=100,
        tolerance=1e-8,
    )
    assert scores["a"] > scores["c"]
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_personalized_pagerank_handles_dangling_nodes() -> None:
    scores = personalized_pagerank(
        {"a": {}, "b": {"a": 1.0}},
        {"b": 1.0},
        damping=0.85,
        max_iterations=100,
        tolerance=1e-8,
    )
    assert set(scores) == {"a", "b"}
    assert abs(sum(scores.values()) - 1.0) < 1e-6
```

- [x] **Step 2: Run PPR tests to verify failure**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_pagerank.py -q
```

Expected: fail because PPR module does not exist.

- [x] **Step 3: Implement PPR**

Implement `personalized_pagerank()` with deterministic sorting of node ids during iteration.

- [x] **Step 4: Run PPR tests**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_pagerank.py -q
```

Expected: pass.

### Task 5: Implement FastGraphRAG scoring

**Files:**

- Create: `graph_memory/retrieval/methods/fast_graphrag/scoring.py`
- Test: `tests/test_fast_graphrag_scoring.py`

- [x] **Step 1: Write scoring tests**

Cover:

```python
def test_candidate_scores_aggregate_entity_and_relation_support() -> None:
    candidates = (
        TextCandidate(item_id="m0", text="Changed It mentions Nicki Minaj.", metadata={"position": 0}),
        TextCandidate(item_id="m1", text="Unrelated.", metadata={"position": 1}),
    )
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:changed-it", "Changed It", "changed it", "document_title", "Changed It", ("m0",)),
            FastGraphRAGEntity("e:nicki-minaj", "Nicki Minaj", "nicki minaj", "mention", "Nicki Minaj", ("m0",)),
        ),
        relations=(
            FastGraphRAGRelation("r:changed:nicki:m0", "e:changed-it", "e:nicki-minaj", "co-occurs", ("m0",), 1.0),
        ),
    )
    scores = score_candidates(
        candidates,
        kg,
        entity_scores={"e:changed-it": 0.6, "e:nicki-minaj": 0.4},
        dense_fallback_scores={"m0": 0.2, "m1": 0.1},
        config=FastGraphRAGScoringConfig(lambda_entity=1.0, lambda_relation=1.0, lambda_dense_fallback=0.05),
    )
    assert scores["m0"] > scores["m1"]
```

- [x] **Step 2: Run scoring tests to verify failure**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_scoring.py -q
```

Expected: fail because scoring module does not exist.

- [x] **Step 3: Implement scoring dataclass and functions**

Add:

```python
@dataclass(frozen=True)
class FastGraphRAGScoringConfig:
    lambda_entity: float = 1.0
    lambda_relation: float = 1.0
    lambda_dense_fallback: float = 0.05
```

Implement:

```python
def score_relations(
    kg: FastGraphRAGKnowledgeGraph,
    entity_scores: Mapping[str, float],
) -> dict[str, float]: ...


def score_candidates(
    candidates: Sequence[TextCandidate],
    kg: FastGraphRAGKnowledgeGraph,
    *,
    entity_scores: Mapping[str, float],
    dense_fallback_scores: Mapping[str, float],
    config: FastGraphRAGScoringConfig,
) -> dict[str, float]: ...
```

- [x] **Step 4: Run scoring tests**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_scoring.py -q
```

Expected: pass.

### Task 6: Implement FastGraphRAG method

**Files:**

- Create: `graph_memory/retrieval/methods/fast_graphrag/method.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/__init__.py`
- Test: `tests/test_fast_graphrag_method.py`

- [x] **Step 1: Write method behavior tests**

Cover:

```python
def test_fast_graphrag_method_returns_full_ranking_and_topk_candidate_edges() -> None:
    request = fast_graphrag_fixture_request()
    method = FastGraphRAGMethod(
        name="fast_graphrag",
        config=FastGraphRAGConfig(),
        dense_ranker=FakeDenseSeedRanker(
            entity_scores={"e:changed-it": 0.9, "e:nicki-minaj": 0.7},
            candidate_scores={"m0": 0.8, "m1": 0.6, "m2": 0.1},
        ),
    )

    result = method.rank_task(request, top_k=2)

    assert [node.node_id for node in result.ranked_nodes] == ["m0", "m1", "m2"]
    assert result.trace.retrieved_edges == [
        {"source": "m0", "target": "m1", "edge_type": "entity_overlap", "weight": 1.0, "directed": False}
    ]
```

Also cover wrong request type:

```python
def test_fast_graphrag_method_rejects_non_fast_graphrag_request() -> None:
    method = FastGraphRAGMethod(name="fast_graphrag", config=FastGraphRAGConfig(), dense_ranker=FakeDenseSeedRanker())
    with pytest.raises(TypeError, match="FastGraphRAGRequest"):
        method.rank_task(TextRankingRequest(task_id="x", query_text="q", candidates=()), top_k=1)
```

- [x] **Step 2: Run method tests to verify failure**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_method.py -q
```

Expected: fail because method module does not exist.

- [x] **Step 3: Implement method config and method**

Add:

```python
@dataclass(frozen=True)
class FastGraphRAGConfig:
    ppr_damping: float = 0.85
    ppr_max_iterations: int = 100
    ppr_tolerance: float = 1e-8
    lexical_exact_match_score: float = 1.0
    lexical_substring_match_score: float = 0.5
    scoring: FastGraphRAGScoringConfig = field(default_factory=FastGraphRAGScoringConfig)
```

Add:

```python
@dataclass(frozen=True)
class FastGraphRAGMethod:
    name: str
    config: FastGraphRAGConfig
    dense_ranker: FastGraphRAGDenseScorer

    def rank_task(self, request: RankingMethodRequest, *, top_k: int) -> RetrievalMethodResult:
        if not isinstance(request, FastGraphRAGRequest):
            raise TypeError(f"{self.name} requires FastGraphRAGRequest, got {type(request).__name__}.")
        ...
```

Ranking flow:

1. Build entity adjacency from request relations.
2. Link query text to request entities.
3. Compute lexical entity seed scores.
4. Compute dense entity seed scores.
5. Merge seed scores and run PPR.
6. Compute dense fallback scores for candidates.
7. Aggregate candidate scores.
8. Return all candidates sorted by score.
9. Extract retrieved edges from `candidate_graph` where both endpoints are in top-k node ids.

- [x] **Step 4: Run method tests**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_method.py -q
```

Expected: pass.

### Task 7: Register method settings and registry builder

**Files:**

- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Modify: `graph_memory/registry/methods.py`
- Modify: `graph_memory/validation/ranking.py`
- Test: `tests/test_fast_graphrag_registry.py`

- [x] **Step 1: Write registry tests**

Cover:

```python
def test_fast_graphrag_registry_builds_graph_backed_execution_tasks() -> None:
    settings = FastGraphRAGRetrievalSettings(
        top_k=2,
        encoder=DenseEncoderSettings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            query_prefix="query: ",
            passage_prefix="passage: ",
            batch_size=8,
        ),
    )
    built = build_retrieval_registry().build(
        settings,
        FastGraphRAGBuildPayload(
            ranking_requests=[text_request_fixture()],
            graphs=[graph_fixture()],
            dense_encoder=FakeSentenceEncoder(),
        ),
    )
    assert built.method.name == "fast_graphrag"
    assert built.execution_tasks[0].method_request.task_id == built.execution_tasks[0].text_request.task_id


def test_fast_graphrag_supports_path_metrics() -> None:
    registry = build_method_registry()
    assert registry.supports_path_metrics(RetrievalMethodId.FAST_GRAPHRAG)
```

- [x] **Step 2: Run registry tests to verify failure**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_registry.py -q
```

Expected: fail because registry entries do not exist.

- [x] **Step 3: Add registry settings**

In `RetrievalMethodId` add:

```python
FAST_GRAPHRAG = "fast_graphrag"
```

Add:

```python
@dataclass(frozen=True)
class FastGraphRAGRetrievalSettings:
    top_k: int
    encoder: DenseEncoderSettings
    ppr_damping: float = 0.85
    ppr_max_iterations: int = 100
    ppr_tolerance: float = 1e-8
    lambda_entity: float = 1.0
    lambda_relation: float = 1.0
    lambda_dense_fallback: float = 0.05
    method: Literal[RetrievalMethodId.FAST_GRAPHRAG] = RetrievalMethodId.FAST_GRAPHRAG
```

Add:

```python
@dataclass(frozen=True)
class FastGraphRAGBuildPayload:
    ranking_requests: list[TextRankingRequest]
    graphs: list[MemoryGraph]
    dense_encoder: SentenceEncoder | None = None
```

Extend `RetrievalJobSettings` and `__all__`.

- [x] **Step 4: Add method registry definition**

Add a graph-aware lifecycle. Either add:

```python
FAST_GRAPHRAG = "fast_graphrag"
```

to `RetrievalLifecycle`, or include `FAST_GRAPHRAG` in the graph-aware support set with a clear condition. Prefer a distinct lifecycle because FastGraphRAG is not a seed reranker and is not checkpoint-trained.

Definition:

```python
MethodDefinition(
    identifier=RetrievalMethodId.FAST_GRAPHRAG,
    lifecycle=RetrievalLifecycle.FAST_GRAPHRAG,
    retrieval_settings_type=FastGraphRAGRetrievalSettings,
    dependencies=RetrievalDependencySpec(
        graphs=GraphInputSource.GRAPH_ARTIFACT,
        selected_config=SelectedConfigSource.NONE,
        model=ModelSource.NONE,
        encoder=EncoderSource.EXPERIMENT_CONFIG,
    ),
    method_config_type=None,
    train_artifact=None,
    seed_method=RetrievalMethodId.DENSE,
)
```

Update `supports_path_metrics()`:

```python
definition.lifecycle in {
    RetrievalLifecycle.GRAPH_RERANK,
    RetrievalLifecycle.RGCN_TRAINABLE,
    RetrievalLifecycle.FAST_GRAPHRAG,
}
```

- [x] **Step 5: Implement registry builder**

In `retrieval_builders.py`, add `_build_fast_graphrag()`.

Builder responsibilities:

- Validate graphs with `_validated_graph_index()`.
- Build one `FastGraphRAGRequest` per `TextRankingRequest`.
- Use `build_fast_graphrag_knowledge_graph()` for each task.
- Construct `FastGraphRAGMethod`.
- Return `BuiltRetrievalMethod` with encoder provenance.

- [x] **Step 6: Run registry tests**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_registry.py -q
```

Expected: pass.

### Task 8: Wire retrieve stage and workflow config

**Files:**

- Modify: `graph_memory/stages/retrieve.py`
- Modify: `scripts/workflow/stage_configs.py`
- Modify: `scripts/workflow/workflows.py`
- Modify: `tests/test_config_run_retrieval.py`

- [x] **Step 1: Add stage-level test**

Extend config run retrieval tests with a `fast_graphrag` retrieve config that includes graph input and encoder settings.

Expected config shape:

```json
{
  "schema_version": 1,
  "stage": "retrieve",
  "dataset": "twowiki",
  "job": {
    "method": "fast_graphrag",
    "top_k": 2,
    "encoder": {
      "model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "query_prefix": "query: ",
      "passage_prefix": "passage: ",
      "batch_size": 8
    }
  },
  "io": {
    "input": "inputs/test.input.json",
    "graphs": "graphs/test.graphs.json",
    "output": "retrieval/fast_graphrag.predictions.json",
    "summary": "retrieval/fast_graphrag.summary.json"
  }
}
```

- [x] **Step 2: Run stage tests to verify failure**

Run:

```powershell
uv run pytest tests/test_config_run_retrieval.py -q
```

Expected: fail because config and retrieve stage do not support `fast_graphrag`.

- [x] **Step 3: Add retrieve stage payload branch**

In `_build_payload()`, add:

```python
if isinstance(job, FastGraphRAGRetrievalSettings):
    return FastGraphRAGBuildPayload(
        ranking_requests=ranking_requests,
        graphs=graphs,
        dense_encoder=dense_encoder,
    )
```

- [x] **Step 4: Add workflow config construction**

In `scripts/workflow/stage_configs.py`, add `fast_graphrag` handling with graph input and encoder settings.

- [x] **Step 5: Run stage tests**

Run:

```powershell
uv run pytest tests/test_config_run_retrieval.py tests/test_fast_graphrag_registry.py -q
```

Expected: pass.

### Task 9: Add no-LLM boundary tests

**Files:**

- Create: `tests/test_fast_graphrag_no_llm_boundary.py`

- [x] **Step 1: Add static scan test**

Add:

```python
from pathlib import Path


FORBIDDEN_PATTERNS = (
    "openai",
    "llm",
    "prompt",
    "completion",
    "chat",
    "domain",
    "example_queries",
    "entity_types",
)


def test_fast_graphrag_code_has_no_llm_or_prompt_boundary() -> None:
    root = Path("graph_memory/retrieval/methods/fast_graphrag")
    sources = list(root.glob("*.py"))
    assert sources
    combined = "\n".join(path.read_text(encoding="utf-8").casefold() for path in sources)
    for pattern in FORBIDDEN_PATTERNS:
        assert pattern not in combined
```

If `entity_type` field names cause a false positive, keep the forbidden pattern as `"entity_types"` plural, not `"entity_type"`.

- [x] **Step 2: Run no-LLM test**

Run:

```powershell
uv run pytest tests/test_fast_graphrag_no_llm_boundary.py -q
```

Expected: pass.

### Task 10: End-to-end retrieval validation

**Files:**

- Modify or add fixtures only if existing tiny HotpotQA/2Wiki fixtures cannot cover a graph-backed request.
- Test: focused CLI smoke in existing config or registry tests.

- [x] **Step 1: Run focused unit suite**

Run:

```powershell
uv run pytest `
  tests/test_fast_graphrag_requests.py `
  tests/test_fast_graphrag_nlp.py `
  tests/test_fast_graphrag_index.py `
  tests/test_fast_graphrag_pagerank.py `
  tests/test_fast_graphrag_scoring.py `
  tests/test_fast_graphrag_method.py `
  tests/test_fast_graphrag_registry.py `
  tests/test_fast_graphrag_no_llm_boundary.py `
  -q
```

Expected: all pass.

- [x] **Step 2: Run retrieval integration tests**

Run:

```powershell
uv run pytest tests/test_config_run_retrieval.py tests/test_retrieval_registry_builders.py -q
```

Expected: all pass.

- [x] **Step 3: Run static checks**

Run:

```powershell
uv run ruff check
uv run basedpyright --level error
python -m compileall -q graph_memory scripts tests
git diff --check
```

Expected:

- ruff: no violations.
- basedpyright: 0 errors, 0 warnings, 0 notes.
- compileall: no output.
- git diff check: no whitespace errors.

- [x] **Step 4: Run real workflow smoke**

Use the smallest available processed fixture with graph artifacts. If 2Wiki is the target, run the already prepared tiny 2Wiki profile; if only HotpotQA tiny is available in the checkout, run HotpotQA first to prove method integration.

Command shape:

```powershell
uv run python scripts/experiment.py run 2wiki_tiny --config configs/experiments/2wiki_tiny.json --method fast_graphrag
```

Expected:

- predictions JSON contains `"method": "fast_graphrag"`.
- each prediction ranks every candidate exactly once.
- `retrieved_subgraph.nodes` contains at most top-k ids.
- `retrieved_subgraph.edges` endpoints are in `retrieved_subgraph.nodes`.
- evaluation can compute node metrics and path metrics according to method capability.

Actual smoke run:

```powershell
uv run python scripts/experiment.py run codex_fast_graphrag_smoke --config configs/experiments/2wiki_tiny.json --method fast_graphrag --force --no-cache
```

Artifacts:

- `runs/codex_fast_graphrag_smoke/predictions/test.fast_graphrag.ranked.json`
- `runs/codex_fast_graphrag_smoke/metrics/test.fast_graphrag.metrics.csv`
- `runs/codex_fast_graphrag_smoke/tables/main_results.csv`
- `runs/codex_fast_graphrag_smoke/tables/path_results.csv`

## Acceptance Criteria

The implementation is complete only when all of these are true:

1. `FastGraphRAGRequest` is the only method-specific runtime input consumed by `FastGraphRAGMethod.rank_task()`.
2. The request contains `task_id`, `query_text`, `candidates`, `candidate_graph`, and `knowledge_graph`.
3. The KG is built only from retrieval-visible query/candidate/graph data.
4. No LLM, prompt, answer generation, or future LLM integration concept appears in the FastGraphRAG package.
5. The method returns a full candidate ranking, not only top-k.
6. The method returns top-k candidate-level retrieved edges through `RetrievalTrace`.
7. `fast_graphrag` is registered as graph-aware and path-metric-capable.
8. Flat methods and Memory Stream behavior remain unchanged.
9. HotpotQA and 2Wiki dataset projectors do not import FastGraphRAG method internals.
10. The real retrieval workflow has been run at least once for `fast_graphrag`, not only unit tests.

## Implementation Notes

### Why `candidate_graph` and `knowledge_graph` both exist

The entity-relation graph is the internal FastGraphRAG reasoning surface. It has entity ids and relation ids.

The candidate graph is the evaluation-visible graph. It has candidate item ids such as `m0`, `m1`, and edges that path metrics can compare against retrieved evidence nodes.

Do not output internal entity edges as `retrieved_subgraph.edges`. The existing validation expects edge endpoints to be retrieved candidate node ids or `q`.

### Why query entities are not request fields

Query entity extraction is deterministic and uses only `query_text` plus the request KG. It is therefore a method-internal computation. Putting query entities into the request would widen the converter surface without improving dataset isolation.

### Why FastGraphRAG is not graph rerank

Graph rerank starts from candidate initial scores and propagates over the candidate graph.

FastGraphRAG starts from query-to-entity seed scores, propagates over an entity-relation graph, then aggregates back to candidates. It should therefore have its own request and lifecycle instead of overloading `GraphRankingRequest.initial_scores`.

### Why dense fallback is allowed

Dense fallback uses the existing non-LLM sentence-transformers encoder. It is a deterministic retrieval signal under fixed model/config inputs and does not create answer text or label leakage.

### Leakage checks

The no-leakage tests should scan both artifacts and source:

- source should not import label records inside `graph_memory/retrieval/methods/fast_graphrag`;
- KG builder should accept `TextRankingRequest` and `MemoryGraph`, not raw dataset records;
- test fixtures should prove a label-only phrase placed only in `gold_answer` cannot appear in KG entities or relations.

## Suggested Commit Boundaries

Use small commits if implementing manually:

1. `feat: add fast-graphrag request contracts`
2. `feat: add deterministic fast-graphrag nlp`
3. `feat: build fast-graphrag knowledge graph`
4. `feat: add fast-graphrag pagerank and scoring`
5. `feat: implement fast-graphrag retrieval method`
6. `feat: register fast-graphrag retrieval`
7. `test: enforce fast-graphrag no-llm boundary`

## Verification Checklist

- [x] `uv run pytest tests/test_fast_graphrag_requests.py -q`
- [x] `uv run pytest tests/test_fast_graphrag_nlp.py -q`
- [x] `uv run pytest tests/test_fast_graphrag_index.py -q`
- [x] `uv run pytest tests/test_fast_graphrag_pagerank.py -q`
- [x] `uv run pytest tests/test_fast_graphrag_scoring.py -q`
- [x] `uv run pytest tests/test_fast_graphrag_method.py -q`
- [x] `uv run pytest tests/test_fast_graphrag_registry.py -q`
- [x] `uv run pytest tests/test_fast_graphrag_no_llm_boundary.py -q`
- [x] `uv run pytest tests/test_config_run_retrieval.py tests/test_retrieval_registry_builders.py -q`
- [x] `uv run ruff check`
- [x] `uv run basedpyright --level error`
- [x] `python -m compileall -q graph_memory scripts tests`
- [x] `git diff --check`
- [x] Real workflow smoke with `--method fast_graphrag`
