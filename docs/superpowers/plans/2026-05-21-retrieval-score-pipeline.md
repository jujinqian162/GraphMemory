# Retrieval Score Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor retrieval execution so existing baselines run through a stable `RetrievalMethod` boundary, with BM25/dense/graph-rerank methods implemented as composable score pipelines.

**Status:** Executed on 2026-05-21 under OpenSpec change `refactor-retrieval-score-pipeline`; verification evidence is recorded in the final task output and OpenSpec task checklist.

**Architecture:** Keep `run_retrieval` as the public service and keep all method names/output artifacts stable. Internally, construct a retrieval method once from the public method name; current score-based baselines use a `ScorePipelineMethod` made from baseline and graph score components.

**Tech Stack:** Python dataclasses/protocols, existing `rank_bm25`, NumPy dense encoder path, existing graph helpers, pytest, OpenSpec.

---

## File Structure

- Modify `tests/test_phase1_real_retrieval.py`: add behavior tests for method construction, graph-free flat execution, graph requirement failures, and graph-rerank equivalence.
- Modify `tests/test_type_contracts.py`: update structural source checks so the intended narrowing and dispatch shape are protected.
- Modify `graph_memory/types.py`: add narrow Protocol/dataclass types if they are shared outside `retrieval.py`.
- Modify `graph_memory/retrieval.py`: add `RetrievalMethod`, score context, score components, method registry/builders, and refactor `run_retrieval`.
- Modify `docs/30-design/architecture.md`: update retrieval/rerank responsibilities.
- Modify `docs/30-design/abstractions.md`: document top-level method boundary and score-pipeline implementation.
- Modify `docs/30-design/testing-strategy.md`: document tests for score-pipeline behavior.
- Modify `docs/40-operations/implementation-handoff.md`: update review flow and extension guidance.
- Modify `openspec/changes/refactor-retrieval-score-pipeline/tasks.md`: mark tasks complete as they are verified.

## Task 1: Write Failing Tests

**Files:**
- Modify: `tests/test_phase1_real_retrieval.py`
- Modify: `tests/test_type_contracts.py`

- [ ] **Step 1: Add score-pipeline behavioral tests**

Add tests that call existing public APIs and expected internal helpers:

```python
def test_flat_methods_do_not_require_graph_inputs():
    result = run_retrieval(
        method="bm25",
        task_inputs=retrieval_task_inputs(),
        graphs=None,
        top_k=2,
    )

    assert result[0]["retrieved_subgraph"]["edges"] == []


def test_graph_pipeline_requires_config_before_processing():
    with pytest.raises(ValueError, match="Graph rerank methods require graph_config"):
        run_retrieval(
            method="bm25_graph_rerank",
            task_inputs=retrieval_task_inputs(),
            graphs=retrieval_graphs(),
            top_k=2,
        )


def test_score_pipeline_graph_method_matches_graph_rerank_helper():
    task_input = retrieval_task_inputs()[0]
    initial = {"m0": 1.0, "m1": 0.2, "m2": 0.8}
    config = GraphRerankConfig(lambda_init=1.0, lambda_query=0.1, lambda_neighbor=0.2, lambda_bridge=0.1)
    method = build_retrieval_method(
        method="bm25_graph_rerank",
        graphs=retrieval_graphs(),
        graph_config=config,
        dense_encoder=FakeEncoder(),
    )

    ranked_nodes, retrieved_edges = method.rank_task_from_scores(task_input, initial, top_k=2)

    assert ranked_nodes == graph_rerank(initial, retrieval_graphs()[0], config)
    assert retrieved_edges == induced_retrieved_subgraph(
        retrieval_graphs()[0],
        [node.node_id for node in ranked_nodes[:2]],
    )["edges"]
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
if (Test-Path .venv\Scripts\python.exe) { .\.venv\Scripts\python.exe -m pytest tests/test_phase1_real_retrieval.py tests/test_type_contracts.py -q } else { python -m pytest tests/test_phase1_real_retrieval.py tests/test_type_contracts.py -q }
```

Expected: failure because `build_retrieval_method` and `rank_task_from_scores` do not exist yet.

## Task 2: Implement Score Pipeline

**Files:**
- Modify: `graph_memory/retrieval.py`
- Modify: `graph_memory/types.py` only if shared types are needed

- [ ] **Step 1: Add retrieval method boundary**

Implement a small internal boundary:

```python
class RetrievalMethod(Protocol):
    name: str

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        ...
```

Keep the method internal to `retrieval.py` unless type reuse is needed elsewhere.

- [ ] **Step 2: Add score context and components**

Implement score components for baseline and graph signals:

```python
@dataclass(frozen=True)
class ScoreContext:
    task_input: MemoryTaskInput
    graph: MemoryGraph | None
    normalized_initial: dict[str, float]
    candidate_nodes: set[str]
    graph_config: GraphRerankConfig | None


class NodeScoreComponent(Protocol):
    weight: float

    def scores(self, context: ScoreContext) -> dict[str, float]:
        ...
```

Use existing `normalize_scores`, `_expanded_candidate_nodes`, `_query_scores`, `_neighbor_scores`, and `_bridge_scores` helpers where practical.

- [ ] **Step 3: Add score-pipeline method**

Implement `ScorePipelineMethod` so flat methods use one baseline retriever component and graph methods combine normalized initial with graph components. It must return every memory node exactly once and sort by descending score then node id.

- [ ] **Step 4: Refactor builder and `run_retrieval`**

Add:

```python
def build_retrieval_method(...) -> RetrievalMethod:
    ...
```

Then make `run_retrieval` validate task inputs, construct the method once, loop over tasks, and call `method.rank_task(...)`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the same focused command from Task 1. Expected: all focused tests pass.

## Task 3: Update Docs

**Files:**
- Modify: `docs/30-design/architecture.md`
- Modify: `docs/30-design/abstractions.md`
- Modify: `docs/30-design/testing-strategy.md`
- Modify: `docs/40-operations/implementation-handoff.md`

- [ ] **Step 1: Update architecture docs**

Document that `retrieval.py` owns public method construction and ranked-result assembly, while score-pipeline methods are one implementation style for score-based baselines.

- [ ] **Step 2: Update abstraction docs**

Add sections:

```text
RetrievalMethod
ScorePipelineMethod
NodeScoreComponent
```

Explain that not all future baselines must be weighted sums.

- [ ] **Step 3: Update handoff/testing docs**

Mention where reviewers should inspect method recipes, graph requirements, and component equivalence tests.

## Task 4: Verify and Close OpenSpec Tasks

**Files:**
- Modify: `openspec/changes/refactor-retrieval-score-pipeline/tasks.md`

- [ ] **Step 1: Run OpenSpec validation**

Run:

```powershell
openspec validate refactor-retrieval-score-pipeline
```

Expected: validation passes.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
if (Test-Path .venv\Scripts\python.exe) { .\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp .pytest-tmp } else { python -m pytest tests -q -p no:cacheprovider --basetemp .pytest-tmp }
```

Expected: all tests pass.

- [ ] **Step 3: Mark OpenSpec tasks complete**

Update every completed checkbox in `openspec/changes/refactor-retrieval-score-pipeline/tasks.md` from `- [ ]` to `- [x]` after verification.

- [ ] **Step 4: Review diff**

Run:

```powershell
git diff -- graph_memory tests docs openspec
```

Expected: changes are scoped to retrieval abstraction, tests, docs, and OpenSpec artifacts.
