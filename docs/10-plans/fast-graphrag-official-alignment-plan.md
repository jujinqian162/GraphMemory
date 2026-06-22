# FastGraphRAG 官方对齐修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复当前 `fast_graphrag` 与 Microsoft GraphRAG FastGraphRAG 的关键差距，让它在保持本 repo no-LLM、evidence retrieval-only 边界的前提下，尽可能贴近官方的 NLP graph extraction、co-occurrence relationship、graph pruning 和 query-side entity mapping 语义。

**Architecture:** 保留 repo-owned `FastGraphRAGRequest`，不把官方 SDK API 形状倒灌进本仓库。`TextRankingRequest + MemoryGraph` 仍由 registry builder 投影成方法私有 request；FastGraphRAG 内部改为 `visible text unit -> noun phrase mentions -> entity/relation KG -> pruning -> query entity seeds -> graph propagation -> candidate ranking`。官方必须依赖 LLM 的 community report / answer generation 只做显式简化，不引入 prompt、LLM provider 或未来 LLM hook。

**Tech Stack:** Python dataclass、现有 `TextCandidate` / `MemoryGraph` / registry / workflow contracts、dependency-free `regex_english` noun phrase extraction、optional spaCy path、sentence-transformers dense scoring、pure-Python graph pruning、现有 Personalized PageRank、pytest、ruff、basedpyright。

---

日期：2026-06-22

状态：修复计划草案。本文档只规划实现顺序和验收门，不表示代码已经修改。

## 1. 官方基准

本计划以当前可查的 Microsoft GraphRAG 官方资料为基准：

- Microsoft GraphRAG release 页面显示最新 release 为 `v3.1.0`，发布时间为 2026-05-28。
- 官方 indexing methods 文档说明 `graphrag index --method fast` 是 FastGraphRAG 入口。
- 官方 FastGraphRAG 的 graph extraction 语义：
  - entity extraction：从 text unit 中抽取 noun phrases；默认 `NLTK + regular expressions`，并提供 spaCy `syntactic_parser` 和 `cfg` 两种替代 extractor。
  - relationship extraction：同一 text unit 中 entity pair co-occurrence 构成关系。
  - entity / relationship summarization：不需要单独摘要。
  - community report generation：收集包含 noun phrase 的 text unit 内容，再用 LLM 生成 summary report。
- 官方配置中 `extract_graph_nlp` 提供 `extractor_type`、`normalize_edge_weights`、`max_word_length`、`include_named_entities`、`exclude_nouns`、`exclude_entity_tags`、`exclude_pos_tags`、`noun_phrase_tags`、`noun_phrase_grammars`。
- 官方配置中 `prune_graph` 提供 `min_node_freq`、`max_node_freq_std`、`min_node_degree`、`max_node_degree_std`、`min_edge_weight_pct`、`remove_ego_nodes`、`lcc_only`。

参考链接：

- <https://microsoft.github.io/graphrag/index/methods/>
- <https://microsoft.github.io/graphrag/config/yaml/>
- <https://github.com/microsoft/graphrag/releases>

## 2. 本仓库必须保留的边界

这些边界不能为了“像官方”而破坏：

- `FastGraphRAGRequest` 是本 repo 的 method-owned request，不是官方 API mirror。
- 始终不用 LLM，也不添加“以后接 LLM”的 adapter、字段、prompt 或 provider。
- 不读取 label-only 字段：`answer`、`supporting_facts`、`evidences`、`evidences_id`、`gold_dependency_edges`。
- 检索输出仍是完整 `ranked_nodes` 加 top-k `retrieved_subgraph`，保持 evaluation contract。
- `candidate_graph` 仍是 path metrics 可见的 graph artifact；FastGraphRAG 内部 KG 是方法内部 graph，不直接替代 evaluation graph。

## 3. 当前实现差距和修复策略

| 当前实现 | 官方语义 | 修复策略 |
|---|---|---|
| 只从 title、source_ref、title prefix、capitalized phrase 抽 entity | 从 text unit 抽 noun phrases，默认 regex/NLTK，spaCy 可选 | 新增 `regex_english` noun phrase extractor，保留 title/source_ref 作为强信号，optional spaCy 只在已安装时启用 |
| `_STOPWORDS` 是小型停用词表，容易被误解为“高频词列表” | `exclude_nouns` 是名词过滤配置；`prune_graph` 才处理高频/低频 graph noise | 明确拆分 `internal_stopwords`、`exclude_nouns`、`prune_graph` 三类语义 |
| 同一 entity pair 在不同 candidate 中生成多条 relation，权重固定 1.0 | text unit co-occurrence 关系应能按 pair 聚合并归一化 | 改成一对 entity 一条 relation，`candidate_ids` 记录所有 text units，权重为 co-occurrence count 或归一化值 |
| PPR seed 主要靠 dense scoring all entities 和简单 lexical substring | 官方 query 侧先做 entity mapping，再取相关实体/关系/text units | 显式使用 `link_query_entities()` 作为强 seed，再混合 dense top-k entity seed |
| 没有 graph pruning | 官方有 node frequency / degree / edge percentile pruning | 新增 deterministic pruning；`remove_ego_nodes` 因当前 KG 无 ego node，第一阶段只允许 false |
| 没有 community report | 官方 FastGraphRAG 仍会用 text unit 内容 prompt LLM 生成 community report | 保持 no-LLM，第一阶段不实现；只保留可选 extractive community context 作为后续非必需任务 |

## 4. 简化原则

1. 官方 `NLTK + regex` 默认需要外部 tokenizer/tagger 资源，容易引入下载和环境不稳定。第一阶段默认实现 dependency-free `regex_english` 近似版；不要把 NLTK 加进 required dependencies。
2. spaCy 已经是本 repo optional `ner` extra。可以实现 optional spaCy extractor，但测试必须用 fake nlp object，不依赖下载模型。
3. 官方 community report 依赖 LLM summary，和用户确定的 no-LLM 边界冲突。第一阶段不实现 community report；如需类似信号，只允许 extractive text-unit context，不叫 summary/report。
4. 官方 chunking 建议小 text unit；本 repo 当前 candidate 通常已经是 sentence/paragraph evidence item。第一阶段把 `TextCandidate` 当作 text unit，不新增 chunking pipeline。
5. 当前 PPR 不属于 Microsoft FastGraphRAG indexing 文档的必需项，但它是本 repo 既有 retrieval 算法核心。保留 PPR，同时把 seed 和 graph construction 修正得更接近官方 KG。

## 5. 目标文件结构

### 新增文件

```text
graph_memory/retrieval/methods/fast_graphrag/config.py
graph_memory/retrieval/methods/fast_graphrag/noun_phrases.py
graph_memory/retrieval/methods/fast_graphrag/pruning.py
tests/test_fast_graphrag_official_alignment.py
tests/test_fast_graphrag_pruning.py
```

### 修改文件

```text
graph_memory/retrieval/methods/fast_graphrag/__init__.py
graph_memory/retrieval/methods/fast_graphrag/nlp.py
graph_memory/retrieval/methods/fast_graphrag/index.py
graph_memory/retrieval/methods/fast_graphrag/method.py
graph_memory/retrieval/methods/fast_graphrag/scoring.py
graph_memory/retrieval/requests.py
graph_memory/registry/retrieval.py
graph_memory/registry/retrieval_builders.py
scripts/workflow/stage_configs.py
tests/test_fast_graphrag_index.py
tests/test_fast_graphrag_method.py
tests/test_fast_graphrag_nlp.py
tests/test_fast_graphrag_scoring.py
tests/test_fast_graphrag_registry.py
tests/test_registry_stage_configs.py
```

## 6. 目标数据结构

### 6.1 FastGraphRAG extraction config

新增到 `graph_memory/retrieval/methods/fast_graphrag/config.py`：

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


FastGraphRAGExtractorType = Literal["regex_english", "syntactic_parser", "cfg"]


@dataclass(frozen=True)
class FastGraphRAGExtractionConfig:
    extractor_type: FastGraphRAGExtractorType = "regex_english"
    normalize_edge_weights: bool = True
    max_word_length: int = 15
    word_delimiter: str = " "
    include_named_entities: bool = True
    exclude_nouns: tuple[str, ...] | None = None
    exclude_entity_tags: tuple[str, ...] = ()
    exclude_pos_tags: tuple[str, ...] = ()
    noun_phrase_tags: tuple[str, ...] = ()
    noun_phrase_grammars: Mapping[str, str] = field(default_factory=dict)
    model_name: str = "en_core_web_md"
```

语义：

- `exclude_nouns is None`：使用内部 stopword / generic noun list。
- `exclude_nouns=()`：不额外排除 noun phrase。
- `extractor_type="regex_english"`：默认路径，不需要外部模型。
- `extractor_type in {"syntactic_parser", "cfg"}`：需要调用方提供 spaCy nlp object 或已安装模型；无法加载时明确失败，不静默退回。

### 6.2 FastGraphRAG pruning config

同文件新增：

```python
@dataclass(frozen=True)
class FastGraphRAGPruningConfig:
    min_node_freq: int = 1
    max_node_freq_std: float | None = None
    min_node_degree: int = 0
    max_node_degree_std: float | None = None
    min_edge_weight_pct: float = 0.0
    remove_ego_nodes: bool = False
    lcc_only: bool = False
```

第一阶段约束：

- `remove_ego_nodes=True` 直接 `ValueError`，因为当前 FastGraphRAG KG 没有官方 graph 里的 ego node 概念。
- `lcc_only=True` 可以实现，保留 largest connected component；如果图为空则返回空 KG。
- `min_edge_weight_pct` 按当前 task 的 relation weight percentile 剪边。

### 6.3 FastGraphRAG method config

把当前 `FastGraphRAGConfig` 从 `method.py` 迁移到 `config.py`：

```python
@dataclass(frozen=True)
class FastGraphRAGScoringConfig:
    lambda_entity: float = 1.0
    lambda_relation: float = 1.0
    lambda_dense_fallback: float = 0.05


@dataclass(frozen=True)
class FastGraphRAGConfig:
    extraction: FastGraphRAGExtractionConfig = field(default_factory=FastGraphRAGExtractionConfig)
    pruning: FastGraphRAGPruningConfig = field(default_factory=FastGraphRAGPruningConfig)
    scoring: FastGraphRAGScoringConfig = field(default_factory=FastGraphRAGScoringConfig)
    entity_seed_top_k: int = 32
    query_link_seed_score: float = 1.0
    dense_entity_seed_weight: float = 1.0
    lexical_substring_match_score: float = 0.5
    ppr_damping: float = 0.85
    ppr_max_iterations: int = 100
    ppr_tolerance: float = 1e-8
```

## 7. 实施任务

### Task 1: 写官方对齐差距测试

**Files:**

- Create: `tests/test_fast_graphrag_official_alignment.py`
- Modify: `tests/test_fast_graphrag_nlp.py`
- Modify: `tests/test_fast_graphrag_index.py`

- [ ] **Step 1: 添加 lower-case noun phrase 抽取测试**

在 `tests/test_fast_graphrag_official_alignment.py` 写入：

```python
from __future__ import annotations

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGExtractionConfig
from graph_memory.retrieval.methods.fast_graphrag.nlp import extract_candidate_mentions
from graph_memory.retrieval.requests import TextCandidate


def test_regex_english_extracts_lowercase_noun_phrases_from_visible_text() -> None:
    candidate = TextCandidate(
        item_id="m0",
        text="the prime minister approved nuclear energy policy after the budget hearing.",
        metadata={"position": 0},
    )

    mentions = extract_candidate_mentions(
        (candidate,),
        config=FastGraphRAGExtractionConfig(extractor_type="regex_english"),
    )

    normalized_names = {mention.normalized_name for mention in mentions}
    assert "prime minister" in normalized_names
    assert "nuclear energy policy" in normalized_names
    assert "budget hearing" in normalized_names
    assert "the" not in normalized_names
```

- [ ] **Step 2: 添加 `exclude_nouns` 语义测试**

同文件继续添加：

```python
def test_exclude_nouns_filters_noun_phrases_but_does_not_define_extraction() -> None:
    candidate = TextCandidate(
        item_id="m0",
        text="the prime minister approved nuclear energy policy.",
        metadata={"position": 0},
    )

    mentions = extract_candidate_mentions(
        (candidate,),
        config=FastGraphRAGExtractionConfig(
            extractor_type="regex_english",
            exclude_nouns=("prime minister",),
        ),
    )

    normalized_names = {mention.normalized_name for mention in mentions}
    assert "prime minister" not in normalized_names
    assert "nuclear energy policy" in normalized_names
```

这个测试明确：“高频词/排除词列表”是过滤器，不是抽取算法。

- [ ] **Step 3: 添加 relation aggregation 测试**

在 `tests/test_fast_graphrag_index.py` 添加：

```python
def test_build_fast_graphrag_kg_aggregates_cooccurring_entity_pairs_across_text_units() -> None:
    request = TextRankingRequest(
        task_id="task-1",
        query_text="Who approved nuclear energy policy?",
        candidates=(
            TextCandidate(
                item_id="m0",
                text="The prime minister approved nuclear energy policy.",
                metadata={"position": 0},
            ),
            TextCandidate(
                item_id="m1",
                text="The prime minister defended nuclear energy policy.",
                metadata={"position": 1},
            ),
        ),
    )
    graph: MemoryGraph = {
        "task_id": "task-1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": request.query_text},
            {"id": "m0", "node_type": "graph_item", "node_kind": "document_sentence", "text": request.candidates[0].text},
            {"id": "m1", "node_type": "graph_item", "node_kind": "document_sentence", "text": request.candidates[1].text},
        ],
        "edges": [],
    }

    kg = build_fast_graphrag_knowledge_graph(request, graph)

    matching = [
        relation
        for relation in kg.relations
        if {relation.source_entity_id, relation.target_entity_id}
        == {"e:prime-minister", "e:nuclear-energy-policy"}
    ]
    assert len(matching) == 1
    assert matching[0].candidate_ids == ("m0", "m1")
    assert matching[0].weight == 1.0
```

- [ ] **Step 4: 运行测试确认失败**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_official_alignment.py tests/test_fast_graphrag_index.py::test_build_fast_graphrag_kg_aggregates_cooccurring_entity_pairs_across_text_units
```

Expected:

- `FastGraphRAGExtractionConfig` 不存在。
- `extract_candidate_mentions(..., config=...)` 还不支持。
- relation aggregation 还没有实现。

### Task 2: 新增 FastGraphRAG config 模块并接入 registry settings

**Files:**

- Create: `graph_memory/retrieval/methods/fast_graphrag/config.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/method.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/scoring.py`
- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Test: `tests/test_fast_graphrag_registry.py`
- Test: `tests/test_registry_stage_configs.py`

- [ ] **Step 1: 写 settings round-trip 测试**

在 `tests/test_registry_stage_configs.py` 的 FastGraphRAG config 测试里断言：

```python
assert config.job.extraction.extractor_type == "regex_english"
assert config.job.extraction.normalize_edge_weights is True
assert config.job.extraction.exclude_nouns is None
assert config.job.pruning.min_node_freq == 1
assert config.job.pruning.min_edge_weight_pct == 0.0
```

- [ ] **Step 2: 新增 `config.py`**

按第 6 节添加 `FastGraphRAGExtractionConfig`、`FastGraphRAGPruningConfig`、`FastGraphRAGScoringConfig`、`FastGraphRAGConfig`。

- [ ] **Step 3: 移动 scoring config import**

`graph_memory/retrieval/methods/fast_graphrag/scoring.py` 不再定义 `FastGraphRAGScoringConfig`，改为：

```python
from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGScoringConfig
```

- [ ] **Step 4: registry settings 暴露官方对齐配置**

`FastGraphRAGRetrievalSettings` 改为：

```python
@dataclass(frozen=True)
class FastGraphRAGRetrievalSettings:
    top_k: int
    encoder: DenseEncoderSettings
    extraction: FastGraphRAGExtractionConfig = field(default_factory=FastGraphRAGExtractionConfig)
    pruning: FastGraphRAGPruningConfig = field(default_factory=FastGraphRAGPruningConfig)
    scoring: FastGraphRAGScoringConfig = field(default_factory=FastGraphRAGScoringConfig)
    entity_seed_top_k: int = 32
    query_link_seed_score: float = 1.0
    dense_entity_seed_weight: float = 1.0
    lexical_substring_match_score: float = 0.5
    ppr_damping: float = 0.85
    ppr_max_iterations: int = 100
    ppr_tolerance: float = 1e-8
    method: Literal[RetrievalMethodId.FAST_GRAPHRAG] = RetrievalMethodId.FAST_GRAPHRAG
```

- [ ] **Step 5: builder 组装 config**

`_build_fast_graphrag()` 传入：

```python
FastGraphRAGConfig(
    extraction=settings.extraction,
    pruning=settings.pruning,
    scoring=settings.scoring,
    entity_seed_top_k=settings.entity_seed_top_k,
    query_link_seed_score=settings.query_link_seed_score,
    dense_entity_seed_weight=settings.dense_entity_seed_weight,
    lexical_substring_match_score=settings.lexical_substring_match_score,
    ppr_damping=settings.ppr_damping,
    ppr_max_iterations=settings.ppr_max_iterations,
    ppr_tolerance=settings.ppr_tolerance,
)
```

- [ ] **Step 6: 运行配置测试**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_registry.py tests/test_registry_stage_configs.py
```

Expected: PASS。

### Task 3: 实现 dependency-free `regex_english` noun phrase extraction

**Files:**

- Create: `graph_memory/retrieval/methods/fast_graphrag/noun_phrases.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/nlp.py`
- Test: `tests/test_fast_graphrag_official_alignment.py`
- Test: `tests/test_fast_graphrag_nlp.py`

- [ ] **Step 1: 新增 noun phrase extraction helper**

`noun_phrases.py` 提供：

```python
@dataclass(frozen=True)
class NounPhrase:
    text: str
    normalized_text: str
    source: Literal["regex_english", "spacy_noun_chunk", "spacy_entity"]


def extract_regex_english_noun_phrases(text: str, config: FastGraphRAGExtractionConfig) -> tuple[NounPhrase, ...]:
    ...
```

默认 regex 规则：

- token pattern: `[A-Za-z][A-Za-z0-9'-]*`
- normalize 使用现有 `normalize_entity_text()`
- 丢弃 stopword-only span
- 丢弃任意 word 长度超过 `max_word_length` 的 span
- 对连续 content token 生成最长 2-4 gram span
- 单 token lower-case noun 默认不收，避免把所有普通词变成 entity
- 单 token capitalized / all-caps / numeric-mixed 可以收
- `exclude_nouns` 按 normalized exact phrase 过滤

- [ ] **Step 2: `nlp.py` 接受 config**

把签名改为：

```python
def extract_candidate_mentions(
    candidates: Sequence[TextCandidate],
    *,
    config: FastGraphRAGExtractionConfig | None = None,
    nlp: object | None = None,
) -> tuple[EntityMention, ...]:
```

默认 `config = FastGraphRAGExtractionConfig()`。

- [ ] **Step 3: 保留强信号 mention 来源**

现有 `title`、`alias`、`source_ref`、`title_prefix`、`capitalized_phrase` 不删除，但它们必须共用同一套 normalization / exclude_nouns / max_word_length 过滤。

- [ ] **Step 4: 把 noun phrase mention source 纳入类型**

`MentionSource` 增加：

```python
Literal[
    "title",
    "source_ref",
    "title_prefix",
    "capitalized_phrase",
    "alias",
    "noun_phrase",
    "spacy_noun_chunk",
    "spacy_entity",
]
```

`_entity_type()` 为 noun phrase 返回 `"noun_phrase"`，spaCy entity 返回 `"named_entity"`。

- [ ] **Step 5: 运行 NLP 测试**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_nlp.py tests/test_fast_graphrag_official_alignment.py
```

Expected: PASS。

### Task 4: 增加 optional spaCy extractor，但不让测试依赖模型下载

**Files:**

- Modify: `graph_memory/retrieval/methods/fast_graphrag/noun_phrases.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/nlp.py`
- Test: `tests/test_fast_graphrag_official_alignment.py`

- [ ] **Step 1: 写 fake spaCy 测试**

添加：

```python
def test_syntactic_parser_uses_supplied_spacy_noun_chunks_without_model_download() -> None:
    candidate = TextCandidate(
        item_id="m0",
        text="The prime minister approved nuclear energy policy.",
        metadata={"position": 0},
    )

    class FakeSpan:
        def __init__(self, text: str) -> None:
            self.text = text
            self.label_ = "NP"
            self.root = type("Root", (), {"pos_": "NOUN", "tag_": "NN"})()

    class FakeDoc:
        noun_chunks = [FakeSpan("prime minister"), FakeSpan("nuclear energy policy")]
        ents = []

    def fake_nlp(text: str) -> FakeDoc:
        assert "prime minister" in text
        return FakeDoc()

    mentions = extract_candidate_mentions(
        (candidate,),
        config=FastGraphRAGExtractionConfig(extractor_type="syntactic_parser"),
        nlp=fake_nlp,
    )

    assert {"prime minister", "nuclear energy policy"} <= {m.normalized_name for m in mentions}
```

- [ ] **Step 2: 实现 spaCy branch**

规则：

- `syntactic_parser` 读取 `doc.noun_chunks`。
- `cfg` 第一阶段也读取 `doc.noun_chunks`，但通过 `noun_phrase_grammars` 过滤 tag；不实现完整官方 CFG parser。
- `include_named_entities=True` 时读取 `doc.ents`。
- `exclude_entity_tags` 过滤 `entity.label_`。
- `exclude_pos_tags` 过滤 `span.root.pos_`。
- `noun_phrase_tags` 过滤 `span.root.tag_`。

- [ ] **Step 3: 明确无模型时失败**

如果 `extractor_type != "regex_english"` 且 `nlp is None`，不要自动下载模型。抛出：

```python
ValueError("FastGraphRAG spaCy extraction requires an nlp object or preloaded model.")
```

- [ ] **Step 4: 运行测试**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_official_alignment.py tests/test_fast_graphrag_nlp.py
```

Expected: PASS。

### Task 5: 按官方 co-occurrence 语义重做 relation aggregation

**Files:**

- Modify: `graph_memory/retrieval/methods/fast_graphrag/index.py`
- Test: `tests/test_fast_graphrag_index.py`
- Test: `tests/test_fast_graphrag_official_alignment.py`

- [ ] **Step 1: 改 builder 签名传 config**

`build_fast_graphrag_knowledge_graph()` 改为：

```python
def build_fast_graphrag_knowledge_graph(
    request: TextRankingRequest,
    graph: MemoryGraph,
    *,
    config: FastGraphRAGExtractionConfig | None = None,
) -> FastGraphRAGKnowledgeGraph:
```

- [ ] **Step 2: 聚合 relation by entity pair**

实现规则：

- 每个 candidate 是一个 text unit。
- 同一 candidate 内 canonical entity ids 去重。
- 对每个 unordered pair 累加 co-occurrence count。
- relation id 不包含 candidate hash，改成 `relation:{first_id}:{second_id}`。
- `candidate_ids` 是所有共现 text unit id，排序稳定。
- `description` 不编造事实，只写：`"{source_name} -- co-occurs with -- {target_name} in {count} text units"`。

- [ ] **Step 3: edge weight normalization**

如果 `config.normalize_edge_weights=True`：

- 原始权重为 co-occurrence count。
- 除以当前 task 最大 co-occurrence count。
- 若最大值为 0，保持 0。

如果 false，保留 count。

- [ ] **Step 4: 运行 index 测试**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_index.py tests/test_fast_graphrag_official_alignment.py
```

Expected: PASS。

### Task 6: 实现 graph pruning

**Files:**

- Create: `graph_memory/retrieval/methods/fast_graphrag/pruning.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/index.py`
- Test: `tests/test_fast_graphrag_pruning.py`

- [ ] **Step 1: 写 pruning 测试**

`tests/test_fast_graphrag_pruning.py` 添加：

```python
from __future__ import annotations

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGPruningConfig
from graph_memory.retrieval.methods.fast_graphrag.pruning import prune_knowledge_graph
from graph_memory.retrieval.requests import FastGraphRAGEntity, FastGraphRAGKnowledgeGraph, FastGraphRAGRelation


def test_prune_knowledge_graph_removes_low_frequency_entities_and_orphan_relations() -> None:
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:a", "A", "a", "noun_phrase", "A", ("m0", "m1")),
            FastGraphRAGEntity("e:b", "B", "b", "noun_phrase", "B", ("m0",)),
            FastGraphRAGEntity("e:c", "C", "c", "noun_phrase", "C", ("m1", "m2")),
        ),
        relations=(
            FastGraphRAGRelation("r:a:b", "e:a", "e:b", "cooccurs", ("m0",), 1.0),
            FastGraphRAGRelation("r:a:c", "e:a", "e:c", "cooccurs", ("m1",), 1.0),
        ),
    )

    pruned = prune_knowledge_graph(kg, FastGraphRAGPruningConfig(min_node_freq=2))

    assert [entity.entity_id for entity in pruned.entities] == ["e:a", "e:c"]
    assert [relation.relation_id for relation in pruned.relations] == ["r:a:c"]
```

再添加 edge percentile 和 `remove_ego_nodes=True` rejected tests。

- [ ] **Step 2: 实现 pruning**

`prune_knowledge_graph()` 输入 KG 和 config，输出新 KG。

实现顺序：

1. 校验 `remove_ego_nodes is False`，否则 `ValueError`。
2. 计算 node frequency：`len(entity.candidate_ids)`。
3. 计算 node degree：relation endpoint 出现次数。
4. 应用 `min_node_freq` 和 `min_node_degree`。
5. 如果 `max_node_freq_std` 非空，移除 `freq > mean(freq) + std(freq) * max_node_freq_std`。
6. 如果 `max_node_degree_std` 非空，移除 `degree > mean(degree) + std(degree) * max_node_degree_std`。
7. 移除 endpoint 不存在的 relation。
8. 按 `min_edge_weight_pct` 移除低权重 relation。
9. 如果 `lcc_only=True`，保留 largest connected component。

- [ ] **Step 3: 在 index builder 中调用 pruning**

`build_fast_graphrag_knowledge_graph()` 先构建 raw KG，再：

```python
return prune_knowledge_graph(raw_kg, pruning_config)
```

实际 builder 需要接收完整 `FastGraphRAGConfig` 或至少 extraction/pruning 二元配置；不要让 method 层二次 prune。

- [ ] **Step 4: 运行 pruning 测试**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_pruning.py tests/test_fast_graphrag_index.py
```

Expected: PASS。

### Task 7: Query entity mapping 成为强 seed

**Files:**

- Modify: `graph_memory/retrieval/methods/fast_graphrag/method.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/nlp.py`
- Test: `tests/test_fast_graphrag_method.py`
- Test: `tests/test_fast_graphrag_scoring.py`

- [ ] **Step 1: 写 query mapping 测试**

在 `tests/test_fast_graphrag_method.py` 添加：

```python
def test_fast_graphrag_query_linked_entities_override_dense_noise() -> None:
    request = fast_graphrag_fixture_request()
    method = FastGraphRAGMethod(
        name="fast_graphrag",
        config=FastGraphRAGConfig(query_link_seed_score=1.0, dense_entity_seed_weight=0.1),
        dense_ranker=FakeDenseSeedRanker(
            entity_scores={"e:changed-it": 0.0, "e:nicki-minaj": 0.1},
            candidate_scores={"m0": 0.0, "m1": 0.0, "m2": 0.9},
        ),
    )

    result = method.rank_task(request, top_k=2)

    assert result.ranked_nodes[0].node_id in {"m0", "m1"}
```

- [ ] **Step 2: 复用 `link_query_entities()`**

`rank_task()` seed 顺序：

1. `linked_entities = link_query_entities(request.query_text, catalog_or_kg_view)`。
2. linked entity 得 `query_link_seed_score`。
3. dense entity scores 只取 top `entity_seed_top_k`。
4. lexical substring seed 继续保留，但作为弱 seed。
5. 合并时不要无脑 `max()`；使用 weighted sum，避免 dense all-entity 噪声覆盖 query-linked seed。

- [ ] **Step 3: 不把 query_entities 放进 request**

query mapping 是方法内部计算，不能扩大 dataset projector contract。

- [ ] **Step 4: 运行 method/scoring 测试**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_method.py tests/test_fast_graphrag_scoring.py
```

Expected: PASS。

### Task 8: Candidate scoring 对齐 entity / relation / text unit 贡献

**Files:**

- Modify: `graph_memory/retrieval/methods/fast_graphrag/scoring.py`
- Test: `tests/test_fast_graphrag_scoring.py`

- [ ] **Step 1: 增加聚合归一化测试**

新增测试：

```python
def test_candidate_scores_do_not_reward_repeated_mentions_without_relation_support() -> None:
    candidates = (
        TextCandidate(item_id="m0", text="A mentions B once.", metadata={"position": 0}),
        TextCandidate(item_id="m1", text="A repeats A A A.", metadata={"position": 1}),
    )
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:a", "A", "a", "noun_phrase", "A", ("m0", "m1")),
            FastGraphRAGEntity("e:b", "B", "b", "noun_phrase", "B", ("m0",)),
        ),
        relations=(
            FastGraphRAGRelation("r:a:b", "e:a", "e:b", "co-occurs", ("m0",), 1.0),
        ),
    )

    scores = score_candidates(
        candidates,
        kg,
        entity_scores={"e:a": 0.5, "e:b": 0.5},
        dense_fallback_scores={"m0": 0.0, "m1": 0.0},
        config=FastGraphRAGScoringConfig(lambda_entity=1.0, lambda_relation=1.0, lambda_dense_fallback=0.0),
    )

    assert scores["m0"] > scores["m1"]
```

- [ ] **Step 2: 调整 scoring**

保留当前公式，但明确：

- entity contribution 按 candidate linked unique entity 求和。
- relation contribution 按 candidate linked relation 求和。
- relation 权重来自 aggregation/pruning 后 KG。
- dense fallback 只用于没有 KG support 或平局时的小权重补充。

- [ ] **Step 3: 运行 scoring 测试**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_scoring.py
```

Expected: PASS。

### Task 9: registry / workflow 配置接通官方对齐参数

**Files:**

- Modify: `graph_memory/registry/retrieval.py`
- Modify: `graph_memory/registry/retrieval_builders.py`
- Modify: `scripts/workflow/stage_configs.py`
- Test: `tests/test_fast_graphrag_registry.py`
- Test: `tests/test_registry_stage_configs.py`
- Test: `tests/test_config_run_retrieval.py`

- [ ] **Step 1: stage config 支持 nested dataclass JSON**

FastGraphRAG retrieve config 应能 round-trip：

```json
{
  "job": {
    "method": "fast_graphrag",
    "top_k": 10,
    "encoder": {
      "model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "query_prefix": "query: ",
      "passage_prefix": "passage: ",
      "batch_size": 64
    },
    "extraction": {
      "extractor_type": "regex_english",
      "normalize_edge_weights": true,
      "include_named_entities": true,
      "exclude_nouns": null
    },
    "pruning": {
      "min_node_freq": 1,
      "min_edge_weight_pct": 0.0,
      "remove_ego_nodes": false,
      "lcc_only": false
    }
  }
}
```

- [ ] **Step 2: builder 将 config 传入 KG builder**

`_fast_graphrag_execution_tasks()` 签名改成接收 method config：

```python
def _fast_graphrag_execution_tasks(
    ranking_requests: list[TextRankingRequest],
    graph_index: GraphIndex,
    knowledge_graph_builder: Callable[[TextRankingRequest, MemoryGraph, FastGraphRAGConfig], FastGraphRAGKnowledgeGraph],
    config: FastGraphRAGConfig,
) -> list[RetrievalExecutionTask]:
```

也可以用 keyword-only builder wrapper，关键是 extraction/pruning config 必须在 request 构造时生效。

- [ ] **Step 3: workflow 默认不需要 tuning**

保持 `fast_graphrag` 为 graph-backed retrieval workflow，不加入 `tune_graph_rerank.py`。

- [ ] **Step 4: 运行 registry/workflow 测试**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_registry.py tests/test_registry_stage_configs.py tests/test_config_run_retrieval.py
```

Expected: PASS。

### Task 10: 添加 no-LLM 边界回归测试

**Files:**

- Modify: `tests/test_fast_graphrag_no_llm_boundary.py`
- Test: `tests/test_fast_graphrag_no_llm_boundary.py`

- [ ] **Step 1: 扩展 forbidden patterns**

继续禁止：

```python
FORBIDDEN_PATTERNS = (
    "openai",
    "llm",
    "prompt",
    "completion",
    "chat",
    "domain",
    "example_queries",
    "entity_types",
    "community_report",
    "summarize",
)
```

注意：官方资料在计划文档里可以出现这些词；测试只扫 `graph_memory/retrieval/methods/fast_graphrag` 源码。

- [ ] **Step 2: 允许 config 名称但不允许 LLM 概念**

`extractor_type`、`exclude_nouns`、`prune_graph` 是官方 NLP / graph 配置，不属于 LLM 边界。

- [ ] **Step 3: 运行边界测试**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_no_llm_boundary.py
```

Expected: PASS。

### Task 11: 小规模 workflow 验证

**Files:**

- No code-only files unless previous tasks expose failures.
- Test command exercises current consuming workflow.

- [ ] **Step 1: 运行 FastGraphRAG 单元测试集合**

Run:

```powershell
$files = Get-ChildItem -LiteralPath tests -Filter 'test_fast_graphrag_*.py' | ForEach-Object { $_.FullName }
uv run pytest -q @files
```

Expected: PASS。

- [ ] **Step 2: 运行相关 registry/workflow tests**

Run:

```powershell
uv run pytest -q tests/test_method_registry.py tests/test_registry_stage_configs.py tests/test_workflow_orchestration.py tests/test_config_run_retrieval.py
```

Expected: PASS。

- [ ] **Step 3: 运行静态检查**

Run:

```powershell
uv run ruff check graph_memory/retrieval/methods/fast_graphrag graph_memory/registry tests/test_fast_graphrag_*.py
uv run basedpyright graph_memory/retrieval/methods/fast_graphrag graph_memory/registry tests/test_fast_graphrag_*.py
```

Expected: no errors。

- [ ] **Step 4: 运行真实 retrieve smoke**

优先使用现有 `tests/test_config_run_retrieval.py` 覆盖的 fixture；如果要跑 workflow 层 smoke，使用当前项目已有 quick profile，不新增本地 uv cache，也不改实验配置语义。

Expected:

- `fast_graphrag` 能从 graph-backed workflow 构造 request。
- 输出 ranked results 包含每个 candidate exactly once。
- `retrieved_subgraph` 仍来自 candidate graph，不混入 label-only edges。

## 8. 不做事项

本轮不要做：

- 不实现官方完整 local/global/DRIFT query answer generation。
- 不实现 LLM community report。
- 不新增 prompt、completion model、LLM provider config。
- 不把 `FastGraphRAGRequest` 改成官方 SDK API 形状。
- 不把 `answer`、`supporting_facts`、`gold_dependency_edges` 用作 extraction seed。
- 不把所有 lower-case unigram 都当 entity；这会造成 graph 噪声并偏离 noun phrase 目标。
- 不新增一个宽泛 `object` / `dict[str, object]` 配置 bag。

## 9. 验收标准

实现完成后，至少满足：

1. lower-case noun phrases 能被默认 `regex_english` extractor 抽出。
2. `exclude_nouns` 作为过滤器生效，并且文档/测试明确它不是抽取方法。
3. 同一 entity pair 在多个 text units 中被聚合成一条 relation，`candidate_ids` 保留 provenance。
4. relation weight 能按配置归一化。
5. graph pruning 能按 node frequency、node degree、edge weight percentile 去噪。
6. query-linked entities 是强 seed，不会被 dense all-entity 噪声轻易覆盖。
7. no-LLM 边界测试继续通过。
8. registry、stage config、workflow orchestration 仍能 round-trip `fast_graphrag`。
9. 全部 FastGraphRAG tests、相关 registry/workflow tests、ruff、basedpyright 通过。

## 10. 后续可选项

这些可以以后再做，不进入第一轮修复：

- 引入 optional `nltk` extra，并在有本地资源时提供更接近官方默认的 NLTK extractor。
- 做 extractive community context：按 connected components / Leiden-like clustering 收集 text units，但不生成 LLM summary。
- 增加 graph snapshot debug artifact，帮助检查 entity/relation graph 与 candidate graph 的差异。
- 在真实 HotpotQA / 2Wiki quick profile 上比较修复前后 Full Support@k、Path Recall@k、latency。

