# FastGraphRAG Official NLTK Regex Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current dependency-free FastGraphRAG regex approximation with an internally owned implementation that follows Microsoft GraphRAG FastGraphRAG's default NLTK/TextBlob + regular-expression noun phrase extraction and noun-graph weighting semantics as closely as this repo's no-LLM evidence-retrieval boundary allows.

**Architecture:** Keep the repo-owned `FastGraphRAGRequest` and evidence-ranking output contract, but make the private FastGraphRAG indexer look like the official `extract_graph_nlp -> build_noun_graph -> prune_graph` path. The official LLM community-report/query-generation parts remain explicitly out of scope; this plan only aligns the NLP graph index and the deterministic retrieval adapter that consumes it.

**Tech Stack:** Python 3.10, Python dataclasses, `nltk>=3.9.1,<3.10`, `textblob>=0.18,<0.21`, current `TextCandidate` / `MemoryGraph` / registry contracts, pure-Python PMI edge weighting, existing spaCy optional path, pytest, ruff, basedpyright.

---

Date: 2026-06-23

Status: Draft implementation plan. This document records implementation order and verification gates; it does not mean code has been changed.

## 1. Official Baseline To Match

Use Microsoft GraphRAG source commit `6d02c2355c3fed4c49007572fbe951d73258a37f` and the public docs as the implementation reference.

Primary source files:

- [`regex_extractor.py`](https://github.com/microsoft/graphrag/blob/6d02c2355c3fed4c49007572fbe951d73258a37f/packages/graphrag/graphrag/index/operations/build_noun_graph/np_extractors/regex_extractor.py)
- [`base.py`](https://github.com/microsoft/graphrag/blob/6d02c2355c3fed4c49007572fbe951d73258a37f/packages/graphrag/graphrag/index/operations/build_noun_graph/np_extractors/base.py)
- [`factory.py`](https://github.com/microsoft/graphrag/blob/6d02c2355c3fed4c49007572fbe951d73258a37f/packages/graphrag/graphrag/index/operations/build_noun_graph/np_extractors/factory.py)
- [`build_noun_graph.py`](https://github.com/microsoft/graphrag/blob/6d02c2355c3fed4c49007572fbe951d73258a37f/packages/graphrag/graphrag/index/operations/build_noun_graph/build_noun_graph.py)
- [`edge_weights.py`](https://github.com/microsoft/graphrag/blob/6d02c2355c3fed4c49007572fbe951d73258a37f/packages/graphrag/graphrag/graphs/edge_weights.py)
- [`prune_graph.py`](https://github.com/microsoft/graphrag/blob/6d02c2355c3fed4c49007572fbe951d73258a37f/packages/graphrag/graphrag/index/operations/prune_graph.py)
- [`defaults.py`](https://github.com/microsoft/graphrag/blob/6d02c2355c3fed4c49007572fbe951d73258a37f/packages/graphrag/graphrag/config/defaults.py)

Public documentation:

- [GraphRAG indexing methods: FastGraphRAG](https://microsoft.github.io/graphrag/index/methods/)
- [GraphRAG YAML configuration: `extract_graph_nlp` and `prune_graph`](https://microsoft.github.io/graphrag/config/yaml/)

Official behavior that must be represented in this repo:

- `regex_english` is the default noun phrase extractor.
- Official `regex_english` uses `TextBlob(text).noun_phrases`, `TextBlob(text).tags`, and NLTK resources, not the current hand-rolled n-gram segmenter.
- Official regex extraction filters noun phrases by:
  - removing configured `exclude_nouns` from phrase tokens by uppercase exact token comparison;
  - keeping phrases that have a proper noun, have more than one cleaned token, or contain compound words;
  - requiring every cleaned token to match `^[a-zA-Z0-9\-]+\n?$`;
  - requiring every cleaned token length to be `<= max_word_length`;
  - emitting uppercase joined phrase text using `word_delimiter`.
- Official `exclude_nouns=None` resolves to `EN_STOP_WORDS`, whose current list is: `stuff`, `thing`, `things`, `bunch`, `bit`, `bits`, `people`, `person`, `okay`, `hey`, `hi`, `hello`, `laughter`, `oh`.
- Official entities have `title`, `frequency`, `text_unit_ids`, `type="NOUN PHRASE"`, and `description=""`.
- Official relationships are co-occurrence pairs over unique noun phrase titles per text unit.
- Official raw relationship weight is text-unit co-occurrence count.
- Official `normalize_edge_weights=True` applies PMI weighting via `calculate_pmi_edge_weights`, not max-count normalization.
- Official prune defaults are stricter than the current repo defaults: `min_node_degree=1` and `min_edge_weight_pct=40` in the operation signature.

Important boundary:

- Official FastGraphRAG still uses LLM community reports for the normal GraphRAG query flow. This repo still must not add prompts, LLM clients, generated answers, or future LLM hooks. The output remains evidence retrieval: `ranked_nodes` plus top-k `retrieved_subgraph`.
- The server runtime is Python 3.10. Do not add the official `graphrag` package as a runtime dependency: the official package at the referenced source snapshot declares `requires-python = ">=3.11,<3.14"`. This plan ports the specific official NLP implementation semantics into this repo instead.

## 2. Python 3.10 Dependency Policy

The dependency policy for this plan is:

```text
requires-python = ">=3.10"
nltk>=3.9.1,<3.10
textblob>=0.18,<0.21
```

Rationale:

- Current repo `pyproject.toml` already declares `requires-python = ">=3.10"`.
- PyPI metadata checked on 2026-06-23 shows current `nltk 3.9.4` has `Requires-Python >=3.10`.
- PyPI metadata checked on 2026-06-23 shows current `textblob 0.20.0` has `Requires-Python >=3.10`.
- The upper caps prevent a future `nltk 3.10+` or `textblob 0.21+` release from silently raising the Python requirement above the server's Python 3.10.
- The version ranges still stay within the official GraphRAG dependency family: official GraphRAG declares `nltk~=3.9` and `textblob~=0.18`, but this repo uses tighter upper caps because the deployment interpreter is fixed at Python 3.10.

Every implementation run must verify dependency resolution with Python 3.10:

```powershell
uv lock --python 3.10 --dry-run
uv sync --python 3.10
uv run python -c "import sys, nltk, textblob; assert sys.version_info[:2] == (3, 10), sys.version; print(sys.version); print(nltk.__version__); print(textblob.__version__)"
```

Expected:

```text
Python version starts with 3.10
NLTK version is 3.9.x
TextBlob version is 0.18.x, 0.19.x, or 0.20.x
```

## 3. Current Implementation Gap

The current `graph_memory/retrieval/methods/fast_graphrag/noun_phrases.py` implements a dependency-free n-gram extractor. That was useful as a first pass, but it is not the official default. It also lowercases normalized output early, has a broader stopword/boundary-word model, and creates phrases the official TextBlob extractor would not necessarily emit.

The current relation aggregation in `graph_memory/retrieval/methods/fast_graphrag/index.py` is closer to official co-occurrence semantics, but its normalized edge weight is `count / max_count`, not official PMI.

The current pruning module has similar knobs but default values differ from official pruning defaults.

## 4. File Structure

### Create

```text
graph_memory/retrieval/methods/fast_graphrag/official_nltk.py
graph_memory/retrieval/methods/fast_graphrag/edge_weights.py
scripts/bootstrap_fast_graphrag_nltk.py
tests/test_fast_graphrag_official_nltk.py
tests/test_fast_graphrag_edge_weights.py
```

### Modify

```text
pyproject.toml
graph_memory/retrieval/methods/fast_graphrag/config.py
graph_memory/retrieval/methods/fast_graphrag/noun_phrases.py
graph_memory/retrieval/methods/fast_graphrag/nlp.py
graph_memory/retrieval/methods/fast_graphrag/index.py
graph_memory/retrieval/methods/fast_graphrag/pruning.py
graph_memory/registry/retrieval.py
tests/test_fast_graphrag_official_alignment.py
tests/test_fast_graphrag_index.py
tests/test_fast_graphrag_pruning.py
tests/test_fast_graphrag_no_llm_boundary.py
tests/test_registry_stage_configs.py
```

### Existing files to read before implementing

```text
docs/10-plans/fast-graphrag-official-alignment-plan.md
graph_memory/retrieval/methods/fast_graphrag/config.py
graph_memory/retrieval/methods/fast_graphrag/noun_phrases.py
graph_memory/retrieval/methods/fast_graphrag/index.py
graph_memory/retrieval/methods/fast_graphrag/pruning.py
```

## 5. Implementation Tasks

### Task 1: Add Official NLTK/TextBlob Dependencies And Resource Bootstrap

**Files:**

- Modify: `pyproject.toml`
- Create: `scripts/bootstrap_fast_graphrag_nltk.py`
- Test: `tests/test_fast_graphrag_official_nltk.py`

- [ ] **Step 1: Write dependency presence test**

Create `tests/test_fast_graphrag_official_nltk.py` with:

```python
from __future__ import annotations


def test_official_nltk_regex_dependencies_are_importable() -> None:
    import nltk  # noqa: F401
    import textblob  # noqa: F401
```

- [ ] **Step 2: Run the dependency test and confirm failure before dependency changes**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_official_nltk.py::test_official_nltk_regex_dependencies_are_importable
```

Expected before editing dependencies:

```text
ModuleNotFoundError: No module named 'nltk'
```

If `nltk` is already present transitively, the failure may instead be:

```text
ModuleNotFoundError: No module named 'textblob'
```

- [ ] **Step 3: Add dependencies**

Modify `pyproject.toml`:

```toml
dependencies = [
    "numpy>=1.26",
    "nltk>=3.9.1,<3.10",
    "rank-bm25>=0.2.2",
    "sentence-transformers==2.7.0",
    "textblob>=0.18,<0.21",
    "torch>=2.11,<2.12",
    "torchvision>=0.26,<0.27",
    "tqdm>=4.66",
    "typing-extensions>=4.10",
]
```

Rationale: official GraphRAG declares `nltk~=3.9` and `textblob~=0.18` in its package dependencies, but this repo runs on Python 3.10 servers, so the plan uses tighter upper caps that preserve Python 3.10 compatibility.

- [ ] **Step 4: Add explicit bootstrap script**

Create `scripts/bootstrap_fast_graphrag_nltk.py`:

```python
from __future__ import annotations

import nltk

REQUIRED_RESOURCES: tuple[str, ...] = (
    "brown",
    "treebank",
    "averaged_perceptron_tagger_eng",
    "punkt",
    "punkt_tab",
)


def main() -> int:
    for resource_name in REQUIRED_RESOURCES:
        nltk.download(resource_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

This script is intentionally explicit. Runtime extraction should not hide data downloads in normal tests unless a test opts into monkeypatching or the user has bootstrapped resources.

- [ ] **Step 5: Install and verify dependencies**

Run:

```powershell
uv lock --python 3.10 --dry-run
uv sync --python 3.10
uv run python -c "import sys, nltk, textblob; assert sys.version_info[:2] == (3, 10), sys.version; print(sys.version); print(nltk.__version__); print(textblob.__version__)"
uv run pytest -q tests/test_fast_graphrag_official_nltk.py::test_official_nltk_regex_dependencies_are_importable
```

Expected:

```text
Python version starts with 3.10
NLTK version is 3.9.x
TextBlob version is 0.18.x, 0.19.x, or 0.20.x
1 passed
```

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml uv.lock scripts/bootstrap_fast_graphrag_nltk.py tests/test_fast_graphrag_official_nltk.py
git commit -m "deps: add official fast-graphrag nltk regex dependencies"
```

### Task 2: Implement Official-Style Regex English Extractor

**Files:**

- Create: `graph_memory/retrieval/methods/fast_graphrag/official_nltk.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/noun_phrases.py`
- Test: `tests/test_fast_graphrag_official_nltk.py`

- [ ] **Step 1: Add unit tests for official filtering semantics**

Append to `tests/test_fast_graphrag_official_nltk.py`:

```python
from __future__ import annotations

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGExtractionConfig
from graph_memory.retrieval.methods.fast_graphrag.official_nltk import (
    OFFICIAL_EN_STOP_WORDS,
    OfficialRegexEnglishExtractor,
)


def test_official_regex_extractor_uses_official_default_exclude_nouns() -> None:
    extractor = OfficialRegexEnglishExtractor(
        FastGraphRAGExtractionConfig(exclude_nouns=None)
    )

    assert extractor.exclude_nouns == tuple(noun.upper() for noun in OFFICIAL_EN_STOP_WORDS)


def test_official_regex_tagging_filters_single_common_noun_but_keeps_proper_noun() -> None:
    extractor = OfficialRegexEnglishExtractor(
        FastGraphRAGExtractionConfig(max_word_length=15, word_delimiter=" ")
    )

    common = extractor.tag_noun_phrase("population", all_proper_nouns=())
    proper = extractor.tag_noun_phrase("Richmond", all_proper_nouns=("RICHMOND",))

    assert common.keep is False
    assert proper.keep is True
    assert proper.cleaned_text == "RICHMOND"


def test_official_regex_tagging_keeps_multiword_phrase_and_compound_word() -> None:
    extractor = OfficialRegexEnglishExtractor(
        FastGraphRAGExtractionConfig(max_word_length=15, word_delimiter=" ")
    )

    multiword = extractor.tag_noun_phrase("nuclear energy policy", all_proper_nouns=())
    compound = extractor.tag_noun_phrase("census-designated", all_proper_nouns=())

    assert multiword.keep is True
    assert multiword.cleaned_text == "NUCLEAR ENERGY POLICY"
    assert compound.keep is True
    assert compound.cleaned_text == "CENSUS-DESIGNATED"


def test_official_regex_tagging_removes_excluded_tokens_before_keep_decision() -> None:
    extractor = OfficialRegexEnglishExtractor(
        FastGraphRAGExtractionConfig(exclude_nouns=("thing",), max_word_length=15)
    )

    phrase = extractor.tag_noun_phrase("thing Richmond", all_proper_nouns=("RICHMOND",))

    assert phrase.keep is True
    assert phrase.cleaned_tokens == ("Richmond",)
    assert phrase.cleaned_text == "RICHMOND"
```

- [ ] **Step 2: Run tests and confirm module missing**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_official_nltk.py
```

Expected:

```text
ModuleNotFoundError: No module named 'graph_memory.retrieval.methods.fast_graphrag.official_nltk'
```

- [ ] **Step 3: Implement official extractor wrapper**

Create `graph_memory/retrieval/methods/fast_graphrag/official_nltk.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import nltk
from textblob import TextBlob

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGExtractionConfig

OFFICIAL_EN_STOP_WORDS: tuple[str, ...] = (
    "stuff",
    "thing",
    "things",
    "bunch",
    "bit",
    "bits",
    "people",
    "person",
    "okay",
    "hey",
    "hi",
    "hello",
    "laughter",
    "oh",
)

REQUIRED_NLTK_RESOURCES: tuple[str, ...] = (
    "brown",
    "treebank",
    "averaged_perceptron_tagger_eng",
    "punkt",
    "punkt_tab",
)


@dataclass(frozen=True)
class TaggedNounPhrase:
    cleaned_tokens: tuple[str, ...]
    cleaned_text: str
    has_proper_nouns: bool
    has_compound_words: bool
    has_valid_tokens: bool

    @property
    def keep(self) -> bool:
        return (
            self.has_proper_nouns
            or len(self.cleaned_tokens) > 1
            or self.has_compound_words
        ) and self.has_valid_tokens


class OfficialRegexEnglishExtractor:
    def __init__(self, config: FastGraphRAGExtractionConfig) -> None:
        self.max_word_length = config.max_word_length
        self.word_delimiter = config.word_delimiter
        exclude_nouns = config.exclude_nouns
        if exclude_nouns is None:
            exclude_nouns = OFFICIAL_EN_STOP_WORDS
        self.exclude_nouns = tuple(noun.upper() for noun in exclude_nouns)

    def ensure_resources(self) -> None:
        for resource_name in REQUIRED_NLTK_RESOURCES:
            _download_if_not_exists(resource_name)
        nltk.corpus.brown.ensure_loaded()
        nltk.corpus.treebank.ensure_loaded()

    def extract(self, text: str) -> tuple[str, ...]:
        self.ensure_resources()
        blob = TextBlob(text)
        proper_nouns = tuple(token[0].upper() for token in blob.tags if token[1] == "NNP")  # type: ignore[attr-defined]
        phrases: set[str] = set()
        for noun_phrase in blob.noun_phrases:  # type: ignore[attr-defined]
            tagged = self.tag_noun_phrase(str(noun_phrase), all_proper_nouns=proper_nouns)
            if tagged.keep:
                phrases.add(tagged.cleaned_text)
        return tuple(sorted(phrases))

    def tag_noun_phrase(
        self,
        noun_phrase: str,
        *,
        all_proper_nouns: Iterable[str] = (),
    ) -> TaggedNounPhrase:
        proper_nouns = {token.upper() for token in all_proper_nouns}
        tokens = tuple(token for token in re.split(r"[\s]+", noun_phrase) if token)
        cleaned_tokens = tuple(token for token in tokens if token.upper() not in self.exclude_nouns)
        has_proper_nouns = any(token.upper() in proper_nouns for token in cleaned_tokens)
        has_compound_words = any(
            "-" in token and len(token.strip()) > 1 and len(token.strip().split("-")) > 1
            for token in cleaned_tokens
        )
        has_valid_tokens = all(
            re.match(r"^[a-zA-Z0-9\-]+\n?$", token) is not None
            for token in cleaned_tokens
        ) and all(len(token) <= self.max_word_length for token in cleaned_tokens)
        cleaned_text = self.word_delimiter.join(cleaned_tokens).replace("\n", "").upper()
        return TaggedNounPhrase(
            cleaned_tokens=cleaned_tokens,
            cleaned_text=cleaned_text,
            has_proper_nouns=has_proper_nouns,
            has_compound_words=has_compound_words,
            has_valid_tokens=has_valid_tokens,
        )

    def __str__(self) -> str:
        return f"regex_en_{list(self.exclude_nouns)}_{self.max_word_length}_{self.word_delimiter}"


def _download_if_not_exists(resource_name: str) -> bool:
    root_categories = (
        "corpora",
        "tokenizers",
        "taggers",
        "chunkers",
        "classifiers",
        "stemmers",
        "stopwords",
        "languages",
        "frequent",
        "gate",
        "models",
        "mt",
        "sentiment",
        "similarity",
    )
    for category in root_categories:
        try:
            nltk.find(f"{category}/{resource_name}")
            return True
        except LookupError:
            continue
    nltk.download(resource_name)
    return False


__all__ = [
    "OFFICIAL_EN_STOP_WORDS",
    "OfficialRegexEnglishExtractor",
    "REQUIRED_NLTK_RESOURCES",
    "TaggedNounPhrase",
]
```

- [ ] **Step 4: Wire `extract_regex_english_noun_phrases` to official extractor**

Modify `graph_memory/retrieval/methods/fast_graphrag/noun_phrases.py`.

Replace the body of `extract_regex_english_noun_phrases()` with:

```python
def extract_regex_english_noun_phrases(
    text: str,
    config: FastGraphRAGExtractionConfig,
) -> tuple[NounPhrase, ...]:
    extractor = OfficialRegexEnglishExtractor(config)
    phrases: list[NounPhrase] = []
    for phrase in extractor.extract(text):
        normalized = normalize_noun_phrase_text(phrase, delimiter=config.word_delimiter)
        if not normalized:
            continue
        phrases.append(
            NounPhrase(
                text=phrase,
                normalized_text=normalized,
                source="regex_english",
            )
        )
    return tuple(phrases)
```

Add import:

```python
from graph_memory.retrieval.methods.fast_graphrag.official_nltk import (
    OFFICIAL_EN_STOP_WORDS,
    OfficialRegexEnglishExtractor,
)
```

Replace `_DEFAULT_EXCLUDE_NOUNS` with:

```python
_DEFAULT_EXCLUDE_NOUNS = frozenset(noun.casefold() for noun in OFFICIAL_EN_STOP_WORDS)
```

- [ ] **Step 5: Run extractor tests**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_official_nltk.py tests/test_fast_graphrag_official_alignment.py
```

Expected:

```text
passed
```

If the test environment has no NLTK data and network is not allowed, run:

```powershell
uv run python scripts/bootstrap_fast_graphrag_nltk.py
uv run pytest -q tests/test_fast_graphrag_official_nltk.py tests/test_fast_graphrag_official_alignment.py
```

- [ ] **Step 6: Commit**

```powershell
git add graph_memory/retrieval/methods/fast_graphrag/official_nltk.py graph_memory/retrieval/methods/fast_graphrag/noun_phrases.py tests/test_fast_graphrag_official_nltk.py tests/test_fast_graphrag_official_alignment.py
git commit -m "feat: align fast-graphrag regex extractor with official nltk path"
```

### Task 3: Align Extraction Config Defaults With Official Text Analyzer Defaults

**Files:**

- Modify: `graph_memory/retrieval/methods/fast_graphrag/config.py`
- Modify: `graph_memory/registry/retrieval.py`
- Test: `tests/test_registry_stage_configs.py`

- [ ] **Step 1: Add config default test**

Modify the FastGraphRAG stage config test in `tests/test_registry_stage_configs.py` to assert:

```python
from graph_memory.retrieval.methods.fast_graphrag.official_nltk import OFFICIAL_EN_STOP_WORDS

assert config.job.extraction.extractor_type == "regex_english"
assert config.job.extraction.max_word_length == 15
assert config.job.extraction.word_delimiter == " "
assert config.job.extraction.include_named_entities is True
assert config.job.extraction.exclude_nouns is None
assert config.job.extraction.exclude_entity_tags == ("DATE",)
assert config.job.extraction.exclude_pos_tags == ("DET", "PRON", "INTJ", "X")
assert config.job.extraction.noun_phrase_tags == ("PROPN", "NOUNS")
assert config.job.extraction.noun_phrase_grammars == {
    "PROPN,PROPN": "PROPN",
    "NOUN,NOUN": "NOUNS",
    "NOUNS,NOUN": "NOUNS",
    "ADJ,ADJ": "ADJ",
    "ADJ,NOUN": "NOUNS",
}
assert tuple(noun.casefold() for noun in OFFICIAL_EN_STOP_WORDS) == (
    "stuff",
    "thing",
    "things",
    "bunch",
    "bit",
    "bits",
    "people",
    "person",
    "okay",
    "hey",
    "hi",
    "hello",
    "laughter",
    "oh",
)
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```powershell
uv run pytest -q tests/test_registry_stage_configs.py
```

Expected failure before implementation:

```text
AssertionError
```

At least `exclude_entity_tags`, `exclude_pos_tags`, `noun_phrase_tags`, or `noun_phrase_grammars` should differ from official defaults.

- [ ] **Step 3: Update config defaults**

Modify `graph_memory/retrieval/methods/fast_graphrag/config.py`:

```python
def _official_noun_phrase_grammars() -> Mapping[str, str]:
    return {
        "PROPN,PROPN": "PROPN",
        "NOUN,NOUN": "NOUNS",
        "NOUNS,NOUN": "NOUNS",
        "ADJ,ADJ": "ADJ",
        "ADJ,NOUN": "NOUNS",
    }


@dataclass(frozen=True)
class FastGraphRAGExtractionConfig:
    extractor_type: FastGraphRAGExtractorType = "regex_english"
    normalize_edge_weights: bool = True
    max_word_length: int = 15
    word_delimiter: str = " "
    include_named_entities: bool = True
    exclude_nouns: tuple[str, ...] | None = None
    exclude_entity_tags: tuple[str, ...] = ("DATE",)
    exclude_pos_tags: tuple[str, ...] = ("DET", "PRON", "INTJ", "X")
    noun_phrase_tags: tuple[str, ...] = ("PROPN", "NOUNS")
    noun_phrase_grammars: Mapping[str, str] = field(default_factory=_official_noun_phrase_grammars)
    model_name: str = "en_core_web_md"
```

- [ ] **Step 4: Run config tests**

Run:

```powershell
uv run pytest -q tests/test_registry_stage_configs.py tests/test_fast_graphrag_registry.py
```

Expected:

```text
passed
```

- [ ] **Step 5: Commit**

```powershell
git add graph_memory/retrieval/methods/fast_graphrag/config.py graph_memory/registry/retrieval.py tests/test_registry_stage_configs.py
git commit -m "feat: align fast-graphrag extraction defaults with official config"
```

### Task 4: Replace Max-Count Edge Normalization With Official PMI Weighting

**Files:**

- Create: `graph_memory/retrieval/methods/fast_graphrag/edge_weights.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/index.py`
- Test: `tests/test_fast_graphrag_edge_weights.py`
- Test: `tests/test_fast_graphrag_index.py`

- [ ] **Step 1: Add PMI tests**

Create `tests/test_fast_graphrag_edge_weights.py`:

```python
from __future__ import annotations

import math

from graph_memory.retrieval.methods.fast_graphrag.edge_weights import pmi_edge_weight


def test_pmi_edge_weight_matches_official_formula() -> None:
    # p(x,y) = 2 / 10 = 0.2
    # p(x) = 4 / 20 = 0.2
    # p(y) = 5 / 20 = 0.25
    # weight = p(x,y) * log2(p(x,y) / (p(x) * p(y)))
    weight = pmi_edge_weight(
        edge_count=2,
        total_edge_weights=10,
        source_frequency=4,
        target_frequency=5,
        total_frequency_occurrences=20,
    )

    assert math.isclose(weight, 0.4, rel_tol=1e-12)


def test_pmi_edge_weight_returns_zero_for_empty_denominator() -> None:
    assert pmi_edge_weight(
        edge_count=0,
        total_edge_weights=0,
        source_frequency=0,
        target_frequency=0,
        total_frequency_occurrences=0,
    ) == 0.0
```

- [ ] **Step 2: Add index-level normalized weight test**

Modify `tests/test_fast_graphrag_index.py`:

```python
from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGConfig


def test_build_fast_graphrag_kg_uses_pmi_when_normalizing_edge_weights() -> None:
    request = TextRankingRequest(
        task_id="task-1",
        query_text="Richmond census",
        candidates=(
            TextCandidate("m0", "Richmond census-designated place in Maine.", {"position": 0}),
            TextCandidate("m1", "Richmond population census result.", {"position": 1}),
            TextCandidate("m2", "Maine river town.", {"position": 2}),
        ),
    )
    graph: MemoryGraph = {
        "task_id": "task-1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": request.query_text},
            {
                "id": "m0",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": request.candidates[0].text,
            },
            {
                "id": "m1",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": request.candidates[1].text,
            },
            {
                "id": "m2",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": request.candidates[2].text,
            },
        ],
        "edges": [],
    }

    kg = build_fast_graphrag_knowledge_graph(
        request,
        graph,
        config=FastGraphRAGConfig(),
    )

    weights = [relation.weight for relation in kg.relations]
    assert weights
    assert any(weight != 1.0 for weight in weights)
```

This test intentionally checks behavior, not a brittle phrase pair, because TextBlob extraction can produce multiple noun phrases.

- [ ] **Step 3: Run tests and confirm failure**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_edge_weights.py tests/test_fast_graphrag_index.py::test_build_fast_graphrag_kg_uses_pmi_when_normalizing_edge_weights
```

Expected before implementation:

```text
ModuleNotFoundError: No module named 'graph_memory.retrieval.methods.fast_graphrag.edge_weights'
```

- [ ] **Step 4: Implement PMI helper**

Create `graph_memory/retrieval/methods/fast_graphrag/edge_weights.py`:

```python
from __future__ import annotations

import math


def pmi_edge_weight(
    *,
    edge_count: int,
    total_edge_weights: int,
    source_frequency: int,
    target_frequency: int,
    total_frequency_occurrences: int,
) -> float:
    if (
        edge_count <= 0
        or total_edge_weights <= 0
        or source_frequency <= 0
        or target_frequency <= 0
        or total_frequency_occurrences <= 0
    ):
        return 0.0
    prop_weight = edge_count / total_edge_weights
    source_prop = source_frequency / total_frequency_occurrences
    target_prop = target_frequency / total_frequency_occurrences
    denominator = source_prop * target_prop
    if denominator <= 0.0:
        return 0.0
    return prop_weight * math.log2(prop_weight / denominator)


__all__ = ["pmi_edge_weight"]
```

- [ ] **Step 5: Apply PMI in relation aggregation**

Modify `graph_memory/retrieval/methods/fast_graphrag/index.py`.

Change `_relations_from_mentions()` to accept entity frequency:

```python
def _relations_from_mentions(
    mentions: Sequence[EntityMention],
    alias_owner: dict[str, str],
    entity_name_by_id: dict[str, str],
    entity_frequency_by_id: dict[str, int],
    *,
    normalize_edge_weights: bool,
) -> tuple[FastGraphRAGRelation, ...]:
```

In `build_fast_graphrag_knowledge_graph()`, compute:

```python
entity_frequency_by_id = {
    entity.entity_id: len(entity.candidate_ids)
    for entity in entities
}
```

Pass it to `_relations_from_mentions(...)`.

Inside `_relations_from_mentions()`, replace max-count normalization with:

```python
total_edge_weights = sum(len(candidate_ids) for candidate_ids in candidate_ids_by_pair.values())
total_frequency_occurrences = sum(entity_frequency_by_id.values())
```

Then compute each relation weight:

```python
count = len(candidate_ids)
weight = float(count)
if normalize_edge_weights:
    weight = pmi_edge_weight(
        edge_count=count,
        total_edge_weights=total_edge_weights,
        source_frequency=entity_frequency_by_id.get(source_id, 0),
        target_frequency=entity_frequency_by_id.get(target_id, 0),
        total_frequency_occurrences=total_frequency_occurrences,
    )
```

Add import:

```python
from graph_memory.retrieval.methods.fast_graphrag.edge_weights import pmi_edge_weight
```

- [ ] **Step 6: Update old relation weight assertions**

If `tests/test_fast_graphrag_index.py::test_build_fast_graphrag_kg_aggregates_cooccurring_entity_pairs_across_text_units` asserts `matching[0].weight == 1.0`, change it to:

```python
assert matching[0].weight > 0.0
```

Add a non-normalized test to preserve raw count semantics:

```python
def test_build_fast_graphrag_kg_can_keep_raw_cooccurrence_count_weights() -> None:
    config = FastGraphRAGConfig(
        extraction=FastGraphRAGExtractionConfig(normalize_edge_weights=False)
    )
    kg = build_fast_graphrag_knowledge_graph(request, graph, config=config)
    matching = [
        relation
        for relation in kg.relations
        if {relation.source_entity_id, relation.target_entity_id}
        == {"e:prime-minister", "e:nuclear-energy-policy"}
    ]
    assert matching[0].weight == 2.0
```

Use the existing relation aggregation fixture's `request` and `graph` setup when adding this test.

- [ ] **Step 7: Run edge/index tests**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_edge_weights.py tests/test_fast_graphrag_index.py
```

Expected:

```text
passed
```

- [ ] **Step 8: Commit**

```powershell
git add graph_memory/retrieval/methods/fast_graphrag/edge_weights.py graph_memory/retrieval/methods/fast_graphrag/index.py tests/test_fast_graphrag_edge_weights.py tests/test_fast_graphrag_index.py
git commit -m "feat: use official pmi weights for fast-graphrag noun graph"
```

### Task 5: Align Pruning Defaults And Ego-Node Behavior

**Files:**

- Modify: `graph_memory/retrieval/methods/fast_graphrag/config.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/pruning.py`
- Test: `tests/test_fast_graphrag_pruning.py`
- Test: `tests/test_registry_stage_configs.py`

- [ ] **Step 1: Add official pruning default test**

Modify `tests/test_registry_stage_configs.py`:

```python
assert config.job.pruning.min_node_freq == 1
assert config.job.pruning.min_node_degree == 1
assert config.job.pruning.min_edge_weight_pct == 40.0
assert config.job.pruning.remove_ego_nodes is False
assert config.job.pruning.lcc_only is False
```

- [ ] **Step 2: Add ego-node pruning behavior test**

Append to `tests/test_fast_graphrag_pruning.py`:

```python
def test_prune_knowledge_graph_can_remove_highest_degree_ego_node() -> None:
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:a", "A", "a", "noun_phrase", "", ("m0", "m1", "m2")),
            FastGraphRAGEntity("e:b", "B", "b", "noun_phrase", "", ("m0",)),
            FastGraphRAGEntity("e:c", "C", "c", "noun_phrase", "", ("m1",)),
            FastGraphRAGEntity("e:d", "D", "d", "noun_phrase", "", ("m2",)),
        ),
        relations=(
            FastGraphRAGRelation("r:a:b", "e:a", "e:b", "", ("m0",), 1.0),
            FastGraphRAGRelation("r:a:c", "e:a", "e:c", "", ("m1",), 1.0),
            FastGraphRAGRelation("r:a:d", "e:a", "e:d", "", ("m2",), 1.0),
        ),
    )

    pruned = prune_knowledge_graph(
        kg,
        FastGraphRAGPruningConfig(
            min_node_degree=0,
            min_edge_weight_pct=0.0,
            remove_ego_nodes=True,
        ),
    )

    assert "e:a" not in {entity.entity_id for entity in pruned.entities}
    assert pruned.relations == ()
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_pruning.py tests/test_registry_stage_configs.py
```

Expected before implementation:

```text
ValueError: remove_ego_nodes=True is not supported
```

or default assertion mismatch.

- [ ] **Step 4: Update pruning config defaults**

Modify `FastGraphRAGPruningConfig` in `graph_memory/retrieval/methods/fast_graphrag/config.py`:

```python
@dataclass(frozen=True)
class FastGraphRAGPruningConfig:
    min_node_freq: int = 1
    max_node_freq_std: float | None = None
    min_node_degree: int = 1
    max_node_degree_std: float | None = None
    min_edge_weight_pct: float = 40.0
    remove_ego_nodes: bool = False
    lcc_only: bool = False
```

- [ ] **Step 5: Implement ego-node removal**

Modify `graph_memory/retrieval/methods/fast_graphrag/pruning.py`.

Remove the current rejection:

```python
if config.remove_ego_nodes:
    raise ValueError(...)
```

After degree calculation, add:

```python
if config.remove_ego_nodes and degree_by_entity:
    ego_entity_id = max(degree_by_entity, key=lambda entity_id: degree_by_entity[entity_id])
    removed_entity_ids.add(ego_entity_id)
```

Apply degree and frequency pruning after that, matching official order:

```text
compute degree over original relations
remove ego node if configured
remove low/high degree nodes
compute frequency thresholds over entities that survived degree filtering
remove low/high frequency nodes
filter relations to surviving endpoints
filter low edge weights by percentile
optionally keep largest connected component
```

- [ ] **Step 6: Run pruning tests**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_pruning.py tests/test_registry_stage_configs.py
```

Expected:

```text
passed
```

- [ ] **Step 7: Commit**

```powershell
git add graph_memory/retrieval/methods/fast_graphrag/config.py graph_memory/retrieval/methods/fast_graphrag/pruning.py tests/test_fast_graphrag_pruning.py tests/test_registry_stage_configs.py
git commit -m "feat: align fast-graphrag pruning with official defaults"
```

### Task 6: Preserve Official Entity And Relationship Table Semantics Internally

**Files:**

- Modify: `graph_memory/retrieval/methods/fast_graphrag/index.py`
- Modify: `graph_memory/retrieval/methods/fast_graphrag/nlp.py`
- Test: `tests/test_fast_graphrag_index.py`

- [ ] **Step 1: Add entity shape test**

Append to `tests/test_fast_graphrag_index.py`:

```python
def test_fast_graphrag_entities_preserve_official_title_frequency_and_empty_description() -> None:
    request = TextRankingRequest(
        task_id="task-1",
        query_text="Richmond census",
        candidates=(
            TextCandidate("m0", "Richmond census-designated place in Maine.", {"position": 0}),
            TextCandidate("m1", "Richmond population census result.", {"position": 1}),
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

    kg = build_fast_graphrag_knowledge_graph(request, graph, config=FastGraphRAGConfig())

    assert kg.entities
    assert all(entity.entity_type == "NOUN PHRASE" for entity in kg.entities)
    assert all(entity.description == "" for entity in kg.entities)
    assert all(entity.name == entity.name.upper() for entity in kg.entities)
    assert all(entity.candidate_ids for entity in kg.entities)
```

- [ ] **Step 2: Run and confirm failure**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_index.py::test_fast_graphrag_entities_preserve_official_title_frequency_and_empty_description
```

Expected before implementation:

```text
AssertionError
```

Current entity type is likely `noun_phrase`, and description is likely a preferred mention instead of empty.

- [ ] **Step 3: Update entity conversion**

Modify `graph_memory/retrieval/methods/fast_graphrag/index.py` entity creation:

```python
FastGraphRAGEntity(
    entity_id=entity.entity_id,
    name=entity.name.upper(),
    normalized_name=entity.normalized_name,
    entity_type="NOUN PHRASE" if entity.entity_type == "noun_phrase" else entity.entity_type,
    description="" if entity.entity_type == "noun_phrase" else entity.description,
    candidate_ids=entity.candidate_ids,
)
```

If title/source_ref legacy mentions remain enabled, document that they are repo-specific augmentation and keep their entity types unchanged. If this test fails because title/source_ref mentions appear, narrow the assertion to noun-phrase entities:

```python
noun_entities = [entity for entity in kg.entities if entity.entity_type == "NOUN PHRASE"]
assert noun_entities
assert all(entity.description == "" for entity in noun_entities)
```

- [ ] **Step 4: Run index tests**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_index.py
```

Expected:

```text
passed
```

- [ ] **Step 5: Commit**

```powershell
git add graph_memory/retrieval/methods/fast_graphrag/index.py graph_memory/retrieval/methods/fast_graphrag/nlp.py tests/test_fast_graphrag_index.py
git commit -m "feat: preserve official noun graph table semantics"
```

### Task 7: Add Official-Alignment Debug Artifact For A Single Task

**Files:**

- Modify: `graph_memory/retrieval/methods/fast_graphrag/index.py`
- Create: `tests/test_fast_graphrag_official_debug.py`

- [ ] **Step 1: Write debug snapshot test**

Create `tests/test_fast_graphrag_official_debug.py`:

```python
from __future__ import annotations

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGConfig
from graph_memory.retrieval.methods.fast_graphrag.index import (
    build_fast_graphrag_knowledge_graph,
    official_noun_graph_snapshot,
)
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest


def test_official_noun_graph_snapshot_exposes_entities_and_relationships() -> None:
    request = TextRankingRequest(
        task_id="task-1",
        query_text="Richmond census",
        candidates=(
            TextCandidate("m0", "Richmond census-designated place in Maine.", {"position": 0}),
            TextCandidate("m1", "Richmond population census result.", {"position": 1}),
        ),
    )
    graph = {
        "task_id": "task-1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": request.query_text},
            {"id": "m0", "node_type": "graph_item", "node_kind": "document_sentence", "text": request.candidates[0].text},
            {"id": "m1", "node_type": "graph_item", "node_kind": "document_sentence", "text": request.candidates[1].text},
        ],
        "edges": [],
    }
    kg = build_fast_graphrag_knowledge_graph(request, graph, config=FastGraphRAGConfig())

    snapshot = official_noun_graph_snapshot(kg)

    assert set(snapshot) == {"entities", "relationships"}
    assert snapshot["entities"]
    assert all(set(row) == {"title", "frequency", "text_unit_ids", "type", "description"} for row in snapshot["entities"])
    assert all(set(row) == {"source", "target", "weight", "text_unit_ids", "description"} for row in snapshot["relationships"])
```

- [ ] **Step 2: Implement snapshot helper**

Modify `graph_memory/retrieval/methods/fast_graphrag/index.py`:

```python
def official_noun_graph_snapshot(kg: FastGraphRAGKnowledgeGraph) -> dict[str, list[dict[str, object]]]:
    entities = [
        {
            "title": entity.name,
            "frequency": len(entity.candidate_ids),
            "text_unit_ids": list(entity.candidate_ids),
            "type": entity.entity_type,
            "description": entity.description,
        }
        for entity in kg.entities
    ]
    relationships = [
        {
            "source": _entity_name_by_id(kg).get(relation.source_entity_id, relation.source_entity_id),
            "target": _entity_name_by_id(kg).get(relation.target_entity_id, relation.target_entity_id),
            "weight": relation.weight,
            "text_unit_ids": list(relation.candidate_ids),
            "description": relation.description,
        }
        for relation in kg.relations
    ]
    return {"entities": entities, "relationships": relationships}


def _entity_name_by_id(kg: FastGraphRAGKnowledgeGraph) -> dict[str, str]:
    return {entity.entity_id: entity.name for entity in kg.entities}
```

Export it:

```python
__all__ = ["build_fast_graphrag_knowledge_graph", "official_noun_graph_snapshot"]
```

- [ ] **Step 3: Run debug tests**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_official_debug.py
```

Expected:

```text
passed
```

- [ ] **Step 4: Commit**

```powershell
git add graph_memory/retrieval/methods/fast_graphrag/index.py tests/test_fast_graphrag_official_debug.py
git commit -m "test: expose fast-graphrag official noun graph snapshot"
```

### Task 8: Keep No-LLM Boundary Explicit

**Files:**

- Modify: `tests/test_fast_graphrag_no_llm_boundary.py`
- Test: `tests/test_fast_graphrag_no_llm_boundary.py`

- [ ] **Step 1: Update allowed and forbidden terms**

Keep these forbidden terms in source under `graph_memory/retrieval/methods/fast_graphrag`:

```python
FORBIDDEN_PATTERNS = (
    "openai",
    "prompt",
    "completion",
    "chat",
    "community_report",
    "summarize",
)
```

Do not forbid these terms:

```python
ALLOWED_OFFICIAL_NLP_TERMS = (
    "nltk",
    "textblob",
    "noun_phrase",
    "extract_graph_nlp",
    "prune_graph",
    "PMI",
)
```

The test does not need to define `ALLOWED_OFFICIAL_NLP_TERMS`; the point is that the forbidden list must not reject official NLP implementation names.

- [ ] **Step 2: Run boundary test**

Run:

```powershell
uv run pytest -q tests/test_fast_graphrag_no_llm_boundary.py
```

Expected:

```text
passed
```

- [ ] **Step 3: Commit**

```powershell
git add tests/test_fast_graphrag_no_llm_boundary.py
git commit -m "test: keep official fast-graphrag nltk path no-llm"
```

### Task 9: Run End-To-End Verification Gates

**Files:**

- No new source files unless verification exposes failures.

- [ ] **Step 1: Run FastGraphRAG unit tests**

Run:

```powershell
$files = Get-ChildItem -LiteralPath tests -Filter 'test_fast_graphrag_*.py' | ForEach-Object { $_.FullName }
uv run pytest -q @files
```

Expected:

```text
passed
```

- [ ] **Step 2: Run registry/workflow contract tests**

Run:

```powershell
uv run pytest -q tests/test_method_registry.py tests/test_registry_stage_configs.py tests/test_workflow_orchestration.py tests/test_config_run_retrieval.py
```

Expected:

```text
passed
```

- [ ] **Step 3: Run lint**

Run:

```powershell
uv run ruff check graph_memory/retrieval/methods/fast_graphrag graph_memory/registry tests/test_fast_graphrag_*.py tests/test_registry_stage_configs.py tests/test_config_run_retrieval.py scripts/bootstrap_fast_graphrag_nltk.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 4: Run type check**

Run:

```powershell
uv run basedpyright --level error graph_memory/retrieval/methods/fast_graphrag graph_memory/registry tests/test_fast_graphrag_*.py tests/test_registry_stage_configs.py tests/test_config_run_retrieval.py scripts/bootstrap_fast_graphrag_nltk.py
```

Expected:

```text
0 errors
```

- [ ] **Step 5: Run a real one-task smoke**

Use an existing HotpotQA graph record and route it through `FastGraphRAGMethod` with a fake dense scorer, so the smoke validates the official NLP graph path without loading a sentence-transformer model.

Run:

```powershell
@'
import json
from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGConfig
from graph_memory.retrieval.methods.fast_graphrag.index import build_fast_graphrag_knowledge_graph
from graph_memory.retrieval.methods.fast_graphrag.method import FastGraphRAGMethod
from graph_memory.retrieval.requests import FastGraphRAGEntity, FastGraphRAGRequest, TextCandidate, TextRankingRequest


class ZeroDenseScorer:
    def score_entities(self, query_text: str, entities: object) -> dict[str, float]:
        return {entity.entity_id: 0.0 for entity in entities}

    def score_candidates(self, query_text: str, candidates: object) -> dict[str, float]:
        return {candidate.item_id: 0.0 for candidate in candidates}


with open("results/hp-fastgraphrag/graphs/test.graphs.json", encoding="utf-8") as file:
    graph = json.load(file)[0]

query_text = next(node["text"] for node in graph["nodes"] if node["id"] == "q")
candidates = tuple(
    TextCandidate(
        item_id=str(node["id"]),
        text=str(node["text"]),
        metadata=node.get("metadata", {}),
    )
    for node in graph["nodes"]
    if node["id"] != "q"
)
ranking_request = TextRankingRequest(
    task_id=str(graph["task_id"]),
    query_text=query_text,
    candidates=candidates,
)
config = FastGraphRAGConfig()
kg = build_fast_graphrag_knowledge_graph(ranking_request, graph, config=config)
method_request = FastGraphRAGRequest(
    task_id=ranking_request.task_id,
    query_text=ranking_request.query_text,
    candidates=ranking_request.candidates,
    candidate_graph=graph,
    knowledge_graph=kg,
)
result = FastGraphRAGMethod(
    name="fast_graphrag",
    config=config,
    dense_ranker=ZeroDenseScorer(),
).rank_task(method_request, top_k=10)

print(f"task_id={ranking_request.task_id}")
print(f"entities={len(kg.entities)}")
print(f"relations={len(kg.relations)}")
print(f"top10={[node.node_id for node in result.ranked_nodes[:10]]}")
print(f"retrieved_edges={len(result.trace.retrieved_edges)}")
'@ | uv run python -
```

Expected output shape:

```text
task_id=<id>
entities=<positive integer>
relations=<positive integer>
top10=<list of candidate ids>
retrieved_edges=<non-negative integer>
```

If NLTK resources are missing, run:

```powershell
uv run python scripts/bootstrap_fast_graphrag_nltk.py
```

Then rerun the smoke.

## 6. Acceptance Criteria

The implementation is complete only when all of these are true:

1. `regex_english` no longer uses the hand-rolled n-gram segmenter by default.
2. `regex_english` uses `TextBlob.noun_phrases` and `TextBlob.tags`, backed by NLTK resources.
3. Official stop words are used when `exclude_nouns is None`.
4. Official phrase filtering semantics are covered by tests: proper noun, multiword phrase, compound word, excluded-token cleanup, invalid token, max word length.
5. Entity rows preserve official noun-graph table semantics: uppercase title, frequency from text unit ids, `type="NOUN PHRASE"`, empty description.
6. Relationship rows preserve official co-occurrence semantics and text unit provenance.
7. `normalize_edge_weights=True` applies official PMI weighting.
8. `normalize_edge_weights=False` preserves raw co-occurrence counts.
9. Pruning defaults and ego-node behavior match official operation semantics unless a test documents a deliberate repo boundary.
10. No LLM, prompt, community report, answer generation, or future LLM hook is added.
11. FastGraphRAG tests, registry/workflow tests, ruff, and basedpyright pass.

## 7. Explicit Non-Goals

- Do not add official GraphRAG SDK calls as runtime dependencies.
- Do not raise this repo's required Python version above 3.10.
- Do not add a dependency whose resolved version requires Python 3.11+ on the server path.
- Do not mirror official `GraphRagConfig` or table-provider APIs into this repo.
- Do not add LLM community reports.
- Do not add prompt templates.
- Do not add generated answers.
- Do not consume `answer`, `supporting_facts`, `evidences`, `evidences_id`, or `gold_dependency_edges` during retrieval.
- Do not mix graph-rerank candidate graph components into this baseline plan; that is a separate retrieval adaptation, not official FastGraphRAG indexing alignment.
- Do not tune metrics in the same change.

## 8. Notes For Result Interpretation

After this plan is implemented, `fast_graphrag` should be described as:

```text
FastGraphRAG official-NLP-indexing baseline adapted to this repo's no-LLM evidence retrieval contract.
```

It should not be described as:

```text
Full Microsoft GraphRAG FastGraphRAG query pipeline.
```

The reason is simple: official FastGraphRAG still relies on community reports and GraphRAG query stages for its standard user-facing behavior, while this repo intentionally returns evidence rankings without LLM-generated reports or answers.
