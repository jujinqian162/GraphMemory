from __future__ import annotations

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
    "community_report",
    "summarize",
)


def test_fast_graphrag_code_has_no_llm_or_prompt_boundary() -> None:
    root = Path("graph_memory/retrieval/methods/fast_graphrag")
    sources = list(root.glob("*.py"))
    assert sources
    combined = "\n".join(path.read_text(encoding="utf-8").casefold() for path in sources)
    for pattern in FORBIDDEN_PATTERNS:
        assert pattern not in combined
