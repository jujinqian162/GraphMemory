from __future__ import annotations

import re
import unicodedata

from graph_memory.retrieval.methods.graphrag.config import GraphRAGConfig
from graph_memory.text.tokens import STOPWORDS

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")
_PROPER_SPAN = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)?"
    + r"(?:\s+(?:of\s+|the\s+|and\s+)?"
    + r"[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)?)*"
)
_STRUCTURED = re.compile(
    r"https?://[^\s\]\[\)\}\>\"']+"
    + r"|(?:/[A-Za-z0-9_.@%+,:=~-]+){2,}"
    + r"|\b[A-Za-z0-9_.-]{3,}\.(?:jsonl?|md|txt|csv|ya?ml|py|sh|html|pdf|xml|toml)\b"
    + r"|\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){1,}\b"
)
_CLAUSE_SEPARATOR = re.compile(r"[\n.!?;:,[\]{}()]+")

# RegexEnglish in FastGraphRAG uses POS tagging. This native implementation keeps
# the same filtering intent without runtime corpus/model downloads.
_EXCLUDED = STOPWORDS | {
    "about",
    "after",
    "again",
    "all",
    "also",
    "another",
    "any",
    "before",
    "both",
    "can",
    "could",
    "did",
    "do",
    "does",
    "done",
    "each",
    "either",
    "else",
    "arguments",
    "assistant",
    "content",
    "file",
    "files",
    "first",
    "had",
    "have",
    "here",
    "how",
    "however",
    "just",
    "later",
    "message",
    "more",
    "most",
    "n",
    "now",
    "only",
    "other",
    "output",
    "please",
    "position",
    "really",
    "role",
    "s",
    "same",
    "should",
    "some",
    "then",
    "there",
    "tool",
    "these",
    "those",
    "very",
    "will",
    "would",
    "actually",
    "context",
    "result",
    "results",
    "thing",
    "things",
    "stuff",
    "okay",
    "hello",
}
_VERB_LIKE = {
    "add",
    "added",
    "adding",
    "build",
    "built",
    "call",
    "called",
    "change",
    "changed",
    "check",
    "checked",
    "create",
    "created",
    "delete",
    "deleted",
    "fail",
    "failed",
    "find",
    "fix",
    "fixed",
    "fixing",
    "found",
    "get",
    "got",
    "include",
    "included",
    "make",
    "made",
    "move",
    "moved",
    "need",
    "needed",
    "read",
    "remove",
    "removed",
    "replace",
    "replaced",
    "return",
    "returned",
    "run",
    "save",
    "saved",
    "set",
    "show",
    "shown",
    "update",
    "updated",
    "identify",
    "identified",
    "let",
    "use",
    "used",
    "write",
    "wrote",
}


def normalize_entity_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"[^\w./:-]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split()).strip(".,;:")


def extract_noun_phrases(text: str, config: GraphRAGConfig) -> tuple[str, ...]:
    """Extract deterministic FastGraphRAG-style noun/structured phrases."""

    weighted: dict[str, tuple[int, int]] = {}

    def add(raw: str, priority: int) -> None:
        normalized = normalize_entity_text(raw)
        tokens = normalized.split()
        if not tokens or len(tokens) > config.max_entity_words:
            return
        if priority > 0 and any(
            len(token) > config.max_word_length for token in tokens
        ):
            return
        if all(token in _EXCLUDED for token in tokens):
            return
        if len(tokens) == 1 and tokens[0] in _VERB_LIKE:
            return
        if not all(re.fullmatch(r"[\w./:-]+", token) for token in tokens):
            return
        current = weighted.get(normalized)
        value = (priority, -len(normalized))
        if current is None or value < current:
            weighted[normalized] = value

    for match in _STRUCTURED.finditer(text):
        add(match.group(0).rstrip(".,;:"), 0)

    for match in _PROPER_SPAN.finditer(text):
        surface = " ".join(match.group(0).split())
        normalized = normalize_entity_text(surface)
        if len(normalized.split()) > 1 or normalized not in _EXCLUDED:
            add(surface, 1)

    for clause in _CLAUSE_SEPARATOR.split(text):
        words = [match.group(0) for match in _WORD.finditer(clause)]
        segment: list[str] = []
        segments: list[list[str]] = []
        for word in words:
            normalized = normalize_entity_text(word)
            if (
                not normalized
                or normalized in _EXCLUDED
                or normalized in _VERB_LIKE
                or len(normalized) > config.max_word_length
            ):
                if segment:
                    segments.append(segment)
                    segment = []
                continue
            segment.append(word)
        if segment:
            segments.append(segment)

        for items in segments:
            for size in (3, 2):
                for start in range(0, len(items) - size + 1):
                    add(" ".join(items[start : start + size]), 2)
            for item in items:
                if "-" in item or "_" in item or _looks_like_identifier(item):
                    add(item, 1)

    ordered = sorted(weighted, key=lambda phrase: (*weighted[phrase], phrase))
    return tuple(ordered[: config.max_entities_per_text_unit])


def _looks_like_identifier(value: str) -> bool:
    return any(character.isupper() for character in value[1:]) or (
        any(character.isdigit() for character in value)
        and any(character.isalpha() for character in value)
    )


__all__ = ["extract_noun_phrases", "normalize_entity_text"]
