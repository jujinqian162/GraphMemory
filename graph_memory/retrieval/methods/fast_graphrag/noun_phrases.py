from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, cast

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGExtractionConfig

NounPhraseSource = Literal["regex_english", "spacy_noun_chunk", "spacy_entity"]

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_INTERNAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "after",
        "before",
        "between",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "who",
        "which",
        "with",
    }
)
_BOUNDARY_WORDS = _INTERNAL_STOPWORDS | frozenset(
    {
        "approved",
        "approves",
        "born",
        "defended",
        "defends",
        "hosted",
        "mentions",
        "performed",
        "repeats",
    }
)
_DEFAULT_EXCLUDE_NOUNS = frozenset({"example", "examples", "item", "items", "thing", "things"})


@dataclass(frozen=True)
class NounPhrase:
    text: str
    normalized_text: str
    source: NounPhraseSource


def extract_regex_english_noun_phrases(
    text: str,
    config: FastGraphRAGExtractionConfig,
) -> tuple[NounPhrase, ...]:
    phrases: list[NounPhrase] = []
    seen: set[str] = set()
    for segment in _content_token_segments(text):
        for start in range(len(segment)):
            max_end = min(len(segment), start + 4)
            min_end = start + 1 if len(segment) == 1 else start + 2
            for end in range(max_end, min_end - 1, -1):
                words = segment[start:end]
                if len(words) == 1 and not _allows_single_token(words[0]):
                    continue
                _append_phrase(
                    phrases,
                    seen,
                    " ".join(word.text for word in words),
                    source="regex_english",
                    config=config,
                )
    return tuple(phrases)


def extract_spacy_noun_phrases(
    text: str,
    config: FastGraphRAGExtractionConfig,
    nlp: Callable[[str], object] | None,
) -> tuple[NounPhrase, ...]:
    if nlp is None:
        raise ValueError("FastGraphRAG spaCy extraction requires an nlp object or preloaded model.")

    doc = nlp(text)
    phrases: list[NounPhrase] = []
    seen: set[str] = set()
    noun_chunks = cast(Iterable[object], getattr(doc, "noun_chunks", ()))
    for span in noun_chunks:
        if _span_filtered(span, config):
            continue
        _append_phrase(
            phrases,
            seen,
            str(getattr(span, "text", "")),
            source="spacy_noun_chunk",
            config=config,
        )
    if config.include_named_entities:
        entities = cast(Iterable[object], getattr(doc, "ents", ()))
        for entity in entities:
            if _entity_filtered(entity, config):
                continue
            _append_phrase(
                phrases,
                seen,
                str(getattr(entity, "text", "")),
                source="spacy_entity",
                config=config,
            )
    return tuple(phrases)


@dataclass(frozen=True)
class _Token:
    text: str
    normalized: str


def _content_token_segments(text: str) -> tuple[tuple[_Token, ...], ...]:
    segments: list[tuple[_Token, ...]] = []
    current: list[_Token] = []
    previous_end = 0
    for match in _TOKEN_RE.finditer(text):
        separator = text[previous_end : match.start()]
        if re.search(r"[.!?;:,]", separator) and current:
            segments.append(tuple(current))
            current = []
        token_text = match.group(0)
        normalized = _normalize_for_filter(token_text)
        if not normalized or normalized in _BOUNDARY_WORDS:
            if current:
                segments.append(tuple(current))
                current = []
            previous_end = match.end()
            continue
        current.append(_Token(text=token_text, normalized=normalized))
        previous_end = match.end()
    if current:
        segments.append(tuple(current))
    return tuple(segments)


def _allows_single_token(token: _Token) -> bool:
    return (
        token.text[:1].isupper()
        or token.text.isupper()
        or any(character.isdigit() for character in token.text)
    )


def _append_phrase(
    phrases: list[NounPhrase],
    seen: set[str],
    text: str,
    *,
    source: NounPhraseSource,
    config: FastGraphRAGExtractionConfig,
) -> None:
    text = " ".join(text.split())
    normalized = normalize_noun_phrase_text(text, delimiter=config.word_delimiter)
    if not _phrase_allowed(text, normalized, config):
        return
    if normalized in seen:
        return
    seen.add(normalized)
    phrases.append(NounPhrase(text=text, normalized_text=normalized, source=source))


def _phrase_allowed(text: str, normalized: str, config: FastGraphRAGExtractionConfig) -> bool:
    if not normalized:
        return False
    words = normalized.split(config.word_delimiter)
    if not words or all(word in _INTERNAL_STOPWORDS for word in words):
        return False
    if any(len(word) > config.max_word_length for word in words):
        return False
    if normalized in _configured_exclude_nouns(config):
        return False
    return bool(text.strip())


def _configured_exclude_nouns(config: FastGraphRAGExtractionConfig) -> frozenset[str]:
    if config.exclude_nouns is None:
        return _DEFAULT_EXCLUDE_NOUNS
    return frozenset(
        normalize_noun_phrase_text(noun, delimiter=config.word_delimiter)
        for noun in config.exclude_nouns
        if noun
    )


def _span_filtered(span: object, config: FastGraphRAGExtractionConfig) -> bool:
    root = getattr(span, "root", None)
    pos = str(getattr(root, "pos_", ""))
    tag = str(getattr(root, "tag_", ""))
    if config.exclude_pos_tags and pos in config.exclude_pos_tags:
        return True
    if config.noun_phrase_tags and tag not in config.noun_phrase_tags:
        return True
    if config.extractor_type == "cfg" and config.noun_phrase_grammars and tag not in config.noun_phrase_grammars:
        return True
    return False


def _entity_filtered(entity: object, config: FastGraphRAGExtractionConfig) -> bool:
    label = str(getattr(entity, "label_", ""))
    return bool(config.exclude_entity_tags and label in config.exclude_entity_tags)


def normalize_noun_phrase_text(text: str, *, delimiter: str = " ") -> str:
    normalized = _normalize_for_filter(text)
    if delimiter != " ":
        return delimiter.join(normalized.split())
    return normalized


def _normalize_for_filter(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return " ".join(normalized.split())


def internal_stopwords() -> frozenset[str]:
    return _INTERNAL_STOPWORDS


def default_exclude_nouns() -> frozenset[str]:
    return _DEFAULT_EXCLUDE_NOUNS


def unique_normalized(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_noun_phrase_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


__all__ = [
    "NounPhrase",
    "default_exclude_nouns",
    "extract_regex_english_noun_phrases",
    "extract_spacy_noun_phrases",
    "internal_stopwords",
    "normalize_noun_phrase_text",
    "unique_normalized",
]
