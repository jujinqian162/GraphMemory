from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

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
        tags = cast(Iterable[tuple[str, str]], blob.tags)
        proper_nouns = tuple(token[0].upper() for token in tags if token[1] == "NNP")
        phrases: set[str] = set()
        for noun_phrase in cast(Iterable[object], blob.noun_phrases):
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
            nltk.data.find(f"{category}/{resource_name}")
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
