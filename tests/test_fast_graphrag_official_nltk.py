from __future__ import annotations

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGExtractionConfig
from graph_memory.retrieval.methods.fast_graphrag.official_nltk import (
    OFFICIAL_EN_STOP_WORDS,
    OfficialRegexEnglishExtractor,
)


def test_official_nltk_regex_dependencies_are_importable() -> None:
    import nltk  # noqa: F401
    import textblob  # noqa: F401


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
    compound = extractor.tag_noun_phrase("energy-policy", all_proper_nouns=())

    assert multiword.keep is True
    assert multiword.cleaned_text == "NUCLEAR ENERGY POLICY"
    assert compound.keep is True
    assert compound.cleaned_text == "ENERGY-POLICY"


def test_official_regex_tagging_removes_excluded_tokens_before_keep_decision() -> None:
    extractor = OfficialRegexEnglishExtractor(
        FastGraphRAGExtractionConfig(exclude_nouns=("thing",), max_word_length=15)
    )

    phrase = extractor.tag_noun_phrase("thing Richmond", all_proper_nouns=("RICHMOND",))

    assert phrase.keep is True
    assert phrase.cleaned_tokens == ("Richmond",)
    assert phrase.cleaned_text == "RICHMOND"
