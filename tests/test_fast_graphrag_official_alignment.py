from __future__ import annotations

import pytest

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
        ents: list[FakeSpan] = []

    def fake_nlp(text: str) -> FakeDoc:
        assert "prime minister" in text
        return FakeDoc()

    mentions = extract_candidate_mentions(
        (candidate,),
        config=FastGraphRAGExtractionConfig(extractor_type="syntactic_parser"),
        nlp=fake_nlp,
    )

    assert {"prime minister", "nuclear energy policy"} <= {mention.normalized_name for mention in mentions}


def test_spacy_extractor_requires_supplied_nlp_object() -> None:
    candidate = TextCandidate(
        item_id="m0",
        text="The prime minister approved nuclear energy policy.",
        metadata={"position": 0},
    )

    with pytest.raises(ValueError, match="requires an nlp object"):
        extract_candidate_mentions(
            (candidate,),
            config=FastGraphRAGExtractionConfig(extractor_type="syntactic_parser"),
        )
