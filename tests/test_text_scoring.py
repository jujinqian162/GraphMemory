from __future__ import annotations

import math


def test_compute_idf_matches_smoothed_log_formula() -> None:
    from graph_memory.text.lexical import compute_idf

    idf = compute_idf(["Ada Lovelace wrote notes", "Ada visited London"])

    assert math.isclose(idf["ada"], math.log(3 / 3) + 1.0)
    assert math.isclose(idf["lovelace"], math.log(3 / 2) + 1.0)


def test_lexical_score_combines_idf_title_aliases_and_entity_overlap() -> None:
    from graph_memory.text.lexical import lexical_score

    assert math.isclose(
        lexical_score(
            "Ada Lovelace in London",
            "London honored Ada Lovelace",
            {"ada": 1.0, "lovelace": 2.0, "london": 3.0},
            title_aliases={"ada lovelace"},
            query_entities={"ada lovelace", "london"},
            passage_entities={"ada lovelace"},
        ),
        9.5,
    )


def test_extract_entities_merges_heuristics_with_injected_spacy_pipeline() -> None:
    from graph_memory.text.entities import extract_entities

    class _SpacyEntity:
        def __init__(self, text: str) -> None:
            self.text = text

    class _SpacyDocument:
        ents = [_SpacyEntity("Charles Babbage")]

    def _fake_nlp(_text: str) -> _SpacyDocument:
        return _SpacyDocument()

    assert extract_entities("Ada Lovelace", use_spacy=True, nlp=_fake_nlp) == {
        "ada lovelace",
        "charles babbage",
    }
