from __future__ import annotations

from graph_memory.retrieval.methods.fast_graphrag.nlp import (
    build_entity_catalog,
    extract_candidate_mentions,
    link_query_entities,
    normalize_entity_text,
)
from graph_memory.retrieval.requests import TextCandidate


def test_normalize_entity_text_collapses_case_punctuation_and_parentheses() -> None:
    assert normalize_entity_text("Andrew Allen (Singer)") == "andrew allen singer"


def test_extract_candidate_mentions_uses_title_alias_and_visible_text_only() -> None:
    candidate = TextCandidate(
        item_id="m0",
        text="Andrew Allen is a Canadian singer from Vernon.",
        metadata={"title": "Andrew Allen (singer)", "position": 0},
    )

    mentions = extract_candidate_mentions((candidate,))

    by_name = {mention.name for mention in mentions}
    assert "Andrew Allen (singer)" in by_name
    assert "Andrew Allen" in by_name
    assert "Vernon" in by_name


def test_link_query_entities_matches_known_title_aliases() -> None:
    candidate = TextCandidate(
        item_id="m0",
        text="Changed It is a song by Nicki Minaj.",
        metadata={"title": "Changed It"},
    )
    catalog = build_entity_catalog((candidate,))

    linked = link_query_entities("Who performed Changed It?", catalog)

    assert [entity.name for entity in linked] == ["Changed It"]


def test_catalog_merges_visible_mentions_with_known_title_aliases() -> None:
    candidates = (
        TextCandidate(
            item_id="m0",
            text="Andrew Allen is a Canadian singer.",
            metadata={"title": "Andrew Allen (singer)", "position": 0},
        ),
        TextCandidate(
            item_id="m1",
            text="Vernon hosted a concert by Andrew Allen.",
            metadata={"title": "Vernon", "position": 1},
        ),
    )
    catalog = build_entity_catalog(candidates)

    linked = link_query_entities("Who is Andrew Allen?", catalog)

    assert [entity.name for entity in linked] == ["Andrew Allen (singer)"]
    assert linked[0].candidate_ids == ("m0", "m1")
