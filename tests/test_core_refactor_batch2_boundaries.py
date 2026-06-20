from __future__ import annotations

import ast
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_DATASET_TEXT_MODULES = {
    "graph_memory.hotpotqa",
    "graph_memory.splits",
    "graph_memory.entities",
}
LEGACY_COMPATIBILITY_FILES = {
    ROOT / "graph_memory" / "hotpotqa.py",
    ROOT / "graph_memory" / "splits.py",
    ROOT / "graph_memory" / "entities.py",
}


def _raw_hotpotqa_record() -> dict[str, object]:
    return {
        "_id": "abc123",
        "question": "Where was Ada Lovelace honored?",
        "answer": "London",
        "context": [
            ["Ada Lovelace", ["Ada Lovelace wrote notes.", "She worked with Charles Babbage."]],
            ["London", ["London hosted an exhibition.", "Paris hosted another event."]],
        ],
        "supporting_facts": [["Ada Lovelace", 0], ["London", 0], ["Ada Lovelace", 0]],
    }


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_dataset_package_preserves_hotpotqa_parsing_conversion_errors_and_split_order():
    from graph_memory.datasets.hotpotqa.compatibility import combined_hotpotqa_records
    from graph_memory.datasets.hotpotqa.converter import convert_hotpotqa_example
    from graph_memory.datasets.hotpotqa.parser import parse_hotpotqa_example
    from graph_memory.datasets.hotpotqa.records import HotpotQAExample
    from graph_memory.datasets.splits import sample_split

    parsed = parse_hotpotqa_example(_raw_hotpotqa_record(), record_index=7)

    assert isinstance(parsed, HotpotQAExample)
    assert parsed.raw_id == "abc123"
    assert [document.title for document in parsed.documents] == ["Ada Lovelace", "London"]
    assert [fact.title for fact in parsed.supporting_facts] == ["Ada Lovelace", "London", "Ada Lovelace"]

    converted = convert_hotpotqa_example(parsed)

    assert converted.ranking_record == {
        "task_id": "hotpot_abc123",
        "question": "Where was Ada Lovelace honored?",
        "candidate_sentences": [
            {
                "sentence_id": "m0",
                "title": "Ada Lovelace",
                "sentence_index": 0,
                "position": 0,
                "text": "Ada Lovelace wrote notes.",
            },
            {
                "sentence_id": "m1",
                "title": "Ada Lovelace",
                "sentence_index": 1,
                "position": 1,
                "text": "She worked with Charles Babbage.",
            },
            {
                "sentence_id": "m2",
                "title": "London",
                "sentence_index": 0,
                "position": 2,
                "text": "London hosted an exhibition.",
            },
            {
                "sentence_id": "m3",
                "title": "London",
                "sentence_index": 1,
                "position": 3,
                "text": "Paris hosted another event.",
            },
        ],
    }
    assert converted.label_record == {
        "task_id": "hotpot_abc123",
        "gold_answer": "London",
        "gold_evidence_sentence_ids": ["m0", "m2"],
        "gold_dependency_edges": [],
    }
    assert combined_hotpotqa_records([converted.ranking_record], [converted.label_record]) == [
        {**converted.ranking_record, **converted.label_record}
    ]
    assert sample_split(["a", "b", "c", "d", "e"], count=2, seed=5, offset=1) == ["b", "d"]

    malformed = {**_raw_hotpotqa_record(), "context": [["Ada Lovelace", ["ok", 3]]]}
    try:
        parse_hotpotqa_example(malformed)
    except ValueError as error:
        assert str(error) == "HotpotQA example _id=abc123 title=Ada Lovelace sentence_id=1 must be text."
    else:
        raise AssertionError("Expected malformed HotpotQA sentence to fail parsing.")


def test_text_package_preserves_tokens_lexical_scores_and_entities():
    from graph_memory.text.entities import extract_entities, heuristic_entities, title_aliases
    from graph_memory.text.lexical import compute_idf, lexical_score
    from graph_memory.text.tokens import content_tokens

    assert content_tokens("The Eiffel Tower was in Paris in 1889, with US visitors.", keep_short={"us"}) == [
        "eiffel",
        "tower",
        "paris",
        "1889",
        "us",
        "visitors",
    ]

    idf = compute_idf(["Ada Lovelace wrote notes", "Ada visited London"])
    assert math.isclose(idf["ada"], math.log(3 / 3) + 1.0)
    assert math.isclose(idf["lovelace"], math.log(3 / 2) + 1.0)
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

    assert title_aliases("The Ada Lovelace") == {"ada lovelace", "ada", "lovelace"}
    assert heuristic_entities("Ada Lovelace met NASA in Paris.") == {"ada lovelace", "nasa", "paris"}

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


def test_dataset_and_text_domains_have_explicit_packages_and_updated_importers():
    assert (ROOT / "graph_memory" / "datasets" / "hotpotqa" / "parser.py").exists()
    assert (ROOT / "graph_memory" / "datasets" / "hotpotqa" / "converter.py").exists()
    assert (ROOT / "graph_memory" / "datasets" / "splits.py").exists()
    assert (ROOT / "graph_memory" / "text" / "tokens.py").exists()
    assert (ROOT / "graph_memory" / "text" / "lexical.py").exists()
    assert (ROOT / "graph_memory" / "text" / "entities.py").exists()
    assert not (ROOT / "graph_memory" / "text.py").exists()

    scanned_roots = [ROOT / "graph_memory", ROOT / "scripts", ROOT / "tests"]
    offenders: dict[str, list[str]] = {}
    for scanned_root in scanned_roots:
        for path in scanned_root.rglob("*.py"):
            if path in LEGACY_COMPATIBILITY_FILES:
                continue
            imported_old_modules = _imported_modules(path) & OLD_DATASET_TEXT_MODULES
            if imported_old_modules:
                offenders[str(path.relative_to(ROOT))] = sorted(imported_old_modules)

    assert offenders == {}
