import pytest

from graph_memory.tuning.grid_search import GridSearchRunner, ParameterGrid


def test_parameter_grid_expands_in_deterministic_order() -> None:
    grid = ParameterGrid(
        parameters={"a": [1, 2], "b": ["x", "y"]},
        fixed={"fixed": 3},
    )

    assert grid.expand() == [
        {"fixed": 3, "a": 1, "b": "x"},
        {"fixed": 3, "a": 1, "b": "y"},
        {"fixed": 3, "a": 2, "b": "x"},
        {"fixed": 3, "a": 2, "b": "y"},
    ]


def test_parameter_grid_rejects_empty_candidate_values() -> None:
    grid = ParameterGrid(parameters={"a": []}, fixed={})

    with pytest.raises(ValueError, match="non-empty sequence"):
        grid.expand()


def test_parameter_grid_rejects_overlapping_fixed_fields() -> None:
    grid = ParameterGrid(parameters={"a": [1]}, fixed={"a": 2})

    with pytest.raises(ValueError, match="overlap"):
        grid.expand()


def test_parameter_grid_rejects_strings_as_candidate_sequences() -> None:
    grid = ParameterGrid(parameters={"a": "abc"}, fixed={})

    with pytest.raises(ValueError, match="non-empty sequence"):
        grid.expand()


def test_grid_search_evaluates_each_candidate_once_and_selects_max_key() -> None:
    calls: list[int] = []
    runner = GridSearchRunner[int, int, int](selection_key=lambda value: value)

    result = runner.run(
        [1, 3, 2],
        lambda candidate: calls.append(candidate) or candidate * 10,
    )

    assert calls == [1, 3, 2]
    assert result.selected.config == 3
    assert [row.config for row in result.candidates] == [1, 3, 2]


def test_grid_search_selects_first_candidate_on_exact_tie() -> None:
    runner = GridSearchRunner[str, int, int](selection_key=lambda value: value)

    result = runner.run(["first", "other"], lambda candidate: len(candidate))

    assert result.selected.config == "first"


def test_grid_search_rejects_empty_candidates() -> None:
    runner = GridSearchRunner[int, int, int](selection_key=lambda value: value)

    with pytest.raises(
        ValueError,
        match="Grid search requires at least one candidate",
    ):
        runner.run([], lambda candidate: candidate)
