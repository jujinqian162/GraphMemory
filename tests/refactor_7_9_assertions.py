from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard


TRAINABLE_SCALAR_REL_TOLERANCE = 1e-3
TRAINABLE_SCALAR_ABS_TOLERANCE = 1e-4


def assert_deterministic_artifact_equal(actual: Any, expected: Any) -> None:
    """Require byte-decoded deterministic contracts to remain exactly equal."""

    assert actual == expected


def deterministic_prediction_facts(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Remove runtime observations before exact scientific-result comparison."""

    return [
        {key: value for key, value in row.items() if key != "latency_ms"}
        for row in rows
    ]


def assert_trainable_metric_rows_close(
    actual: Sequence[Mapping[str, object]],
    expected: Sequence[Mapping[str, object]],
) -> None:
    """Compare trainable scalar observations while keeping structure exact."""

    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert set(actual_row) == set(expected_row)
        for key, expected_value in expected_row.items():
            actual_value = actual_row[key]
            if _is_numeric_scalar(actual_value) and _is_numeric_scalar(expected_value):
                assert math.isclose(
                    float(actual_value),
                    float(expected_value),
                    rel_tol=TRAINABLE_SCALAR_REL_TOLERANCE,
                    abs_tol=TRAINABLE_SCALAR_ABS_TOLERANCE,
                ), (key, actual_value, expected_value)
            else:
                assert actual_value == expected_value, key


def _is_numeric_scalar(value: object) -> TypeGuard[int | float]:
    return not isinstance(value, bool) and isinstance(value, (int, float))


__all__ = [
    "TRAINABLE_SCALAR_ABS_TOLERANCE",
    "TRAINABLE_SCALAR_REL_TOLERANCE",
    "assert_deterministic_artifact_equal",
    "assert_trainable_metric_rows_close",
    "deterministic_prediction_facts",
]
