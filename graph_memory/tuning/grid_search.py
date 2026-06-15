from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Generic, TypeVar, cast

from typing_extensions import Protocol, Self


class SupportsLessThan(Protocol):
    def __lt__(self, other: Self, /) -> bool: ...

ConfigT = TypeVar("ConfigT")
EvaluationT = TypeVar("EvaluationT")
KeyT = TypeVar("KeyT", bound=SupportsLessThan)


@dataclass(frozen=True)
class ParameterGrid:
    parameters: Mapping[str, Sequence[object]]
    fixed: Mapping[str, object]

    def expand(self) -> list[dict[str, object]]:
        overlap = self.parameters.keys() & self.fixed.keys()
        if overlap:
            fields = ", ".join(sorted(overlap))
            raise ValueError(f"Parameter and fixed fields overlap: {fields}.")

        parameter_names = list(self.parameters)
        candidate_values: list[Sequence[object]] = []
        for name in parameter_names:
            values = cast(object, self.parameters[name])
            if (
                isinstance(values, (str, bytes))
                or not isinstance(values, Sequence)
                or not values
            ):
                raise ValueError(
                    f"Parameter grid requires a non-empty sequence for {name}."
                )
            candidate_values.append(values)

        return [
            {
                **self.fixed,
                **dict(zip(parameter_names, combination, strict=True)),
            }
            for combination in product(*candidate_values)
        ]


@dataclass(frozen=True)
class EvaluatedCandidate(Generic[ConfigT, EvaluationT]):
    config: ConfigT
    evaluation: EvaluationT


@dataclass(frozen=True)
class GridSearchResult(Generic[ConfigT, EvaluationT]):
    selected: EvaluatedCandidate[ConfigT, EvaluationT]
    candidates: list[EvaluatedCandidate[ConfigT, EvaluationT]]


@dataclass(frozen=True)
class GridSearchRunner(Generic[ConfigT, EvaluationT, KeyT]):
    selection_key: Callable[[EvaluationT], KeyT]

    def run(
        self,
        candidates: Iterable[ConfigT],
        evaluate: Callable[[ConfigT], EvaluationT],
    ) -> GridSearchResult[ConfigT, EvaluationT]:
        materialized = list(candidates)
        if not materialized:
            raise ValueError("Grid search requires at least one candidate.")

        evaluated = [
            EvaluatedCandidate(config=config, evaluation=evaluate(config))
            for config in materialized
        ]
        selected = max(
            evaluated,
            key=lambda candidate: self.selection_key(candidate.evaluation),
        )
        return GridSearchResult(selected=selected, candidates=evaluated)
