from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels


@dataclass(frozen=True)
class HotpotQADocument:
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class HotpotQASupportingFact:
    title: str
    sentence_id: int


@dataclass(frozen=True)
class HotpotQAExample:
    raw_id: str
    question: str
    answer: str
    documents: tuple[HotpotQADocument, ...]
    supporting_facts: tuple[HotpotQASupportingFact, ...]


@dataclass(frozen=True)
class ConvertedHotpotQAExample:
    task_input: MemoryTaskInput
    task_labels: MemoryTaskLabels


@dataclass(frozen=True)
class HotpotQAConversionResult:
    task_inputs: list[MemoryTaskInput]
    task_labels: list[MemoryTaskLabels]

