from __future__ import annotations

from graph_memory.datasets.hotpotqa.compatibility import combined_memory_tasks
from graph_memory.datasets.hotpotqa.converter import convert_hotpotqa_example, convert_hotpotqa_examples
from graph_memory.datasets.hotpotqa.parser import parse_hotpotqa_example, parse_hotpotqa_examples
from graph_memory.datasets.hotpotqa.records import (
    ConvertedHotpotQAExample,
    HotpotQAConversionResult,
    HotpotQADocument,
    HotpotQAExample,
    HotpotQASupportingFact,
)

__all__ = [
    "ConvertedHotpotQAExample",
    "HotpotQAConversionResult",
    "HotpotQADocument",
    "HotpotQAExample",
    "HotpotQASupportingFact",
    "combined_memory_tasks",
    "convert_hotpotqa_example",
    "convert_hotpotqa_examples",
    "parse_hotpotqa_example",
    "parse_hotpotqa_examples",
]

