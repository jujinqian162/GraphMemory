from __future__ import annotations

from graph_memory.datasets.hotpotqa.compatibility import combined_hotpotqa_records
from graph_memory.datasets.hotpotqa.compatibility import coerce_hotpotqa_label_records, coerce_hotpotqa_ranking_records
from graph_memory.datasets.hotpotqa.converter import convert_hotpotqa_example, convert_hotpotqa_examples
from graph_memory.datasets.hotpotqa.parser import parse_hotpotqa_example, parse_hotpotqa_examples
from graph_memory.datasets.hotpotqa.projectors import (
    HotpotQAToEvidenceEvaluationRequest,
    HotpotQAToGraphBuildRequest,
    HotpotQAToGraphRankingRequest,
    HotpotQAToTemporalMemoryRankingRequest,
    HotpotQAToTextRankingRequest,
)
from graph_memory.datasets.hotpotqa.records import (
    CombinedHotpotQARecord,
    ConvertedHotpotQAExample,
    HotpotQACandidateSentence,
    HotpotQAConversionResult,
    HotpotQADocument,
    HotpotQAExample,
    HotpotQALabelRecord,
    HotpotQARankingRecord,
    HotpotQASupportingFact,
)

__all__ = [
    "CombinedHotpotQARecord",
    "ConvertedHotpotQAExample",
    "HotpotQACandidateSentence",
    "HotpotQAConversionResult",
    "HotpotQADocument",
    "HotpotQAExample",
    "HotpotQALabelRecord",
    "HotpotQARankingRecord",
    "HotpotQASupportingFact",
    "HotpotQAToEvidenceEvaluationRequest",
    "HotpotQAToGraphBuildRequest",
    "HotpotQAToGraphRankingRequest",
    "HotpotQAToTemporalMemoryRankingRequest",
    "HotpotQAToTextRankingRequest",
    "combined_hotpotqa_records",
    "coerce_hotpotqa_label_records",
    "coerce_hotpotqa_ranking_records",
    "convert_hotpotqa_example",
    "convert_hotpotqa_examples",
    "parse_hotpotqa_example",
    "parse_hotpotqa_examples",
]
