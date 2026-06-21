from __future__ import annotations

from graph_memory.datasets.twowiki.compatibility import (
    coerce_twowiki_label_records,
    coerce_twowiki_ranking_records,
    combined_twowiki_records,
)
from graph_memory.datasets.twowiki.converter import convert_twowiki_example, convert_twowiki_examples
from graph_memory.datasets.twowiki.parser import parse_twowiki_example, parse_twowiki_examples
from graph_memory.datasets.twowiki.projectors import (
    TwoWikiToEvidenceEvaluationRequest,
    TwoWikiToGraphBuildRequest,
    TwoWikiToGraphRankingRequest,
    TwoWikiToTemporalMemoryRankingRequest,
    TwoWikiToTextRankingRequest,
)
from graph_memory.datasets.twowiki.records import (
    CombinedTwoWikiRecord,
    ConvertedTwoWikiExample,
    TwoWikiCandidateSentence,
    TwoWikiConversionResult,
    TwoWikiDocument,
    TwoWikiEvidenceTriple,
    TwoWikiExample,
    TwoWikiLabelRecord,
    TwoWikiRankingRecord,
    TwoWikiSupportingFact,
)

__all__ = [
    "CombinedTwoWikiRecord",
    "ConvertedTwoWikiExample",
    "TwoWikiCandidateSentence",
    "TwoWikiConversionResult",
    "TwoWikiDocument",
    "TwoWikiEvidenceTriple",
    "TwoWikiExample",
    "TwoWikiLabelRecord",
    "TwoWikiRankingRecord",
    "TwoWikiSupportingFact",
    "TwoWikiToEvidenceEvaluationRequest",
    "TwoWikiToGraphBuildRequest",
    "TwoWikiToGraphRankingRequest",
    "TwoWikiToTemporalMemoryRankingRequest",
    "TwoWikiToTextRankingRequest",
    "coerce_twowiki_label_records",
    "coerce_twowiki_ranking_records",
    "combined_twowiki_records",
    "convert_twowiki_example",
    "convert_twowiki_examples",
    "parse_twowiki_example",
    "parse_twowiki_examples",
]
