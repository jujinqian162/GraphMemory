from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

from graph_memory.experiment.config import ClosedModel, EvidenceRgcnVariant
from graph_memory.retrieval.methods.ids import RetrievalMethodId

InspectionKind = Literal[
    "methods",
    "datasets",
    "profiles",
    "configs",
    "variants",
    "jobs",
]


class InspectCommandConfig(ClosedModel):
    kind: InspectionKind
    name: str | None = None


def inspect_catalog(
    kind: InspectionKind,
    *,
    repository_root: Path,
    name: str | None = None,
) -> object:
    config_root = repository_root.resolve() / "configs"
    if kind == "methods":
        return [_method_row(method) for method in RetrievalMethodId]
    if kind in {"datasets", "profiles"}:
        directory = config_root / ("dataset" if kind == "datasets" else "profile")
        return sorted(path.stem for path in directory.glob("*.yaml"))
    if kind == "configs":
        return sorted(path.stem for path in config_root.glob("*.yaml"))
    if kind == "variants":
        evidence_variants = [str(value) for value in get_args(EvidenceRgcnVariant)]
        return {
            RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER: evidence_variants,
            RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER: evidence_variants,
        }
    if kind == "jobs":
        if name is None:
            raise ValueError("inspect kind=jobs requires name=<multirun-name>")
        named_root = repository_root.resolve() / "runs" / name
        return sorted(
            path.parent.parent.name
            for path in named_root.glob("*/workflow/summary.yaml")
            if path.is_file()
        )
    raise ValueError(f"unsupported inspection kind: {kind}")


def _method_row(method: RetrievalMethodId) -> dict[str, object]:
    evidence_only = method in {
        RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
        RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
    }
    provenance_only = method in {
        RetrievalMethodId.PROVENANCE_PATH,
        RetrievalMethodId.PROVENANCE_RGCN,
    }
    families = (
        ["evidence_retrieval"]
        if evidence_only
        else ["execution_provenance"]
        if provenance_only
        else ["evidence_retrieval", "execution_provenance"]
    )
    return {
        "method": method.value,
        "request_type": {
            RetrievalMethodId.BM25: "TextRankingRequest",
            RetrievalMethodId.DENSE: "TextRankingRequest",
            RetrievalMethodId.DENSE_FT: "TextRankingRequest",
            RetrievalMethodId.GRAPHRAG: "GraphRAGRequest",
            RetrievalMethodId.PROVENANCE_PATH: "ProvenancePathRequest",
            RetrievalMethodId.PROVENANCE_RGCN: "ProvenanceRgcnRequest",
            RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER: (
                "EvidenceGraphRankingRequest"
            ),
            RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER: (
                "EvidenceGraphRankingRequest"
            ),
        }[method],
        "supported_families": families,
    }
