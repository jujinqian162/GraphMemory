from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
import numpy as np
import pytest

from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceQueryMetadata,
)
from graph_memory.embeddings import SentenceEncoder
from graph_memory.experiment.config import (
    DenseFinetuneMethodConfig,
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.models.dense_finetune.metadata import (
    DenseFinetuneModelMetadata,
    DenseFinetuneSelectionMetadata,
    write_dense_ft_model_metadata,
)
from graph_memory.registry.retrieval_builders import build_retrieval
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.ids import DenseCandidateView
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest
from graph_memory.stages.evaluate import run_evaluate_stage
from graph_memory.trajectories import SourceSpan


ROOT = Path(__file__).resolve().parents[1]


class KeywordEncoder:
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        del batch_size, show_progress_bar
        rows: list[np.ndarray] = []
        for text in texts:
            vector = np.asarray(
                [float("target" in text), float("other" in text)],
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(vector))
            rows.append(vector / norm if normalize_embeddings and norm else vector)
        return np.asarray(rows, dtype=np.float32)


def _method(variant: DenseCandidateView) -> DenseFinetuneMethodConfig:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                "name=dense-ft-metadata-test",
                "dataset=isetrace",
                "profile=smoke",
                "device=cpu",
                "method=dense_ft",
                f"method.variant={variant}",
            ],
        )
    resolved = resolve_experiment_config(
        parse_composed_config(composed), repository_root=ROOT
    )
    assert isinstance(resolved.method, DenseFinetuneMethodConfig)
    return resolved.method


def _write_metadata(model_dir: Path, *, variant: DenseCandidateView) -> None:
    write_dense_ft_model_metadata(
        model_dir=model_dir,
        metadata=DenseFinetuneModelMetadata(
            base_model="fixture-model",
            query_prefix="query: ",
            passage_prefix="passage: ",
            batch_size=8,
            device="cpu",
            variant=variant,
            selection=DenseFinetuneSelectionMetadata(
                selected_metric="dev_recall_at_5",
                higher_is_better=True,
            ),
        ),
    )


def test_dense_ft_retrieval_accepts_matching_variant_and_keeps_method_name(
    tmp_path: Path,
) -> None:
    _write_metadata(tmp_path, variant="provenance_unit")

    retriever, provenance, requests = build_retrieval(
        _method("provenance_unit"),
        text_requests=[],
        checkpoint=tmp_path,
        dense_encoder=cast(SentenceEncoder, object()),
        device="cpu",
    )

    assert retriever.name == "dense_ft"
    assert provenance["method"] == "dense_ft"
    assert requests == []


def test_provenance_unit_dense_ft_checkpoint_runs_exact_span_evaluation(
    tmp_path: Path,
) -> None:
    _write_metadata(tmp_path, variant="provenance_unit")
    source_span = SourceSpan(
        event_id="event:output",
        json_pointer="/content",
        char_start=0,
        char_end=6,
    )
    request = TextRankingRequest(
        task_id="task:provenance-unit",
        query_text="target",
        candidates=(
            TextCandidate(
                item_id="output-content:c1:chunk:0",
                text="target",
                metadata={},
                source_spans=(source_span,),
            ),
            TextCandidate(
                item_id="output-content:c2:chunk:0",
                text="other",
                metadata={},
                source_spans=(
                    SourceSpan(
                        event_id="event:other",
                        json_pointer="/content",
                        char_start=0,
                        char_end=5,
                    ),
                ),
            ),
        ),
    )
    retriever, _provenance, execution_requests = build_retrieval(
        _method("provenance_unit"),
        text_requests=[request],
        checkpoint=tmp_path,
        dense_encoder=KeywordEncoder(),
        device="cpu",
    )

    predictions = run_retrieval(
        retrieval_method=retriever,
        requests=execution_requests,
        top_k=2,
    )
    metrics, _failures, per_task = run_evaluate_stage(
        dataset="isetrace",
        top_k=2,
        failure_case_limit=10,
        predictions=predictions,
        labels=[
            ISETraceLabelRecord(
                task_id=request.task_id,
                graph_id="graph:fixture",
                gold_evidence_spans=(source_span,),
            ).model_dump(mode="json")
        ],
        graphs=[],
        query_metadata=(
            ISETraceQueryMetadata(
                task_id=request.task_id,
                graph_id="graph:fixture",
                memory_mode="direct_recall",
            ),
        ),
    )

    assert predictions[0].method == "dense_ft"
    assert predictions[0].ranked_nodes[0].node_id == "output-content:c1:chunk:0"
    assert metrics[0].recall_at_2 == pytest.approx(1.0)
    assert per_task[0].graph_id == "graph:fixture"
