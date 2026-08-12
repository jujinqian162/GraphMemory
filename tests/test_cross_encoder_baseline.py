from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

import graph_memory.experiment.workflow as experiment_workflow
import graph_memory.registry.retrieval_builders as retrieval_builders
from graph_memory.experiment.config import (
    CrossEncoderMethodConfig,
    PairBuildConfig,
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.models.cross_encoder.metadata import (
    CrossEncoderModelMetadata,
    load_cross_encoder_model_metadata,
    write_cross_encoder_model_metadata,
)
from graph_memory.models.cross_encoder.training import (
    _TaskLocalCrossEncoderEvaluator,
    build_cross_encoder_examples,
)
from graph_memory.registry.retrieval_builders import build_retrieval
from graph_memory.retrieval.methods.flat.cross_encoder import CrossEncoderTaskRetriever
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest
from graph_memory.training_pairs.contracts import TrainPairRecord

ROOT = Path(__file__).resolve().parents[1]


class Captured(Exception):
    pass


class FixtureCrossEncoder:
    def predict(self, sentences, **kwargs):
        del kwargs
        return np.asarray(
            [float("target" in candidate) for _query, candidate in sentences]
        )


def _resolve(*overrides: str):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                "name=cross-encoder-test",
                "dataset=isetrace",
                "profile=smoke",
                "device=cpu",
                "method=cross_encoder",
                *overrides,
            ],
        )
    return resolve_experiment_config(
        parse_composed_config(composed), repository_root=ROOT
    )


def test_cross_encoder_variants_share_training_settings() -> None:
    flat = _resolve()
    provenance = _resolve("method.variant=provenance_unit")

    assert isinstance(flat.method, CrossEncoderMethodConfig)
    assert isinstance(provenance.method, CrossEncoderMethodConfig)
    assert flat.method.variant == "flat"
    assert provenance.method.variant == "provenance_unit"
    assert flat.method.backbone == provenance.method.backbone
    assert flat.method.trainer == provenance.method.trainer
    assert flat.method.pairs == provenance.method.pairs
    assert flat.method.pairs.hard_graph_neighbor_per_positive == 0


def test_cross_encoder_is_isetrace_only() -> None:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                "name=cross-encoder-test",
                "dataset=hotpotqa",
                "profile=smoke",
                "device=cpu",
                "method=cross_encoder",
            ],
        )
    with pytest.raises(ValueError, match="does not support provenance"):
        resolve_experiment_config(
            parse_composed_config(composed), repository_root=ROOT
        )


def test_cross_encoder_scores_every_candidate_with_deterministic_ties() -> None:
    request = TextRankingRequest(
        task_id="task",
        query_text="query",
        candidates=(
            TextCandidate(item_id="b", text="other", metadata={}),
            TextCandidate(item_id="c", text="target", metadata={}),
            TextCandidate(item_id="a", text="other", metadata={}),
        ),
    )
    retriever = CrossEncoderTaskRetriever(FixtureCrossEncoder(), batch_size=2)

    result = retriever.rank_task(request, top_k=1)

    assert [row.node_id for row in result.ranked_nodes] == ["c", "a", "b"]
    assert len(result.ranked_nodes) == len(request.candidates)


def test_cross_encoder_examples_preserve_all_persisted_pairs() -> None:
    request = TextRankingRequest(
        task_id="task",
        query_text="query",
        candidates=(
            TextCandidate(item_id="p", text="positive", metadata={}),
            TextCandidate(item_id="n", text="negative", metadata={}),
        ),
    )
    pairs = (
        TrainPairRecord(task_id="task", node_id="p", label=1, sample_type="positive"),
        TrainPairRecord(
            task_id="task", node_id="n", label=0, sample_type="hard_bm25"
        ),
    )

    examples = build_cross_encoder_examples(
        ranking_requests=[request], train_pairs=pairs
    )

    assert [(row.candidate_id, row.label) for row in examples] == [("p", 1.0), ("n", 0.0)]


def test_cross_encoder_dev_evaluator_batches_all_task_pairs() -> None:
    requests = (
        TextRankingRequest(
            task_id="task-a",
            query_text="a",
            candidates=(
                TextCandidate(item_id="a1", text="target", metadata={}),
                TextCandidate(item_id="a2", text="other", metadata={}),
            ),
        ),
        TextRankingRequest(
            task_id="task-b",
            query_text="b",
            candidates=(
                TextCandidate(item_id="b1", text="other", metadata={}),
                TextCandidate(item_id="b2", text="target", metadata={}),
            ),
        ),
    )
    labels = (
        EvidenceLabel(
            task_id="task-a",
            gold_answer="",
            gold_evidence_item_ids=("a1",),
            gold_dependency_edges=(),
        ),
        EvidenceLabel(
            task_id="task-b",
            gold_answer="",
            gold_evidence_item_ids=("b2",),
            gold_dependency_edges=(),
        ),
    )
    model = FixtureCrossEncoder()
    evaluator = _TaskLocalCrossEncoderEvaluator(
        requests=requests, labels=labels, batch_size=8
    )

    assert evaluator(model) == 1.0
    assert evaluator.metric_values["dev_recall_at_5"] == 1.0


def test_cross_encoder_metadata_records_variant(tmp_path: Path) -> None:
    write_cross_encoder_model_metadata(
        model_dir=tmp_path,
        metadata=CrossEncoderModelMetadata(
            base_model="fixture",
            max_length=512,
            train_batch_size=4,
            eval_batch_size=8,
            device="cpu",
            variant="provenance_unit",
        ),
    )

    metadata = load_cross_encoder_model_metadata(tmp_path)

    assert metadata.method == "cross_encoder"
    assert metadata.variant == "provenance_unit"
    assert metadata.selected_metric == "dev_recall_at_5"


def test_cross_encoder_retrieval_rejects_checkpoint_variant_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    write_cross_encoder_model_metadata(
        model_dir=tmp_path,
        metadata=CrossEncoderModelMetadata(
            base_model="fixture",
            max_length=512,
            train_batch_size=4,
            eval_batch_size=8,
            device="cpu",
            variant="flat",
        ),
    )
    method = _resolve("method.variant=provenance_unit").method
    assert isinstance(method, CrossEncoderMethodConfig)
    monkeypatch.setattr(
        retrieval_builders,
        "load_cross_encoder_model_metadata",
        load_cross_encoder_model_metadata,
    )

    with pytest.raises(ValueError, match="variant='flat'.*provenance_unit"):
        build_retrieval(
            method,
            text_requests=[],
            checkpoint=tmp_path,
            device="cpu",
        )


def test_cross_encoder_workflow_is_graph_free_and_uses_selected_view(
    monkeypatch, tmp_path: Path
) -> None:
    config = _resolve("method.variant=provenance_unit")
    observed: dict[str, object] = {}

    monkeypatch.setattr(experiment_workflow, "ensure_inputs", lambda config: None)
    monkeypatch.setattr(
        experiment_workflow,
        "prefect_storage_settings",
        lambda *, refresh_cache: nullcontext(),
    )
    monkeypatch.setattr(
        experiment_workflow,
        "_resolve_split_sources",
        lambda config: {split: object() for split in ("train", "dev", "test")},
    )
    monkeypatch.setattr(experiment_workflow, "_trajectory_source", lambda config: None)
    monkeypatch.setattr(
        experiment_workflow, "_authoring_metadata_source", lambda config: None
    )
    monkeypatch.setattr(
        experiment_workflow, "prepare_split_task", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        experiment_workflow, "resolve_encoder_source", lambda encoder: object()
    )

    def pairs(**kwargs):
        observed["pair_config"] = kwargs["config"]
        observed["pair_graph"] = kwargs["evidence_graphs"]
        return object()

    def train(**kwargs):
        observed["train_config"] = kwargs["config"]
        raise Captured

    monkeypatch.setattr(experiment_workflow, "build_training_pairs_task", pairs)
    monkeypatch.setattr(experiment_workflow, "train_cross_encoder_task", train)
    monkeypatch.setattr(
        experiment_workflow,
        "build_evidence_graphs_task",
        lambda **kwargs: pytest.fail("Cross-Encoder must not build graphs"),
    )

    with pytest.raises(Captured):
        experiment_workflow.run_experiment.fn(
            config,
            run_output=tmp_path / "run",
        )

    pair_config = observed["pair_config"]
    assert isinstance(pair_config, PairBuildConfig)
    assert pair_config.method == "cross_encoder"
    assert pair_config.candidate_view == "provenance_unit"
    assert observed["pair_graph"] is None
    assert observed["train_config"] == config.method
