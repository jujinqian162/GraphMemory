from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

import graph_memory.registry.retrieval_builders as retrieval_builders
from graph_memory.experiment.config import (
    CrossEncoderMethodConfig,
    parse_composed_config,
    resolve_experiment_config,
)
from graph_memory.models.cross_encoder.metadata import (
    CrossEncoderModelMetadata,
    load_cross_encoder_model_metadata,
    write_cross_encoder_model_metadata,
)
from graph_memory.registry.retrieval_builders import build_retrieval
from graph_memory.retrieval.methods.flat.cross_encoder import (
    CrossEncoderTaskRetriever,
)
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest

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
