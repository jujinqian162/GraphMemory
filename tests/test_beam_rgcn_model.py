from __future__ import annotations

import torch
from dataclasses import replace
import inspect
from typing import cast
from torch import nn

from graph_memory.models.graph_retriever.factory import build_model_from_config
from tests.test_phase2_rgcn_model import tiny_training_batch
from tests.test_phase2_rgcn_training import tiny_model_config
import graph_memory.models.graph_retriever.inference as inference_module


def _model():
    return build_model_from_config(replace(tiny_model_config(), encoder_dim=3))


def test_cached_graph_encoding_is_reused_for_hypothesis_scoring() -> None:
    model = _model()
    batch = tiny_training_batch()
    calls = 0

    def count_calls(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    handle = cast(nn.Module, model.graph_encoder).register_forward_hook(count_calls)
    encoded = model.encode_graph(batch)
    empty = model.score_decoder_actions(
        encoded, task_index=0, selected_local_indices=()
    )
    selected = model.score_decoder_actions(
        encoded, task_index=0, selected_local_indices=(0,)
    )
    handle.remove()

    assert calls == 1
    assert encoded.candidate_node_ids_by_task == [["m0", "m1", "m2"]]
    assert "q" not in encoded.candidate_node_ids_by_task[0]
    assert empty.candidate_logits.shape == (3,)
    assert selected.candidate_logits.shape == (3,)
    assert not torch.equal(empty.candidate_logits, selected.candidate_logits)
    assert empty.stop_logit.ndim == 0


def test_selected_set_context_handles_empty_and_non_empty_hypotheses() -> None:
    model = _model()
    encoded = model.encode_graph(tiny_training_batch())

    empty_context, empty_last = model.selected_context(
        encoded, task_index=0, selected_local_indices=()
    )
    context, last = model.selected_context(
        encoded, task_index=0, selected_local_indices=(0, 1)
    )

    assert torch.count_nonzero(empty_context) == 0
    assert torch.count_nonzero(empty_last) == 0
    assert torch.count_nonzero(context) > 0
    assert torch.equal(last, encoded.candidate_states_for_task(0)[1])


def test_beam_inference_has_no_training_or_label_input_boundary() -> None:
    source = inspect.getsource(inference_module)
    assert "graph_retriever.training" not in source
    signature = inspect.signature(_model().score_decoder_actions)
    assert "label" not in signature.parameters
    assert "gold" not in source
    assert "decomposition" not in source
