from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from graph_memory.models.graph_retriever.config.records import NodeFeatureConfig
from graph_memory.retrieval.signals import SeedSignal


@dataclass(frozen=True)
class NodeFeatureTensors:
    """
    Numeric node and scorer features for one tensorized task graph.
    单个张量化 task graph 的数值 node 与 scorer 特征。
    """

    node_ids: list[str]
    node_features: Tensor
    scorer_features: Tensor
    node_feature_names: tuple[str, ...]
    scorer_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class NodeFeatureBuilder:
    """
    Builds ordered numeric features from frozen seed signals.
    从冻结 seed signal 构造有序数值特征。
    """

    config: NodeFeatureConfig

    def build_node_features(self, *, node_ids: list[str], seed_signals: list[SeedSignal]) -> NodeFeatureTensors:
        signal_by_node_id = {signal.node_id: signal for signal in seed_signals}
        node_rows = [self._feature_row(node_id, self.config.node_feature_names, signal_by_node_id) for node_id in node_ids]
        scorer_rows = [
            self._feature_row(node_id, self.config.scorer_feature_names, signal_by_node_id) for node_id in node_ids
        ]
        return NodeFeatureTensors(
            node_ids=node_ids,
            node_features=torch.tensor(node_rows, dtype=torch.float32),
            scorer_features=torch.tensor(scorer_rows, dtype=torch.float32),
            node_feature_names=self.config.node_feature_names,
            scorer_feature_names=self.config.scorer_feature_names,
        )

    def _feature_row(
        self,
        node_id: str,
        feature_names: tuple[str, ...],
        signal_by_node_id: dict[str, SeedSignal],
    ) -> list[float]:
        row: list[float] = []
        signal = signal_by_node_id.get(node_id)
        for feature_name in feature_names:
            if feature_name == "seed_score":
                row.append(0.0 if signal is None else signal.score)
            elif feature_name == "seed_rank_percentile":
                row.append(1.0 if signal is None else signal.rank_percentile)
            elif feature_name == "is_question_node":
                row.append(1.0 if node_id == "q" else 0.0)
            else:
                raise ValueError(f"Unsupported node feature name: {feature_name}")
        return row
