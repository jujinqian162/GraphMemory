from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from abstraction.domain.common.capability_names import PredictionKind, RequestKind
from abstraction.domain.common.identifiers import MethodId


@dataclass(frozen=True)
class MethodCapability:
    method_id: MethodId
    consumed_request_kind: RequestKind
    produced_prediction_kind: PredictionKind
    requires_graph_artifact: bool
    requires_training_artifact: bool
    requires_tuning_artifact: bool
    may_consume_seed_scores: bool


class MethodRegistry(Protocol):
    def register_method(self, capability: MethodCapability) -> None:
        ...

    def get_method_capability(self, method_id: MethodId) -> MethodCapability:
        ...

    def list_methods_for_request(self, request_kind: RequestKind) -> Sequence[MethodCapability]:
        ...


class CapabilityMethodRegistry:  # implement MethodRegistry
    def __init__(self) -> None:
        self.capability_by_method_id: dict[MethodId, MethodCapability] = {}

    def register_method(self, capability: MethodCapability) -> None:
        self.capability_by_method_id[capability.method_id] = capability

    def get_method_capability(self, method_id: MethodId) -> MethodCapability:
        return self.capability_by_method_id[method_id]

    def list_methods_for_request(self, request_kind: RequestKind) -> Sequence[MethodCapability]:
        return [
            capability
            for capability in self.capability_by_method_id.values()
            if capability.consumed_request_kind == request_kind
        ]
