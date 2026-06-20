from __future__ import annotations

from typing import Protocol, Sequence

from abstraction.domain.common.capability_names import ViewKind
from abstraction.domain.common.identifiers import DatasetId, TaskId
from abstraction.domain.task_views.eval_views import EvalLabelView
from abstraction.domain.task_views.views import TaskView


class TaskViewCatalog(Protocol):
    def list_view_kinds(self, dataset_id: DatasetId) -> Sequence[ViewKind]:
        ...

    def get_task_view(self, dataset_id: DatasetId, task_id: TaskId, view_kind: ViewKind) -> TaskView:
        ...


class EvalViewCatalog(Protocol):
    def get_eval_view(self, dataset_id: DatasetId, task_id: TaskId) -> EvalLabelView:
        ...


class TaskViewValidator(Protocol):
    def validate_task_view(self, view: TaskView) -> None:
        ...

    def validate_eval_view(self, view: EvalLabelView) -> None:
        ...


class DatasetTaskViewCatalog:  # implement TaskViewCatalog
    def list_view_kinds(self, dataset_id: DatasetId) -> Sequence[ViewKind]:
        raise NotImplementedError
    def get_task_view(self, dataset_id: DatasetId, task_id: TaskId, view_kind: ViewKind) -> TaskView:
        raise NotImplementedError
class DatasetEvalViewCatalog:  # implement EvalViewCatalog
    def get_eval_view(self, dataset_id: DatasetId, task_id: TaskId) -> EvalLabelView:
        raise NotImplementedError
class BoundaryOnlyTaskViewValidator:  # implement TaskViewValidator
    def validate_task_view(self, view: TaskView) -> None:
        raise NotImplementedError
    def validate_eval_view(self, view: EvalLabelView) -> None:
        raise NotImplementedError