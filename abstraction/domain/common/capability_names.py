from __future__ import annotations

from enum import Enum


class ViewKind(str, Enum):
    EVIDENCE_RANKING = "evidence_ranking_view"
    CONTEXT_GATHERING = "context_gathering_view"
    GRAPH_BUILD = "graph_build_view"
    TRAINING = "training_view"
    ANSWER_EVALUATION = "answer_evaluation_view"


class RequestKind(str, Enum):
    TEXT_RANKING = "text_ranking_request"
    GRAPH_RANKING = "graph_ranking_request"
    TEMPORAL_MEMORY_RANKING = "temporal_memory_ranking_request"
    CONTEXT_GATHERING = "context_gathering_request"
    ANSWER = "answer_request"


class PredictionKind(str, Enum):
    RANKING = "ranking_prediction"
    CONTEXT = "context_prediction"
    ANSWER = "answer_prediction"


class SplitRole(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class StageKind(str, Enum):
    PREPARE_DATASET = "prepare_dataset"
    BUILD_TASK_VIEW = "build_task_view"
    PROJECT_REQUEST = "project_request"
    BUILD_GRAPH = "build_graph"
    TRAIN_METHOD = "train_method"
    TUNE_METHOD = "tune_method"
    RUN_RETRIEVAL = "run_retrieval"
    PROJECT_EVALUATION = "project_evaluation"
    RUN_METRICS = "run_metrics"

