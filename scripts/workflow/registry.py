from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from graph_memory.retrieval_registry import get_supported_methods
from scripts.workflow.types import (
    AblationSuiteSpec,
    ChangeDimension,
    RgcnAblationVariant,
    VariantSpec,
    WorkflowSpec,
)
from scripts.workflow.workflows import GRAPH_RERANK_WORKFLOW, RGCN_WORKFLOW, STATELESS_RETRIEVAL_WORKFLOW


METHOD_WORKFLOW_REGISTRY: dict[str, WorkflowSpec] = {
    "bm25": STATELESS_RETRIEVAL_WORKFLOW,
    "dense": STATELESS_RETRIEVAL_WORKFLOW,
    "bm25_graph_rerank": GRAPH_RERANK_WORKFLOW,
    "dense_graph_rerank": GRAPH_RERANK_WORKFLOW,
    "dense_rgcn_graph_retriever": RGCN_WORKFLOW,
}

_MODEL_GRAPH_VIEW = frozenset({ChangeDimension.MODEL_GRAPH_VIEW})
_MODEL_STRUCTURE = frozenset({ChangeDimension.MODEL_STRUCTURE})
_PAIR_SAMPLING = frozenset({ChangeDimension.PAIR_SAMPLING})

RGCN_ABLATION_SUITE = AblationSuiteSpec(
    method="dense_rgcn_graph_retriever",
    variants=(
        VariantSpec(RgcnAblationVariant.FULL_RGCN, frozenset(), baseline_alias=True),
        VariantSpec(
            RgcnAblationVariant.WO_BRIDGE,
            _MODEL_GRAPH_VIEW,
            {"model": {"ablation": RgcnAblationVariant.WO_BRIDGE.value}},
        ),
        VariantSpec(
            RgcnAblationVariant.WO_ENTITY_OVERLAP,
            _MODEL_GRAPH_VIEW,
            {"model": {"ablation": RgcnAblationVariant.WO_ENTITY_OVERLAP.value}},
        ),
        VariantSpec(
            RgcnAblationVariant.WO_SEQUENTIAL,
            _MODEL_GRAPH_VIEW,
            {"model": {"ablation": RgcnAblationVariant.WO_SEQUENTIAL.value}},
        ),
        VariantSpec(
            RgcnAblationVariant.WO_QUERY_OVERLAP,
            _MODEL_GRAPH_VIEW,
            {"model": {"ablation": RgcnAblationVariant.WO_QUERY_OVERLAP.value}},
        ),
        VariantSpec(
            RgcnAblationVariant.WO_GRAPH,
            _MODEL_STRUCTURE,
            {"model": {"ablation": RgcnAblationVariant.WO_GRAPH.value, "num_layers": 0}},
        ),
        VariantSpec(
            RgcnAblationVariant.WO_EDGE_TYPE,
            _MODEL_STRUCTURE,
            {"model": {"ablation": RgcnAblationVariant.WO_EDGE_TYPE.value}},
        ),
        VariantSpec(
            RgcnAblationVariant.WO_EDGE_WEIGHT,
            _MODEL_STRUCTURE,
            {"model": {"ablation": RgcnAblationVariant.WO_EDGE_WEIGHT.value}},
        ),
        VariantSpec(
            RgcnAblationVariant.WO_SEED_SCORE,
            _MODEL_STRUCTURE,
            {"model": {"ablation": RgcnAblationVariant.WO_SEED_SCORE.value}},
        ),
        VariantSpec(
            RgcnAblationVariant.WO_HARD_NEGATIVES,
            _PAIR_SAMPLING,
            {
                "pair_sampling": {
                    "hard_bm25_per_positive": 0,
                    "hard_dense_per_positive": 0,
                    "hard_graph_neighbor_per_positive": 0,
                }
            },
        ),
    ),
)

ABLATION_SUITE_REGISTRY: dict[str, AblationSuiteSpec] = {
    RGCN_ABLATION_SUITE.method: RGCN_ABLATION_SUITE,
}


def get_workflow(method: str) -> WorkflowSpec:
    try:
        return METHOD_WORKFLOW_REGISTRY[method]
    except KeyError as error:
        allowed = ", ".join(sorted(METHOD_WORKFLOW_REGISTRY))
        raise ValueError(f"Unsupported workflow method={method!r}; allowed values: {allowed}") from error


def get_ablation_suite(method: str) -> AblationSuiteSpec | None:
    return ABLATION_SUITE_REGISTRY.get(method)


def get_variant_spec(method: str, variant: str) -> VariantSpec:
    suite = get_ablation_suite(method)
    if suite is None:
        raise ValueError(f"Method does not register an ablation suite: {method}")
    for spec in suite.variants:
        if spec.identifier.value == variant:
            return spec
    allowed = ", ".join(spec.identifier.value for spec in suite.variants)
    raise ValueError(f"Unknown variant={variant!r} for method={method}; allowed values: {allowed}")


def discover_ablation_variants(method: str | None = None) -> list[dict[str, Any]]:
    suites = (
        [get_ablation_suite(method)]
        if method is not None
        else [ABLATION_SUITE_REGISTRY[name] for name in sorted(ABLATION_SUITE_REGISTRY)]
    )
    if method is not None and suites[0] is None:
        allowed = ", ".join(sorted(ABLATION_SUITE_REGISTRY))
        raise ValueError(f"Method does not register an ablation suite: {method}; allowed values: {allowed}")
    return [
        {
            "method": suite.method,
            "variant": variant.identifier.value,
            "changed_dimensions": [dimension.value for dimension in sorted(variant.changed_dimensions, key=str)],
            "baseline_alias": variant.baseline_alias,
        }
        for suite in suites
        if suite is not None
        for variant in suite.variants
    ]


def validate_workflow_registry(
    *,
    runtime_methods: Sequence[str] | None = None,
    registrations: Mapping[str, WorkflowSpec] | None = None,
    suites: Mapping[str, AblationSuiteSpec] | None = None,
) -> None:
    selected_runtime_methods = set(runtime_methods or get_supported_methods())
    selected_registrations = registrations or METHOD_WORKFLOW_REGISTRY
    selected_suites = suites or ABLATION_SUITE_REGISTRY
    missing = sorted(selected_runtime_methods - set(selected_registrations))
    extra = sorted(set(selected_registrations) - selected_runtime_methods)
    if missing or extra:
        raise ValueError(f"Workflow registry mismatch: missing={missing}, extra={extra}")
    for method, suite in selected_suites.items():
        if method != suite.method:
            raise ValueError(f"Ablation suite registry key={method} does not match suite method={suite.method}.")
        if method not in selected_registrations:
            raise ValueError(f"Ablation suite references unregistered workflow method={method}.")
        baseline_aliases = [variant for variant in suite.variants if variant.baseline_alias]
        if len(baseline_aliases) != 1:
            raise ValueError(f"Ablation suite method={method} must declare exactly one baseline alias.")
