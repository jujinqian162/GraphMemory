from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from graph_memory.registry import Registry
from graph_memory.registry.ablations import ABLATION_SUITE_PATCHES, AblationSuitePatch, AblationVariantPatch
from graph_memory.registry.methods import RetrievalLifecycle
from scripts.workflow.types import (
    AblationSuiteSpec,
    ChangeDimension,
    RgcnAblationVariant,
    VariantSpec,
    WorkflowSpec,
)
from scripts.workflow.workflows import (
    DENSE_FT_WORKFLOW,
    GRAPH_RERANK_WORKFLOW,
    RGCN_WORKFLOW,
    STATELESS_RETRIEVAL_WORKFLOW,
)


WORKFLOW_BY_LIFECYCLE: dict[RetrievalLifecycle, WorkflowSpec] = {
    RetrievalLifecycle.STATELESS: STATELESS_RETRIEVAL_WORKFLOW,
    RetrievalLifecycle.GRAPH_RERANK: GRAPH_RERANK_WORKFLOW,
    RetrievalLifecycle.RGCN_TRAINABLE: RGCN_WORKFLOW,
    RetrievalLifecycle.DENSE_FINETUNE: DENSE_FT_WORKFLOW,
}


def _project_ablation_suite(suite: AblationSuitePatch) -> AblationSuiteSpec:
    return AblationSuiteSpec(
        method=suite.method,
        variants=tuple(_project_variant_patch(variant) for variant in suite.variants),
    )


def _project_variant_patch(variant: AblationVariantPatch) -> VariantSpec:
    return VariantSpec(
        identifier=RgcnAblationVariant(variant.identifier),
        changed_dimensions=frozenset(ChangeDimension(dimension) for dimension in variant.changed_dimensions),
        training_config_override=variant.training_config_override,
        baseline_alias=variant.baseline_alias,
    )


ABLATION_SUITE_REGISTRY: dict[str, AblationSuiteSpec] = {
    method: _project_ablation_suite(suite)
    for method, suite in ABLATION_SUITE_PATCHES.items()
}


def get_workflow(method: str) -> WorkflowSpec:
    try:
        definition = Registry.methods.get(method)
        return WORKFLOW_BY_LIFECYCLE[definition.lifecycle]
    except (KeyError, ValueError) as error:
        allowed = ", ".join(method_id.value for method_id in Registry.methods.list_ids())
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
    selected_runtime_methods = set(runtime_methods or (method.value for method in Registry.methods.list_ids()))
    selected_registrations = registrations or {
        method.value: get_workflow(method.value)
        for method in Registry.methods.list_ids()
    }
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
