from __future__ import annotations

import json
from itertools import groupby
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from graph_memory.experiment.config import (
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.layout import (
    MultirunIdentity,
    RunLayout,
    concise_override_dirname,
)
from graph_memory.experiment.planning import WorkflowPlanner, format_plan
from graph_memory.experiment.service import initialize_experiment
from graph_memory.registry import Registry
from graph_memory.registry.ablations import (
    ABLATION_SUITE_PATCHES,
    ExecutableAblationVariant,
)
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.experiment.stage_models import (
    AblationSelection,
    Bm25RetrieveStageConfig,
    DenseGraphRerankRetrieveStageConfig,
    PairStageConfig,
    OrdinaryRgcnTrainStageConfig,
)
from graph_memory.registry.methods import ArtifactKind
from graph_memory.registry.retrieval import RetrievalMethodId


REPO_ROOT = Path(__file__).parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
LEGACY_FIXTURE = json.loads(
    (
        REPO_ROOT / "tests/fixtures/refactor_7_9/legacy_workflow_contracts.json"
    ).read_text(encoding="utf-8")
)


def _resolved(
    *,
    config_name: str = "config",
    overrides: list[str] | None = None,
):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        composed = compose(
            config_name=config_name,
            overrides=["name=planning-test", "profile=smoke", *(overrides or [])],
        )
    return resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=REPO_ROOT,
    )


def _groups(plan) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for stage, items_iter in groupby(plan.invocations, key=lambda item: item.stage):
        items = list(items_iter)
        groups.append(
            {
                "stage": stage,
                "splits": [item.split for item in items if item.split is not None],
                "methods": [
                    item.method.value for item in items if item.method is not None
                ],
            }
        )
    return groups


def _expected_groups(contract_name: str) -> list[dict[str, object]]:
    contract = LEGACY_FIXTURE[contract_name]
    source = (
        contract.get("groups") or LEGACY_FIXTURE[contract["same_groups_as"]]["groups"]
    )
    return [
        {
            "stage": group["stage"],
            "splits": group["splits"],
            "methods": group["methods"],
        }
        for group in source
    ]


def _complete(invocation) -> None:
    for output in invocation.outputs:
        output.path.parent.mkdir(parents=True, exist_ok=True)
        if output.kind == "file":
            output.path.write_text("[]", encoding="utf-8")
        else:
            output.path.mkdir(parents=True, exist_ok=True)
    with stage_lifecycle(invocation):
        pass


def test_run_layout_owns_single_multirun_variant_and_artifact_paths(
    tmp_path: Path,
) -> None:
    single = RunLayout(tmp_path, "demo")
    multi = RunLayout(
        tmp_path,
        "sweep",
        identity=MultirunIdentity(
            job_num=2,
            suffix="num_layers=4",
        ),
    )

    assert single.run_dir == tmp_path.resolve() / "runs" / "demo"
    assert single.resolved_config == single.run_dir / "config/resolved.yaml"
    assert (
        single.stage_config("retrieve", method=RetrievalMethodId.DENSE)
        == single.run_dir / "config/stages/retrieve/dense.yaml"
    )
    assert (
        single.stage_config(
            "train",
            method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            variant="wo_graph",
        )
        == single.run_dir
        / "config/stages/ablations/dense_rgcn_graph_retriever/wo_graph/train.yaml"
    )
    assert multi.run_dir.name == "2_num_layers=4"
    assert (
        single.checkpoint(RetrievalMethodId.DENSE_FT, kind="directory").name
        == "best_model"
    )
    assert (
        single.checkpoint(
            RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            kind="file",
        ).name
        == "best.pt"
    )
    with pytest.raises(ValueError, match="run name"):
        RunLayout(tmp_path, "../escape")


def test_concise_override_dirname_is_deterministic_and_retains_dataset() -> None:
    assert concise_override_dirname(
        "name=sweep,dataset=hotpotqa,profile=full,methods=[dense],"
        "method_configs.dense_rgcn_graph_retriever.train.model.num_layers=3,"
        "method_configs.dense_rgcn_graph_retriever.train.trainer.learning_rate=1e-3"
    ) == "dataset=hotpotqa,num_layers=3,learning_rate=1e-3"


def test_concise_override_dirname_accepts_dataset_only_multirun() -> None:
    assert concise_override_dirname(
        "name=beam-rgcn,dataset=musique,methods=[dense_rgcn_graph_retriever]"
    ) == "dataset=musique"


def test_concise_override_dirname_sanitizes_values_and_rejects_leaf_collisions() -> None:
    assert concise_override_dirname("model.output=a/b:c") == "output=a_b_c"
    with pytest.raises(ValueError, match="ambiguous.*num_layers"):
        concise_override_dirname("encoder.num_layers=2,model.num_layers=3")


def test_initialize_persists_complete_overrides_below_concise_layout(
    tmp_path: Path,
) -> None:
    config = _resolved()
    layout = RunLayout(
        tmp_path,
        config.name,
        identity=MultirunIdentity(job_num=0, suffix="num_layers=2"),
    )
    overrides = (
        "name=planning-test",
        "dataset=hotpotqa",
        "method_configs.dense_rgcn_graph_retriever.train.model.num_layers=2",
    )

    initialize_experiment(config, layout=layout, overrides=overrides)

    assert layout.run_dir.name == "0_num_layers=2"
    assert "dataset=hotpotqa" in layout.overrides.read_text(encoding="utf-8")
    assert "num_layers=2" in layout.overrides.read_text(encoding="utf-8")
    assert layout.resolved_config.is_file()


def test_typed_method_registry_projects_all_eight_runtime_contracts() -> None:
    assert Registry.methods.list_ids() == tuple(RetrievalMethodId)
    seeded = Registry.methods.get(RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER)
    assert seeded.train_dependencies == (RetrievalMethodId.DENSE_FT,)
    assert seeded.train_artifact is not None
    assert seeded.train_artifact.kind is ArtifactKind.FILE
    dense_ft = Registry.methods.get(RetrievalMethodId.DENSE_FT)
    assert dense_ft.train_artifact is not None
    assert dense_ft.train_artifact.kind is ArtifactKind.DIRECTORY
    assert Registry.methods.expand_train_dependencies(
        (RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,)
    ) == (
        RetrievalMethodId.DENSE_FT,
        RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
    )


@pytest.mark.parametrize(
    ("contract_name", "overrides"),
    [
        ("hotpotqa_default", []),
        ("twowiki", ["dataset=2wiki"]),
        ("musique", ["dataset=musique"]),
        (
            "hotpotqa_all_methods",
            [
                "dataset=hotpotqa-memory-stream",
                "methods=[bm25,dense,memory_stream,bm25_graph_rerank,dense_graph_rerank,dense_rgcn_graph_retriever,dense_ft,dense_ft_rgcn_graph_retriever]",
            ],
        ),
    ],
)
def test_typed_plan_preserves_frozen_stage_method_split_order(
    tmp_path: Path,
    contract_name: str,
    overrides: list[str],
) -> None:
    plan = WorkflowPlanner(
        _resolved(overrides=overrides),
        RunLayout(tmp_path, "planning-test"),
    ).build()
    assert _groups(plan) == _expected_groups(contract_name)
    assert all(
        item.argv[2] == "--config" and item.argv[3].endswith(".yaml")
        for item in plan.invocations
    )


def test_seeded_rgcn_plan_keeps_hidden_dense_ft_dependency(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "planning-test")
    unbounded = WorkflowPlanner(
        _resolved(overrides=["methods=[dense_ft_rgcn_graph_retriever]"]),
        layout,
    ).build()
    for invocation in unbounded.invocations:
        if invocation.stage in {"prepare", "graphs"}:
            _complete(invocation)
    plan = WorkflowPlanner(
        _resolved(
            overrides=[
                "methods=[dense_ft_rgcn_graph_retriever]",
                "stages.from=pairs",
                "stages.to=evaluate",
            ]
        ),
        layout,
    ).build()
    assert all(item.method is not None for item in plan.invocations)
    assert [
        [item.stage, item.method.value if item.method is not None else None]
        for item in plan.invocations
    ] == LEGACY_FIXTURE["seeded_rgcn_dependency"]


def test_method_discriminated_stage_configs_do_not_carry_unrelated_fields(
    tmp_path: Path,
) -> None:
    plan = WorkflowPlanner(_resolved(), RunLayout(tmp_path, "planning-test")).build()
    bm25 = next(
        item.config
        for item in plan.invocations
        if item.stage == "retrieve" and item.method is RetrievalMethodId.BM25
    )
    dense_graph = next(
        item.config
        for item in plan.invocations
        if item.stage == "retrieve"
        and item.method is RetrievalMethodId.DENSE_GRAPH_RERANK
    )
    pair = next(item.config for item in plan.invocations if item.stage == "pairs")

    assert isinstance(bm25, Bm25RetrieveStageConfig)
    assert set(bm25.model_dump()) == {
        "stage",
        "method",
        "variant",
        "dataset",
        "tasks",
        "output",
        "top_k",
    }
    assert isinstance(dense_graph, DenseGraphRerankRetrieveStageConfig)
    assert dense_graph.graphs and dense_graph.selected_config and dense_graph.encoder
    assert isinstance(pair, PairStageConfig)


def test_intermediate_stage_requires_bound_artifacts_before_planning(
    tmp_path: Path,
) -> None:
    config = _resolved(
        overrides=[
            "methods=[dense_rgcn_graph_retriever]",
            "stages.from=retrieve",
            "stages.to=retrieve",
        ]
    )
    layout = RunLayout(tmp_path, "planning-test")
    with pytest.raises(ValueError, match="external dependency.*inputs"):
        WorkflowPlanner(config, layout).build()

    unbounded = WorkflowPlanner(
        _resolved(overrides=["methods=[dense_rgcn_graph_retriever]"]),
        layout,
    ).build()
    for invocation in unbounded.invocations:
        if invocation.identifier in {
            "prepare:test",
            "graphs:test",
            "train:dense_rgcn_graph_retriever",
        }:
            _complete(invocation)
    plan = WorkflowPlanner(config, layout).build()
    assert [(item.stage, item.method) for item in plan.invocations] == [
        ("retrieve", RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER)
    ]


def test_ablation_planning_preserves_invalidation_alias_and_variant_namespaces(
    tmp_path: Path,
) -> None:
    config = _resolved(
        overrides=[
            "methods=[dense_rgcn_graph_retriever]",
            "ablation.variants=[wo_graph,wo_hard_negatives]",
        ]
    )
    plan = WorkflowPlanner(config, RunLayout(tmp_path, "planning-test")).build()

    variant_rows = [
        (item.variant, item.stage)
        for item in plan.invocations
        if item.variant is not None
    ]
    assert variant_rows == [
        ("wo_graph", "train"),
        ("wo_graph", "retrieve"),
        ("wo_graph", "evaluate"),
        ("wo_hard_negatives", "pairs"),
        ("wo_hard_negatives", "train"),
        ("wo_hard_negatives", "retrieve"),
        ("wo_hard_negatives", "evaluate"),
    ]
    wo_graph_train = next(
        item.config
        for item in plan.invocations
        if item.variant == "wo_graph" and item.stage == "train"
    )
    wo_hard_pairs = next(
        item.config
        for item in plan.invocations
        if item.variant == "wo_hard_negatives" and item.stage == "pairs"
    )
    assert isinstance(wo_graph_train, OrdinaryRgcnTrainStageConfig)
    assert wo_graph_train.train is not None
    assert wo_graph_train.train.model.ablation == "wo_graph"
    assert wo_graph_train.train.model.num_layers == 0
    assert isinstance(wo_hard_pairs, PairStageConfig)
    assert wo_hard_pairs.sampling.hard_bm25_per_positive == 0
    assert wo_hard_pairs.sampling.hard_dense_per_positive == 0
    assert wo_hard_pairs.sampling.hard_graph_neighbor_per_positive == 0
    assert [
        (alias.stage, alias.variant, alias.artifact.role) for alias in plan.aliases
    ] == [
        ("pairs", "full_rgcn", "train_pairs"),
        ("train", "full_rgcn", "checkpoint"),
        ("retrieve", "full_rgcn", "predictions"),
        ("evaluate", "full_rgcn", "metrics"),
        ("pairs", "wo_graph", "train_pairs"),
    ]
    assert plan.ablation_selections == (
        AblationSelection(
            method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            variant="full_rgcn",
        ),
        AblationSelection(
            method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            variant="wo_graph",
        ),
        AblationSelection(
            method=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            variant="wo_hard_negatives",
        ),
    )


def test_all_registered_executable_ablation_variants_are_plannable(
    tmp_path: Path,
) -> None:
    method = RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER
    config = _resolved(
        overrides=[
            "methods=[dense_rgcn_graph_retriever]",
            "ablation.variants=all",
        ]
    )
    plan = WorkflowPlanner(config, RunLayout(tmp_path, "planning-test")).build()
    expected = {
        variant.identifier
        for variant in ABLATION_SUITE_PATCHES[method].variants
        if isinstance(variant, ExecutableAblationVariant)
    }
    planned = {
        item.variant
        for item in plan.invocations
        if item.variant is not None and item.stage == "train"
    }
    assert planned == expected


def test_ablation_only_requires_ordinary_baseline_metric(tmp_path: Path) -> None:
    config = _resolved(
        overrides=[
            "methods=[dense_rgcn_graph_retriever]",
            "ablation.variants=[wo_graph]",
            "ablation.only=true",
        ]
    )
    with pytest.raises(ValueError, match="baseline metrics"):
        WorkflowPlanner(config, RunLayout(tmp_path, "planning-test")).build()


def test_plan_formatter_exposes_stage_script_config_and_full_argv(
    tmp_path: Path,
) -> None:
    plan = WorkflowPlanner(
        _resolved(overrides=["methods=[bm25]", "stages.to=prepare"]),
        RunLayout(tmp_path, "planning-test"),
    ).build()
    rendered = format_plan(plan)
    assert "[1] stage=prepare split=train" in rendered
    assert "script:" in rendered
    assert "config:" in rendered
    assert "command:" in rendered
    assert "  --config" in rendered
