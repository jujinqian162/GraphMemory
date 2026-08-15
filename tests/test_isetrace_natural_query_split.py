from __future__ import annotations


from graph_memory.datasets.isetrace import (
    allocate_trajectory_splits,
)
from graph_memory.query_synthesis.provenance.authoring import (
    AuthoringGold,
    AuthoringQueryRecord,
)
from graph_memory.text.chunking import TokenChunkingConfig


class CharacterOffsetTokenizer:
    is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> dict[str, object]:
        del add_special_tokens, return_offsets_mapping, truncation
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


_CHUNKING = TokenChunkingConfig(
    tokenizer_name="test-character-tokenizer",
    max_tokens=512,
    overlap_tokens=64,
)


def _query(trajectory, *, query_id: str) -> AuthoringQueryRecord:
    call = trajectory.tool_calls[0]
    output = next(
        item for item in trajectory.tool_outputs if item.call_id == call.call_id
    )
    sections = [
        *(
            f"[I{index} | user_intent]\n{intent.text}"
            for index, intent in enumerate(trajectory.intents, start=1)
        ),
        f"[A1 | tool_call | {call.tool_name}]\n{call.raw_arguments}",
        f"[E1 | tool_output | {output.tool_name}]\n{output.content}",
    ]
    return AuthoringQueryRecord(
        id=query_id,
        text="\n\n".join(sections),
        query=f"Which recorded output answers {query_id}?",
        gold=(AuthoringGold(source="E1", quote=output.content),),
    )


def _trajectory_splits(
    *,
    natural: tuple[int, int, int] = (1, 1, 1),
) -> dict[str, int]:
    return {
        split: natural[index] for index, split in enumerate(("train", "dev", "test"))
    }


def test_trajectory_split_counts_are_disjoint() -> None:
    trajectory_ids = {f"trajectory:{index}" for index in range(6)}
    splits = allocate_trajectory_splits(
        trajectory_ids,
        split_counts=_trajectory_splits(natural=(2, 1, 1)),
        split_seed=41,
    )

    assert len(splits["train"]) == 2
    assert len(splits["dev"]) == 1
    assert len(splits["test"]) == 1
    assert not (splits["train"] & splits["dev"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["dev"] & splits["test"])


def test_trajectory_split_depends_only_on_split_seed() -> None:
    trajectory_ids = {f"trajectory:{index}" for index in range(30)}
    counts = _trajectory_splits(natural=(10, 10, 10))

    seed_13_a = allocate_trajectory_splits(
        trajectory_ids, split_counts=counts, split_seed=13
    )
    seed_13_b = allocate_trajectory_splits(
        trajectory_ids, split_counts=counts, split_seed=13
    )
    seed_17 = allocate_trajectory_splits(
        trajectory_ids, split_counts=counts, split_seed=17
    )

    assert seed_13_a == seed_13_b
    assert seed_13_a != seed_17


def test_smaller_train_count_keeps_dev_and_test_fixed_and_train_nested() -> None:
    trajectory_ids = [f"trajectory:{index}" for index in range(40)]
    smaller = allocate_trajectory_splits(
        trajectory_ids,
        split_counts=_trajectory_splits(natural=(10, 6, 5)),
        split_seed=13,
    )
    larger = allocate_trajectory_splits(
        trajectory_ids,
        split_counts=_trajectory_splits(natural=(20, 6, 5)),
        split_seed=13,
    )

    assert smaller["test"] == larger["test"] == frozenset(trajectory_ids[:5])
    assert smaller["dev"] == larger["dev"]
    assert smaller["train"] < larger["train"]
    assert not (larger["train"] & larger["dev"])
    assert not (larger["train"] & larger["test"])
    assert not (larger["dev"] & larger["test"])
