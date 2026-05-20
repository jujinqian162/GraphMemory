from __future__ import annotations

import random


def sample_split(examples: list[dict], count: int, seed: int, offset: int = 0) -> list[dict]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    if offset + count > len(examples):
        raise ValueError(
            f"Requested split offset+count={offset + count} exceeds available examples={len(examples)}."
        )

    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected_indices = indices[offset : offset + count]
    return [examples[index] for index in selected_indices]
