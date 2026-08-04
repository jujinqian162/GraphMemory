from __future__ import annotations

from graph_memory.trajectories.contracts import SourceSpan


def source_spans_overlap(left: SourceSpan, right: SourceSpan) -> bool:
    """Return whether two exact spans overlap in the same source field."""

    if left.event_id != right.event_id or left.json_pointer != right.json_pointer:
        return False
    if (
        left.char_start is None
        or left.char_end is None
        or right.char_start is None
        or right.char_end is None
    ):
        return False
    return max(left.char_start, right.char_start) < min(left.char_end, right.char_end)


__all__ = ["source_spans_overlap"]
