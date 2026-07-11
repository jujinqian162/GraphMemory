from __future__ import annotations

# ruff: noqa: E402 -- the public filename inspect.py otherwise shadows stdlib inspect

import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry).resolve() != SCRIPT_DIRECTORY]
sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))

from graph_memory.experiment.commands import parse_command
from graph_memory.experiment.reset import ResetCommandConfig, reset_named_run


def main() -> None:
    command = parse_command(ResetCommandConfig, sys.argv[1:])
    target = reset_named_run(
        command.name,
        repository_root=SCRIPT_DIRECTORY.parent,
    )
    print(f"reset={target}")


if __name__ == "__main__":
    main()
