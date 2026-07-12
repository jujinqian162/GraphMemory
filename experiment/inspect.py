from __future__ import annotations

# ruff: noqa: E402 -- this public filename must be removed from import search first

import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry).resolve() != SCRIPT_DIRECTORY]
sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))

import json

from graph_memory.experiment.commands import parse_command
from graph_memory.experiment.inspect import InspectCommandConfig, inspect_catalog


def main() -> None:
    command = parse_command(InspectCommandConfig, sys.argv[1:])
    catalog = inspect_catalog(
        command.kind,
        repository_root=SCRIPT_DIRECTORY.parent,
        name=command.name,
    )
    print(json.dumps(catalog, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
