from __future__ import annotations

import nltk

REQUIRED_RESOURCES: tuple[str, ...] = (
    "brown",
    "treebank",
    "averaged_perceptron_tagger_eng",
    "punkt",
    "punkt_tab",
)


def main() -> int:
    for resource_name in REQUIRED_RESOURCES:
        nltk.download(resource_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
