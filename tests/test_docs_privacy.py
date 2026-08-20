from __future__ import annotations

from pathlib import Path
import re


_LOCAL_PATH = re.compile(
    r"(?:/Users/|/Volumes/|/home/|/private/|/tmp/|file://|~/|[A-Za-z]:\\\\Users\\\\)"
)


def test_research_and_experiment_docs_do_not_expose_local_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    leaks: list[str] = []
    for directory in (root / "docs" / "research", root / "docs" / "experiments"):
        for path in sorted(directory.rglob("*.md")):
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                if _LOCAL_PATH.search(line):
                    leaks.append(f"{path.relative_to(root)}:{line_number}")

    assert leaks == []
