"""Fail unless the public tree contains only the OpenRoutiQ namespace."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETIRED_NAMESPACE = "intelli" + "route"
_PRIVATE_DIRECTORIES = {
    ".git",
    ".benchmark-data",
    ".mypy_cache",
    ".openroutiq",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "benchmarks",
    "benchmark-incidents",
    "dist",
    "docs",
    "node_modules",
}
_TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def find_retired_namespace(root: Path) -> list[str]:
    """Return public paths and text files containing the retired namespace."""

    root = root.resolve()
    found: set[str] = set()
    needle = RETIRED_NAMESPACE.casefold()
    for directory, directory_names, filenames in os.walk(root):
        directory_names[:] = [name for name in directory_names if name not in _PRIVATE_DIRECTORIES]
        directory_path = Path(directory)
        for directory_name in directory_names:
            relative = (directory_path / directory_name).relative_to(root).as_posix()
            if needle in relative.casefold():
                found.add(relative + "/")
        for filename in filenames:
            path = directory_path / filename
            relative = path.relative_to(root).as_posix()
            if needle in relative.casefold():
                found.add(relative)
            if path.suffix.casefold() not in _TEXT_SUFFIXES:
                continue
            if needle in path.read_text(encoding="utf-8").casefold():
                found.add(relative)
    return sorted(found)


def validate(root: Path = ROOT) -> dict[str, object]:
    matches = find_retired_namespace(root)
    if matches:
        raise ValueError("retired namespace remains in: " + ", ".join(matches))
    return {"status": "pass", "root": str(root.resolve()), "matches": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = validate(args.root)
    except (OSError, UnicodeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
