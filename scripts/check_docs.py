"""Check that repository-local Markdown links resolve to public files."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^\s)]+)")
HTML_LINK = re.compile(
    r"<(?:a|img|source)\b[^>]*\b(?:href|src)=[\"'](?P<target>[^\"']+)[\"']",
    re.IGNORECASE,
)
SKIP_PARTS = {".git", ".benchmark-data", "benchmark-incidents", "llmrouterbench-recorded"}
SKIP_NAMES = {"selection-benchmark-v2.md"}
REPOSITORY_RAW_PREFIX = "/antat-ai/openroutiq/main/"


def _markdown_files() -> list[Path]:
    files = list(ROOT.glob("*.md"))
    for directory in (".github", "examples"):
        root = ROOT / directory
        if root.is_dir():
            files.extend(root.rglob("*.md"))
    return sorted(
        {
            path
            for path in files
            if not (set(path.parts) & SKIP_PARTS)
            and path.name not in SKIP_NAMES
            and not path.name.startswith("production-readiness-audit-")
        }
    )


def validate() -> dict[str, int | str]:
    broken: list[str] = []
    checked = 0
    repository_raw_links = 0
    files = _markdown_files()
    for source in files:
        text = source.read_text(encoding="utf-8")
        matches = sorted(
            (*LINK.finditer(text), *HTML_LINK.finditer(text)), key=lambda item: item.start()
        )
        for match in matches:
            target = match.group("target").strip("<>")
            parsed = urlsplit(target)
            if parsed.netloc == "raw.githubusercontent.com" and parsed.path.startswith(
                REPOSITORY_RAW_PREFIX
            ):
                repository_raw_links += 1
                checked += 1
                relative = unquote(parsed.path.removeprefix(REPOSITORY_RAW_PREFIX))
                destination = (ROOT / relative).resolve()
                if not destination.is_file():
                    broken.append(
                        f"{source.relative_to(ROOT)}:"
                        f"{text.count(chr(10), 0, match.start()) + 1}: missing {target}"
                    )
                continue
            if parsed.netloc == "raw.githubusercontent.com" and parsed.path.startswith(
                "/antat-ai/"
            ):
                broken.append(
                    f"{source.relative_to(ROOT)}:"
                    f"{text.count(chr(10), 0, match.start()) + 1}: "
                    f"repository asset URL is not canonical: {target}"
                )
                continue
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            relative = unquote(parsed.path)
            if not relative:
                continue
            checked += 1
            destination = (source.parent / relative).resolve()
            try:
                destination.relative_to(ROOT)
            except ValueError:
                broken.append(
                    f"{source.relative_to(ROOT)}:{text.count(chr(10), 0, match.start()) + 1}: "
                    f"link escapes repository: {target}"
                )
                continue
            if not destination.exists():
                broken.append(
                    f"{source.relative_to(ROOT)}:{text.count(chr(10), 0, match.start()) + 1}: "
                    f"missing {target}"
                )
    if broken:
        raise ValueError("broken local Markdown links:\n" + "\n".join(broken))
    return {
        "status": "pass",
        "markdown_files": len(files),
        "local_links": checked,
        "repository_raw_links": repository_raw_links,
    }


def main() -> int:
    try:
        result = validate()
    except (OSError, UnicodeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
