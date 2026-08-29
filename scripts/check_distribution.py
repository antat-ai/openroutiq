"""Validate OpenRoutiQ wheel and source-distribution contents before publishing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RETIRED_NAMESPACE = "intelli" + "route"
_SHA256_CHUNK = 1024 * 1024
_MAX_TEXT_MEMBER_BYTES = 64 * 1024 * 1024
_CREDENTIAL = re.compile(r"\b(?:sk-or-v1-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{20,})")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if not name or member.is_absolute() or ".." in member.parts or "\\" in name:
        raise ValueError(f"distribution contains unsafe archive member: {name!r}")
    return member


def _scan_member(name: str, data: bytes) -> None:
    member = _safe_member(name)
    if RETIRED_NAMESPACE in name.casefold():
        raise ValueError(f"distribution member uses the retired namespace: {name}")
    filename = member.name
    if member.suffix.casefold() not in _TEXT_SUFFIXES and filename not in {
        "METADATA",
        "entry_points.txt",
    }:
        return
    if len(data) > _MAX_TEXT_MEMBER_BYTES:
        raise ValueError(f"text distribution member exceeds safety limit: {name}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"declared text distribution member is not UTF-8: {name}") from exc
    if RETIRED_NAMESPACE in text.casefold():
        raise ValueError(f"distribution text contains the retired namespace: {name}")
    if _CREDENTIAL.search(text):
        raise ValueError(f"distribution text resembles a live provider credential: {name}")


def _project_identity() -> tuple[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return str(project["name"]), str(project["version"])


def validate_wheel(path: Path, *, project_name: str, version: str) -> dict[str, Any]:
    path = path.resolve()
    expected_prefix = f"{project_name}-{version}-"
    if not path.name.startswith(expected_prefix) or path.suffix != ".whl":
        raise ValueError(f"unexpected wheel filename: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate archive members")
        for name in names:
            _scan_member(name, archive.read(name))
        dist_info = f"{project_name}-{version}.dist-info"
        required = {
            f"{project_name}/__init__.py",
            f"{project_name}/adaptive/__init__.py",
            f"{project_name}/benchmark/__init__.py",
            f"{project_name}/benchmark/cli.py",
            f"{project_name}/benchmark/core.py",
            f"{project_name}/benchmark/templates/__init__.py",
            f"{project_name}/benchmark/templates/catalog.json",
            f"{project_name}/benchmark/templates/recorded.json",
            f"{project_name}/benchmark/templates/replay.json",
            f"{project_name}/observability/__init__.py",
            f"{project_name}/observability/dispatcher.py",
            f"{project_name}/observability/events.py",
            f"{project_name}/observability/http_json.py",
            f"{project_name}/observability/otel.py",
            f"{project_name}/observability/vendors.py",
            f"{project_name}/py.typed",
            f"{project_name}/providers/__init__.py",
            f"{project_name}/proxy/__init__.py",
            f"{project_name}/quickstart/__init__.py",
            f"{project_name}/router/__init__.py",
            f"{project_name}/selection/__init__.py",
            f"{dist_info}/METADATA",
            f"{dist_info}/entry_points.txt",
        }
        missing = sorted(required - set(names))
        if missing:
            raise ValueError("wheel is missing required members: " + ", ".join(missing))
        allowed_roots = {project_name, dist_info}
        unexpected_roots = sorted(
            {
                PurePosixPath(name).parts[0]
                for name in names
                if PurePosixPath(name).parts and PurePosixPath(name).parts[0] not in allowed_roots
            }
        )
        if unexpected_roots:
            raise ValueError(
                "wheel contains unexpected archive roots: " + ", ".join(unexpected_roots)
            )
        metadata_text = archive.read(f"{dist_info}/METADATA").decode("utf-8")
        metadata = Parser().parsestr(metadata_text, headersonly=True)
        if metadata.get_all("Name", []) != [project_name] or metadata.get_all("Version", []) != [
            version
        ]:
            raise ValueError("wheel metadata name/version does not match pyproject.toml")
        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        expected_entries = {
            f"openroutiq = {project_name}.cli:main",
            f"openroutiq-benchmark = {project_name}.benchmark.cli:main",
        }
        if not expected_entries <= set(entry_points.splitlines()):
            raise ValueError("wheel console entry points are incomplete or incorrect")
    return {"file": path.name, "sha256": _sha256(path), "members": len(names)}


def validate_sdist(path: Path, *, project_name: str, version: str) -> dict[str, Any]:
    path = path.resolve()
    expected_name = f"{project_name}-{version}.tar.gz"
    if path.name != expected_name:
        raise ValueError(f"unexpected source-distribution filename: {path.name}")
    root = f"{project_name}-{version}"
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("source distribution contains duplicate archive members")
        for member in members:
            parsed = _safe_member(member.name)
            if not parsed.parts or parsed.parts[0] != root:
                raise ValueError(f"source-distribution member escapes its root: {member.name}")
            if member.isdir():
                if RETIRED_NAMESPACE in member.name.casefold():
                    raise ValueError(
                        f"source-distribution directory uses retired namespace: {member.name}"
                    )
                continue
            if not member.isfile():
                raise ValueError(
                    f"source distribution contains a non-regular member: {member.name}"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read source-distribution member: {member.name}")
            _scan_member(member.name, extracted.read())
        required = {
            f"{root}/pyproject.toml",
            f"{root}/README.md",
            f"{root}/src/{project_name}/__init__.py",
            f"{root}/src/{project_name}/benchmark/cli.py",
            f"{root}/src/{project_name}/benchmark/templates/catalog.json",
            f"{root}/src/{project_name}/benchmark/templates/recorded.json",
            f"{root}/src/{project_name}/benchmark/templates/replay.json",
            f"{root}/examples/domains/customer_support.py",
            f"{root}/examples/domains/financial_document_review.py",
            f"{root}/examples/domains/healthcare_document_assist.py",
            f"{root}/examples/frameworks/langchain_runnable.py",
            f"{root}/examples/frameworks/langgraph_workflow.py",
            f"{root}/examples/observability/exporter_fanout.py",
            f"{root}/scripts/check_namespace_cutover.py",
            f"{root}/scripts/check_distribution.py",
        }
        missing = sorted(required - set(names))
        if missing:
            raise ValueError(
                "source distribution is missing required members: " + ", ".join(missing)
            )
        forbidden_parts = {
            ".benchmark-data",
            ".git",
            ".env",
            ".openroutiq",
            "benchmarks",
            "build",
            "dist",
            "docs",
        }
        leaked = sorted(
            name for name in names if forbidden_parts & set(PurePosixPath(name).parts[1:])
        )
        if leaked:
            raise ValueError(
                "source distribution contains private/build paths: " + ", ".join(leaked)
            )
    return {"file": path.name, "sha256": _sha256(path), "members": len(names)}


def validate(dist_dir: Path) -> dict[str, Any]:
    project_name, version = _project_identity()
    dist_dir = dist_dir.resolve()
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("dist must contain exactly one wheel and one .tar.gz source distribution")
    return {
        "status": "pass",
        "project": project_name,
        "version": version,
        "wheel": validate_wheel(wheels[0], project_name=project_name, version=version),
        "sdist": validate_sdist(sdists[0], project_name=project_name, version=version),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    try:
        result = validate(args.dist)
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
