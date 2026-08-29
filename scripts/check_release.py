"""Validate version and metadata consistency before building or publishing."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_ACTION_REF = re.compile(r"uses:\s+[^\s@]+@([^\s#]+)")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_CREDENTIAL = re.compile(r"\b(?:sk-or-v1-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{20,})")
_PUBLIC_TEXT_SUFFIXES = {
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
_PRIVATE_PARTS = {
    ".agents",
    ".codex",
    ".freebuff",
    ".git",
    ".benchmark-data",
    ".mypy_cache",
    ".openroutiq",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "benchmarks",
    "benchmark-incidents",
    "build",
    "dist",
    "docs",
    "node_modules",
}
_PRIVATE_FILE_PATTERNS = {
    "requirement.md",
    "scripts/check_benchmark_evidence.py",
    "tests/test_benchmark*.py",
    "tests/test_portable_report.py",
    "tests/test_risk_replay.py",
    "tests/test_selection_comparator_protocol.py",
    "tests/test_selection_eval.py",
    "tests/test_selection_evidence_publication.py",
    "tests/test_selection_live_ledger_audit.py",
    "tests/test_selection_manifest.py",
    "tests/test_selection_plots.py",
    "tests/test_selection_report*.py",
    "tests/test_xroutebench.py",
}
_DOMAIN_PACKAGES = {
    "adaptive",
    "benchmark",
    "observability",
    "providers",
    "proxy",
    "quickstart",
    "router",
    "selection",
}
_ROOT_MODULES = {"__init__.py", "__main__.py", "cli.py"}


def _validate_no_public_credentials() -> int:
    checked = 0
    for directory, directory_names, filenames in os.walk(ROOT):
        directory_names[:] = [name for name in directory_names if name not in _PRIVATE_PARTS]
        for filename in filenames:
            path = Path(directory) / filename
            relative = path.relative_to(ROOT).as_posix()
            if any(fnmatch.fnmatchcase(relative, pattern) for pattern in _PRIVATE_FILE_PATTERNS):
                continue
            if filename == ".env":
                continue
            if path.suffix.casefold() not in _PUBLIC_TEXT_SUFFIXES:
                continue
            checked += 1
            text = path.read_text(encoding="utf-8")
            if "\N{EM DASH}" in text:
                raise ValueError(f"public file contains an em dash: {path.relative_to(ROOT)}")
            if _CREDENTIAL.search(text):
                raise ValueError(
                    f"public file resembles a live provider credential: {path.relative_to(ROOT)}"
                )
    return checked


def validate_package_structure(package_root: Path | None = None) -> dict[str, int]:
    """Validate domain packages and reject ambiguous relative imports."""

    package_root = (package_root or ROOT / "src" / "openroutiq").resolve()
    if not package_root.is_dir():
        raise ValueError(f"package directory is missing: {package_root}")

    missing_packages = sorted(
        name for name in _DOMAIN_PACKAGES if not (package_root / name / "__init__.py").is_file()
    )
    if missing_packages:
        raise ValueError("domain packages are missing: " + ", ".join(missing_packages))

    root_modules = {path.name for path in package_root.glob("*.py")}
    unexpected_modules = sorted(root_modules - _ROOT_MODULES)
    if unexpected_modules:
        raise ValueError(
            "source modules must live in domain packages: " + ", ".join(unexpected_modules)
        )

    module_paths = sorted(package_root.rglob("*.py"))
    relative_imports: list[str] = []
    for path in module_paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            raise ValueError(f"cannot parse package module {path}: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                relative_imports.append(
                    f"{path.relative_to(package_root).as_posix()}:{node.lineno}"
                )
    if relative_imports:
        raise ValueError(
            "package modules must use absolute imports: " + ", ".join(relative_imports)
        )

    return {"domain_packages": len(_DOMAIN_PACKAGES), "modules": len(module_paths)}


def validate(tag: str | None = None) -> dict[str, str]:
    package_structure = validate_package_structure()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build_requirements = pyproject.get("build-system", {}).get("requires", [])
    if not build_requirements or any(
        not isinstance(requirement, str) or "==" not in requirement
        for requirement in build_requirements
    ):
        raise ValueError("build-system requirements must be exact pins")
    project = pyproject.get("project", {})
    if project.get("name") != "openroutiq":
        raise ValueError("project name must be openroutiq")
    expected_scripts = {
        "openroutiq": "openroutiq.cli:main",
        "openroutiq-benchmark": "openroutiq.benchmark.cli:main",
    }
    if project.get("scripts") != expected_scripts:
        raise ValueError("console scripts must expose only the OpenRoutiQ entry points")
    package_version = str(project.get("version", ""))
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[A-Za-z0-9.-]+)?", package_version):
        raise ValueError(f"invalid project version: {package_version!r}")

    sys.path.insert(0, str(ROOT / "src"))
    try:
        import openroutiq
    finally:
        sys.path.pop(0)
    runtime_version = openroutiq.__version__

    versions = {package_version, runtime_version}
    if len(versions) != 1:
        raise ValueError(
            f"version mismatch: pyproject={package_version}, runtime={runtime_version}"
        )
    if tag is not None and tag.removeprefix("refs/tags/") != f"v{package_version}":
        raise ValueError(f"release tag must be v{package_version}, got {tag!r}")
    if project.get("license") != "MIT" or "MIT License" not in (ROOT / "LICENSE").read_text(
        encoding="utf-8"
    ):
        raise ValueError("pyproject and LICENSE must both declare MIT")

    required = (
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "assets/banner.png",
        "constraints/release.txt",
        "scripts/check_namespace_cutover.py",
        "scripts/check_distribution.py",
        "src/openroutiq/py.typed",
    )
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise ValueError(f"required release files are missing: {', '.join(missing)}")

    constraints = (ROOT / "constraints" / "release.txt").read_text(encoding="utf-8")
    release_series = ".".join(package_version.split(".")[:2])
    if f"OpenRoutiQ {release_series} release" not in constraints:
        raise ValueError("release constraints header does not match the package release series")
    for required_pin in (
        "opentelemetry-api==",
        "opentelemetry-exporter-otlp-proto-grpc==",
        "opentelemetry-exporter-otlp-proto-http==",
        "opentelemetry-sdk==",
        "pip==",
        "setuptools==",
    ):
        if not any(
            line.strip().startswith(required_pin)
            for line in constraints.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ):
            raise ValueError(f"release constraints are missing an exact {required_pin[:-2]} pin")

    workflow_paths = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    if not workflow_paths:
        raise ValueError("no GitHub Actions workflows found")
    action_count = 0
    for workflow_path in workflow_paths:
        workflow = workflow_path.read_text(encoding="utf-8")
        refs = _ACTION_REF.findall(workflow)
        action_count += len(refs)
        unpinned = [ref for ref in refs if _COMMIT_SHA.fullmatch(ref) is None]
        if unpinned:
            raise ValueError(
                f"{workflow_path.relative_to(ROOT)} has non-immutable Action refs: "
                + ", ".join(unpinned)
            )
        for line in workflow.splitlines():
            if "pip install" in line and '-e ".[' in line:
                if "-c constraints/release.txt" not in line:
                    raise ValueError(
                        f"{workflow_path.relative_to(ROOT)} installs release extras "
                        "without constraints/release.txt"
                    )
    if action_count == 0:
        raise ValueError("release workflows contain no pinned Actions")

    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    if "recursive-include constraints *.txt" not in manifest:
        raise ValueError("source distribution must include release constraints")
    if "recursive-include assets *.png" not in manifest:
        raise ValueError("source distribution must include brand assets")
    if "prune benchmarks" not in manifest or "prune docs" not in manifest:
        raise ValueError("source distribution must exclude private benchmark and docs trees")
    if "recursive-include src/openroutiq/benchmark/templates *.json" not in manifest:
        raise ValueError("source distribution must include benchmark starter templates")

    credential_scanned_files = _validate_no_public_credentials()

    return {
        "status": "pass",
        "name": str(project.get("name")),
        "version": package_version,
        "tag": f"v{package_version}" if tag is not None else "not-checked",
        "pinned_actions": str(action_count),
        "domain_packages": str(package_structure["domain_packages"]),
        "package_modules": str(package_structure["modules"]),
        "credential_scanned_files": str(credential_scanned_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag, expected to be v<project version>")
    args = parser.parse_args()
    try:
        result = validate(args.tag)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
