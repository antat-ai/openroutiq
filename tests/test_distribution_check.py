import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.check_distribution import validate_sdist, validate_wheel


PROJECT = "openroutiq"
VERSION = "0.1.0"
RETIRED = "intelli" + "route"


def _wheel(
    path: Path,
    *,
    retired: bool = False,
    unexpected_root: bool = False,
    metadata_newline: str = "\n",
    duplicate_metadata: bool = False,
) -> None:
    dist_info = f"{PROJECT}-{VERSION}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{PROJECT}/__init__.py", "")
        for package in (
            "adaptive",
            "benchmark",
            "providers",
            "proxy",
            "quickstart",
            "router",
            "selection",
        ):
            archive.writestr(f"{PROJECT}/{package}/__init__.py", "")
        archive.writestr(f"{PROJECT}/benchmark/cli.py", "")
        archive.writestr(f"{PROJECT}/benchmark/core.py", "")
        archive.writestr(f"{PROJECT}/benchmark/templates/__init__.py", "")
        archive.writestr(f"{PROJECT}/benchmark/templates/catalog.json", "{}")
        archive.writestr(f"{PROJECT}/benchmark/templates/recorded.json", "{}")
        archive.writestr(f"{PROJECT}/benchmark/templates/replay.json", "{}")
        archive.writestr(f"{PROJECT}/py.typed", "")
        metadata_lines = ["Metadata-Version: 2.4", f"Name: {PROJECT}", f"Version: {VERSION}"]
        if duplicate_metadata:
            metadata_lines.append(f"Name: {PROJECT}")
        metadata_lines.append("")
        archive.writestr(
            f"{dist_info}/METADATA",
            metadata_newline.join(metadata_lines),
        )
        archive.writestr(
            f"{dist_info}/entry_points.txt",
            "[console_scripts]\n"
            f"openroutiq = {PROJECT}.cli:main\n"
            f"openroutiq-benchmark = {PROJECT}.benchmark.cli:main\n",
        )
        if retired:
            archive.writestr(f"{PROJECT}/old.py", RETIRED)
        if unexpected_root:
            archive.writestr("other_package/__init__.py", "")


def _sdist(path: Path, *, private_cache: bool = False) -> None:
    root = f"{PROJECT}-{VERSION}"
    files = {
        f"{root}/pyproject.toml": "",
        f"{root}/README.md": "OpenRoutiQ",
        f"{root}/src/{PROJECT}/__init__.py": "",
        f"{root}/src/{PROJECT}/benchmark/cli.py": "",
        f"{root}/src/{PROJECT}/benchmark/templates/catalog.json": "{}",
        f"{root}/src/{PROJECT}/benchmark/templates/recorded.json": "{}",
        f"{root}/src/{PROJECT}/benchmark/templates/replay.json": "{}",
        f"{root}/scripts/check_namespace_cutover.py": "",
        f"{root}/scripts/check_distribution.py": "",
    }
    if private_cache:
        files[f"{root}/.benchmark-data/raw.json"] = "{}"
    with tarfile.open(path, "w:gz") as archive:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


class DistributionCheckTest(unittest.TestCase):
    def test_valid_minimal_archives_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / f"{PROJECT}-{VERSION}-py3-none-any.whl"
            sdist = root / f"{PROJECT}-{VERSION}.tar.gz"
            _wheel(wheel)
            _sdist(sdist)
            self.assertGreater(
                validate_wheel(wheel, project_name=PROJECT, version=VERSION)["members"], 0
            )
            self.assertGreater(
                validate_sdist(sdist, project_name=PROJECT, version=VERSION)["members"], 0
            )

    def test_valid_crlf_wheel_metadata_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / f"{PROJECT}-{VERSION}-py3-none-any.whl"
            _wheel(wheel, metadata_newline="\r\n")
            self.assertGreater(
                validate_wheel(wheel, project_name=PROJECT, version=VERSION)["members"], 0
            )

    def test_duplicate_identity_metadata_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / f"{PROJECT}-{VERSION}-py3-none-any.whl"
            _wheel(wheel, duplicate_metadata=True)
            with self.assertRaisesRegex(ValueError, "metadata name/version"):
                validate_wheel(wheel, project_name=PROJECT, version=VERSION)

    def test_retired_namespace_inside_wheel_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / f"{PROJECT}-{VERSION}-py3-none-any.whl"
            _wheel(wheel, retired=True)
            with self.assertRaisesRegex(ValueError, "retired namespace"):
                validate_wheel(wheel, project_name=PROJECT, version=VERSION)

    def test_unexpected_wheel_root_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / f"{PROJECT}-{VERSION}-py3-none-any.whl"
            _wheel(wheel, unexpected_root=True)
            with self.assertRaisesRegex(ValueError, "unexpected archive roots"):
                validate_wheel(wheel, project_name=PROJECT, version=VERSION)

    def test_private_benchmark_cache_inside_sdist_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            sdist = Path(directory) / f"{PROJECT}-{VERSION}.tar.gz"
            _sdist(sdist, private_cache=True)
            with self.assertRaisesRegex(ValueError, "private/build paths"):
                validate_sdist(sdist, project_name=PROJECT, version=VERSION)


if __name__ == "__main__":
    unittest.main()
