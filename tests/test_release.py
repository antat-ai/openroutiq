import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import openroutiq
from scripts.check_release import validate_package_structure


ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTest(unittest.TestCase):
    def test_public_exports_exist_and_are_unique(self):
        self.assertEqual(len(openroutiq.__all__), len(set(openroutiq.__all__)))
        for name in openroutiq.__all__:
            self.assertTrue(hasattr(openroutiq, name), name)

    def test_release_metadata_is_consistent(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_release.py"),
                "--tag",
                f"v{openroutiq.__version__}",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual("pass", result["status"])
        self.assertEqual(openroutiq.__version__, result["version"])

    def test_local_documentation_links_resolve(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_docs.py")],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual("pass", result["status"])

    def test_package_structure_uses_domain_packages_and_absolute_imports(self):
        result = validate_package_structure(ROOT / "src" / "openroutiq")
        self.assertEqual(8, result["domain_packages"])
        self.assertGreater(result["modules"], 8)

    def test_package_structure_rejects_relative_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            for name in (
                "adaptive",
                "benchmark",
                "observability",
                "providers",
                "proxy",
                "quickstart",
                "router",
                "selection",
            ):
                domain = package / name
                domain.mkdir()
                (domain / "__init__.py").write_text("", encoding="utf-8")
            for name in ("__init__.py", "__main__.py", "cli.py"):
                (package / name).write_text("", encoding="utf-8")
            (package / "router" / "core.py").write_text(
                "from .failures import FailureType\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "absolute imports"):
                validate_package_structure(package)


if __name__ == "__main__":
    unittest.main()
