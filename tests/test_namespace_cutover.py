import tempfile
import unittest
from pathlib import Path

from scripts.check_namespace_cutover import find_retired_namespace, validate


RETIRED = "Intelli" + "Route"


class NamespaceCutoverTest(unittest.TestCase):
    def test_clean_openroutiq_tree_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("OpenRoutiQ", encoding="utf-8")
            self.assertEqual([], find_retired_namespace(root))
            self.assertEqual("pass", validate(root)["status"])

    def test_retired_content_and_path_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text(RETIRED, encoding="utf-8")
            package = root / RETIRED.casefold()
            package.mkdir()
            (package / "module.py").write_text("pass\n", encoding="utf-8")
            matches = find_retired_namespace(root)
        self.assertEqual(
            ["README.md", f"{RETIRED.casefold()}/", f"{RETIRED.casefold()}/module.py"],
            matches,
        )

    def test_private_provenance_is_not_a_public_namespace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / ".benchmark-data"
            private.mkdir()
            (private / "raw.json").write_text(RETIRED, encoding="utf-8")
            self.assertEqual([], find_retired_namespace(root))


if __name__ == "__main__":
    unittest.main()
