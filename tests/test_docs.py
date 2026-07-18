import re
import unittest
from pathlib import Path

from scripts.check_docs import validate


ROOT = Path(__file__).resolve().parents[1]


class DocumentationTest(unittest.TestCase):
    def test_repository_links_and_graph_assets_resolve(self):
        result = validate()
        self.assertEqual("pass", result["status"])
        self.assertEqual(0, int(result["repository_raw_links"]))
        self.assertGreaterEqual(int(result["local_links"]), 15)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        banner = ROOT / "assets" / "banner.png"
        self.assertIn("(assets/banner.png)", readme)
        self.assertNotIn("raw.githubusercontent.com/antat-ai/openroutiq", readme)
        self.assertIn(
            'src="assets/results/selection-router-quality.png"',
            readme,
        )
        self.assertTrue(banner.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

        result_section = readme[
            readme.index("## Benchmark results") : readme.index("## Why routing complexity")
        ]
        self.assertIn("### Router selection benchmark", result_section)
        self.assertIn("| Router | Accuracy | Execution cost", result_section)
        self.assertIn("54.0% lower", result_section)
        self.assertIn("LLMRouter SVM", result_section)
        self.assertIn("Semantic Router configurations", result_section)
        self.assertIn("RouteLLM's best accuracy at 65.1% lower measured cost", result_section)
        self.assertIn(
            "xRouteBench's best published macro by 2.63 percentage points", result_section
        )
        self.assertNotRegex(result_section, r"\bUSD\s+\d")
        for forbidden in (
            "oracle",
            "fixed model",
            "segment rules",
            "thompson",
            "ucb1",
            "avengers",
            "linucb",
            "internal",
            "diagnostic",
            "adaptive promotion",
            "reasoning-level",
        ):
            self.assertNotIn(forbidden, result_section.casefold())
        expected_plots = {
            "assets/results/selection-router-quality.png",
            "assets/results/selection-quality-cost-frontier.png",
            "assets/results/live-openrouter-comparison.png",
            "assets/results/live-framework-task-comparison.png",
            "assets/results/live-multimodal-task-comparison.png",
            "assets/results/live-routellm-task-comparison.png",
            "assets/results/semantic-router-quality-cost.png",
            "assets/results/xroutebench-held-out-performance.png",
        }
        linked_plots = set(re.findall(r"assets/results/[a-z0-9-]+\.png", result_section))
        self.assertEqual(expected_plots, linked_plots)
        for relative in linked_plots:
            self.assertTrue((ROOT / relative).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_release_profile_and_routing_math_are_explicit(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        operations = (ROOT / "docs" / "operations.md").read_text(encoding="utf-8")
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("The package is release-ready", readme)
        self.assertNotIn("Current status:", readme)
        self.assertNotIn("remains alpha", readme.casefold())
        self.assertIn("Development Status :: 5 - Production/Stable", project)
        self.assertIn("release-ready for application", operations)

        self.assertIn("## Why routing complexity explodes in today's AI ecosystem", readme)
        self.assertIn(r"\log(n)", readme)
        self.assertIn(r"\text{Static matrix size}=T\times V", readme)
        self.assertIn(r"\text{Possible static routing policies}=V^T", readme)
        self.assertIn(r"\underset{v\in\mathcal V_{\pi}(x)}{\arg\max}", readme)
        self.assertLess(readme.index("## Benchmark results"), readme.index("## Why routing"))
        self.assertLess(readme.index("## Why routing"), readme.index("## Contributing"))

        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("benchmarks/", gitignore)
        self.assertIn(".benchmark-data/", gitignore)
        self.assertIn("Benchmark datasets", gitignore)

        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("prune benchmarks", manifest)
        self.assertIn("exclude THIRD_PARTY_DATA.md", manifest)


if __name__ == "__main__":
    unittest.main()
