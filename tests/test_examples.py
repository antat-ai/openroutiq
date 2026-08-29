import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from openroutiq import ObservabilityError, Router

from examples.common.observability import build_observability_from_environment
from examples.domains.customer_support import route_support_request
from examples.domains.financial_document_review import route_financial_document
from examples.domains.healthcare_document_assist import route_clinical_document
from examples.frameworks.langchain_runnable import build_routed_chain


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = Router.from_file(PROJECT_ROOT / "models.example.json", review_margin=0)

    def test_domain_examples_route_offline_with_explicit_safety_policies(self):
        support = route_support_request(
            self.router,
            [{"role": "user", "content": "Check synthetic order REF-1007."}],
        )
        finance = route_financial_document(
            self.router,
            "Synthetic statement: revenue 100 USD; operating expense 80 USD.",
        )
        healthcare = route_clinical_document(
            self.router,
            "Synthetic discharge document with a follow-up date.",
        )

        self.assertEqual("example", support.selected.provider)
        self.assertTrue(finance.analyst_approval_required)
        self.assertTrue(finance.decision.review_required)
        self.assertTrue(healthcare.clinician_approval_required)
        self.assertFalse(healthcare.autonomous_care_decision_allowed)
        self.assertTrue(healthcare.decision.review_required)

    def test_framework_examples_import_without_optional_frameworks(self):
        langchain = importlib.import_module("examples.frameworks.langchain_runnable")
        langgraph = importlib.import_module("examples.frameworks.langgraph_workflow")

        self.assertTrue(callable(langchain.build_routed_chain))
        self.assertTrue(callable(langgraph.build_workflow))

    def test_langchain_registry_dispatches_only_to_a_trusted_runnable(self):
        class RunnableLambda:
            def __init__(self, function):
                self.function = function

            def invoke(self, value):
                return self.function(value)

            def __or__(self, other):
                return RunnableLambda(lambda value: other.invoke(self.invoke(value)))

        package = types.ModuleType("langchain_core")
        runnables = types.ModuleType("langchain_core.runnables")
        runnables.RunnableLambda = RunnableLambda
        package.runnables = runnables
        offline = RunnableLambda(lambda _messages: {"content": "offline"})
        with patch.dict(
            sys.modules,
            {"langchain_core": package, "langchain_core.runnables": runnables},
        ):
            chain = build_routed_chain(
                self.router,
                {profile.id: offline for profile in self.router.profiles},
            )
            result = chain.invoke(
                {"messages": [{"role": "user", "content": "Explain retry safety."}]}
            )

        self.assertIn(result["selected_model_id"], {profile.id for profile in self.router.profiles})
        self.assertEqual("offline", result["result"]["content"])

    def test_observability_configuration_is_off_without_explicit_backends(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(build_observability_from_environment())

    def test_observability_configuration_rejects_unknown_backends_without_secrets(self):
        with patch.dict(
            os.environ,
            {"OPENROUTIQ_OBSERVABILITY_BACKENDS": "unknown"},
            clear=True,
        ):
            with self.assertRaisesRegex(ObservabilityError, "unsupported") as raised:
                build_observability_from_environment()
        self.assertNotIn("API_KEY", str(raised.exception))

    def test_examples_use_package_imports_instead_of_relative_imports(self):
        offenders = []
        for path in sorted((PROJECT_ROOT / "examples").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "from ." in text:
                offenders.append(path.relative_to(PROJECT_ROOT).as_posix())
        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
