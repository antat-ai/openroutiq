import unittest

from openroutiq import (
    CapabilityGate,
    CapabilityRequirements,
    FailureType,
    NoEligibleModelError,
    Router,
    classify_failure,
)


def _profile(model_id, supported):
    return {
        "id": model_id,
        "provider": "openrouter",
        "model": model_id,
        "api_style": "openrouter",
        "quality": {"general": 80},
        "latency_ms": 100,
        "input_price_per_million": 1,
        "output_price_per_million": 1,
        "max_context_tokens": 4096,
        "capabilities": ["text", "tools", "json_schema"],
        "supported_parameters": supported,
    }


class CapabilityGateTest(unittest.TestCase):
    def test_declared_provider_parameters_are_fail_closed_before_scoring(self):
        router = Router(
            [
                _profile("tools-only", ["tools", "tool_choice"]),
                _profile(
                    "structured",
                    ["tools", "tool_choice", "structured_outputs"],
                ),
            ],
            review_margin=0,
        )
        decision = router.route(
            "call the tool",
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            tool_choice="required",
            response_format={"type": "json_schema", "json_schema": {"schema": {}}},
        )
        self.assertEqual("structured", decision.selected.model_id)
        excluded = next(item for item in decision.excluded if item.model_id == "tools-only")
        self.assertEqual(FailureType.CAPABILITY_MISMATCH, excluded.failure_type)

    def test_gate_reports_context_and_capability_mismatch(self):
        profile = Router([_profile("one", ["tools"])]).profiles[0]
        result = CapabilityGate().evaluate(
            profile,
            CapabilityRequirements(
                capabilities=frozenset({"vision"}),
                context_tokens=5000,
            ),
        )
        self.assertFalse(result.eligible)
        self.assertEqual(FailureType.CAPABILITY_MISMATCH, result.failure_type)
        with self.assertRaises(NoEligibleModelError):
            Router([_profile("one", ["tools"])]).route(
                "hello", constraints={"required_capabilities": ["vision"]}
            )

    def test_failure_classifier_normalizes_common_provider_errors(self):
        self.assertEqual(FailureType.TIMEOUT, classify_failure(TimeoutError()))
        self.assertEqual(FailureType.RATE_LIMIT, classify_failure("too many requests"))
        self.assertEqual(
            FailureType.PROTOCOL_FAILURE,
            classify_failure(RuntimeError("bad input"), status_code=400),
        )


if __name__ == "__main__":
    unittest.main()
