import tempfile
import unittest
import json
import math
from pathlib import Path

from openroutiq import (
    Router,
    SelectionIntelligence,
    SelectionPolicy,
    SelectionTrainingObservation,
    SelfLearningRouter,
)


def _profiles():
    base = {
        "provider": "test",
        "quality": {"general": 50},
        "latency_ms": 100,
        "input_price_per_million": 1,
        "output_price_per_million": 1,
        "max_context_tokens": 10_000,
        "capabilities": ["text", "tools"],
    }
    return [
        {**base, "id": "code-model", "model": "code"},
        {**base, "id": "legal-model", "model": "legal"},
    ]


def _training_rows():
    rows = []
    for index in range(40):
        for model_id, success in (("code-model", True), ("legal-model", False)):
            rows.append(
                SelectionTrainingObservation(
                    request=f"repair python concurrency bug worker {index}",
                    model_id=model_id,
                    success=success,
                    quality_score=100 if success else 0,
                    actual_cost_usd=0.001,
                    latency_ms=100,
                )
            )
        for model_id, success in (("code-model", False), ("legal-model", True)):
            rows.append(
                SelectionTrainingObservation(
                    request=f"review legal contract indemnity clause {index}",
                    model_id=model_id,
                    success=success,
                    quality_score=100 if success else 0,
                    actual_cost_usd=0.001,
                    latency_ms=100,
                )
            )
    return rows


class SelectionIntelligenceTest(unittest.TestCase):
    def test_uncertainty_weighted_exploration_reports_exact_propensity(self):
        intelligence = SelectionIntelligence(
            _profiles(),
            policy=SelectionPolicy(exploration_rate=1.0),
            random_seed=7,
        )
        intelligence.observe(
            SelectionTrainingObservation(
                request="calibrated context",
                model_id="code-model",
                success=True,
                quality_score=100,
            )
        )
        choice = intelligence.select("calibrated context", explore=True)
        weights = {
            item.model_id: 1 / math.sqrt(1 + item.calibration_samples)
            for item in choice.predictions
        }
        expected = weights[choice.model_id] / sum(weights.values())
        self.assertTrue(choice.explored)
        self.assertAlmostEqual(expected, choice.selection_probability)

    def test_lightweight_predictor_learns_request_model_complementarity(self):
        intelligence = SelectionIntelligence(
            _profiles(),
            policy=SelectionPolicy(cost_weight=0, latency_weight=0, risk_aversion=0),
            random_seed=7,
        ).fit(_training_rows(), epochs=2)
        code = intelligence.select("repair python worker deadlock")
        legal = intelligence.select("review contract indemnity language")
        self.assertEqual("code-model", code.model_id)
        self.assertEqual("legal-model", legal.model_id)
        self.assertGreater(code.predictions[0].success_probability, 0.8)
        self.assertGreater(legal.predictions[0].success_probability, 0.8)

    def test_state_round_trip_never_persists_request_text(self):
        intelligence = SelectionIntelligence(_profiles(), random_seed=7)
        intelligence.observe(
            SelectionTrainingObservation(
                request="private customer contract secret phrase",
                model_id="legal-model",
                success=True,
                quality_score=90,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = intelligence.save(Path(directory) / "selection.json")
            raw = path.read_text(encoding="utf-8")
            restored = SelectionIntelligence.load(path, _profiles())
        self.assertNotIn("private customer", raw)
        self.assertEqual(intelligence.to_dict(), restored.to_dict())

    def test_self_learning_router_preserves_hard_capability_gate(self):
        trained = SelectionIntelligence(
            _profiles(),
            policy=SelectionPolicy(cost_weight=0, latency_weight=0, risk_aversion=0),
            random_seed=7,
        ).fit(_training_rows(), epochs=2)
        router = SelfLearningRouter(Router(_profiles(), review_margin=0), trained)
        decision = router.route("repair python deadlock", task="general")
        self.assertEqual("code-model", decision.selected.model_id)
        self.assertEqual("self_learning", decision.strategy)
        self.assertFalse(decision.review_required)
        router.record_evaluation(
            "repair python deadlock",
            decision,
            quality_score=100,
            success=True,
        )
        with self.assertRaisesRegex(ValueError, "No eligible model"):
            router.route(
                "inspect an image",
                constraints={"required_capabilities": ["vision"]},
            )

    def test_arbitrary_size_integer_seed_is_stable(self):
        seed = 10**100
        first = SelectionIntelligence(_profiles(), random_seed=seed).select("hello")
        second = SelectionIntelligence(_profiles(), random_seed=seed).select("hello")
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_state_loader_rejects_non_finite_or_out_of_range_weights(self):
        intelligence = SelectionIntelligence(_profiles(), random_seed=7)
        with tempfile.TemporaryDirectory() as directory:
            path = intelligence.save(Path(directory) / "selection.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["states"]["code-model"]["success_weights"] = {"16384": 1.0}
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "out of range"):
                SelectionIntelligence.load(path, _profiles())


if __name__ == "__main__":
    unittest.main()
