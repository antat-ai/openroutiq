import json
import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openroutiq import (
    AdaptiveModelRegistry,
    AdaptivePolicy,
    AdaptiveRouter,
    AdaptiveStoreError,
    ChanceConstraint,
    OpenRoutiQError,
    RiskPolicy,
)
from openroutiq.cli import main as cli_main


def profile(
    model_id: str,
    *,
    quality: float,
    cost: float = 1.0,
    reasoning_level: str | None = None,
    provider_options=None,
):
    raw = {
        "id": model_id,
        "provider": "test",
        "model": model_id,
        "api_style": "openai_compatible",
        "quality": {"general": quality, "coding": quality},
        "latency_ms": 100,
        "input_price_per_million": cost,
        "output_price_per_million": cost,
        "max_context_tokens": 128_000,
        "capabilities": ["text", "tools", "json_schema"],
        "confidence": 90,
        "provider_options": provider_options or {},
    }
    if reasoning_level is not None:
        raw["reasoning_level"] = reasoning_level
    return raw


class AdaptiveRegistryTest(unittest.TestCase):
    def test_registry_bounds_untrusted_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = AdaptiveModelRegistry(Path(directory) / "adaptive.sqlite3")
            with self.assertRaisesRegex(OpenRoutiQError, "at most 512 characters"):
                registry.record("m" * 513, "general", quality_score=90)
            with self.assertRaisesRegex(OpenRoutiQError, "at most 512 characters"):
                registry.record("model", "t" * 513, quality_score=90)

    def test_fixed_clock_makes_decayed_evidence_reproducible(self):
        as_of = datetime(2026, 8, 25, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as directory:
            registry = AdaptiveModelRegistry(
                Path(directory) / "adaptive.sqlite3",
                clock=lambda: as_of,
            )
            registry.encounter(profile("private", quality=90), source="private")
            first = registry.record("private", "general", quality_score=90)
            second = registry.record("private", "general", quality_score=90)

        self.assertEqual(1.0, first.effective_samples)
        self.assertEqual(2.0, second.effective_samples)
        self.assertEqual(0.0, second.quality_variance)

    def test_cli_adaptive_registry_is_explicit_opt_in(self):
        catalog = Path(__file__).resolve().parents[1] / "models.example.json"
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "adaptive.sqlite3"
            fake_uvicorn = SimpleNamespace(run=Mock())
            fake_app = object()
            with (
                patch.dict(sys.modules, {"uvicorn": fake_uvicorn}),
                patch("openroutiq.proxy.create_app", return_value=fake_app) as create_app,
            ):
                code = cli_main(
                    [
                        "serve",
                        "--catalog",
                        str(catalog),
                        "--adaptive-registry",
                        str(registry),
                    ]
                )
            route_engine = create_app.call_args.args[0]
            registry_created = registry.is_file()

        self.assertEqual(0, code)
        self.assertIsInstance(route_engine, AdaptiveRouter)
        self.assertTrue(registry_created)

    def test_private_model_starts_provisional_and_never_stores_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive.sqlite3"
            router = AdaptiveRouter(
                [profile("trusted", quality=60)],
                registry=path,
                review_margin=0,
            )
            status = router.encounter_opaque(
                model_id="private/legal-v7",
                provider="customer-local",
                model="legal-v7",
                max_context_tokens=64_000,
                capabilities=["text"],
                tasks=["legal_review"],
                latency_ms=50,
                input_price_per_million=0.1,
                output_price_per_million=0.1,
                prior_quality=95,
                base_url="http://model.internal/v1",
                local=True,
            )
            self.assertEqual("provisional", status.task_state)
            self.assertEqual("trusted", router.route("fix this", task="coding").selected.model_id)
            legal = router.route(
                "review this clause",
                task="legal_review",
                pinned_model="private/legal-v7",
            )
            self.assertEqual("private/legal-v7", legal.selected.model_id)

            pinned = router.route(
                "private contract text that must not be stored",
                task="coding",
                pinned_model="private/legal-v7",
            )
            self.assertEqual("private/legal-v7", pinned.selected.model_id)
            self.assertTrue(pinned.review_required)

            router.record_evaluation(
                "private contract text that must not be stored",
                "private/legal-v7",
                95,
                task="coding",
                latency_ms=45,
                actual_cost_usd=0.001,
                success=True,
            )
            learned = router.status("private/legal-v7", task="coding")
            self.assertEqual(0, learned.quality_variance)
            self.assertEqual(45, learned.latency_p95_ms)
            self.assertEqual(0.001, learned.cost_p95_usd)
            with closing(sqlite3.connect(path)) as connection:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(adaptive_observations)"
                    ).fetchall()
                }
                self.assertNotIn("prompt", columns)
                self.assertNotIn("response", columns)
                row = connection.execute(
                    "SELECT task, quality_score, latency_ms FROM adaptive_observations"
                ).fetchone()
            self.assertEqual(("coding", 95.0, 45.0), row)

    def test_outcomes_promote_an_opaque_model_for_only_the_evaluated_task(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = AdaptivePolicy(automatic_promotion=True, prior_samples=2, promotion_samples=8)
            router = AdaptiveRouter(
                [profile("trusted", quality=60, cost=2)],
                registry=Path(directory) / "adaptive.sqlite3",
                policy=policy,
                review_margin=0,
            )
            router.encounter(profile("private", quality=90, cost=0.1), source="private")
            for _ in range(8):
                status = router.record_evaluation(
                    "repair this Python function",
                    "private",
                    95,
                    task="coding",
                    latency_ms=50,
                    actual_cost_usd=0.0001,
                    success=True,
                )

            self.assertEqual("trusted", status.task_state)
            self.assertEqual("private", router.route("fix this", task="coding").selected.model_id)
            self.assertEqual("provisional", router.status("private", task="general").task_state)
            self.assertEqual("trusted", router.route("hello", task="general").selected.model_id)

    def test_risk_policy_uses_adaptive_distribution_without_context_store(self):
        with tempfile.TemporaryDirectory() as directory:
            router = AdaptiveRouter(
                [
                    profile("volatile", quality=95, cost=0.5),
                    profile("steady", quality=82, cost=0.5),
                ],
                registry=Path(directory) / "adaptive.sqlite3",
                review_margin=0,
            )
            for score, success, latency in (
                (100, True, 70),
                (98, True, 80),
                (95, True, 90),
                (20, False, 900),
                (10, False, 1_200),
            ):
                router.record_evaluation(
                    "request",
                    "volatile",
                    score,
                    task="general",
                    latency_ms=latency,
                    actual_cost_usd=0.01,
                    success=success,
                )
            for _ in range(5):
                router.record_evaluation(
                    "request",
                    "steady",
                    82,
                    task="general",
                    latency_ms=100,
                    actual_cost_usd=0.012,
                    success=True,
                )

            decision = router.route(
                "request",
                task="general",
                strategy="risk_aware",
                risk_policy=RiskPolicy(
                    constraints=(ChanceConstraint("success", minimum_probability=0.8),),
                    risk_aversion=0.8,
                    minimum_samples=5,
                    require_observed_probabilities=True,
                ),
            )

            self.assertEqual("steady", decision.selected.model_id)
            self.assertEqual(5, decision.selected.forecast.evidence_samples)
            self.assertEqual(100, decision.selected.forecast.latency_p95_ms)
            self.assertTrue(any(item.model_id == "volatile" for item in decision.excluded))

    def test_automatic_promotion_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = AdaptivePolicy(prior_samples=1, promotion_samples=3)
            registry = AdaptiveModelRegistry(Path(directory) / "adaptive.sqlite3", policy=policy)
            registry.encounter(profile("private", quality=90), source="private")
            for _ in range(5):
                status = registry.record("private", "coding", quality_score=100, success=True)

            self.assertFalse(policy.automatic_promotion)
            self.assertEqual("provisional", status.task_state)

        with self.assertRaisesRegex(OpenRoutiQError, "automatic_promotion"):
            AdaptivePolicy(automatic_promotion=1)

    def test_stale_evidence_automatically_leaves_normal_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive.sqlite3"
            policy = AdaptivePolicy(
                automatic_promotion=True,
                prior_samples=1,
                promotion_samples=3,
                evidence_half_life_days=1,
                trusted_evidence_stale_after_days=2,
            )
            router = AdaptiveRouter(
                [profile("trusted", quality=60, cost=2)],
                registry=path,
                policy=policy,
                review_margin=0,
            )
            router.encounter(profile("private", quality=90, cost=0.1), source="private")
            for _ in range(3):
                promoted = router.record_evaluation(
                    "repair this function",
                    "private",
                    98,
                    task="coding",
                    success=True,
                )
            self.assertEqual("trusted", promoted.task_state)

            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "UPDATE adaptive_observations SET created_at = ?",
                    ("2000-01-01T00:00:00+00:00",),
                )
                connection.commit()

            stale = router.status("private", task="coding")
            self.assertEqual("provisional", stale.task_state)
            self.assertLess(stale.effective_samples, 0.01)
            self.assertEqual(
                ("active", "provisional"),
                router.registry.routing_states("coding")["private"],
            )
            self.assertEqual(
                "trusted",
                router.route("fix this", task="coding").selected.model_id,
            )

    def test_reasoning_levels_learn_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = AdaptivePolicy(automatic_promotion=True, prior_samples=1, promotion_samples=3)
            registry = AdaptiveModelRegistry(Path(directory) / "adaptive.sqlite3", policy=policy)
            registry.encounter(
                profile("private:low", quality=80, reasoning_level="low"),
                source="private",
            )
            registry.encounter(
                profile("private:high", quality=90, reasoning_level="high"),
                source="private",
            )
            for _ in range(3):
                registry.record(
                    "private:high",
                    "coding",
                    quality_score=98,
                    success=True,
                )

            self.assertEqual("trusted", registry.task_state("private:high", "coding"))
            self.assertEqual("provisional", registry.task_state("private:low", "coding"))

    def test_failures_and_quality_drift_degrade_only_the_affected_task(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = AdaptivePolicy(
                automatic_promotion=True,
                prior_samples=1,
                promotion_samples=3,
                consecutive_failure_limit=3,
                drift_window=2,
                drift_min_history=3,
                drift_quality_drop=20,
            )
            registry = AdaptiveModelRegistry(Path(directory) / "adaptive.sqlite3", policy=policy)
            registry.encounter(profile("private", quality=90), source="private")
            for _ in range(5):
                registry.record("private", "coding", quality_score=95, success=True)
            self.assertEqual("trusted", registry.task_state("private", "coding"))
            registry.record("private", "coding", quality_score=10, success=True)
            degraded = registry.record("private", "coding", quality_score=10, success=True)
            self.assertEqual("degraded", degraded.task_state)
            self.assertEqual("provisional", registry.task_state("private", "general"))

            for _ in range(3):
                failed = registry.record("private", "general", success=False)
            self.assertEqual("degraded", failed.task_state)
            self.assertEqual(3, failed.consecutive_failures)

    def test_exploration_is_explicit_budgeted_and_disabled_for_high_risk(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = AdaptivePolicy(
                exploration_rate=0,
                exploration_daily_budget_usd=0.001,
                exploration_max_request_cost_usd=0.001,
            )
            router = AdaptiveRouter(
                [profile("trusted", quality=60, cost=1)],
                registry=Path(directory) / "adaptive.sqlite3",
                policy=policy,
                review_margin=0,
            )
            router.encounter(profile("private", quality=95, cost=1), source="private")

            exploration = router.route("fix", task="coding", explore=True)
            self.assertEqual("private", exploration.selected.model_id)
            self.assertIn("adaptive:exploration", exploration.analysis.signals)
            self.assertTrue(exploration.review_required)

            budget_exhausted = router.route("fix", task="coding", explore=True)
            self.assertEqual("trusted", budget_exhausted.selected.model_id)
            high_risk = router.route("fix", task="coding", explore=True, high_risk=True)
            self.assertEqual("trusted", high_risk.selected.model_id)

    def test_exploration_budget_uses_exact_integer_nano_dollars(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = AdaptivePolicy(
                exploration_daily_budget_usd=0.000000001,
                exploration_max_request_cost_usd=0.000000001,
            )
            registry = AdaptiveModelRegistry(Path(directory) / "adaptive.sqlite3", policy=policy)
            registry.encounter(profile("private", quality=80), source="private")

            self.assertTrue(registry.reserve_exploration("private", "coding", 0.0000000001))
            self.assertFalse(registry.reserve_exploration("private", "coding", 0.0000000001))
            with closing(sqlite3.connect(registry.path)) as connection:
                nanos = connection.execute(
                    "SELECT reserved_cost_nano_usd FROM adaptive_explorations"
                ).fetchone()[0]
            self.assertEqual(1, nanos)

    def test_exploration_budget_is_atomic_across_registry_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive.sqlite3"
            policy = AdaptivePolicy(
                exploration_daily_budget_usd=0.000000010,
                exploration_max_request_cost_usd=0.000000001,
            )
            owner = AdaptiveModelRegistry(path, policy=policy)
            owner.encounter(profile("private", quality=80), source="private")
            registries = [AdaptiveModelRegistry(path, policy=policy) for _ in range(20)]

            with ThreadPoolExecutor(max_workers=20) as executor:
                accepted = list(
                    executor.map(
                        lambda registry: registry.reserve_exploration(
                            "private", "coding", 0.000000001
                        ),
                        registries,
                    )
                )

            self.assertEqual(10, sum(accepted))
            with closing(sqlite3.connect(path)) as connection:
                count, nanos = connection.execute(
                    "SELECT COUNT(*), SUM(reserved_cost_nano_usd) FROM adaptive_explorations"
                ).fetchone()
            self.assertEqual((10, 10), (count, nanos))

    def test_legacy_registry_is_migrated_without_losing_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive.sqlite3"
            timestamp = datetime.now(UTC).isoformat()
            with closing(sqlite3.connect(path)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE adaptive_meta (
                        key TEXT PRIMARY KEY,
                        value INTEGER NOT NULL
                    );
                    INSERT INTO adaptive_meta (key, value) VALUES ('revision', 0);
                    CREATE TABLE adaptive_models (
                        model_id TEXT PRIMARY KEY,
                        profile_json TEXT NOT NULL,
                        source TEXT NOT NULL,
                        operating_state TEXT NOT NULL,
                        default_task_state TEXT NOT NULL,
                        first_seen_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE adaptive_task_states (
                        model_id TEXT NOT NULL,
                        task TEXT NOT NULL,
                        state TEXT NOT NULL,
                        reason TEXT,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (model_id, task)
                    );
                    CREATE TABLE adaptive_observations (
                        id INTEGER PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        task TEXT NOT NULL,
                        quality_score REAL,
                        latency_ms REAL,
                        actual_cost_usd REAL,
                        success INTEGER,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE adaptive_explorations (
                        id INTEGER PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        task TEXT NOT NULL,
                        reserved_cost_usd REAL NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO adaptive_models VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "private",
                        json.dumps(profile("private", quality=80)),
                        "legacy",
                        "active",
                        "provisional",
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
                connection.execute(
                    "INSERT INTO adaptive_observations "
                    "(model_id, task, quality_score, latency_ms, actual_cost_usd, "
                    "success, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("private", "coding", 90, 12, 0.01, 1, timestamp),
                )
                connection.execute(
                    "INSERT INTO adaptive_explorations "
                    "(model_id, task, reserved_cost_usd, created_at) VALUES (?, ?, ?, ?)",
                    ("private", "coding", 0.0000000001, timestamp),
                )
                connection.commit()

            registry = AdaptiveModelRegistry(path)
            migrated = registry.status("private", task="coding")
            self.assertEqual(1, migrated.samples)
            with closing(sqlite3.connect(path)) as connection:
                observation_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(adaptive_observations)")
                }
                exploration_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(adaptive_explorations)")
                }
                schema_version = connection.execute(
                    "SELECT value FROM adaptive_meta WHERE key = 'schema_version'"
                ).fetchone()[0]
                observation_count = connection.execute(
                    "SELECT observation_count FROM adaptive_models WHERE model_id = 'private'"
                ).fetchone()[0]
                reserved_nanos = connection.execute(
                    "SELECT reserved_cost_nano_usd FROM adaptive_explorations"
                ).fetchone()[0]
            self.assertTrue({"input_tokens", "output_tokens"} <= observation_columns)
            self.assertIn("reserved_cost_nano_usd", exploration_columns)
            self.assertEqual(3, schema_version)
            self.assertEqual(1, observation_count)
            self.assertEqual(1, reserved_nanos)

    def test_corrupt_persisted_profile_is_reported_as_store_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive.sqlite3"
            registry = AdaptiveModelRegistry(path)
            registry.encounter(profile("private", quality=80), source="private")
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "UPDATE adaptive_models SET profile_json = 'not-json' "
                    "WHERE model_id = 'private'"
                )
                connection.commit()

            with self.assertRaisesRegex(AdaptiveStoreError, "invalid profile"):
                registry.status("private")

    def test_identity_collisions_and_persisted_secrets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = AdaptiveModelRegistry(Path(directory) / "adaptive.sqlite3")
            registry.encounter(profile("private", quality=80), source="private")
            collision = {
                **profile("private", quality=80),
                "model": "different-weights",
            }
            with self.assertRaisesRegex(OpenRoutiQError, "already bound"):
                registry.encounter(collision, source="private")
            with self.assertRaisesRegex(OpenRoutiQError, "credential-like"):
                registry.encounter(
                    profile(
                        "secret-model",
                        quality=80,
                        provider_options={"api_key": "do-not-store"},
                    )
                )

    def test_dormant_model_reactivates_only_as_previously_calibrated(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = AdaptiveModelRegistry(Path(directory) / "adaptive.sqlite3")
            contract = profile("private", quality=80)
            registry.encounter(contract, source="private")
            registry.set_operating_state("private", "dormant")
            reactivated = registry.encounter(contract, source="private")
            self.assertEqual("active", reactivated.operating_state)
            self.assertEqual("provisional", reactivated.task_state)


if __name__ == "__main__":
    unittest.main()
