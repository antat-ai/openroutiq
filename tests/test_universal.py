import asyncio
import os
import sqlite3
import stat
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path

from openroutiq import OpenRoutiQError, OutcomeStore, ProviderRequest, Router


def universal_profile(**overrides):
    base = {
        "id": "custom/model",
        "provider": "custom",
        "model": "provider-model",
        "api_style": "acme.agent_v2",
        "quality": {"legal_review": 90, "general": 70},
        "latency_ms": 100,
        "input_price_per_million": 1,
        "output_price_per_million": 2,
        "max_context_tokens": 100_000,
        "capabilities": [
            "text",
            "audio",
            "video",
            "documents",
            "audio_output",
        ],
        "confidence": 80,
    }
    return {**base, **overrides}


class UniversalRoutingTest(unittest.TestCase):
    def test_catalog_defined_task_and_custom_api_adapter(self):
        router = Router(
            [universal_profile()],
            task_examples={
                "general": ["hello", "explain this"],
                "legal_review": ["review this contract", "find risky clauses"],
            },
        )
        decision = router.route("Review this indemnification clause", task="legal_review")

        def adapter(**values):
            return ProviderRequest(
                api_style="acme.agent_v2",
                provider="custom",
                model="provider-model",
                base_url="https://example.invalid",
                kwargs=values["kwargs"],
                metadata=values["metadata"],
                invoker=lambda client, kwargs: client.execute(kwargs),
            )

        plan = decision.prepare("Review this clause", adapter=adapter)
        client = type("Client", (), {"execute": lambda self, kwargs: kwargs["model"]})()
        self.assertEqual("provider-model", plan.invoke(client))

    def test_audio_video_documents_and_output_modalities_are_hard_requirements(self):
        request = [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "audio_url": "https://example.invalid/a.wav"},
                    {"type": "input_video", "video_url": "https://example.invalid/v.mp4"},
                    {"type": "input_file", "file_data": {"mime_type": "application/pdf"}},
                ],
            }
        ]
        decision = Router([universal_profile()]).route(
            request,
            task="general",
            output_modalities=["text", "audio"],
        )
        self.assertTrue(
            {"audio", "video", "documents", "audio_output"}.issubset(
                decision.analysis.required_capabilities
            )
        )

    def test_sync_provider_runs_off_event_loop_in_ainvoke(self):
        router = Router(
            [
                {
                    **universal_profile(api_style="openai_compatible"),
                    "id": "sync/model",
                }
            ]
        )
        plan = router.route("hello", task="general").prepare("hello")
        event_thread = threading.get_ident()
        invoked_thread = None

        def create(**kwargs):
            nonlocal invoked_thread
            invoked_thread = threading.get_ident()
            return kwargs["model"]

        client = type(
            "Client",
            (),
            {
                "chat": type(
                    "Chat",
                    (),
                    {"completions": type("Completions", (), {"create": staticmethod(create)})()},
                )()
            },
        )()
        self.assertEqual("provider-model", asyncio.run(plan.ainvoke(client)))
        self.assertNotEqual(event_thread, invoked_thread)

    def test_custom_token_and_cost_estimators_support_nonstandard_pricing(self):
        calls = []

        def token_counter(context, tools):
            calls.append((context.request, tools))
            return 12_345

        def cost_estimator(model, input_tokens, output_tokens):
            return model.pricing["request"] + (input_tokens + output_tokens) / 1_000_000

        router = Router(
            [universal_profile(pricing={"request": 0.25})],
            token_counter=token_counter,
            cost_estimator=cost_estimator,
        )
        decision = router.route("hello", task="general", expected_output_tokens=100)
        self.assertEqual(12_345, decision.estimated_input_tokens)
        self.assertAlmostEqual(0.262445, decision.selected.predicted_cost)
        self.assertEqual(1, len(calls))

    def test_outcome_store_has_bounded_lookup_and_explicit_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcomes.sqlite3"
            store = OutcomeStore(path)
            if os.name != "nt":
                self.assertEqual(0o600, stat.S_IMODE(path.stat().st_mode))
            for score in (10, 20, 30):
                store.add(
                    embedding_id="encoder-v1",
                    model_id="custom/model",
                    embedding=[1, 0],
                    quality_score=score,
                    latency_ms=100,
                    actual_cost_usd=score / 1_000,
                    success=score >= 30,
                    failure_class=None if score >= 30 else "evaluator_failure",
                    input_tokens=10,
                    output_tokens=20,
                    selection_probability=0.5,
                    metadata={},
                )
            for _ in range(3):
                store.add(
                    embedding_id="encoder-v1",
                    model_id="irrelevant/model",
                    embedding=[1, 0],
                    quality_score=99,
                    latency_ms=1,
                    metadata={},
                )
            estimates, _, has_data = store.estimates(
                embedding_id="encoder-v1",
                embedding=[1, 0],
                model_ids=["custom/model"],
                neighbors=3,
                minimum_similarity=0,
                max_rows=2,
            )
            self.assertTrue(has_data)
            self.assertEqual(2, estimates["custom/model"].samples)
            self.assertAlmostEqual(25, estimates["custom/model"].quality_score)
            self.assertIsNotNone(estimates["custom/model"].success_probability)
            self.assertEqual(
                ("evaluator_failure", 0.5),
                estimates["custom/model"].failure_probabilities[0],
            )
            self.assertAlmostEqual(0.03, estimates["custom/model"].cost_p95_usd)
            self.assertEqual(2, len(estimates["custom/model"].scenarios))
            with self.assertRaisesRegex(OpenRoutiQError, "max_rows must be an integer >= 1"):
                store.estimates(
                    embedding_id="encoder-v1",
                    embedding=[1, 0],
                    model_ids=["custom/model"],
                    neighbors=1,
                    minimum_similarity=0,
                    max_rows=-1,
                )

            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "UPDATE route_outcomes SET created_at = '2000-01-01 00:00:00' WHERE id = 1"
                )
                connection.commit()
            self.assertEqual(1, store.prune(created_before="2001-01-01 00:00:00"))

    def test_outcome_store_migrates_existing_point_estimate_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    """
                    CREATE TABLE route_outcomes (
                        id INTEGER PRIMARY KEY,
                        embedding_id TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        quality_score REAL NOT NULL,
                        latency_ms REAL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                connection.commit()

            OutcomeStore(path)
            with closing(sqlite3.connect(path)) as connection:
                columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(route_outcomes)").fetchall()
                }
            self.assertTrue(
                {
                    "actual_cost_usd",
                    "success",
                    "input_tokens",
                    "output_tokens",
                    "failure_class",
                    "selection_probability",
                }.issubset(columns)
            )


if __name__ == "__main__":
    unittest.main()
