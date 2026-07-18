import asyncio
import io
import json
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openroutiq import (
    CatalogError,
    ChanceConstraint,
    ContextAnalysis,
    OpenRoutiQError,
    NoEligibleModelError,
    ProxyLimits,
    RouteContext,
    RiskPolicy,
    Router,
    TaskClassifier,
    analyze_context,
    classify_task,
    get_router,
    init_catalog,
    local_sentence_embedder,
    route as quick_route,
)
from openroutiq.cli import main as cli_main


def task_examples():
    return {
        "general": ["Hello there", "Answer this ordinary question"],
        "coding": ["Repair the worker race condition", "Implement the Python service"],
        "research": ["Find recent papers with citations", "Investigate reliable sources"],
        "vision": ["Inspect the screenshot", "Read the architecture diagram"],
    }


def profiles():
    base = {
        "provider": "test",
        "max_context_tokens": 100_000,
        "capabilities": ["text", "json", "tools"],
        "confidence": 90,
    }
    return [
        {
            **base,
            "id": "fast",
            "model": "fast-model",
            "quality": {"general": 60, "coding": 65},
            "latency_ms": 100,
            "input_price_per_million": 0.1,
            "output_price_per_million": 0.2,
        },
        {
            **base,
            "id": "deep:low",
            "model": "deep-model",
            "reasoning_level": "low",
            "quality": {"general": 75, "coding": 80, "vision": 76},
            "latency_ms": 500,
            "input_price_per_million": 1,
            "output_price_per_million": 2,
        },
        {
            **base,
            "id": "deep:high",
            "model": "deep-model",
            "reasoning_level": "high",
            "quality": {"general": 95, "coding": 98, "vision": 95},
            "latency_ms": 2000,
            "input_price_per_million": 10,
            "output_price_per_million": 20,
            "capabilities": ["text", "vision", "json", "tools"],
        },
    ]


def provider_profile(provider, api_style, *, reasoning_mode="effort", provider_options=None):
    return {
        "id": f"{provider}/model:high",
        "provider": provider,
        "model": "provider-model",
        "api_style": api_style,
        "reasoning_level": "high",
        "reasoning_mode": reasoning_mode,
        "provider_options": provider_options or {},
        "quality": {"general": 90, "tool_use": 92, "extraction": 91},
        "latency_ms": 500,
        "input_price_per_million": 1,
        "output_price_per_million": 2,
        "max_context_tokens": 100_000,
        "capabilities": ["text", "tools", "parallel_tools", "json_schema", "streaming"],
    }


class RouterTest(unittest.TestCase):
    def test_quickstart_catalog_and_one_function_route(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            self.assertEqual(path, init_catalog(path, provider="openrouter"))
            catalog = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(["openrouter"], [model["provider"] for model in catalog["models"]])
            self.assertIn("task_examples", catalog)
            self.assertEqual(
                "template/openrouter:high", quick_route("hello", catalog=path).selected.model_id
            )
            self.assertIs(get_router(path), get_router(path))
            with self.assertRaises(OpenRoutiQError):
                init_catalog(path)

    def test_weights_change_winner_and_reasoning_levels_are_variants(self):
        router = Router(profiles(), review_margin=0)
        quality = router.route("fix Python code", weights={"quality": 100, "latency": 0, "cost": 0})
        speed = router.route("fix Python code", weights={"quality": 0, "latency": 100, "cost": 0})
        self.assertEqual("deep:high", quality.selected.model_id)
        self.assertEqual("high", quality.selected.reasoning_level)
        self.assertEqual("fast", speed.selected.model_id)

    def test_full_context_learns_vision_and_requires_explicit_risk(self):
        analysis = analyze_context(
            [
                {"role": "system", "content": "Review production security carefully."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/image.png"},
                        },
                        {"type": "text", "text": "Analyze this architecture diagram."},
                    ],
                },
            ],
            task_classifier=TaskClassifier(task_examples()),
        )
        self.assertEqual("vision", analysis.task)
        self.assertIn("vision", analysis.required_capabilities)
        self.assertFalse(analysis.high_risk)
        self.assertEqual(2, analysis.message_count)
        self.assertGreater(analysis.complexity, 20)
        self.assertGreater(dict(analysis.task_scores)["vision"], 0.5)

    def test_auto_strategy_adapts_but_explicit_flags_win(self):
        router = Router(profiles(), review_margin=0)
        simple = router.route("hello")
        complex_request = router.route(
            "Analyze this multi-step architecture with constraints and tradeoffs"
        )
        forced = router.route(
            "Analyze this architecture",
            task="coding",
            complexity=5,
            strategy="speed",
            constraints={"reasoning_levels": ["low"], "min_quality": 70},
        )
        self.assertGreater(complex_request.weights.quality, simple.weights.quality)
        self.assertEqual("coding", forced.task)
        self.assertEqual("speed", forced.strategy)
        self.assertEqual("deep:low", forced.selected.model_id)

    def test_low_confidence_analyzer_and_outcome_feedback(self):
        calls = []

        def analyzer(request, initial):
            calls.append(request)
            return ContextAnalysis(
                task="reasoning",
                complexity=95,
                confidence=92,
                required_capabilities=initial.required_capabilities,
                high_risk=False,
                estimated_input_tokens=initial.estimated_input_tokens,
                message_count=initial.message_count,
                signals=initial.signals + ("classifier-enriched",),
            )

        router = Router(profiles(), review_margin=0, context_analyzer=analyzer)
        decision = router.route("hello")
        self.assertEqual("reasoning", decision.task)
        self.assertIn("classifier-enriched", decision.analysis.signals)
        self.assertEqual(1, len(calls))

        router.record_outcome("fast", "general", 100, alpha=1)
        learned = router.route("hello", task="general", strategy="quality")
        self.assertEqual("fast", learned.selected.model_id)
        self.assertEqual(1, len(calls), "explicit task should skip the context analyzer")

    def test_constraints_filter_before_scoring(self):
        items = profiles()
        items[1] = {**items[1], "local": True}
        router = Router(items, task_examples=task_examples())
        decision = router.route("hello", constraints={"local_only": True})
        self.assertEqual("deep:low", decision.selected.model_id)
        with self.assertRaises(NoEligibleModelError):
            router.route(
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
                constraints={"local_only": True},
            )

    def test_review_for_small_margin_high_risk_and_low_confidence(self):
        items = profiles()[:2]
        items[0] = {**items[0], "quality": {"general": 80}, "confidence": 40}
        items[1] = {
            **items[1],
            "id": "zdeep:low",
            "quality": {"general": 80},
            "latency_ms": 100,
            "input_price_per_million": 0.1,
            "output_price_per_million": 0.2,
        }
        decision = Router(items, review_margin=1, min_confidence=60).route(
            "hello", task="general", high_risk=True
        )
        self.assertTrue(decision.review_required)
        self.assertEqual("fast", decision.selected.model_id)
        self.assertEqual(3, len(decision.review_reasons))

    def test_latency_feedback_changes_speed_route(self):
        router = Router(profiles(), review_margin=0, latency_alpha=1)
        router.record_latency("fast", 5000)
        decision = router.route("hello", weights={"quality": 0, "latency": 100, "cost": 0})
        self.assertEqual("deep:low", decision.selected.model_id)

    def test_catalog_validation_and_serialization(self):
        with self.assertRaises(CatalogError):
            Router([profiles()[0], profiles()[0]])
        for unsafe in (
            {"provider_options": {"api_key": "do-not-store"}},
            {"provider_options": {"headers": {"authorization": "Bearer do-not-store"}}},
            {"base_url": "https://user:password@example.invalid/v1"},
            {"base_url": "https://example.invalid/v1?api_key=do-not-store"},
        ):
            with self.assertRaises(CatalogError, msg=unsafe):
                Router([{**profiles()[0], **unsafe}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps({"models": profiles()}), encoding="utf-8")
            result = Router.from_file(path).route("hello").to_dict()
            self.assertIn("selected", result)
            self.assertIn("analysis", result)

    def test_task_classifier(self):
        classifier = TaskClassifier(task_examples())
        with self.assertRaises(CatalogError):
            TaskClassifier({"coding": ["one label is not enough"]})
        self.assertEqual("general", classify_task("Repair the worker race condition"))
        self.assertEqual("coding", classify_task("Repair the worker race condition", classifier))
        self.assertEqual("research", classify_task("Find recent papers and citations", classifier))
        self.assertEqual("general", classify_task("Hello there"))
        self.assertEqual(0, classifier.predict("999999").confidence)

    def test_learned_task_distribution_selects_the_specialist(self):
        items = profiles()[:2]
        items[0] = {
            **items[0],
            "id": "builder",
            "quality": {"general": 60, "coding": 98, "research": 25},
            "latency_ms": 100,
            "input_price_per_million": 1,
            "output_price_per_million": 1,
        }
        items[1] = {
            **items[1],
            "id": "investigator",
            "quality": {"general": 60, "coding": 25, "research": 98},
            "latency_ms": 100,
            "input_price_per_million": 1,
            "output_price_per_million": 1,
        }
        router = Router(items, task_examples=task_examples(), review_margin=0)
        self.assertEqual(
            "builder",
            router.route("Repair the worker race condition", strategy="quality").selected.model_id,
        )
        self.assertEqual(
            "investigator",
            router.route(
                "Investigate recent papers and reliable sources", strategy="quality"
            ).selected.model_id,
        )

    def test_contextual_outcomes_route_workflow_steps_tools_and_ood(self):
        with self.assertRaises(OpenRoutiQError):
            RouteContext("hello", workflow_step="")
        with self.assertRaises(OpenRoutiQError):
            local_sentence_embedder(Path("missing-local-embedding-model"))

        def embed(text):
            lowered = text.lower()
            if "workflow step: unknown" in lowered:
                return [0, 0, 0, 1]
            return [
                int("workflow step: plan" in lowered),
                int("workflow step: act" in lowered),
                int('"name":"lookup"' in lowered),
                0,
            ]

        items = []
        for model_id, source in (("builder", profiles()[0]), ("reviewer", profiles()[1])):
            items.append(
                {
                    **source,
                    "id": model_id,
                    "quality": {"general": 70, "tool_use": 70},
                    "latency_ms": 100,
                    "input_price_per_million": 1,
                    "output_price_per_million": 1,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outcomes.sqlite3"
            router = Router(
                items,
                embedder=embed,
                outcome_store=database,
                embedding_id="workflow-test-v1",
                outcome_similarity_threshold=0.8,
                outcome_prior_samples=1,
                out_of_domain_model="reviewer",
                review_margin=0,
            )
            plan = RouteContext("Handle this request", agent_role="planner", workflow_step="plan")
            act = RouteContext("Handle this request", agent_role="worker", workflow_step="act")
            lookup = [{"name": "lookup", "description": "Read an account"}]
            for model_id, plan_score, act_score, tool_score in (
                ("builder", 20, 95, 15),
                ("reviewer", 95, 20, 98),
            ):
                router.record_evaluation(plan, model_id, plan_score)
                router.record_evaluation(act, model_id, act_score)
                router.record_evaluation(act, model_id, tool_score, tools=lookup)

            plan_decision = router.route(plan, task="general", strategy="quality")
            act_decision = router.route(act, task="general", strategy="quality")
            tool_decision = router.route(
                act,
                task="general",
                strategy="quality",
                tools=lookup,
                tool_choice="required",
            )
            self.assertEqual("reviewer", plan_decision.selected.model_id)
            self.assertEqual("builder", act_decision.selected.model_id)
            self.assertEqual("reviewer", tool_decision.selected.model_id)
            self.assertEqual(1, tool_decision.selected.context_samples)
            self.assertEqual(98, tool_decision.selected.context_quality_score)

            unknown = router.route(
                RouteContext("Handle this request", workflow_step="unknown"),
                task="general",
            )
            self.assertTrue(unknown.out_of_domain)
            self.assertEqual("reviewer", unknown.selected.model_id)
            self.assertTrue(
                any("context similarity" in reason for reason in unknown.review_reasons)
            )

            with sqlite3.connect(database) as connection:
                rows = connection.execute(
                    "SELECT embedding_json, metadata_json FROM route_outcomes"
                ).fetchall()
            connection.close()
            self.assertEqual(6, len(rows))
            self.assertNotIn("Handle this request", json.dumps(rows))
            self.assertNotIn(b"Handle this request", database.read_bytes())

        with self.assertRaises(NoEligibleModelError):
            Router(items).route(
                RouteContext("hello", budget_remaining=0),
                task="general",
            )

    def test_chance_constraints_choose_the_reliable_model(self):
        def embed(_text):
            return [1, 0]

        items = []
        for model_id, quality in (("volatile", 96), ("steady", 84)):
            items.append(
                {
                    **profiles()[0],
                    "id": model_id,
                    "quality": {"general": quality},
                    "latency_ms": 100,
                    "input_price_per_million": 1,
                    "output_price_per_million": 1,
                    "confidence": 90,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            router = Router(
                items,
                embedder=embed,
                outcome_store=Path(directory) / "risk.sqlite3",
                outcome_similarity_threshold=0,
                outcome_prior_samples=1,
                review_margin=0,
            )
            for score, success in zip((100, 98, 96, 90, 36), (True, True, True, False, False)):
                router.record_evaluation(
                    "same request",
                    "volatile",
                    score,
                    latency_ms=90,
                    actual_cost_usd=0.01,
                    success=success,
                    selection_probability=0.5,
                )
            for _ in range(5):
                router.record_evaluation(
                    "same request",
                    "steady",
                    84,
                    latency_ms=110,
                    actual_cost_usd=0.012,
                    success=True,
                    selection_probability=0.5,
                )

            deterministic = router.route(
                "same request",
                task="general",
                weights={"quality": 100, "latency": 0, "cost": 0},
            )
            self.assertEqual("volatile", deterministic.selected.model_id)

            policy = RiskPolicy(
                constraints=(
                    ChanceConstraint("success", minimum_probability=0.8),
                    ChanceConstraint(
                        "cost_at_most",
                        threshold=0.02,
                        minimum_probability=0.99,
                    ),
                ),
                risk_aversion=0.8,
                minimum_samples=5,
                require_observed_probabilities=True,
            )
            robust = router.route(
                "same request",
                task="general",
                strategy="risk_aware",
                risk_policy=policy,
            )
            self.assertEqual("steady", robust.selected.model_id)
            self.assertEqual(policy, robust.risk_policy)
            self.assertIsNotNone(robust.selected.forecast)
            self.assertGreater(robust.selected.forecast.success_probability, 0.8)
            self.assertIn("success", robust.selected.forecast.observed_metrics)
            volatile = next(item for item in robust.excluded if item.model_id == "volatile")
            self.assertTrue(any("P(success)" in reason for reason in volatile.reasons))
            self.assertIn("forecast", robust.to_dict()["selected"])

    def test_cvar_prefers_stable_tail_and_missing_success_fails_closed(self):
        def embed(_text):
            return [1, 0]

        items = []
        for model_id, quality in (("volatile", 90), ("steady", 84)):
            items.append(
                {
                    **profiles()[0],
                    "id": model_id,
                    "quality": {"general": quality},
                    "latency_ms": 100,
                    "input_price_per_million": 1,
                    "output_price_per_million": 1,
                    "confidence": 90,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            router = Router(
                items,
                embedder=embed,
                outcome_store=Path(directory) / "cvar.sqlite3",
                outcome_similarity_threshold=0,
                outcome_prior_samples=1,
                review_margin=0,
            )
            for score in (100, 100, 100, 100, 20):
                router.record_evaluation("same request", "volatile", score)
            for _ in range(5):
                router.record_evaluation("same request", "steady", 84)

            decision = router.route(
                "same request",
                task="general",
                weights={"quality": 100, "latency": 0, "cost": 0},
                risk_policy=RiskPolicy(risk_aversion=1, cvar_alpha=0.8),
            )
            self.assertEqual("steady", decision.selected.model_id)
            self.assertGreater(
                next(
                    item for item in decision.ranked if item.model_id == "volatile"
                ).forecast.cvar_loss,
                decision.selected.forecast.cvar_loss,
            )

        strict = RiskPolicy(
            constraints=(ChanceConstraint("success", minimum_probability=0.5),),
            require_observed_probabilities=True,
        )
        with self.assertRaises(NoEligibleModelError) as raised:
            Router(items).route("unobserved", task="general", risk_policy=strict)
        self.assertTrue(
            all("no probability estimate" in item.reasons[0] for item in raised.exception.excluded)
        )

    def test_agent_tool_state_and_capability_flags(self):
        history = [
            {"role": "user", "content": "Check both systems"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "opaque", "signature": "signed"},
                    {"type": "tool_use", "id": "tool-1", "name": "check", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}],
            },
        ]
        tools = [
            {"type": "function", "function": {"name": "check", "parameters": {"type": "object"}}}
        ]
        decision = Router([provider_profile("openrouter", "openrouter")]).route(
            history,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=True,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": {"type": "object"}},
            },
            stream=True,
            reasoning_effort="high",
            pinned_model="openrouter/model:high",
        )
        self.assertEqual(
            {"tools", "parallel_tools", "json_schema", "streaming"},
            set(decision.analysis.required_capabilities),
        )
        self.assertIn("model-pinned", decision.analysis.signals)
        self.assertGreater(
            decision.estimated_input_tokens, analyze_context(history).estimated_input_tokens
        )

    def test_provider_request_shapes_preserve_history_and_tools(self):
        messages = [{"role": "user", "content": "Use the tool"}]
        tools = [
            {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        ]
        schema = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": {"type": "object"}},
        }

        openai = (
            Router([provider_profile("openai", "openai_responses")])
            .route(messages, tools=tools, response_format=schema, reasoning_effort="high")
            .prepare(messages, tools=tools, response_format=schema)
        )
        self.assertIs(messages, openai.kwargs["input"])
        self.assertIs(tools, openai.kwargs["tools"])
        self.assertEqual("high", openai.kwargs["reasoning"]["effort"])
        self.assertEqual("answer", openai.kwargs["text"]["format"]["name"])

        anthropic = (
            Router([provider_profile("anthropic", "anthropic_messages", reasoning_mode="adaptive")])
            .route(messages, tools=tools, response_format=schema, reasoning_effort="high")
            .prepare(
                messages,
                tools=tools,
                tool_choice="required",
                parallel_tool_calls=False,
                response_format=schema,
            )
        )
        self.assertIs(messages, anthropic.kwargs["messages"])
        self.assertEqual({"type": "adaptive"}, anthropic.kwargs["thinking"])
        self.assertTrue(anthropic.kwargs["tool_choice"]["disable_parallel_tool_use"])
        self.assertEqual({"type": "object"}, anthropic.kwargs["output_config"]["format"]["schema"])

        openrouter = (
            Router(
                [provider_profile("openrouter", "openrouter", provider_options={"sort": "latency"})]
            )
            .route(messages, tools=tools, reasoning_effort="high")
            .prepare(messages, tools=tools)
        )
        self.assertEqual("https://openrouter.ai/api/v1", openrouter.base_url)
        self.assertTrue(openrouter.kwargs["provider"]["require_parameters"])
        self.assertEqual("latency", openrouter.kwargs["provider"]["sort"])

        requesty = (
            Router([provider_profile("requesty", "requesty")])
            .route(messages, reasoning_effort="high")
            .prepare(messages, agent_id="researcher", run_id="run-7", tags=["multi-agent"])
        )
        self.assertEqual("https://router.requesty.ai/v1", requesty.base_url)
        self.assertEqual("high", requesty.kwargs["reasoning_effort"])
        self.assertEqual("run-7", requesty.kwargs["requesty"]["trace_id"])
        self.assertEqual("researcher", requesty.kwargs["requesty"]["extra"]["agent_id"])

        self.assertEqual(
            "responses",
            openai.invoke(
                SimpleNamespace(responses=SimpleNamespace(create=lambda **_: "responses"))
            ),
        )
        self.assertEqual(
            "messages",
            anthropic.invoke(
                SimpleNamespace(messages=SimpleNamespace(create=lambda **_: "messages"))
            ),
        )
        chat_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: "chat"))
        )
        self.assertEqual("chat", openrouter.invoke(chat_client))
        self.assertEqual("chat", requesty.invoke(chat_client))
        safe_extra = (
            Router([provider_profile("openai", "openai_responses")])
            .route(messages, reasoning_effort="high")
            .prepare(messages, extra={"temperature": 0.2})
        )
        self.assertEqual(0.2, safe_extra.kwargs["temperature"])
        for protected in (
            "api_key",
            "base_url",
            "max_output_tokens",
            "provider",
            "reasoning",
            "stream",
            "tools",
        ):
            with self.assertRaisesRegex(OpenRoutiQError, "extra cannot override"):
                Router([provider_profile("openai", "openai_responses")]).route(
                    messages,
                    reasoning_effort="high",
                ).prepare(messages, extra={protected: "override"})
        with self.assertRaisesRegex(OpenRoutiQError, "credential-like field"):
            Router([provider_profile("openrouter", "openrouter")]).route(
                messages,
                reasoning_effort="high",
            ).prepare(messages, provider_options={"api_key": "do-not-send"})
        litellm = (
            Router([provider_profile("litellm", "litellm")])
            .route(messages, reasoning_effort="high")
            .prepare(messages)
        )
        self.assertEqual("litellm", litellm.invoke(lambda **_: "litellm"))

    def test_tool_quality_changes_quality_route_and_reasoning_cannot_drift(self):
        weak_tools = {
            **provider_profile("one", "openai_compatible"),
            "id": "one",
            "quality": {"general": 99, "coding": 99, "tool_use": 20},
        }
        strong_tools = {
            **provider_profile("two", "litellm"),
            "id": "two",
            "quality": {"general": 90, "coding": 90, "tool_use": 95},
        }
        router = Router([weak_tools, strong_tools], review_margin=0)
        self.assertEqual("one", router.route("Fix code", strategy="quality").selected.model_id)
        decision = router.route("Fix code", strategy="quality", tools=[{"name": "edit"}])
        self.assertEqual("two", decision.selected.model_id)
        with self.assertRaises(OpenRoutiQError):
            decision.prepare([{"role": "user", "content": "Fix code"}], reasoning_effort="low")

    def test_example_catalog_routes_by_real_scores_and_is_deterministic(self):
        catalog = Path(__file__).resolve().parents[1] / "models.example.json"
        router = Router.from_file(catalog, review_margin=0)
        request = "Repair the race condition in this Python worker"

        quality = router.route(
            request,
            task="coding",
            weights={"quality": 100, "latency": 0, "cost": 0},
            input_tokens=1_000,
            expected_output_tokens=500,
        )
        speed = router.route(
            request,
            task="coding",
            weights={"quality": 0, "latency": 100, "cost": 0},
            input_tokens=1_000,
            expected_output_tokens=500,
        )
        cost = router.route(
            request,
            task="coding",
            weights={"quality": 0, "latency": 0, "cost": 100},
            input_tokens=1_000,
            expected_output_tokens=500,
        )

        self.assertEqual("example/deep:high", quality.selected.model_id)
        self.assertEqual("example/fast", speed.selected.model_id)
        self.assertEqual("example/fast", cost.selected.model_id)
        fast = next(item for item in quality.ranked if item.model_id == "example/fast")
        self.assertAlmostEqual(0.00045, fast.predicted_cost)
        expected = quality.to_dict()
        for _ in range(20):
            self.assertEqual(
                expected,
                router.route(
                    request,
                    task="coding",
                    weights={"quality": 100, "latency": 0, "cost": 0},
                    input_tokens=1_000,
                    expected_output_tokens=500,
                ).to_dict(),
            )

    def test_stable_tie_breaking_uses_lexicographically_smaller_id(self):
        model = profiles()[0]
        router = Router(
            [
                {**model, "id": "z-model"},
                {**model, "id": "a-model"},
            ],
            review_margin=0,
        )
        decision = router.route("hello", task="general")
        self.assertEqual("a-model", decision.selected.model_id)
        self.assertEqual(["a-model", "z-model"], [item.model_id for item in decision.ranked])

    def test_every_hard_filter_reports_why_the_model_was_excluded(self):
        base = profiles()[0]
        cases = [
            ({**base, "available": False}, {}, "unavailable"),
            (base, {"candidate_ids": ["another-model"]}, "not in candidate_ids"),
            (base, {"allowed_providers": ["another-provider"]}, "provider not allowed"),
            (base, {"blocked_providers": ["test"]}, "provider blocked"),
            (base, {"local_only": True}, "not local"),
            (base, {"required_capabilities": ["vision"]}, "missing capabilities: vision"),
            (base, {"min_context_tokens": 100_001}, "context 100000 < required 100001"),
            (base, {"max_predicted_cost": 0}, "exceeds hard limit"),
            (base, {"min_quality": 100}, "quality 60.00 < required 100.00"),
            (base, {"max_latency_ms": 99}, "latency 100.00ms exceeds limit 99.00ms"),
            (base, {"reasoning_levels": ["high"]}, "reasoning level not allowed"),
        ]
        for model, constraints, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                with self.assertRaises(NoEligibleModelError) as raised:
                    Router([model]).route("hello", task="general", constraints=constraints)
                self.assertIn(expected_reason, str(raised.exception))

        with self.assertRaises(NoEligibleModelError) as raised:
            Router([base]).route(
                RouteContext("hello", latency_deadline_ms=99),
                task="general",
            )
        self.assertIn("latency 100.00ms exceeds limit 99.00ms", str(raised.exception))

    def test_failed_context_analyzer_falls_back_without_losing_hard_capabilities(self):
        def broken_analyzer(_request, _initial):
            raise RuntimeError("classifier internals must not escape")

        router = Router(
            [profiles()[2]],
            task_examples=task_examples(),
            context_analyzer=broken_analyzer,
            analysis_threshold=100,
            review_margin=0,
        )
        decision = router.route(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "x"}},
                        {"type": "text", "text": "Inspect this screenshot"},
                    ],
                }
            ]
        )
        self.assertEqual("vision", decision.task)
        self.assertIn("vision", decision.analysis.required_capabilities)
        self.assertIn("context-analyzer-fallback", decision.analysis.signals)

    def test_generic_provider_plan_async_invocation_and_budget_reasoning(self):
        messages = [{"role": "user", "content": "Solve this"}]
        original = deepcopy(messages)
        compatible = {
            **provider_profile("gateway", "openai_compatible"),
            "id": "gateway/high",
            "base_url": "https://gateway.example/v1",
        }
        plan = (
            Router([compatible])
            .route(messages, task="general", reasoning_effort="high")
            .prepare(messages, reasoning_effort="high")
        )
        self.assertIs(messages, plan.kwargs["messages"])
        self.assertEqual(original, messages)
        self.assertEqual("https://gateway.example/v1", plan.base_url)
        self.assertNotIn("messages", plan.summary()["parameters"])

        async def create(**kwargs):
            return kwargs["model"]

        async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        self.assertEqual("provider-model", asyncio.run(plan.ainvoke(async_client)))

        budget_profile = {
            **provider_profile("anthropic", "anthropic_messages", reasoning_mode="budget"),
            "id": "anthropic/budget",
            "reasoning_budget_tokens": 128,
        }
        budget_plan = (
            Router([budget_profile])
            .route(
                messages,
                task="general",
                expected_output_tokens=512,
                reasoning_effort="high",
            )
            .prepare(messages, reasoning_effort="high")
        )
        self.assertEqual({"type": "enabled", "budget_tokens": 128}, budget_plan.kwargs["thinking"])
        with self.assertRaises(OpenRoutiQError):
            Router([budget_profile]).route(
                messages,
                task="general",
                expected_output_tokens=128,
                reasoning_effort="high",
            )

    def test_cli_route_outputs_json_and_errors_are_machine_readable(self):
        catalog = Path(__file__).resolve().parents[1] / "models.example.json"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "route",
                    "Repair the Python race condition",
                    "--catalog",
                    str(catalog),
                    "--task",
                    "coding",
                    "--quality",
                    "100",
                    "--latency",
                    "0",
                    "--cost",
                    "0",
                ]
            )
        self.assertEqual(0, code)
        payload = json.loads(stdout.getvalue())
        self.assertEqual("example/deep:high", payload["selected"]["id"])
        self.assertEqual("coding", payload["task"])

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli_main(
                [
                    "route",
                    "hello",
                    "--catalog",
                    str(catalog),
                    "--max-cost",
                    "0",
                ]
            )
        self.assertEqual(2, code)
        self.assertIn("No eligible model", json.loads(stderr.getvalue())["error"])

    def test_cli_proxy_refuses_public_bind_without_authentication(self):
        stderr = io.StringIO()
        fake_uvicorn = SimpleNamespace(run=Mock())
        with (
            patch.dict(os.environ, {"TEST_OPENROUTIQ_PROXY_KEY": ""}),
            patch.dict("sys.modules", {"uvicorn": fake_uvicorn}),
            redirect_stderr(stderr),
        ):
            code = cli_main(
                [
                    "serve",
                    "--host",
                    "0.0.0.0",
                    "--api-key-env",
                    "TEST_OPENROUTIQ_PROXY_KEY",
                ]
            )
        self.assertEqual(2, code)
        self.assertIn("requires a bearer token", json.loads(stderr.getvalue())["error"])
        fake_uvicorn.run.assert_not_called()

    def test_cli_proxy_allows_public_bind_with_authentication(self):
        catalog = Path(__file__).resolve().parents[1] / "models.example.json"
        fake_app = object()
        fake_uvicorn = SimpleNamespace(run=Mock())
        with (
            patch.dict(os.environ, {"TEST_OPENROUTIQ_PROXY_KEY": "local-secret"}),
            patch.dict("sys.modules", {"uvicorn": fake_uvicorn}),
            patch("openroutiq.proxy.create_app", return_value=fake_app) as create_app,
        ):
            code = cli_main(
                [
                    "serve",
                    "--catalog",
                    str(catalog),
                    "--host",
                    "0.0.0.0",
                    "--api-key-env",
                    "TEST_OPENROUTIQ_PROXY_KEY",
                ]
            )
        self.assertEqual(0, code)
        create_app.assert_called_once_with(
            str(catalog),
            api_key_env="TEST_OPENROUTIQ_PROXY_KEY",
            limits=ProxyLimits(),
        )
        fake_uvicorn.run.assert_called_once_with(
            fake_app,
            host="0.0.0.0",
            port=8080,
            log_level="info",
        )

    def test_concurrent_telemetry_and_persisted_evaluations(self):
        def embed(_text):
            return [1, 2, 3]

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "concurrent.sqlite3"
            router = Router(
                profiles()[:2],
                embedder=embed,
                outcome_store=database,
                embedding_id="concurrency-v1",
                review_margin=0,
            )

            def update(index):
                model_id = "fast" if index % 2 == 0 else "deep:low"
                router.record_latency(model_id, 100 + index)
                router.record_outcome(model_id, "general", index % 101)
                return router.record_evaluation(
                    RouteContext("same request", workflow_step=f"step-{index % 4}"),
                    model_id,
                    index % 101,
                    latency_ms=100 + index,
                )

            with ThreadPoolExecutor(max_workers=8) as pool:
                row_ids = list(pool.map(update, range(80)))

            self.assertEqual(80, len(set(row_ids)))
            with closing(sqlite3.connect(database)) as connection:
                row_count = connection.execute("SELECT COUNT(*) FROM route_outcomes").fetchone()[0]
            self.assertEqual(80, row_count)
            decision = router.route(
                RouteContext("same request", workflow_step="step-1"),
                task="general",
            )
            self.assertGreater(decision.selected.context_samples, 0)


if __name__ == "__main__":
    unittest.main()
