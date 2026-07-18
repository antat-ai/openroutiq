import asyncio
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from openroutiq import AdaptiveRouter, FailureType, OpenRoutiQError, ProxyLimits, Router
from openroutiq.proxy.app import _RequestBodyLimitMiddleware, _stream, create_app


def profile(model_id, provider_model, quality, price):
    return {
        "id": model_id,
        "provider": "test",
        "model": provider_model,
        "api_style": "litellm",
        "quality": {"general": quality},
        "latency_ms": 100,
        "input_price_per_million": price,
        "output_price_per_million": price,
        "max_context_tokens": 100_000,
        "capabilities": ["text", "streaming"],
        "confidence": 80,
    }


class ProxyTest(unittest.TestCase):
    def test_proxy_records_execution_telemetry_for_adaptive_router(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        async def execute(**kwargs):
            return {
                "model": kwargs["model"],
                "choices": [],
                "usage": {
                    "cost": 0.002,
                    "prompt_tokens": 800,
                    "completion_tokens": 200,
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            router = AdaptiveRouter(
                [profile("one", "provider/one", 80, 1)],
                registry=Path(directory) / "adaptive.sqlite3",
                review_margin=0,
            )
            client = TestClient(create_app(router, executor=execute))
            response = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
            )
            status = router.status("one", task="general")
            learned_profile = next(item for item in router.profiles if item.id == "one")

        self.assertEqual(200, response.status_code)
        self.assertEqual(1.0, status.success_rate)
        self.assertIsNotNone(status.average_latency_ms)
        self.assertEqual(0.002, status.average_cost_usd)
        self.assertAlmostEqual(1.2, learned_profile.input_price_per_million)
        self.assertAlmostEqual(1.2, learned_profile.output_price_per_million)

    def test_openai_compatible_auto_and_explicit_model(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        async def execute(**kwargs):
            return {
                "id": "response-1",
                "model": kwargs["model"],
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }

        router = Router(
            [
                profile("cheap", "provider/cheap", 60, 1),
                profile("strong", "provider/strong", 95, 10),
            ],
            review_margin=0,
        )
        client = TestClient(create_app(router, executor=execute))
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "openroutiq": {"strategy": "quality"},
            },
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual("strong", response.headers["x-openroutiq-model"])
        self.assertEqual("provider/strong", response.json()["model"])
        self.assertIn("ranked", response.json()["openroutiq"])

        risk_aware = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "openroutiq": {
                    "strategy": "risk_aware",
                    "risk_policy": {"risk_aversion": 0.8, "cvar_alpha": 0.95},
                },
            },
        )
        self.assertEqual(200, risk_aware.status_code)
        self.assertIsNotNone(risk_aware.json()["openroutiq"]["selected"]["forecast"])

        explicit = client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(200, explicit.status_code)
        self.assertEqual("cheap", explicit.headers["x-openroutiq-model"])

    def test_proxy_rejects_unknown_model_and_routing_option(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        async def execute(**kwargs):
            return {"model": kwargs["model"], "choices": []}

        client = TestClient(
            create_app(Router([profile("one", "provider/one", 80, 1)]), executor=execute)
        )
        unknown = client.post(
            "/v1/chat/completions",
            json={"model": "missing", "messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(400, unknown.status_code)
        bad_option = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "openroutiq": {"secret_override": True},
            },
        )
        self.assertEqual(400, bad_option.status_code)
        bad_stream = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": "false",
            },
        )
        self.assertEqual(400, bad_stream.status_code)
        self.assertEqual(
            FailureType.PROTOCOL_FAILURE.value,
            bad_stream.headers["x-openroutiq-error-type"],
        )

    def test_proxy_bounds_untrusted_routing_labels(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        calls = 0

        async def execute(**kwargs):
            nonlocal calls
            calls += 1
            return {"model": kwargs["model"], "choices": []}

        client = TestClient(
            create_app(Router([profile("one", "provider/one", 80, 1)]), executor=execute)
        )
        for payload in (
            {"model": "m" * 513, "messages": [{"role": "user", "content": "hello"}]},
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "openroutiq": {"task": "t" * 513},
            },
        ):
            response = client.post("/v1/chat/completions", json=payload)
            self.assertEqual(400, response.status_code, payload)
            self.assertEqual(
                FailureType.PROTOCOL_FAILURE.value,
                response.headers["x-openroutiq-error-type"],
            )
        self.assertEqual(0, calls)

    def test_proxy_rejects_client_provider_transport_controls(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        calls = 0

        async def execute(**kwargs):
            nonlocal calls
            calls += 1
            return {"model": kwargs["model"], "choices": []}

        client = TestClient(
            create_app(Router([profile("one", "provider/one", 80, 1)]), executor=execute)
        )
        for field, value in (
            ("base_url", "http://127.0.0.1:1"),
            ("api_key", "client-controlled-secret"),
            ("extra_headers", {"authorization": "Bearer client-controlled-secret"}),
            ("custom_llm_provider", "client-controlled-provider"),
            ("provider", {"order": ["client-controlled-provider"]}),
            ("fallbacks", [{"provider/one": ["provider/two"]}]),
            ("max_retries", 100),
            ("aws_bedrock_runtime_endpoint", "http://169.254.169.254/latest"),
            ("aws_sts_endpoint", "http://127.0.0.1:9000"),
            ("aws_region_name", "attacker-controlled-region"),
            ("azure_endpoint", "http://10.0.0.7"),
            ("vertex_project", "attacker-controlled-project"),
            ("future_litellm_transport_control", "client-controlled"),
        ):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "hello"}],
                    field: value,
                },
            )
            self.assertEqual(400, response.status_code, field)
            self.assertEqual(
                FailureType.PROTOCOL_FAILURE.value,
                response.headers["x-openroutiq-error-type"],
                field,
            )
        self.assertEqual(0, calls)

    def test_proxy_treats_client_input_tokens_as_a_conservative_floor(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        calls = 0

        async def execute(**kwargs):
            nonlocal calls
            calls += 1
            return {"model": kwargs["model"], "choices": []}

        constrained = profile("tiny", "provider/tiny", 80, 1)
        constrained["max_context_tokens"] = 1_000
        client = TestClient(create_app(Router([constrained]), executor=execute))
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "x" * 8_000}],
                "openroutiq": {"input_tokens": 0, "expected_output_tokens": 1},
            },
        )

        self.assertEqual(422, response.status_code)
        self.assertEqual(0, calls)

    def test_proxy_injects_the_routed_output_cap_for_every_provider_api(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        calls = {}

        async def chat_execute(**kwargs):
            calls["chat"] = kwargs
            return {"model": kwargs["model"], "choices": []}

        async def responses_execute(**kwargs):
            calls["responses"] = kwargs
            return {"id": "resp-1", "model": kwargs["model"], "output": []}

        def messages_execute(**kwargs):
            calls["messages"] = kwargs
            return {"id": "msg-1", "model": kwargs["model"], "content": []}

        app = create_app(
            Router(
                [profile("one", "provider/one", 80, 1)],
                default_output_tokens=37,
            ),
            executor=chat_execute,
            responses_executor=responses_execute,
            messages_executor=messages_execute,
        )
        client = TestClient(app)

        chat = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
        )
        responses = client.post(
            "/v1/responses",
            json={"model": "auto", "input": "hello"},
        )
        messages = client.post(
            "/v1/messages",
            json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
        )

        self.assertEqual(200, chat.status_code)
        self.assertEqual(200, responses.status_code)
        self.assertEqual(200, messages.status_code)
        self.assertEqual(37, calls["chat"]["max_tokens"])
        self.assertEqual(37, calls["responses"]["max_output_tokens"])
        self.assertEqual(37, calls["messages"]["max_tokens"])

        explicit_chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 19,
            },
        )
        self.assertEqual(200, explicit_chat.status_code)
        self.assertEqual(19, calls["chat"]["max_completion_tokens"])
        self.assertNotIn("max_tokens", calls["chat"])

    def test_proxy_rejects_unsafe_or_inconsistent_declared_token_counts(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        calls = 0

        async def execute(**kwargs):
            nonlocal calls
            calls += 1
            return {"model": kwargs["model"], "choices": []}

        client = TestClient(
            create_app(
                Router([profile("one", "provider/one", 80, 1)]),
                executor=execute,
                limits=ProxyLimits(max_declared_tokens=1000),
            )
        )
        base = {"model": "auto", "messages": [{"role": "user", "content": "hello"}]}
        payloads = (
            {**base, "max_tokens": True},
            {**base, "max_tokens": 1001},
            {**base, "max_tokens": 100, "max_completion_tokens": 200},
            {**base, "openroutiq": {"input_tokens": 1001}},
            {
                **base,
                "max_tokens": 100,
                "openroutiq": {"expected_output_tokens": 10},
            },
        )
        for payload in payloads:
            response = client.post("/v1/chat/completions", json=payload)
            self.assertEqual(400, response.status_code, payload)
            self.assertEqual(
                FailureType.PROTOCOL_FAILURE.value,
                response.headers["x-openroutiq-error-type"],
            )
        self.assertEqual(0, calls)

    def test_proxy_runs_routing_off_the_event_loop(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        routing_thread = None
        executor_thread = None

        class ThreadRecordingRouter(Router):
            def route(self, *args, **kwargs):
                nonlocal routing_thread
                routing_thread = threading.get_ident()
                return super().route(*args, **kwargs)

        async def execute(**kwargs):
            nonlocal executor_thread
            executor_thread = threading.get_ident()
            return {"model": kwargs["model"], "choices": []}

        client = TestClient(
            create_app(
                ThreadRecordingRouter([profile("one", "provider/one", 80, 1)]),
                executor=execute,
            )
        )
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(200, response.status_code)
        self.assertIsNotNone(routing_thread)
        self.assertIsNotNone(executor_thread)
        self.assertNotEqual(routing_thread, executor_thread)

    def test_routing_timeout_holds_slot_until_worker_exits(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        route_calls = 0
        executor_calls = 0
        blocker = threading.Event()

        class SlowRouter(Router):
            def route(self, *args, **kwargs):
                nonlocal route_calls
                route_calls += 1
                if route_calls == 1:
                    blocker.wait(timeout=0.5)
                return super().route(*args, **kwargs)

        async def execute(**kwargs):
            nonlocal executor_calls
            executor_calls += 1
            return {"model": kwargs["model"], "choices": []}

        app = create_app(
            SlowRouter([profile("one", "provider/one", 80, 1)]),
            executor=execute,
            limits=ProxyLimits(
                max_request_bytes=1024,
                max_concurrency=1,
                queue_timeout_seconds=0.02,
                routing_timeout_seconds=0.01,
                provider_timeout_seconds=0.1,
                stream_idle_timeout_seconds=0.1,
            ),
        )
        payload = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
        with TestClient(app) as client:
            first = client.post("/v1/chat/completions", json=payload)
            second = client.post("/v1/chat/completions", json=payload)
            blocker.set()
            time.sleep(0.02)
            third = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(504, first.status_code)
        self.assertEqual(503, second.status_code)
        self.assertEqual(200, third.status_code)
        self.assertEqual(2, route_calls)
        self.assertEqual(1, executor_calls)

    def test_proxy_enforces_request_and_provider_deadlines(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        async def execute(**_kwargs):
            await asyncio.sleep(0.05)
            return {"choices": []}

        app = create_app(
            Router([profile("one", "provider/one", 80, 1)]),
            executor=execute,
            limits=ProxyLimits(
                max_request_bytes=128,
                max_concurrency=1,
                queue_timeout_seconds=0.01,
                provider_timeout_seconds=0.01,
                stream_idle_timeout_seconds=0.01,
            ),
        )
        client = TestClient(app)
        too_large = client.post(
            "/v1/chat/completions",
            content=b"{" + b"x" * 256 + b"}",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(413, too_large.status_code)
        timed_out = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(504, timed_out.status_code)
        self.assertEqual(FailureType.TIMEOUT.value, timed_out.headers["x-openroutiq-error-type"])

    def test_provider_timeout_releases_concurrency_slot(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        calls = 0

        async def execute(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(0.05)
            return {"model": kwargs["model"], "choices": []}

        app = create_app(
            Router([profile("one", "provider/one", 80, 1)]),
            executor=execute,
            limits=ProxyLimits(
                max_request_bytes=1024,
                max_concurrency=1,
                queue_timeout_seconds=0.02,
                provider_timeout_seconds=0.01,
                stream_idle_timeout_seconds=0.01,
            ),
        )
        client = TestClient(app)
        payload = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
        first = client.post("/v1/chat/completions", json=payload)
        second = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(504, first.status_code)
        self.assertEqual(200, second.status_code)
        self.assertEqual(2, calls)

    def test_sync_provider_timeout_holds_slot_until_worker_exits(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        calls = 0
        blocker = threading.Event()

        def execute(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                blocker.wait(timeout=0.5)
            return {"model": kwargs["model"], "choices": []}

        app = create_app(
            Router([profile("one", "provider/one", 80, 1)]),
            executor=execute,
            limits=ProxyLimits(
                max_request_bytes=1024,
                max_concurrency=1,
                queue_timeout_seconds=0.02,
                provider_timeout_seconds=0.01,
                stream_idle_timeout_seconds=0.01,
            ),
        )
        payload = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
        with TestClient(app) as client:
            first = client.post("/v1/chat/completions", json=payload)
            second = client.post("/v1/chat/completions", json=payload)
            blocker.set()
            time.sleep(0.02)
            third = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(504, first.status_code)
        self.assertEqual(503, second.status_code)
        self.assertEqual(200, third.status_code)
        self.assertEqual(2, calls)

    def test_proxy_normalizes_failure_telemetry(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        async def execute(**_kwargs):
            error = RuntimeError("rate limit from upstream")
            error.status_code = 429
            raise error

        with tempfile.TemporaryDirectory() as directory:
            router = AdaptiveRouter(
                [profile("one", "provider/one", 80, 1)],
                registry=Path(directory) / "adaptive.sqlite3",
                review_margin=0,
            )
            response = TestClient(create_app(router, executor=execute)).post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            )
            status = router.status("one", task="general")
        self.assertEqual(429, response.status_code)
        self.assertEqual(FailureType.RATE_LIMIT.value, response.headers["x-openroutiq-error-type"])
        self.assertEqual(((FailureType.RATE_LIMIT.value, 1.0),), status.failure_probabilities)

    def test_chunked_request_body_limit_fails_before_endpoint_execution(self):
        endpoint_called = False
        messages = iter(
            [
                {"type": "http.request", "body": b"1234", "more_body": True},
                {"type": "http.request", "body": b"5678", "more_body": False},
            ]
        )
        sent = []

        async def endpoint(_scope, receive, send):
            nonlocal endpoint_called
            while (await receive()).get("more_body"):
                pass
            endpoint_called = True
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        asyncio.run(
            _RequestBodyLimitMiddleware(endpoint, maximum=6)(
                {"type": "http", "headers": []}, receive, send
            )
        )
        self.assertFalse(endpoint_called)
        self.assertEqual(413, sent[0]["status"])

    def test_proxy_auth_uses_environment_only(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        async def execute(**kwargs):
            return {"model": kwargs["model"], "choices": []}

        with patch.dict(os.environ, {"TEST_OPENROUTIQ_KEY": "local-secret"}):
            app = create_app(
                Router([profile("one", "provider/one", 80, 1)]),
                executor=execute,
                api_key_env="TEST_OPENROUTIQ_KEY",
            )
            client = TestClient(app)
            denied = client.get("/v1/models")
            allowed = client.get(
                "/v1/models",
                headers={"Authorization": "Bearer local-secret"},
            )
        self.assertEqual(401, denied.status_code)
        self.assertEqual(200, allowed.status_code)

    def test_proxy_rejects_empty_authentication_configuration(self):
        async def execute(**kwargs):
            return {"model": kwargs["model"], "choices": []}

        with patch.dict(os.environ, {"TEST_OPENROUTIQ_KEY": ""}):
            with self.assertRaisesRegex(OpenRoutiQError, "must not be empty"):
                create_app(
                    Router([profile("one", "provider/one", 80, 1)]),
                    executor=execute,
                    api_key_env="TEST_OPENROUTIQ_KEY",
                )

    def test_proxy_fails_closed_if_authentication_becomes_empty(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        async def execute(**kwargs):
            return {"model": kwargs["model"], "choices": []}

        with patch.dict(os.environ, {"TEST_OPENROUTIQ_KEY": "local-secret"}):
            app = create_app(
                Router([profile("one", "provider/one", 80, 1)]),
                executor=execute,
                api_key_env="TEST_OPENROUTIQ_KEY",
            )
            client = TestClient(app)
            os.environ["TEST_OPENROUTIQ_KEY"] = ""
            response = client.get("/v1/models")
        self.assertEqual(503, response.status_code)

    def test_sync_stream_iterator_runs_off_event_loop_and_is_closed(self):
        ticked = threading.Event()

        class SyncStream:
            def __init__(self):
                self.index = 0
                self.closed = False
                self.event_loop_progressed = False

            def __iter__(self):
                return self

            def __next__(self):
                if self.index:
                    raise StopIteration
                self.index += 1
                time.sleep(0.05)
                self.event_loop_progressed = ticked.is_set()
                return {"content": "ok"}

            def close(self):
                self.closed = True

        async def collect():
            source = SyncStream()

            async def ticker():
                await asyncio.sleep(0.005)
                ticked.set()

            ticker_task = asyncio.create_task(ticker())
            chunks = [chunk async for chunk in _stream(source, typed=False)]
            await ticker_task
            return source, chunks

        source, chunks = asyncio.run(collect())
        self.assertTrue(source.event_loop_progressed)
        self.assertTrue(source.closed)
        self.assertEqual(b"data: [DONE]\n\n", chunks[-1])

    def test_responses_and_anthropic_messages_endpoints(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")

        async def chat_execute(**kwargs):
            return {"model": kwargs["model"], "choices": []}

        async def responses_execute(**kwargs):
            return {"id": "resp-1", "model": kwargs["model"], "output": []}

        def messages_execute(**kwargs):
            return {
                "id": "msg-1",
                "model": kwargs["model"],
                "type": "message",
                "content": [{"type": "text", "text": "ok"}],
            }

        app = create_app(
            Router([profile("one", "provider/one", 80, 1)]),
            executor=chat_execute,
            responses_executor=responses_execute,
            messages_executor=messages_execute,
        )
        client = TestClient(app)
        responses = client.post(
            "/v1/responses",
            json={"model": "auto", "input": "hello", "max_output_tokens": 20},
        )
        self.assertEqual(200, responses.status_code)
        self.assertEqual("provider/one", responses.json()["model"])
        messages = client.post(
            "/v1/messages",
            json={
                "model": "auto",
                "system": "Be concise",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 20,
            },
        )
        self.assertEqual(200, messages.status_code)
        self.assertEqual("provider/one", messages.json()["model"])


if __name__ == "__main__":
    unittest.main()
