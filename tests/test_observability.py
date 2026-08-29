import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from openroutiq import (
    AdaptiveRouter,
    FailureType,
    InMemoryExporter,
    LangSmithExporter,
    LangtraceExporter,
    OTLPExporter,
    OTLPHTTPJSONExporter,
    Observability,
    ObservabilityError,
    ObservabilityPrivacy,
    OpenTelemetryExporter,
    Router,
)


PII_PROMPT = "Customer alice@example.test has SSN 111-22-3333 and token sk-private-canary"
PII_RESPONSE = "Private diagnosis for alice@example.test"
PII_MODEL = "tenant/alice@example.test/private-model"
PII_PROVIDER = "provider-for-alice@example.test"
PII_TASK = "claim-for-alice@example.test"
EXPORT_SECRET = "export-secret-must-not-appear"


def profile(
    model_id: str = "safe-model",
    provider: str = "safe-provider",
    task: str = "general",
) -> dict[str, object]:
    return {
        "id": model_id,
        "provider": provider,
        "model": "provider-model",
        "api_style": "litellm",
        "quality": {"general": 88, task: 88},
        "latency_ms": 200,
        "input_price_per_million": 1,
        "output_price_per_million": 2,
        "max_context_tokens": 100_000,
        "capabilities": ["text", "streaming"],
        "confidence": 90,
    }


class _FailingExporter:
    def export(self, _event) -> None:
        raise RuntimeError(f"transport failed with {EXPORT_SECRET}")


class _BlockingExporter:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.shutdown_called = False

    def export(self, _event) -> None:
        self.started.set()
        self.release.wait(timeout=5)

    def shutdown(self) -> None:
        self.shutdown_called = True


class _Response:
    status = 200

    def close(self) -> None:
        return None


class _RecordingOpener:
    def __init__(self) -> None:
        self.requests = []
        self.timeouts = []

    def open(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        return _Response()


class ObservabilityTest(unittest.TestCase):
    def test_router_decision_is_identical_with_observability_enabled(self):
        request = [
            {"role": "system", "content": "Review production code."},
            {"role": "user", "content": "Fix this worker race condition."},
        ]
        baseline = Router([profile()], review_margin=0).route(
            request,
            task="general",
            strategy="quality",
        )
        memory = InMemoryExporter()
        with Observability([memory]) as observability:
            observed = Router(
                [profile()],
                review_margin=0,
                observability=observability,
            ).route(request, task="general", strategy="quality")
            self.assertTrue(observability.flush())

        self.assertEqual(baseline.to_dict(), observed.to_dict())
        self.assertEqual(["route"], [event.event_type for event in memory.events])

    def test_default_events_pseudonymize_identifiers_and_reject_content(self):
        memory = InMemoryExporter()
        with Observability(
            [memory],
            privacy=ObservabilityPrivacy(pseudonymization_key=b"stable-test-key-123456"),
        ) as observability:
            router = Router(
                [profile(PII_MODEL, PII_PROVIDER, PII_TASK)],
                review_margin=0,
                observability=observability,
            )
            decision = router.route(PII_PROMPT, task=PII_TASK)
            self.assertTrue(
                observability.record_execution(
                    decision,
                    duration_ms=12.5,
                    success=False,
                    failure_type="alice",
                    actual_cost_usd=0.003,
                    input_tokens=20,
                    output_tokens=5,
                )
            )
            self.assertTrue(
                observability.record_evaluation(
                    model_id=PII_MODEL,
                    provider=PII_PROVIDER,
                    task=PII_TASK,
                    quality_score=0,
                    success=False,
                    failure_type=FailureType.PROVIDER_FAILURE,
                )
            )
            self.assertTrue(observability.flush())

        serialized = json.dumps([event.to_dict() for event in memory.events], sort_keys=True)
        for canary in (
            PII_PROMPT,
            "alice@example.test",
            "111-22-3333",
            "sk-private-canary",
            PII_MODEL,
            PII_PROVIDER,
            PII_TASK,
            "alice",
        ):
            self.assertNotIn(canary, serialized)
        route, execution, evaluation = memory.events
        self.assertIn("openroutiq.model.id_hash", route.attributes)
        self.assertIn("openroutiq.provider.name_hash", route.attributes)
        self.assertIn("openroutiq.task.name_hash", route.attributes)
        self.assertNotIn("openroutiq.execution.failure_type", execution.attributes)
        self.assertEqual(
            "provider_failure",
            evaluation.attributes["openroutiq.evaluation.failure_type"],
        )
        self.assertEqual(route.trace_id, execution.trace_id)
        self.assertEqual(route.span_id, execution.parent_span_id)

    def test_raw_identifiers_require_explicit_opt_in_and_key_repr_is_safe(self):
        privacy = ObservabilityPrivacy(
            include_model_identifiers=True,
            include_provider_identifiers=True,
            include_task_identifiers=True,
            pseudonymization_key=EXPORT_SECRET,
        )
        self.assertNotIn(EXPORT_SECRET, repr(privacy))
        memory = InMemoryExporter()
        with Observability([memory], privacy=privacy) as observability:
            Router(
                [profile("model-a", "provider-a", "coding")],
                observability=observability,
            ).route("synthetic input", task="coding")
            self.assertTrue(observability.flush())

        attributes = memory.events[0].attributes
        self.assertEqual("model-a", attributes["openroutiq.model.id"])
        self.assertEqual("provider-a", attributes["openroutiq.provider.name"])
        self.assertEqual("coding", attributes["openroutiq.task.name"])

    def test_export_failures_never_change_or_raise_from_routing(self):
        baseline = Router([profile()], review_margin=0).route("hello").to_dict()
        memory = InMemoryExporter()
        with Observability([_FailingExporter(), memory]) as observability:
            decision = Router(
                [profile()],
                review_margin=0,
                observability=observability,
            ).route("hello")
            self.assertTrue(observability.flush())
            stats = observability.stats

        self.assertEqual(baseline, decision.to_dict())
        self.assertEqual(1, stats.export_failures)
        self.assertEqual(1, len(memory.events))

    def test_shutdown_timeout_is_retryable_and_does_not_race_exporter_close(self):
        exporter = _BlockingExporter()
        observability = Observability([exporter])
        decision = Router([profile()]).route("hello")
        self.assertTrue(observability.record_route(decision))
        self.assertTrue(exporter.started.wait(timeout=1))

        self.assertFalse(observability.shutdown(timeout_seconds=0.01))
        self.assertFalse(exporter.shutdown_called)
        exporter.release.set()
        self.assertTrue(observability.shutdown(timeout_seconds=1))
        self.assertTrue(exporter.shutdown_called)

    def test_adaptive_router_emits_only_the_final_route(self):
        memory = InMemoryExporter()
        with tempfile.TemporaryDirectory() as directory:
            with Observability([memory]) as observability:
                router = AdaptiveRouter(
                    [profile()],
                    registry=Path(directory) / "adaptive.sqlite3",
                    observability=observability,
                    review_margin=0,
                )
                router.route("hello", task="general")
                self.assertTrue(observability.flush())

        self.assertEqual(["route"], [event.event_type for event in memory.events])

    def test_proxy_emits_correlated_route_and_execution_without_bodies(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("proxy extras are not installed")
        from openroutiq.proxy import create_app

        async def execute(**_kwargs):
            return {
                "model": "provider-model",
                "choices": [{"message": {"role": "assistant", "content": PII_RESPONSE}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "cost": 0.001},
            }

        memory = InMemoryExporter()
        with Observability([memory]) as observability:
            router = Router([profile()], review_margin=0, observability=observability)
            response = TestClient(create_app(router, executor=execute)).post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": PII_PROMPT}],
                },
            )
            self.assertEqual(200, response.status_code)
            self.assertTrue(observability.flush())

        route, execution = memory.events
        self.assertEqual(["route", "execution"], [route.event_type, execution.event_type])
        self.assertEqual(route.trace_id, execution.trace_id)
        self.assertEqual(route.span_id, execution.parent_span_id)
        serialized = json.dumps([event.to_dict() for event in memory.events])
        self.assertNotIn(PII_PROMPT, serialized)
        self.assertNotIn(PII_RESPONSE, serialized)

    def test_provider_usage_serialization_failure_cannot_affect_response_path(self):
        from openroutiq.proxy.app import _observe_execution

        class BrokenProviderResult:
            def model_dump(self):
                raise RuntimeError(PII_RESPONSE)

        plain_router = Router([profile()])
        plain_decision = plain_router.route("synthetic")
        _observe_execution(
            plain_router,
            plain_decision,
            started=0,
            success=True,
            result=BrokenProviderResult(),
        )

        memory = InMemoryExporter()
        with Observability([memory]) as observability:
            observed_router = Router([profile()], observability=observability)
            observed_decision = observed_router.route("synthetic")
            _observe_execution(
                observed_router,
                observed_decision,
                started=0,
                success=True,
                result=BrokenProviderResult(),
            )
            self.assertTrue(observability.flush())

        execution = memory.events[1]
        self.assertNotIn("openroutiq.execution.input_tokens", execution.attributes)
        self.assertNotIn("openroutiq.execution.output_tokens", execution.attributes)
        self.assertNotIn("openroutiq.execution.actual_cost_usd", execution.attributes)
        self.assertNotIn(PII_RESPONSE, json.dumps(execution.to_dict()))

    def test_otlp_json_payload_and_repr_never_contain_authentication_secret(self):
        memory = InMemoryExporter()
        with Observability([memory]) as observability:
            decision = Router([profile()], observability=observability).route(PII_PROMPT)
            observability.record_execution(
                decision,
                duration_ms=10,
                success=True,
                input_tokens=4,
                output_tokens=2,
            )
            self.assertTrue(observability.flush())

        opener = _RecordingOpener()
        exporter = OTLPHTTPJSONExporter(
            "https://collector.example.test/private-path-must-not-appear/v1/traces",
            headers={"Authorization": f"Bearer {EXPORT_SECRET}"},
            _opener=opener,
        )
        exporter.export(memory.events[1])
        payload = opener.requests[0].data.decode("utf-8")
        headers = {name.casefold(): value for name, value in opener.requests[0].header_items()}

        self.assertEqual(f"Bearer {EXPORT_SECRET}", headers["authorization"])
        self.assertNotIn(EXPORT_SECRET, payload)
        self.assertNotIn(EXPORT_SECRET, repr(exporter))
        self.assertNotIn("private-path-must-not-appear", repr(exporter))
        self.assertNotIn(PII_PROMPT, payload)
        self.assertEqual(10.0, opener.timeouts[0])

    def test_otlp_configuration_rejects_unsafe_transport_inputs(self):
        with self.assertRaisesRegex(ObservabilityError, "non-loopback"):
            OTLPHTTPJSONExporter("http://collector.example.test/v1/traces")
        with self.assertRaisesRegex(ObservabilityError, "query"):
            OTLPHTTPJSONExporter("https://collector.example.test/v1/traces?token=secret")
        with self.assertRaisesRegex(ObservabilityError, "unique ignoring case"):
            OTLPHTTPJSONExporter(
                "https://collector.example.test/v1/traces",
                headers={"X-Key": "one", "x-key": "two"},
            )
        with self.assertRaisesRegex(ObservabilityError, "safe value"):
            OTLPHTTPJSONExporter(
                "https://collector.example.test/v1/traces",
                headers={"Authorization": "secret\x00suffix"},
            )
        with self.assertRaisesRegex(ObservabilityError, "privacy"):
            Observability([InMemoryExporter()], privacy=object())  # type: ignore[arg-type]

    def test_langtrace_preset_uses_otlp_json_without_secret_in_repr_or_payload(self):
        memory = InMemoryExporter()
        with Observability([memory]) as observability:
            Router([profile()], observability=observability).route("synthetic")
            self.assertTrue(observability.flush())
        opener = _RecordingOpener()
        exporter = LangtraceExporter(EXPORT_SECRET, _opener=opener)
        exporter.export(memory.events[0])
        request = opener.requests[0]
        headers = {name.casefold(): value for name, value in request.header_items()}

        self.assertEqual("https://app.langtrace.ai/api/trace", request.full_url)
        self.assertEqual(EXPORT_SECRET, headers["x-api-key"])
        self.assertNotIn(EXPORT_SECRET, request.data.decode("utf-8"))
        self.assertNotIn(EXPORT_SECRET, repr(exporter))

    def test_langsmith_preset_uses_documented_endpoint_and_headers(self):
        with patch.object(OTLPExporter, "__init__", return_value=None) as initialize:
            exporter = LangSmithExporter(EXPORT_SECRET, project_name="routing-production")

        endpoint = initialize.call_args.args[0]
        options = initialize.call_args.kwargs
        self.assertEqual("https://api.smith.langchain.com/otel/v1/traces", endpoint)
        self.assertEqual(EXPORT_SECRET, options["headers"]["x-api-key"])
        self.assertEqual("routing-production", options["headers"]["Langsmith-Project"])
        self.assertNotIn(EXPORT_SECRET, vars(exporter).values())

    def test_open_telemetry_adapter_ends_span_even_if_status_update_fails(self):
        try:
            import opentelemetry.trace  # noqa: F401
        except ImportError:
            self.skipTest("OpenTelemetry API is not installed")

        class FailingStatusSpan:
            def __init__(self) -> None:
                self.ended = False

            def set_status(self, _status) -> None:
                raise RuntimeError("status failure")

            def end(self, **_kwargs) -> None:
                self.ended = True

        class Tracer:
            def __init__(self, span) -> None:
                self.span = span

            def start_span(self, *_args, **_kwargs):
                return self.span

        memory = InMemoryExporter()
        with Observability([memory]) as observability:
            decision = Router([profile()], observability=observability).route("synthetic")
            observability.record_execution(decision, duration_ms=2, success=False)
            self.assertTrue(observability.flush())
        span = FailingStatusSpan()
        exporter = OpenTelemetryExporter(Tracer(span))

        with self.assertRaisesRegex(RuntimeError, "status failure"):
            exporter.export(memory.events[1])
        self.assertTrue(span.ended)

    def test_installed_otlp_extra_constructs_an_isolated_pipeline_without_network_io(self):
        try:
            exporter = OTLPExporter("http://127.0.0.1:4318/v1/traces")
        except ObservabilityError as exc:
            if "pip install openroutiq[observability]" in str(exc):
                self.skipTest("OTLP observability extra is not installed")
            raise
        try:
            self.assertEqual("http/protobuf", exporter.protocol)
            self.assertEqual("http://127.0.0.1:4318/v1/traces", exporter.endpoint)
        finally:
            exporter.shutdown()


if __name__ == "__main__":
    unittest.main()
