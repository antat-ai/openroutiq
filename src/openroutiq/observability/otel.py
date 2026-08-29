"""OpenTelemetry span emission and OTLP HTTP/protobuf or gRPC configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from contextvars import ContextVar
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlsplit

from openroutiq.observability.events import AttributeValue, ObservabilityError, ObservabilityEvent


_TRACE_ID: ContextVar[int | None] = ContextVar("openroutiq_otel_trace_id", default=None)
_SPAN_ID: ContextVar[int | None] = ContextVar("openroutiq_otel_span_id", default=None)
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}\Z")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _package_version() -> str:
    try:
        return version("openroutiq")
    except PackageNotFoundError:
        return "0.1.0"


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObservabilityError(f"{name} must be non-empty text")
    parsed = value.strip()
    if len(parsed) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in parsed
    ):
        raise ObservabilityError(f"{name} is too long or contains a control character")
    return parsed


def _headers(value: Mapping[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ObservabilityError("headers must be a mapping")
    result: dict[str, str] = {}
    normalized_names: set[str] = set()
    for name, item in value.items():
        if not isinstance(name, str) or _HEADER_NAME.fullmatch(name) is None:
            raise ObservabilityError("OTLP header names must contain HTTP token characters")
        normalized_name = name.casefold()
        if normalized_name in normalized_names:
            raise ObservabilityError("OTLP header names must be unique ignoring case")
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 8_192
            or any(ord(character) < 32 or ord(character) == 127 for character in item)
        ):
            raise ObservabilityError(f"OTLP header {name} must contain a non-empty safe value")
        normalized_names.add(normalized_name)
        result[name] = item
    return result


def _validate_http_endpoint(endpoint: str, *, allow_insecure: bool) -> str:
    parsed_endpoint = _text(endpoint, "endpoint", maximum=2_048)
    if any(character.isspace() for character in parsed_endpoint) or "\\" in parsed_endpoint:
        raise ObservabilityError("HTTP OTLP endpoint cannot contain whitespace or backslashes")
    try:
        parsed = urlsplit(parsed_endpoint)
        hostname = parsed.hostname
    except ValueError as exc:
        raise ObservabilityError("HTTP OTLP endpoint is malformed") from exc
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ObservabilityError("HTTP OTLP endpoint must be an absolute http or https URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ObservabilityError("OTLP endpoint cannot contain credentials, a query, or a fragment")
    if (
        parsed.scheme == "http"
        and hostname.casefold() not in _LOOPBACK_HOSTS
        and not allow_insecure
    ):
        raise ObservabilityError("non-loopback HTTP OTLP endpoints require allow_insecure=True")
    return parsed_endpoint


def _validate_grpc_endpoint(endpoint: str, *, allow_insecure: bool) -> tuple[str, bool]:
    parsed_endpoint = _text(endpoint, "endpoint", maximum=2_048)
    if any(character.isspace() for character in parsed_endpoint) or "\\" in parsed_endpoint:
        raise ObservabilityError("gRPC OTLP endpoint cannot contain whitespace or backslashes")
    if "://" not in parsed_endpoint:
        host = parsed_endpoint.rsplit(":", 1)[0].strip("[]").casefold()
        if host not in _LOOPBACK_HOSTS and not allow_insecure:
            raise ObservabilityError(
                "scheme-free non-loopback gRPC endpoints require allow_insecure=True"
            )
        return parsed_endpoint, True
    try:
        parsed = urlsplit(parsed_endpoint)
        hostname = parsed.hostname
    except ValueError as exc:
        raise ObservabilityError("gRPC OTLP endpoint is malformed") from exc
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ObservabilityError("gRPC OTLP endpoint must be host:port or an http/https URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ObservabilityError(
            "gRPC OTLP endpoint cannot contain credentials, query, or fragment"
        )
    insecure = parsed.scheme == "http"
    if insecure and hostname.casefold() not in _LOOPBACK_HOSTS and not allow_insecure:
        raise ObservabilityError("non-loopback insecure gRPC endpoints require allow_insecure=True")
    return parsed_endpoint, insecure


def _endpoint_origin(endpoint: str | None) -> str | None:
    """Return a repr-safe endpoint origin without a possibly sensitive path."""

    if endpoint is None or "://" not in endpoint:
        return endpoint
    parsed = urlsplit(endpoint)
    return f"{parsed.scheme}://{parsed.netloc}"


class _EventIdGenerator:
    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback

    def generate_span_id(self) -> int:
        return _SPAN_ID.get() or self._fallback.generate_span_id()

    def generate_trace_id(self) -> int:
        return _TRACE_ID.get() or self._fallback.generate_trace_id()


class OpenTelemetryExporter:
    """Turn privacy-filtered OpenRoutiQ events into completed OpenTelemetry spans."""

    def __init__(
        self,
        tracer: Any,
        *,
        provider: Any = None,
        owns_provider: bool = False,
        preserve_event_ids: bool = False,
    ) -> None:
        if not callable(getattr(tracer, "start_span", None)):
            raise ObservabilityError("tracer must provide start_span")
        try:
            from opentelemetry.trace import (
                NonRecordingSpan,
                SpanContext,
                SpanKind,
                Status,
                StatusCode,
                TraceFlags,
                TraceState,
                set_span_in_context,
            )
        except ImportError as exc:
            raise ObservabilityError(
                "OpenTelemetry export requires 'pip install openroutiq[observability]'"
            ) from exc
        self._tracer = tracer
        self._provider = provider
        self._owns_provider = owns_provider
        self._preserve_event_ids = preserve_event_ids
        self._NonRecordingSpan = NonRecordingSpan
        self._SpanContext = SpanContext
        self._SpanKind = SpanKind
        self._Status = Status
        self._StatusCode = StatusCode
        self._TraceFlags = TraceFlags
        self._TraceState = TraceState
        self._set_span_in_context = set_span_in_context

    def _span_attributes(self, event: ObservabilityEvent) -> dict[str, AttributeValue]:
        return dict(event.attributes)

    def _parent_context(self, event: ObservabilityEvent) -> Any:
        if not self._preserve_event_ids or event.parent_span_id is None:
            return None
        parent = self._NonRecordingSpan(
            self._SpanContext(
                trace_id=int(event.trace_id, 16),
                span_id=int(event.parent_span_id, 16),
                is_remote=False,
                trace_flags=self._TraceFlags(self._TraceFlags.SAMPLED),
                trace_state=self._TraceState(),
            )
        )
        return self._set_span_in_context(parent)

    def export(self, event: ObservabilityEvent) -> None:
        trace_token = None
        span_token = None
        if self._preserve_event_ids:
            trace_token = _TRACE_ID.set(int(event.trace_id, 16))
            span_token = _SPAN_ID.set(int(event.span_id, 16))
        try:
            span = self._tracer.start_span(
                event.name,
                context=self._parent_context(event),
                kind=self._SpanKind.INTERNAL,
                attributes=self._span_attributes(event),
                start_time=event.started_at_unix_ns,
            )
            try:
                success = event.attributes.get(
                    "openroutiq.execution.success",
                    event.attributes.get("openroutiq.evaluation.success"),
                )
                if success is False:
                    span.set_status(self._Status(self._StatusCode.ERROR))
                elif success is True:
                    span.set_status(self._Status(self._StatusCode.OK))
            finally:
                span.end(end_time=event.ended_at_unix_ns)
        finally:
            if span_token is not None:
                _SPAN_ID.reset(span_token)
            if trace_token is not None:
                _TRACE_ID.reset(trace_token)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        if self._provider is None:
            return True
        method = getattr(self._provider, "force_flush", None)
        if not callable(method):
            return True
        return method(timeout_millis=timeout_millis) is not False

    def shutdown(self) -> None:
        if self._owns_provider and self._provider is not None:
            method = getattr(self._provider, "shutdown", None)
            if callable(method):
                method()


class OTLPExporter(OpenTelemetryExporter):
    """Configure an isolated OTLP HTTP/protobuf or gRPC span pipeline."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        protocol: str = "http/protobuf",
        headers: Mapping[str, str] | None = None,
        service_name: str = "openroutiq",
        timeout_seconds: float = 10.0,
        allow_insecure: bool = False,
    ) -> None:
        if protocol not in {"http/protobuf", "grpc"}:
            raise ObservabilityError("protocol must be 'http/protobuf' or 'grpc'")
        if not isinstance(allow_insecure, bool):
            raise ObservabilityError("allow_insecure must be a boolean")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ObservabilityError("timeout_seconds must be a positive finite number")
        parsed_headers = _headers(headers)
        parsed_service_name = _text(service_name, "service_name", maximum=256)
        parsed_endpoint: str | None = None
        grpc_insecure = False
        if endpoint is not None:
            if protocol == "http/protobuf":
                parsed_endpoint = _validate_http_endpoint(endpoint, allow_insecure=allow_insecure)
            else:
                parsed_endpoint, grpc_insecure = _validate_grpc_endpoint(
                    endpoint, allow_insecure=allow_insecure
                )
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

            if protocol == "http/protobuf":
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
            else:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
        except ImportError as exc:
            extra = "observability" if protocol == "http/protobuf" else "observability-grpc"
            raise ObservabilityError(
                f"OTLP {protocol} export requires 'pip install openroutiq[{extra}]'"
            ) from exc
        exporter_options: dict[str, Any] = {
            "headers": parsed_headers or None,
            "timeout": float(timeout_seconds),
        }
        if parsed_endpoint is not None:
            exporter_options["endpoint"] = parsed_endpoint
        if protocol == "grpc" and parsed_endpoint is not None:
            exporter_options["insecure"] = grpc_insecure
        span_exporter = OTLPSpanExporter(**exporter_options)
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": parsed_service_name,
                    "service.version": _package_version(),
                    "telemetry.sdk.language": "python",
                }
            ),
            id_generator=_EventIdGenerator(RandomIdGenerator()),
        )
        provider.add_span_processor(BatchSpanProcessor(span_exporter))
        self.endpoint = parsed_endpoint
        self.protocol = protocol
        self.service_name = parsed_service_name
        self.header_names = tuple(sorted(parsed_headers))
        super().__init__(
            provider.get_tracer("openroutiq.observability", _package_version()),
            provider=provider,
            owns_provider=True,
            preserve_event_ids=True,
        )

    def __repr__(self) -> str:
        return (
            f"OTLPExporter(endpoint_origin={_endpoint_origin(self.endpoint)!r}, "
            f"protocol={self.protocol!r}, header_names={self.header_names!r})"
        )
