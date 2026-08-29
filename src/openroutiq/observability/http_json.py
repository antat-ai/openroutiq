"""Dependency-free OTLP/HTTP JSON span export."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.request import OpenerDirector, Request, build_opener

from openroutiq.observability.events import AttributeValue, ObservabilityError, ObservabilityEvent
from openroutiq.observability.otel import (
    _endpoint_origin,
    _headers,
    _package_version,
    _text,
    _validate_http_endpoint,
)


def _otlp_value(value: AttributeValue) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": value}


def _otlp_attributes(attributes: Mapping[str, AttributeValue]) -> list[dict[str, Any]]:
    return [
        {"key": name, "value": _otlp_value(value)} for name, value in sorted(attributes.items())
    ]


class OTLPHTTPJSONExporter:
    """POST standard OTLP JSON spans using only Python's standard library."""

    def __init__(
        self,
        endpoint: str,
        *,
        headers: Mapping[str, str] | None = None,
        service_name: str = "openroutiq",
        timeout_seconds: float = 10.0,
        allow_insecure: bool = False,
        _opener: OpenerDirector | Any | None = None,
    ) -> None:
        if not isinstance(allow_insecure, bool):
            raise ObservabilityError("allow_insecure must be a boolean")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
            or timeout_seconds != timeout_seconds
            or timeout_seconds == float("inf")
        ):
            raise ObservabilityError("timeout_seconds must be a positive finite number")
        parsed_headers = _headers(headers)
        content_type = next(
            (value for name, value in parsed_headers.items() if name.casefold() == "content-type"),
            None,
        )
        if content_type is not None and content_type.casefold() != "application/json":
            raise ObservabilityError("OTLP/HTTP JSON Content-Type must be application/json")
        self.endpoint = _validate_http_endpoint(endpoint, allow_insecure=allow_insecure)
        self.service_name = _text(service_name, "service_name", maximum=256)
        self.timeout_seconds = float(timeout_seconds)
        self._headers = {
            **parsed_headers,
            "Content-Type": "application/json",
        }
        self.header_names = tuple(sorted(self._headers))
        self._opener = _opener or build_opener()

    def _span_attributes(self, event: ObservabilityEvent) -> dict[str, AttributeValue]:
        return dict(event.attributes)

    def payload(self, event: ObservabilityEvent) -> dict[str, Any]:
        success = event.attributes.get(
            "openroutiq.execution.success",
            event.attributes.get("openroutiq.evaluation.success"),
        )
        span: dict[str, Any] = {
            "traceId": event.trace_id,
            "spanId": event.span_id,
            "name": event.name,
            "kind": 1,
            "startTimeUnixNano": str(event.started_at_unix_ns),
            "endTimeUnixNano": str(event.ended_at_unix_ns),
            "attributes": _otlp_attributes(self._span_attributes(event)),
            "status": {"code": 2 if success is False else 1 if success is True else 0},
            "flags": 1,
        }
        if event.parent_span_id is not None:
            span["parentSpanId"] = event.parent_span_id
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": _otlp_attributes(
                            {
                                "service.name": self.service_name,
                                "service.version": _package_version(),
                                "telemetry.sdk.language": "python",
                            }
                        )
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "openroutiq.observability",
                                "version": _package_version(),
                            },
                            "spans": [span],
                        }
                    ],
                }
            ]
        }

    def export(self, event: ObservabilityEvent) -> None:
        request = Request(
            self.endpoint,
            data=json.dumps(self.payload(event), separators=(",", ":")).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        response = self._opener.open(request, timeout=self.timeout_seconds)
        try:
            status = getattr(response, "status", 200)
            if not isinstance(status, int) or status < 200 or status >= 300:
                raise ObservabilityError("OTLP/HTTP JSON endpoint rejected the span")
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True

    def shutdown(self) -> None:
        return None

    def __repr__(self) -> str:
        return (
            f"OTLPHTTPJSONExporter(endpoint_origin={_endpoint_origin(self.endpoint)!r}, "
            f"header_names={self.header_names!r})"
        )
