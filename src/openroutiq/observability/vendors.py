"""Privacy-preserving presets for LangSmith and Langtrace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openroutiq.observability.events import AttributeValue, ObservabilityError, ObservabilityEvent
from openroutiq.observability.http_json import OTLPHTTPJSONExporter
from openroutiq.observability.otel import OTLPExporter, _endpoint_origin, _text


def _vendor_headers(required: Mapping[str, str], extra: Mapping[str, str] | None) -> dict[str, str]:
    result = dict(extra or {})
    supplied = {name.casefold() for name in result}
    conflicts = sorted(name for name in required if name.casefold() in supplied)
    if conflicts:
        raise ObservabilityError(
            "extra_headers cannot override vendor authentication headers: " + ", ".join(conflicts)
        )
    result.update(required)
    return result


class LangSmithExporter(OTLPExporter):
    """Send safe OpenTelemetry spans to LangSmith's documented OTLP endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        project_name: str = "default",
        endpoint: str = "https://api.smith.langchain.com/otel/v1/traces",
        service_name: str = "openroutiq",
        timeout_seconds: float = 10.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        key = _text(api_key, "api_key", maximum=4_096)
        project = _text(project_name, "project_name", maximum=256)
        super().__init__(
            endpoint,
            protocol="http/protobuf",
            headers=_vendor_headers(
                {"x-api-key": key, "Langsmith-Project": project},
                extra_headers,
            ),
            service_name=service_name,
            timeout_seconds=timeout_seconds,
        )
        self.project_name = project

    def _span_attributes(self, event: ObservabilityEvent) -> dict[str, AttributeValue]:
        attributes = super()._span_attributes(event)
        attributes["langsmith.trace.name"] = event.name
        attributes["langsmith.span.kind"] = "llm" if event.event_type == "execution" else "chain"
        model_id = event.attributes.get("openroutiq.model.id")
        provider = event.attributes.get("openroutiq.provider.name")
        if isinstance(model_id, str):
            attributes["gen_ai.request.model"] = model_id
        if isinstance(provider, str):
            attributes["gen_ai.system"] = provider
        return attributes

    def __repr__(self) -> str:
        return (
            f"LangSmithExporter(endpoint_origin={_endpoint_origin(self.endpoint)!r}, "
            f"header_names={self.header_names!r})"
        )


class LangtraceExporter(OTLPHTTPJSONExporter):
    """Send OTLP/JSON spans to Langtrace Cloud or a self-hosted Langtrace endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = "https://app.langtrace.ai/api/trace",
        service_name: str = "openroutiq",
        timeout_seconds: float = 10.0,
        allow_insecure: bool = False,
        extra_headers: Mapping[str, str] | None = None,
        _opener: Any = None,
    ) -> None:
        key = _text(api_key, "api_key", maximum=4_096)
        super().__init__(
            endpoint,
            headers=_vendor_headers({"x-api-key": key}, extra_headers),
            service_name=service_name,
            timeout_seconds=timeout_seconds,
            allow_insecure=allow_insecure,
            _opener=_opener,
        )

    def _span_attributes(self, event: ObservabilityEvent) -> dict[str, AttributeValue]:
        attributes = super()._span_attributes(event)
        attributes["gen_ai.operation.name"] = (
            "chat" if event.event_type == "execution" else event.event_type
        )
        model_id = event.attributes.get("openroutiq.model.id")
        provider = event.attributes.get("openroutiq.provider.name")
        if isinstance(model_id, str):
            attributes["gen_ai.request.model"] = model_id
        if isinstance(provider, str):
            attributes["gen_ai.system"] = provider
        return attributes

    def __repr__(self) -> str:
        return (
            f"LangtraceExporter(endpoint_origin={_endpoint_origin(self.endpoint)!r}, "
            f"header_names={self.header_names!r})"
        )
