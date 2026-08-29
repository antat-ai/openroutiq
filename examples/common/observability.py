"""Build privacy-bounded exporters from process configuration.

Inject secrets through a production secret manager or the process environment. OpenRoutiQ does
not load ``.env`` files, and these helpers never print credential values.
"""

from __future__ import annotations

import os

from openroutiq import (
    LangSmithExporter,
    LangtraceExporter,
    OTLPExporter,
    Observability,
    ObservabilityError,
    ObservabilityPrivacy,
)


SUPPORTED_BACKENDS = frozenset({"langsmith", "langtrace", "otlp"})


def _required_secret(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ObservabilityError(f"{name} must be supplied by the process secret manager")
    return value


def _backend_names() -> tuple[str, ...]:
    configured = os.environ.get("OPENROUTIQ_OBSERVABILITY_BACKENDS", "")
    names = tuple(
        dict.fromkeys(item.strip().casefold() for item in configured.split(",") if item.strip())
    )
    unknown = sorted(set(names) - SUPPORTED_BACKENDS)
    if unknown:
        raise ObservabilityError(
            "unsupported OPENROUTIQ_OBSERVABILITY_BACKENDS values: " + ", ".join(unknown)
        )
    return names


def build_observability_from_environment() -> Observability | None:
    """Return an opt-in fan-out dispatcher, or ``None`` when no backend is configured."""

    exporters = []
    for backend in _backend_names():
        if backend == "otlp":
            exporters.append(
                OTLPExporter(
                    endpoint=os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"),
                    service_name=os.environ.get("OTEL_SERVICE_NAME", "openroutiq-example"),
                )
            )
        elif backend == "langsmith":
            exporters.append(
                LangSmithExporter(
                    _required_secret("LANGSMITH_API_KEY"),
                    project_name=os.environ.get("LANGSMITH_PROJECT", "openroutiq-example"),
                )
            )
        elif backend == "langtrace":
            exporters.append(
                LangtraceExporter(
                    _required_secret("LANGTRACE_API_KEY"),
                    endpoint=os.environ.get(
                        "LANGTRACE_OTLP_ENDPOINT",
                        "https://app.langtrace.ai/api/trace",
                    ),
                )
            )

    if not exporters:
        return None
    hash_key = os.environ.get("OPENROUTIQ_OBSERVABILITY_HASH_KEY")
    privacy = ObservabilityPrivacy(pseudonymization_key=hash_key)
    return Observability(exporters, privacy=privacy, max_queue_size=2_048)
