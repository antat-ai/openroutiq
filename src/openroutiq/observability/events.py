"""Privacy-bounded observability events for routing, execution, and evaluation."""

from __future__ import annotations

import hashlib
import math
import re
import secrets
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


OBSERVABILITY_SCHEMA_VERSION = 1
_EVENT_TYPES = frozenset({"route", "execution", "evaluation"})
_HEX_16 = re.compile(r"[0-9a-f]{16}\Z")
_HEX_32 = re.compile(r"[0-9a-f]{32}\Z")
_SAFE_ENUM = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_SAFE_STRATEGIES = frozenset(
    {"auto", "balanced", "cost", "custom", "latency", "quality", "risk_aware"}
)
_FAILURE_TYPES = frozenset(
    {
        "capability_mismatch",
        "model_failure",
        "protocol_failure",
        "provider_failure",
        "rate_limit",
        "routing_failure",
        "timeout",
        "tool_failure",
        "unknown_failure",
    }
)
_ATTRIBUTE_NAMES = frozenset(
    {
        "openroutiq.schema.version",
        "openroutiq.event.type",
        "openroutiq.operation.id",
        "openroutiq.model.id",
        "openroutiq.model.id_hash",
        "openroutiq.provider.name",
        "openroutiq.provider.name_hash",
        "openroutiq.task.name",
        "openroutiq.task.name_hash",
        "openroutiq.strategy",
        "openroutiq.strategy_hash",
        "openroutiq.reasoning.level",
        "openroutiq.route.duration_ms",
        "openroutiq.route.review_required",
        "openroutiq.route.out_of_domain",
        "openroutiq.route.high_risk",
        "openroutiq.route.candidate_count",
        "openroutiq.route.excluded_count",
        "openroutiq.route.estimated_input_tokens",
        "openroutiq.route.expected_output_tokens",
        "openroutiq.route.predicted_cost_usd",
        "openroutiq.route.predicted_latency_ms",
        "openroutiq.route.total_score",
        "openroutiq.route.quality_score",
        "openroutiq.route.context_similarity",
        "openroutiq.route.selection_probability",
        "openroutiq.execution.duration_ms",
        "openroutiq.execution.success",
        "openroutiq.execution.streaming",
        "openroutiq.execution.failure_type",
        "openroutiq.execution.actual_cost_usd",
        "openroutiq.execution.input_tokens",
        "openroutiq.execution.output_tokens",
        "openroutiq.evaluation.quality_score",
        "openroutiq.evaluation.latency_ms",
        "openroutiq.evaluation.actual_cost_usd",
        "openroutiq.evaluation.success",
        "openroutiq.evaluation.failure_type",
        "openroutiq.evaluation.input_tokens",
        "openroutiq.evaluation.output_tokens",
    }
)
AttributeValue = str | bool | int | float


class ObservabilityError(ValueError):
    """Raised for invalid observability configuration or event data."""


def _finite_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservabilityError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise ObservabilityError(f"{name} must be finite and at least {minimum}")
    return parsed


def _optional_number(value: Any, name: str, *, minimum: float = 0.0) -> float | None:
    return None if value is None else _finite_number(value, name, minimum=minimum)


def _optional_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**63 - 1:
        raise ObservabilityError(f"{name} must be an integer from zero through 2^63 - 1")
    return value


def _safe_attribute_value(name: str, value: Any) -> AttributeValue:
    if name not in _ATTRIBUTE_NAMES:
        raise ObservabilityError(f"unsupported observability attribute: {name}")
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < -(2**63) or value > 2**63 - 1:
            raise ObservabilityError(f"observability attribute {name} exceeds signed 64-bit range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ObservabilityError(f"observability attribute {name} must be finite")
        return value
    if (
        isinstance(value, str)
        and value
        and len(value) <= 256
        and "\r" not in value
        and "\n" not in value
    ):
        return value
    raise ObservabilityError(f"observability attribute {name} has an unsafe value")


@dataclass(frozen=True)
class ObservabilityPrivacy:
    """Controls explicit identifier export; request and response content is never supported."""

    include_model_identifiers: bool = False
    include_provider_identifiers: bool = False
    include_task_identifiers: bool = False
    pseudonymization_key: bytes | str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "include_model_identifiers",
            "include_provider_identifiers",
            "include_task_identifiers",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ObservabilityError(f"{name} must be a boolean")
        key = self.pseudonymization_key
        if isinstance(key, str):
            key = key.encode("utf-8")
            object.__setattr__(self, "pseudonymization_key", key)
        if key is not None and (not isinstance(key, bytes) or len(key) < 16):
            raise ObservabilityError("pseudonymization_key must contain at least 16 bytes")


@dataclass(frozen=True)
class ObservabilityEvent:
    """One immutable, allowlisted event that cannot contain prompts or response content."""

    event_type: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    started_at_unix_ns: int
    ended_at_unix_ns: int
    attributes: Mapping[str, AttributeValue]

    def __post_init__(self) -> None:
        if self.event_type not in _EVENT_TYPES:
            raise ObservabilityError("unsupported observability event type")
        if _HEX_32.fullmatch(self.trace_id) is None:
            raise ObservabilityError("trace_id must contain 32 lowercase hexadecimal characters")
        if int(self.trace_id, 16) == 0:
            raise ObservabilityError("trace_id cannot be all zeroes")
        if _HEX_16.fullmatch(self.span_id) is None:
            raise ObservabilityError("span_id must contain 16 lowercase hexadecimal characters")
        if int(self.span_id, 16) == 0:
            raise ObservabilityError("span_id cannot be all zeroes")
        if self.parent_span_id is not None and _HEX_16.fullmatch(self.parent_span_id) is None:
            raise ObservabilityError(
                "parent_span_id must contain 16 lowercase hexadecimal characters"
            )
        if self.parent_span_id is not None and int(self.parent_span_id, 16) == 0:
            raise ObservabilityError("parent_span_id cannot be all zeroes")
        for name, value in (
            ("started_at_unix_ns", self.started_at_unix_ns),
            ("ended_at_unix_ns", self.ended_at_unix_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ObservabilityError(f"{name} must be an integer of at least zero")
        if self.ended_at_unix_ns < self.started_at_unix_ns:
            raise ObservabilityError("ended_at_unix_ns cannot precede started_at_unix_ns")
        if not isinstance(self.attributes, Mapping):
            raise ObservabilityError("attributes must be a mapping")
        parsed = {
            str(name): _safe_attribute_value(str(name), value)
            for name, value in self.attributes.items()
        }
        if parsed.get("openroutiq.event.type") != self.event_type:
            raise ObservabilityError("event type attribute does not match the event")
        if parsed.get("openroutiq.operation.id") != self.trace_id:
            raise ObservabilityError("operation id attribute does not match the trace id")
        object.__setattr__(self, "attributes", MappingProxyType(parsed))

    @property
    def name(self) -> str:
        return f"openroutiq.{self.event_type}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "started_at_unix_ns": self.started_at_unix_ns,
            "ended_at_unix_ns": self.ended_at_unix_ns,
            "attributes": dict(self.attributes),
        }


class EventFactory:
    """Build allowlisted events while pseudonymizing identifiers before queueing."""

    def __init__(self, privacy: ObservabilityPrivacy | None = None) -> None:
        self.privacy = privacy or ObservabilityPrivacy()
        configured_key = self.privacy.pseudonymization_key
        if isinstance(configured_key, str):
            configured_key = configured_key.encode("utf-8")
        self._key: bytes = configured_key or secrets.token_bytes(32)

    def new_trace_id(self) -> str:
        trace_id = secrets.token_hex(16)
        return trace_id if int(trace_id, 16) else "1".zfill(32)

    @staticmethod
    def new_span_id() -> str:
        span_id = secrets.token_hex(8)
        return span_id if int(span_id, 16) else "1".zfill(16)

    def _fingerprint(self, namespace: str, value: str) -> str:
        digest = hashlib.blake2b(
            value.encode("utf-8", errors="replace"),
            key=self._key,
            digest_size=16,
            person=f"orq-{namespace}".encode("ascii")[:16],
        )
        return digest.hexdigest()

    def _identifier(
        self,
        attributes: dict[str, AttributeValue],
        *,
        prefix: str,
        namespace: str,
        value: Any,
        include: bool,
    ) -> None:
        if not isinstance(value, str) or not value:
            return
        if include and len(value) <= 256 and "\r" not in value and "\n" not in value:
            attributes[prefix] = value
        else:
            attributes[f"{prefix}_hash"] = self._fingerprint(namespace, value)

    def _base_attributes(self, event_type: str, trace_id: str) -> dict[str, AttributeValue]:
        return {
            "openroutiq.schema.version": OBSERVABILITY_SCHEMA_VERSION,
            "openroutiq.event.type": event_type,
            "openroutiq.operation.id": trace_id,
        }

    @staticmethod
    def _window(duration_ms: float | None) -> tuple[int, int, float | None]:
        ended = time.time_ns()
        duration = _optional_number(duration_ms, "duration_ms")
        started = ended if duration is None else max(0, ended - round(duration * 1_000_000))
        return started, ended, duration

    def route(
        self,
        decision: Any,
        *,
        duration_ms: float | None,
        trace_id: str,
        span_id: str,
    ) -> ObservabilityEvent:
        started, ended, duration = self._window(duration_ms)
        selected = decision.selected
        attributes = self._base_attributes("route", trace_id)
        self._identifier(
            attributes,
            prefix="openroutiq.model.id",
            namespace="model",
            value=getattr(selected, "model_id", None),
            include=self.privacy.include_model_identifiers,
        )
        self._identifier(
            attributes,
            prefix="openroutiq.provider.name",
            namespace="provider",
            value=getattr(selected, "provider", None),
            include=self.privacy.include_provider_identifiers,
        )
        self._identifier(
            attributes,
            prefix="openroutiq.task.name",
            namespace="task",
            value=getattr(decision, "task", None),
            include=self.privacy.include_task_identifiers,
        )
        strategy = getattr(decision, "strategy", None)
        if isinstance(strategy, str) and strategy in _SAFE_STRATEGIES:
            attributes["openroutiq.strategy"] = strategy
        elif isinstance(strategy, str) and strategy:
            attributes["openroutiq.strategy_hash"] = self._fingerprint("strategy", strategy)
        reasoning = getattr(selected, "reasoning_level", None)
        if isinstance(reasoning, str) and _SAFE_ENUM.fullmatch(reasoning):
            attributes["openroutiq.reasoning.level"] = reasoning
        if duration is not None:
            attributes["openroutiq.route.duration_ms"] = duration
        attributes.update(
            {
                "openroutiq.route.review_required": bool(decision.review_required),
                "openroutiq.route.out_of_domain": bool(decision.out_of_domain),
                "openroutiq.route.high_risk": bool(decision.analysis.high_risk),
                "openroutiq.route.candidate_count": len(decision.ranked),
                "openroutiq.route.excluded_count": len(decision.excluded),
                "openroutiq.route.estimated_input_tokens": int(decision.estimated_input_tokens),
                "openroutiq.route.expected_output_tokens": int(decision.expected_output_tokens),
                "openroutiq.route.predicted_cost_usd": float(selected.predicted_cost),
                "openroutiq.route.predicted_latency_ms": float(selected.expected_latency_ms),
                "openroutiq.route.total_score": float(selected.total_score),
                "openroutiq.route.quality_score": float(selected.quality_score),
                "openroutiq.route.context_similarity": float(decision.context_similarity),
                "openroutiq.route.selection_probability": float(decision.selection_probability),
            }
        )
        return ObservabilityEvent(
            event_type="route",
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            started_at_unix_ns=started,
            ended_at_unix_ns=ended,
            attributes=attributes,
        )

    def execution(
        self,
        decision: Any,
        *,
        duration_ms: float,
        success: bool,
        actual_cost_usd: float | None,
        input_tokens: int | None,
        output_tokens: int | None,
        failure_type: str | None,
        streaming: bool,
        trace_id: str,
        parent_span_id: str | None,
    ) -> ObservabilityEvent:
        started, ended, duration = self._window(duration_ms)
        assert duration is not None
        selected = decision.selected
        attributes = self._base_attributes("execution", trace_id)
        self._identifier(
            attributes,
            prefix="openroutiq.model.id",
            namespace="model",
            value=getattr(selected, "model_id", None),
            include=self.privacy.include_model_identifiers,
        )
        self._identifier(
            attributes,
            prefix="openroutiq.provider.name",
            namespace="provider",
            value=getattr(selected, "provider", None),
            include=self.privacy.include_provider_identifiers,
        )
        attributes.update(
            {
                "openroutiq.execution.duration_ms": duration,
                "openroutiq.execution.success": success,
                "openroutiq.execution.streaming": streaming,
            }
        )
        if isinstance(failure_type, str) and failure_type.casefold() in _FAILURE_TYPES:
            attributes["openroutiq.execution.failure_type"] = failure_type.casefold()
        actual_cost = _optional_number(actual_cost_usd, "actual_cost_usd")
        parsed_input = _optional_integer(input_tokens, "input_tokens")
        parsed_output = _optional_integer(output_tokens, "output_tokens")
        if actual_cost is not None:
            attributes["openroutiq.execution.actual_cost_usd"] = actual_cost
        if parsed_input is not None:
            attributes["openroutiq.execution.input_tokens"] = parsed_input
        if parsed_output is not None:
            attributes["openroutiq.execution.output_tokens"] = parsed_output
        return ObservabilityEvent(
            event_type="execution",
            trace_id=trace_id,
            span_id=self.new_span_id(),
            parent_span_id=parent_span_id,
            started_at_unix_ns=started,
            ended_at_unix_ns=ended,
            attributes=attributes,
        )

    def evaluation(
        self,
        *,
        model_id: str,
        provider: str | None,
        task: str | None,
        quality_score: float,
        latency_ms: float | None,
        actual_cost_usd: float | None,
        success: bool | None,
        failure_type: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> ObservabilityEvent:
        trace_id = self.new_trace_id()
        started, ended, _ = self._window(None)
        attributes = self._base_attributes("evaluation", trace_id)
        self._identifier(
            attributes,
            prefix="openroutiq.model.id",
            namespace="model",
            value=model_id,
            include=self.privacy.include_model_identifiers,
        )
        self._identifier(
            attributes,
            prefix="openroutiq.provider.name",
            namespace="provider",
            value=provider,
            include=self.privacy.include_provider_identifiers,
        )
        self._identifier(
            attributes,
            prefix="openroutiq.task.name",
            namespace="task",
            value=task,
            include=self.privacy.include_task_identifiers,
        )
        attributes["openroutiq.evaluation.quality_score"] = _finite_number(
            quality_score, "quality_score"
        )
        parsed_latency = _optional_number(latency_ms, "latency_ms")
        parsed_cost = _optional_number(actual_cost_usd, "actual_cost_usd")
        parsed_input = _optional_integer(input_tokens, "input_tokens")
        parsed_output = _optional_integer(output_tokens, "output_tokens")
        if parsed_latency is not None:
            attributes["openroutiq.evaluation.latency_ms"] = parsed_latency
        if parsed_cost is not None:
            attributes["openroutiq.evaluation.actual_cost_usd"] = parsed_cost
        if success is not None:
            if not isinstance(success, bool):
                raise ObservabilityError("success must be a boolean or None")
            attributes["openroutiq.evaluation.success"] = success
        if isinstance(failure_type, str) and failure_type.casefold() in _FAILURE_TYPES:
            attributes["openroutiq.evaluation.failure_type"] = failure_type.casefold()
        if parsed_input is not None:
            attributes["openroutiq.evaluation.input_tokens"] = parsed_input
        if parsed_output is not None:
            attributes["openroutiq.evaluation.output_tokens"] = parsed_output
        return ObservabilityEvent(
            event_type="evaluation",
            trace_id=trace_id,
            span_id=self.new_span_id(),
            parent_span_id=None,
            started_at_unix_ns=started,
            ended_at_unix_ns=ended,
            attributes=attributes,
        )
