from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from openroutiq.benchmark.core import BenchmarkError
from openroutiq.benchmark.protocol import OPENROUTER_BENCHMARK_PROTOCOL


_MONEY_QUANTUM = Decimal("0.000000001")


class BenchmarkBudgetError(BenchmarkError):
    """A request exceeded a reviewed benchmark call or cost boundary."""


class BenchmarkProtocolError(BenchmarkError):
    """Repeated upstream failures made the live benchmark unsafe to continue."""


def _money(value: Decimal) -> str:
    return format(value.quantize(_MONEY_QUANTUM), "f")


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


def _canonical_response_model(
    *,
    call_kind: str,
    requested_model: str,
    executed_model: Any,
    provider: Any,
) -> tuple[Any, str | None]:
    """Canonicalize OpenRouter's provider-native embedding response alias.

    OpenRouter accepts namespaced model IDs, but the OpenAI embeddings endpoint
    reports the provider-native ID without the ``openai/`` namespace. The two
    values identify the same reviewed model; only this exact, provider-scoped
    transformation is accepted.
    """

    if (
        call_kind == "embedding"
        and provider == "OpenAI"
        and isinstance(executed_model, str)
        and requested_model.startswith("openai/")
        and executed_model == requested_model.removeprefix("openai/")
    ):
        return requested_model, "openrouter_openai_namespace_stripped"
    return executed_model, None


def normalize_gateway_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return the effective form of a raw, immutable gateway event.

    This also permits an interrupted run to resume after a historical event was
    falsely rejected solely because OpenRouter stripped the OpenAI namespace
    from an embedding response, or after a nominally successful response omitted
    the model/usage provenance required to verify it. Those malformed successes
    remain counted, zero-score protocol failures. Real price, global-cost, and
    model violations remain fail-closed.
    """

    normalized = dict(event)
    status = normalized.get("status")
    if not isinstance(status, int) or isinstance(status, bool) or not 200 <= status < 300:
        return normalized
    executed_model = normalized.get("executed_model")
    missing_model_provenance = not isinstance(executed_model, str) or not executed_model.strip()
    missing_usage_provenance = normalized.get("usage_cost_present") is False
    if missing_model_provenance or missing_usage_provenance:
        normalized["status"] = 502
        normalized["executed_model_allowed"] = None
        normalized["response_model_normalization"] = "malformed_success_missing_provenance"
        normalized["error"] = normalized.get("error") or (
            "upstream HTTP 2xx response omitted required model or usage provenance"
        )
        normalized["boundary_breached"] = bool(
            normalized.get("price_boundary_exceeded")
            or normalized.get("global_cost_boundary_exceeded")
        )
        return normalized
    canonical_model, reason = _canonical_response_model(
        call_kind=str(normalized.get("call_kind", "")),
        requested_model=str(normalized.get("requested_model", "")),
        executed_model=executed_model,
        provider=normalized.get("provider"),
    )
    if reason is None:
        return normalized
    normalized["reported_executed_model"] = normalized.get("executed_model")
    normalized["executed_model"] = canonical_model
    normalized["executed_model_allowed"] = True
    normalized["response_model_normalization"] = reason
    normalized["boundary_breached"] = bool(
        normalized.get("price_boundary_exceeded") or normalized.get("global_cost_boundary_exceeded")
    )
    return normalized


def gateway_event_reconciliation(event: Mapping[str, Any]) -> dict[str, Any] | None:
    """Describe a deterministic raw-to-effective event correction, if any."""

    raw = dict(event)
    effective = normalize_gateway_event(raw)
    if effective == raw:
        return None

    def digest(value: Mapping[str, Any]) -> str:
        payload = json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return {
        "schema_version": 1,
        "sequence": raw.get("sequence"),
        "observation_key": raw.get("case_id"),
        "reason": effective.get("response_model_normalization"),
        "reported_executed_model": raw.get("executed_model"),
        "canonical_executed_model": effective.get("executed_model"),
        "counts_toward_approved_calls": True,
        "reported_cost_usd": raw.get("reported_cost_usd"),
        "raw_event_sha256": digest(raw),
        "effective_event_sha256": digest(effective),
    }


@dataclass(frozen=True)
class GatewayCaseBudget:
    track: str
    system: str
    case_id: str
    allowed_models: tuple[str, ...]
    maximum_calls: int
    maximum_cost_per_call_usd: Decimal
    maximum_output_tokens: int
    reasoning_effort: str | None = None
    auto_cost_tier: str | None = None
    provider_sort: str | None = None
    provider_max_prompt_price_per_million_usd: Decimal | None = None
    provider_max_completion_price_per_million_usd: Decimal | None = None
    provider_max_request_price_usd: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.track or not self.system or not self.case_id:
            raise BenchmarkBudgetError("gateway case identifiers must be non-empty")
        if not self.allowed_models or len(set(self.allowed_models)) != len(self.allowed_models):
            raise BenchmarkBudgetError("gateway allowed_models must be non-empty and unique")
        if self.maximum_calls < 1:
            raise BenchmarkBudgetError("gateway maximum_calls must be >= 1")
        if self.maximum_cost_per_call_usd < 0:
            raise BenchmarkBudgetError("gateway maximum cost must be non-negative")
        if self.maximum_output_tokens < 1:
            raise BenchmarkBudgetError("gateway maximum output tokens must be >= 1")
        if self.auto_cost_tier not in {None, "low", "medium", "high", "xhigh", "max"}:
            raise BenchmarkBudgetError("gateway auto_cost_tier is invalid")
        if self.provider_sort not in {None, "price"}:
            raise BenchmarkBudgetError("gateway provider sort is invalid")
        token_prices = (
            self.provider_max_prompt_price_per_million_usd,
            self.provider_max_completion_price_per_million_usd,
        )
        if (token_prices[0] is None) != (token_prices[1] is None):
            raise BenchmarkBudgetError(
                "gateway provider prompt and completion price caps must be supplied together"
            )
        for price in (*token_prices, self.provider_max_request_price_usd):
            if price is not None and price < 0:
                raise BenchmarkBudgetError("gateway provider price caps must be non-negative")


@dataclass(frozen=True)
class GatewayReservation:
    sequence: int
    context: GatewayCaseBudget
    requested_model: str
    call_kind: str
    requested_output_tokens: int | None
    started_at: float
    context_token: str | None = None


@dataclass
class _ActiveGatewayCase:
    context: GatewayCaseBudget
    calls: int = 0
    inflight_sequences: set[int] = field(default_factory=set)
    sequences: list[int] = field(default_factory=list)


class BenchmarkBudgetLedger:
    """Thread-safe fail-closed accounting for a live benchmark relay."""

    def __init__(
        self,
        *,
        maximum_calls: int,
        maximum_reserved_cost_usd: Decimal | str | float,
        event_path: str | Path | None = None,
        initial_events: Sequence[Mapping[str, Any]] = (),
        maximum_consecutive_upstream_failures: int = int(
            OPENROUTER_BENCHMARK_PROTOCOL["maximum_consecutive_upstream_failures"]
        ),
        acknowledged_circuit_breaker_sequence: int | None = None,
    ) -> None:
        if (
            isinstance(maximum_calls, bool)
            or not isinstance(maximum_calls, int)
            or maximum_calls < 0
        ):
            raise BenchmarkBudgetError("maximum_calls must be a non-negative integer")
        cost = _decimal(maximum_reserved_cost_usd)
        if cost is None:
            raise BenchmarkBudgetError("maximum_reserved_cost_usd must be non-negative")
        if (
            isinstance(maximum_consecutive_upstream_failures, bool)
            or not isinstance(maximum_consecutive_upstream_failures, int)
            or maximum_consecutive_upstream_failures < 1
        ):
            raise BenchmarkBudgetError(
                "maximum_consecutive_upstream_failures must be a positive integer"
            )
        self.maximum_calls = maximum_calls
        self.maximum_reserved_cost_usd = cost
        self.maximum_consecutive_upstream_failures = maximum_consecutive_upstream_failures
        self.event_path = None if event_path is None else Path(event_path)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._active_cases: dict[str | None, _ActiveGatewayCase] = {}
        self._inflight_sequences: set[int] = set()
        self._events: list[dict[str, Any]] = []
        self._reserved_cost = Decimal("0")
        self._reported_cost = Decimal("0")
        self._budget_breached = False
        self._consecutive_upstream_failures = 0
        self._circuit_breaker_tripped = False
        self._historical_circuit_breaker_trips: list[int] = []
        self._acknowledged_circuit_breaker_sequences: list[int] = []
        self._settled_sequences: set[int] = set()
        normalized_initial_events = sorted(
            (dict(raw) for raw in initial_events), key=lambda item: int(item.get("sequence", 0))
        )
        expected_sequences = list(range(1, len(normalized_initial_events) + 1))
        if [item.get("sequence") for item in normalized_initial_events] != expected_sequences:
            raise BenchmarkBudgetError(
                "initial gateway events must have contiguous one-based sequences"
            )
        for index, event in enumerate(normalized_initial_events, start=1):
            reserved = _decimal(event.get("reserved_cost_usd"))
            reported = _decimal(event.get("reported_cost_usd"))
            if reserved is None or reported is None:
                raise BenchmarkBudgetError("initial gateway event costs are invalid")
            self._reserved_cost += reserved
            self._reported_cost += reported
            self._events.append(event)
            self._settled_sequences.add(index)
            if (
                reported > reserved
                or event.get("executed_model_allowed") is False
                or event.get("boundary_breached") is True
            ):
                self._budget_breached = True
            status = event.get("status")
            if isinstance(status, int) and not isinstance(status, bool) and 200 <= status < 300:
                self._consecutive_upstream_failures = 0
            else:
                self._consecutive_upstream_failures += 1
            if (
                event.get("circuit_breaker_tripped") is True
                or self._consecutive_upstream_failures >= self.maximum_consecutive_upstream_failures
            ):
                self._circuit_breaker_tripped = True
                self._historical_circuit_breaker_trips.append(index)
        if normalized_initial_events:
            terminal_event = normalized_initial_events[-1]
            terminal_trip = bool(terminal_event.get("circuit_breaker_tripped")) or (
                self._consecutive_upstream_failures >= self.maximum_consecutive_upstream_failures
            )
            self._circuit_breaker_tripped = terminal_trip
            if not terminal_trip:
                # Events can only exist after a historical terminal trip if a prior
                # runner explicitly acknowledged it. Preserve that audit fact when
                # reconstructing an interrupted or subsequently stopped ledger.
                self._acknowledged_circuit_breaker_sequences.extend(
                    self._historical_circuit_breaker_trips
                )
        self._total_calls = len(self._events)
        self._next_settlement_sequence = self._total_calls + 1
        if self._total_calls > self.maximum_calls:
            raise BenchmarkBudgetError("initial gateway events exceed the approved call cap")
        if self._reserved_cost > self.maximum_reserved_cost_usd:
            raise BenchmarkBudgetError("initial gateway events exceed the approved cost cap")
        if self._reported_cost > self.maximum_reserved_cost_usd:
            self._budget_breached = True
        if acknowledged_circuit_breaker_sequence is not None:
            if (
                isinstance(acknowledged_circuit_breaker_sequence, bool)
                or not isinstance(acknowledged_circuit_breaker_sequence, int)
                or acknowledged_circuit_breaker_sequence < 1
            ):
                raise BenchmarkProtocolError(
                    "acknowledged circuit-breaker sequence must be a positive integer"
                )
            if self._budget_breached:
                raise BenchmarkProtocolError(
                    "a circuit breaker cannot be acknowledged after a budget breach"
                )
            if not normalized_initial_events or not self._circuit_breaker_tripped:
                raise BenchmarkProtocolError(
                    "no historical circuit-breaker trip is available to acknowledge"
                )
            terminal_event = normalized_initial_events[-1]
            if (
                acknowledged_circuit_breaker_sequence != self._total_calls
                or terminal_event.get("circuit_breaker_tripped") is not True
            ):
                raise BenchmarkProtocolError(
                    "circuit-breaker acknowledgement must match the terminal settled trip"
                )
            self._acknowledged_circuit_breaker_sequences.append(
                acknowledged_circuit_breaker_sequence
            )
            self._consecutive_upstream_failures = 0
            self._circuit_breaker_tripped = False

    def begin_case(self, context: GatewayCaseBudget, *, token: str | None = None) -> None:
        with self._condition:
            if token is not None and (not token or not token.isascii() or len(token) > 128):
                raise BenchmarkProtocolError("benchmark case token is invalid")
            if token is None and self._inflight_sequences:
                raise BenchmarkProtocolError(
                    "cannot begin a benchmark case while another case is active"
                )
            active = self._active_cases.get(token)
            if active is not None and active.inflight_sequences:
                raise BenchmarkProtocolError("cannot replace an active benchmark case")
            # Preserve the original convenience API: a fully settled case may
            # be replaced without an explicit ``end_case`` call.  Only active
            # reservations make a transition unsafe.
            self._active_cases[token] = _ActiveGatewayCase(context=context)

    def end_case(
        self, *, token: str | None = None, timeout_seconds: float = 620.0
    ) -> tuple[int, ...]:
        """Wait for every accepted reservation before releasing the case.

        Frameworks may issue concurrent requests and may return as soon as one
        branch fails.  Clearing the case immediately can otherwise let grading
        race ahead of a still-billable provider request.  The upstream relay
        timeout is 600 seconds, so this bounded join leaves a small settlement
        margin and fails closed if a handler never reports completion.
        """

        if timeout_seconds <= 0:
            raise BenchmarkProtocolError("case settlement timeout must be positive")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            active = self._active_cases.get(token)
            if active is None:
                return ()
            while active.inflight_sequences:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._circuit_breaker_tripped = True
                    raise BenchmarkProtocolError(
                        "benchmark case ended with unsettled provider reservations: "
                        + ", ".join(str(item) for item in sorted(active.inflight_sequences))
                    )
                self._condition.wait(timeout=remaining)
            sequences = tuple(active.sequences)
            del self._active_cases[token]
            return sequences

    def assert_open(self) -> None:
        """Raise as soon as a live-call boundary or failure circuit has closed."""
        with self._condition:
            if self._budget_breached:
                raise BenchmarkBudgetError(
                    "a live call exceeded its reviewed boundary; benchmark is stopped"
                )
            if self._circuit_breaker_tripped:
                raise BenchmarkProtocolError(
                    "consecutive upstream failures tripped the live-run circuit breaker; "
                    "benchmark is stopped"
                )

    def _validate_model(
        self,
        payload: dict[str, Any],
        context: GatewayCaseBudget,
        call_kind: str,
    ) -> str:
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise BenchmarkBudgetError("relay request requires a non-empty model")
        allowed = set(context.allowed_models)
        if model == "openrouter/auto" and call_kind == "chat_completion":
            plugins = payload.get("plugins", [])
            if not isinstance(plugins, list):
                raise BenchmarkBudgetError("OpenRouter Auto request requires a plugin list")
            auto = [
                item
                for item in plugins
                if isinstance(item, dict) and item.get("id") == "auto-router"
            ]
            if len(auto) > 1:
                raise BenchmarkBudgetError(
                    "OpenRouter Auto request cannot contain duplicate auto-router plugins"
                )
            if auto:
                routed_pool = auto[0].get("allowed_models")
                if routed_pool is not None and (
                    not isinstance(routed_pool, list) or set(routed_pool) != allowed
                ):
                    raise BenchmarkBudgetError(
                        "OpenRouter Auto allowed_models must match the frozen eligible pool"
                    )
        elif model not in allowed:
            raise BenchmarkBudgetError(f"model {model} is outside the reviewed case pool")
        return model

    def reserve(
        self,
        payload: dict[str, Any],
        *,
        call_kind: str = "chat_completion",
        context_token: str | None = None,
    ) -> GatewayReservation:
        with self._lock:
            if call_kind not in {"chat_completion", "embedding"}:
                raise BenchmarkBudgetError("relay call kind is not supported")
            active = self._active_cases.get(context_token)
            if active is None:
                raise BenchmarkBudgetError("relay request has no active benchmark case")
            context = active.context
            if self._budget_breached:
                raise BenchmarkBudgetError(
                    "a prior call exceeded its reviewed price boundary; benchmark is stopped"
                )
            if self._circuit_breaker_tripped:
                raise BenchmarkProtocolError(
                    "a prior upstream failure sequence tripped the live-run circuit breaker; "
                    "benchmark is stopped"
                )
            if payload.get("stream"):
                raise BenchmarkBudgetError("streaming is disabled in the reproducible benchmark")
            model = self._validate_model(payload, context, call_kind)
            requested_output_tokens: int | None = None
            if call_kind == "chat_completion":
                supplied_output_caps = {
                    key: payload[key]
                    for key in ("max_tokens", "max_completion_tokens")
                    if payload.get(key) is not None
                }
                cap_values = list(supplied_output_caps.values())
                if any(value != cap_values[0] for value in cap_values[1:]) if cap_values else False:
                    raise BenchmarkBudgetError(
                        "relay output-token cap fields must contain one identical reviewed value"
                    )
                raw_output = next(iter(supplied_output_caps.values()), None)
                if (
                    isinstance(raw_output, bool)
                    or not isinstance(raw_output, int)
                    or raw_output < 1
                ):
                    raise BenchmarkBudgetError(
                        "every relay chat request must set a positive output-token cap"
                    )
                if raw_output > context.maximum_output_tokens:
                    raise BenchmarkBudgetError(
                        f"request output cap {raw_output} exceeds reviewed cap "
                        f"{context.maximum_output_tokens}"
                    )
                requested_output_tokens = raw_output
            else:
                raw_input = payload.get("input")
                if not isinstance(raw_input, str) and not (
                    isinstance(raw_input, list)
                    and raw_input
                    and all(isinstance(item, str) and item for item in raw_input)
                ):
                    raise BenchmarkBudgetError(
                        "benchmark embedding input must be text or a non-empty text list"
                    )
            if active.calls >= context.maximum_calls:
                raise BenchmarkBudgetError(
                    f"case {context.case_id} exceeded its {context.maximum_calls}-call ceiling"
                )
            if self._total_calls >= self.maximum_calls:
                raise BenchmarkBudgetError(
                    f"benchmark exceeded its {self.maximum_calls}-call ceiling"
                )
            next_cost = self._reserved_cost + context.maximum_cost_per_call_usd
            if next_cost > self.maximum_reserved_cost_usd:
                raise BenchmarkBudgetError(
                    f"benchmark reservation ${_money(next_cost)} exceeds approved "
                    f"${_money(self.maximum_reserved_cost_usd)} cap"
                )
            active.calls += 1
            self._total_calls += 1
            self._reserved_cost = next_cost
            reservation = GatewayReservation(
                sequence=self._total_calls,
                context=context,
                requested_model=model,
                call_kind=call_kind,
                requested_output_tokens=requested_output_tokens,
                started_at=time.perf_counter(),
                context_token=context_token,
            )
            self._inflight_sequences.add(reservation.sequence)
            active.inflight_sequences.add(reservation.sequence)
            active.sequences.append(reservation.sequence)
            return reservation

    def finish(
        self,
        reservation: GatewayReservation,
        *,
        status: int,
        response: dict[str, Any] | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        raw_reported_cost = _decimal(usage.get("cost"))
        usage_cost_present = raw_reported_cost is not None
        reported_cost = (
            Decimal("0")
            if raw_reported_cost is None
            else raw_reported_cost.quantize(_MONEY_QUANTUM)
        )
        reported_executed_model = response.get("model") if isinstance(response, dict) else None
        provider = response.get("provider") if isinstance(response, dict) else None
        executed_model, response_model_normalization = _canonical_response_model(
            call_kind=reservation.call_kind,
            requested_model=reservation.requested_model,
            executed_model=reported_executed_model,
            provider=provider,
        )
        executed_model_allowed = (
            executed_model in reservation.context.allowed_models if 200 <= status < 300 else None
        )
        raw_completion_tokens = usage.get("completion_tokens")
        completion_tokens = (
            raw_completion_tokens
            if isinstance(raw_completion_tokens, int)
            and not isinstance(raw_completion_tokens, bool)
            and raw_completion_tokens >= 0
            else None
        )
        output_cap_exceeded = (
            reservation.requested_output_tokens is not None
            and completion_tokens is not None
            and completion_tokens > reservation.requested_output_tokens
        )
        price_boundary_exceeded = reported_cost > reservation.context.maximum_cost_per_call_usd
        event = {
            "sequence": reservation.sequence,
            "track": reservation.context.track,
            "system": reservation.context.system,
            "case_id": reservation.context.case_id,
            "call_kind": reservation.call_kind,
            "requested_model": reservation.requested_model,
            "executed_model": executed_model,
            "reported_executed_model": reported_executed_model,
            "response_model_normalization": response_model_normalization,
            "executed_model_allowed": executed_model_allowed,
            "provider": provider,
            "response_id": response.get("id") if isinstance(response, dict) else None,
            "status": status,
            "latency_ms": (time.perf_counter() - reservation.started_at) * 1000,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion_tokens,
            "requested_output_tokens": reservation.requested_output_tokens,
            "output_cap_exceeded": output_cap_exceeded,
            "reported_cost_usd": _money(reported_cost),
            "usage_cost_present": usage_cost_present,
            "reserved_cost_usd": _money(reservation.context.maximum_cost_per_call_usd),
            "price_boundary_exceeded": price_boundary_exceeded,
            "error": error,
        }
        with self._condition:
            if reservation.sequence in self._settled_sequences:
                raise BenchmarkProtocolError(
                    f"gateway reservation {reservation.sequence} was already settled"
                )
            while reservation.sequence != self._next_settlement_sequence:
                self._condition.wait()
            self._reported_cost += reported_cost
            if 200 <= status < 300:
                self._consecutive_upstream_failures = 0
            else:
                self._consecutive_upstream_failures += 1
            circuit_breaker_tripped = self._circuit_breaker_tripped or (
                self._consecutive_upstream_failures >= self.maximum_consecutive_upstream_failures
            )
            if circuit_breaker_tripped:
                self._circuit_breaker_tripped = True
            global_cost_boundary_exceeded = self._reported_cost > self.maximum_reserved_cost_usd
            boundary_breached = (
                price_boundary_exceeded
                or global_cost_boundary_exceeded
                or executed_model_allowed is False
            )
            event["global_cost_boundary_exceeded"] = global_cost_boundary_exceeded
            event["boundary_breached"] = boundary_breached
            event["consecutive_upstream_failures"] = self._consecutive_upstream_failures
            event["circuit_breaker_tripped"] = circuit_breaker_tripped
            if boundary_breached:
                self._budget_breached = True
            self._settled_sequences.add(reservation.sequence)
            self._events.append(event)
            if self.event_path is not None:
                self.event_path.parent.mkdir(parents=True, exist_ok=True)
                with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(event, separators=(",", ":")))
                    handle.write("\n")
            self._inflight_sequences.discard(reservation.sequence)
            active = self._active_cases.get(reservation.context_token)
            if active is not None:
                active.inflight_sequences.discard(reservation.sequence)
            self._next_settlement_sequence += 1
            self._condition.notify_all()
        return event

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "maximum_calls": self.maximum_calls,
                "calls_made": self._total_calls,
                "maximum_reserved_cost_usd": _money(self.maximum_reserved_cost_usd),
                "reserved_cost_usd": _money(self._reserved_cost),
                "reported_cost_usd": _money(self._reported_cost),
                "budget_breached": self._budget_breached,
                "circuit_breaker_tripped": self._circuit_breaker_tripped,
                "consecutive_upstream_failures": self._consecutive_upstream_failures,
                "historical_circuit_breaker_trips": list(self._historical_circuit_breaker_trips),
                "acknowledged_circuit_breaker_sequences": list(
                    self._acknowledged_circuit_breaker_sequences
                ),
                "output_cap_warnings": sum(
                    event.get("output_cap_exceeded") is True for event in self._events
                ),
                "inflight_reservations": sorted(self._inflight_sequences),
                "events": list(self._events),
            }


class _RelayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        ledger: BenchmarkBudgetLedger,
        upstream_base_url: str,
        api_key: str,
        relay_key: str,
    ) -> None:
        self.ledger = ledger
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.api_key = api_key
        self.relay_key = relay_key
        super().__init__(address, _RelayHandler)


class _RelayHandler(BaseHTTPRequestHandler):
    server: _RelayServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _json_response(self, status: int, value: Any) -> None:
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"/healthz", "/v1/models"}:
            self._json_response(200, {"object": "list", "data": []})
            return
        self._json_response(404, {"error": {"message": "unsupported benchmark relay path"}})

    def do_POST(self) -> None:  # noqa: N802
        authorization = self.headers.get("Authorization", "")
        supplied = authorization.removeprefix("Bearer ")
        context_token: str | None = None
        if secrets.compare_digest(supplied, self.server.relay_key):
            pass
        elif "." in supplied:
            supplied_key, context_token = supplied.split(".", 1)
            if not context_token or not secrets.compare_digest(supplied_key, self.server.relay_key):
                self._json_response(401, {"error": {"message": "invalid relay credential"}})
                return
        else:
            self._json_response(401, {"error": {"message": "invalid relay credential"}})
            return
        normalized = self.path.split("?", 1)[0].rstrip("/")
        if normalized in {"/v1/chat/completions", "/chat/completions"}:
            call_kind = "chat_completion"
            upstream_path = "/chat/completions"
        elif normalized in {"/v1/embeddings", "/embeddings"}:
            call_kind = "embedding"
            upstream_path = "/embeddings"
        else:
            self._json_response(404, {"error": {"message": "unsupported benchmark relay path"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 2 or length > 32 * 1024 * 1024:
                raise BenchmarkBudgetError("invalid benchmark relay request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise BenchmarkBudgetError("benchmark relay payload must be an object")
            reservation = self.server.ledger.reserve(
                payload, call_kind=call_kind, context_token=context_token
            )
        except (
            BenchmarkBudgetError,
            BenchmarkProtocolError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            self._json_response(429, {"error": {"message": str(exc), "type": "budget_error"}})
            return

        if call_kind == "chat_completion":
            payload["stream"] = False
            payload["usage"] = {"include": True}
            # OpenRouter's aggregate API accepts both spellings, but provider
            # capability metadata advertises max_tokens. With require_parameters,
            # sending both fields can make every otherwise eligible endpoint fail.
            payload.pop("max_completion_tokens", None)
            payload["max_tokens"] = reservation.requested_output_tokens
            provider = payload.get("provider", {})
            if not isinstance(provider, dict):
                provider = {}
            provider["require_parameters"] = bool(
                OPENROUTER_BENCHMARK_PROTOCOL["provider_require_parameters"]
            )
            if reservation.context.provider_sort is not None:
                provider["sort"] = reservation.context.provider_sort
            prompt_price = reservation.context.provider_max_prompt_price_per_million_usd
            completion_price = reservation.context.provider_max_completion_price_per_million_usd
            if prompt_price is not None and completion_price is not None:
                max_price = {
                    "prompt": float(prompt_price),
                    "completion": float(completion_price),
                }
                if reservation.context.provider_max_request_price_usd is not None:
                    max_price["request"] = float(reservation.context.provider_max_request_price_usd)
                provider["max_price"] = max_price
            payload["provider"] = provider
        if call_kind == "chat_completion" and reservation.requested_model == "openrouter/auto":
            plugins = [
                item
                for item in payload.get("plugins", [])
                if not (isinstance(item, dict) and item.get("id") == "auto-router")
            ]
            auto_plugin: dict[str, Any] = {
                "id": "auto-router",
                "allowed_models": list(reservation.context.allowed_models),
            }
            if reservation.context.auto_cost_tier is not None:
                auto_plugin["cost_tier"] = reservation.context.auto_cost_tier
            plugins.append(auto_plugin)
            payload["plugins"] = plugins
        effort = reservation.context.reasoning_effort
        if call_kind == "chat_completion" and effort is not None:
            payload["reasoning"] = {"effort": effort}
        request = urllib.request.Request(
            f"{self.server.upstream_base_url}{upstream_path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.server.api_key}",
                "Content-Type": "application/json",
                "Accept-Encoding": "identity",
                "X-OpenRouter-Metadata": "enabled",
                "X-Title": "OpenRoutiQ benchmark",
            },
            method="POST",
        )
        response_value: dict[str, Any] | None = None
        downstream_status: int
        downstream_value: dict[str, Any]
        settlement_error: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=600) as upstream:
                raw = upstream.read()
                status = upstream.status
            decoded = json.loads(raw.decode("utf-8"))
            response_value = decoded if isinstance(decoded, dict) else None
            if response_value is None:
                raise ValueError("upstream response was not a JSON object")
            # ChatOpenRouter currently expects this OpenAI compatibility field even
            # though some OpenRouter providers omit it.
            if call_kind == "chat_completion":
                response_value.setdefault("system_fingerprint", None)
            downstream_status = status
            downstream_value = response_value
        except urllib.error.HTTPError as exc:
            detail = exc.read(1024 * 1024)
            try:
                decoded = json.loads(detail.decode("utf-8"))
                response_value = decoded if isinstance(decoded, dict) else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                response_value = None
            downstream_status = exc.code
            downstream_value = response_value or {"error": {"message": f"upstream HTTP {exc.code}"}}
            settlement_error = f"upstream HTTP {exc.code}"
        except Exception as exc:
            downstream_status = 502
            downstream_value = {"error": {"message": "upstream relay failure"}}
            settlement_error = f"{type(exc).__name__}: {exc}"
        self.server.ledger.finish(
            reservation,
            status=downstream_status,
            response=response_value,
            error=settlement_error,
        )
        try:
            self._json_response(downstream_status, downstream_value)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # The upstream call has already been immutably settled. A framework
            # closing its loopback socket cannot turn that success into a second
            # provider event or charge.
            self.close_connection = True
            return


class OpenRouterBenchmarkGateway:
    """Loopback-only OpenRouter relay guarded by an exact call/cost reservation ledger."""

    def __init__(
        self,
        ledger: BenchmarkBudgetLedger,
        *,
        api_key: str,
        relay_key: str,
        upstream_base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        if not api_key:
            raise BenchmarkBudgetError("OpenRouter benchmark gateway requires a key")
        if not relay_key:
            raise BenchmarkBudgetError("OpenRouter benchmark gateway requires a relay key")
        self._server = _RelayServer(("127.0.0.1", 0), ledger, upstream_base_url, api_key, relay_key)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="openroutiq-benchmark-gateway",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        address: Any = self._server.server_address
        raw_host, port = address[0], address[1]
        host = raw_host.decode("ascii") if isinstance(raw_host, bytes) else str(raw_host)
        return f"http://{host}:{port}/v1"

    def start(self) -> "OpenRouterBenchmarkGateway":
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    def __enter__(self) -> "OpenRouterBenchmarkGateway":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.close()
