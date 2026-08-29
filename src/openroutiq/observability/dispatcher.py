"""Non-blocking, failure-isolated observability dispatch."""

from __future__ import annotations

import inspect
import math
import time
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Protocol, runtime_checkable

from openroutiq.observability.events import (
    EventFactory,
    ObservabilityEvent,
    ObservabilityError,
    ObservabilityPrivacy,
)


@runtime_checkable
class EventExporter(Protocol):
    """Minimal exporter contract; network work runs only on the dispatcher thread."""

    def export(self, event: ObservabilityEvent) -> None: ...


@dataclass(frozen=True)
class ObservabilityStats:
    accepted_events: int
    dropped_events: int
    rejected_events: int
    export_failures: int


@dataclass(frozen=True)
class _Barrier:
    completed: Event


_STOP = object()


class InMemoryExporter:
    """Thread-safe exporter for local validation and tests; it performs no I/O."""

    def __init__(self) -> None:
        self._events: list[ObservabilityEvent] = []
        self._lock = Lock()

    def export(self, event: ObservabilityEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[ObservabilityEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True

    def shutdown(self) -> None:
        return None


class Observability:
    """Fan out privacy-filtered events without blocking routing or provider responses."""

    def __init__(
        self,
        exporters: Iterable[EventExporter],
        *,
        privacy: ObservabilityPrivacy | None = None,
        max_queue_size: int = 2_048,
    ) -> None:
        parsed = tuple(exporters)
        if not parsed:
            raise ObservabilityError("observability requires at least one exporter")
        if any(not callable(getattr(exporter, "export", None)) for exporter in parsed):
            raise ObservabilityError("each observability exporter must define export(event)")
        if privacy is not None and not isinstance(privacy, ObservabilityPrivacy):
            raise ObservabilityError("privacy must be an ObservabilityPrivacy object or None")
        if (
            isinstance(max_queue_size, bool)
            or not isinstance(max_queue_size, int)
            or max_queue_size < 1
        ):
            raise ObservabilityError("max_queue_size must be an integer of at least one")
        self.exporters = parsed
        self.privacy = privacy or ObservabilityPrivacy()
        self._factory = EventFactory(self.privacy)
        self._queue: Queue[Any] = Queue(maxsize=max_queue_size)
        self._state_lock = Lock()
        self._lifecycle_lock = Lock()
        self._stats_lock = Lock()
        self._operation_lock = Lock()
        self._operations: dict[int, tuple[weakref.ReferenceType[Any], str, str]] = {}
        self._max_operations = max_queue_size
        self._accepting = True
        self._closed = False
        self._stop_requested = False
        self._accepted_events = 0
        self._dropped_events = 0
        self._rejected_events = 0
        self._export_failures = 0
        self._worker = Thread(
            target=self._run,
            name="openroutiq-observability",
            daemon=True,
        )
        self._worker.start()

    def __enter__(self) -> Observability:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.shutdown()

    def _increment(self, name: str) -> None:
        with self._stats_lock:
            setattr(self, name, getattr(self, name) + 1)

    @property
    def stats(self) -> ObservabilityStats:
        with self._stats_lock:
            return ObservabilityStats(
                accepted_events=self._accepted_events,
                dropped_events=self._dropped_events,
                rejected_events=self._rejected_events,
                export_failures=self._export_failures,
            )

    def _submit(self, event: ObservabilityEvent) -> bool:
        with self._state_lock:
            if not self._accepting:
                self._increment("_rejected_events")
                return False
            try:
                self._queue.put_nowait(event)
            except Full:
                self._increment("_dropped_events")
                return False
        self._increment("_accepted_events")
        return True

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                if isinstance(item, _Barrier):
                    item.completed.set()
                    continue
                for exporter in self.exporters:
                    try:
                        exporter.export(item)
                    except Exception:
                        # Export errors and their messages may contain transport data. Keep
                        # them out of application logs and expose only a numeric failure count.
                        self._increment("_export_failures")
            finally:
                self._queue.task_done()

    def _remember_operation(self, decision: Any, trace_id: str, span_id: str) -> None:
        key = id(decision)

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            with self._operation_lock:
                current = self._operations.get(key)
                if current is not None and current[0] is reference:
                    self._operations.pop(key, None)

        try:
            reference = weakref.ref(decision, remove)
        except TypeError:
            return
        with self._operation_lock:
            if key not in self._operations and len(self._operations) >= self._max_operations:
                self._operations.pop(next(iter(self._operations)))
            self._operations[key] = (reference, trace_id, span_id)

    def _take_operation(self, decision: Any) -> tuple[str, str | None]:
        with self._operation_lock:
            current = self._operations.pop(id(decision), None)
        if current is not None and current[0]() is decision:
            return current[1], current[2]
        return self._factory.new_trace_id(), None

    def record_route(self, decision: Any, *, duration_ms: float | None = None) -> bool:
        """Queue a completed route decision; request content is neither accepted nor inspected."""

        try:
            trace_id = self._factory.new_trace_id()
            span_id = self._factory.new_span_id()
            event = self._factory.route(
                decision,
                duration_ms=duration_ms,
                trace_id=trace_id,
                span_id=span_id,
            )
        except Exception:
            self._increment("_rejected_events")
            return False
        accepted = self._submit(event)
        if accepted:
            self._remember_operation(decision, trace_id, span_id)
        return accepted

    def record_execution(
        self,
        decision: Any,
        *,
        duration_ms: float,
        success: bool,
        actual_cost_usd: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        failure_type: Any = None,
        streaming: bool = False,
    ) -> bool:
        """Queue numeric execution telemetry without accepting a response or exception object."""

        try:
            if not isinstance(success, bool) or not isinstance(streaming, bool):
                raise ObservabilityError("success and streaming must be booleans")
            normalized_failure = getattr(failure_type, "value", failure_type)
            if normalized_failure is not None and not isinstance(normalized_failure, str):
                normalized_failure = None
            trace_id, parent_span_id = self._take_operation(decision)
            event = self._factory.execution(
                decision,
                duration_ms=duration_ms,
                success=success,
                actual_cost_usd=actual_cost_usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                failure_type=normalized_failure,
                streaming=streaming,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
            )
        except Exception:
            self._increment("_rejected_events")
            return False
        return self._submit(event)

    def record_evaluation(
        self,
        *,
        model_id: str,
        provider: str | None = None,
        task: str | None = None,
        quality_score: float,
        latency_ms: float | None = None,
        actual_cost_usd: float | None = None,
        success: bool | None = None,
        failure_type: Any = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> bool:
        """Queue an evaluated outcome without accepting the evaluated request or response."""

        try:
            normalized_failure = getattr(failure_type, "value", failure_type)
            if normalized_failure is not None and not isinstance(normalized_failure, str):
                normalized_failure = None
            event = self._factory.evaluation(
                model_id=model_id,
                provider=provider,
                task=task,
                quality_score=quality_score,
                latency_ms=latency_ms,
                actual_cost_usd=actual_cost_usd,
                success=success,
                failure_type=normalized_failure,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception:
            self._increment("_rejected_events")
            return False
        return self._submit(event)

    @staticmethod
    def _remaining_millis(deadline: float) -> int:
        return max(0, round((deadline - time.monotonic()) * 1_000))

    @staticmethod
    def _call_optional(exporter: Any, name: str, timeout_millis: int) -> bool:
        method = getattr(exporter, name, None)
        if not callable(method):
            return True
        try:
            supports_timeout = "timeout_millis" in inspect.signature(method).parameters
        except (TypeError, ValueError):
            supports_timeout = False
        result = method(timeout_millis=timeout_millis) if supports_timeout else method()
        return result is not False

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        """Wait for events queued before this call and flush exporter-owned buffers."""

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ObservabilityError("timeout_seconds must be a positive finite number")
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    return True
                if not self._accepting or not self._worker.is_alive():
                    return False
            deadline = time.monotonic() + float(timeout_seconds)
            barrier = _Barrier(Event())
            try:
                self._queue.put(barrier, timeout=max(0.0, deadline - time.monotonic()))
            except Full:
                return False
            if not barrier.completed.wait(timeout=max(0.0, deadline - time.monotonic())):
                return False
            complete = True
            for exporter in self.exporters:
                try:
                    complete = (
                        self._call_optional(
                            exporter,
                            "force_flush",
                            self._remaining_millis(deadline),
                        )
                        and complete
                    )
                except Exception:
                    self._increment("_export_failures")
                    complete = False
            return complete and time.monotonic() <= deadline

    def shutdown(self, timeout_seconds: float = 5.0) -> bool:
        """Stop accepting events, drain the queue, and close owned exporter resources."""

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ObservabilityError("timeout_seconds must be a positive finite number")
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    return True
                self._accepting = False
            deadline = time.monotonic() + float(timeout_seconds)
            complete = True
            if self._worker.is_alive() and not self._stop_requested:
                barrier = _Barrier(Event())
                try:
                    self._queue.put(barrier, timeout=max(0.0, deadline - time.monotonic()))
                except Full:
                    return False
                if not barrier.completed.wait(timeout=max(0.0, deadline - time.monotonic())):
                    return False
                for exporter in self.exporters:
                    try:
                        complete = (
                            self._call_optional(
                                exporter,
                                "force_flush",
                                self._remaining_millis(deadline),
                            )
                            and complete
                        )
                    except Exception:
                        self._increment("_export_failures")
                        complete = False
                try:
                    self._queue.put(_STOP, timeout=max(0.0, deadline - time.monotonic()))
                except Full:
                    return False
                self._stop_requested = True
            self._worker.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._worker.is_alive():
                return False
            for exporter in self.exporters:
                try:
                    complete = (
                        self._call_optional(
                            exporter,
                            "shutdown",
                            self._remaining_millis(deadline),
                        )
                        and complete
                    )
                except Exception:
                    self._increment("_export_failures")
                    complete = False
            with self._state_lock:
                self._closed = True
            return complete and time.monotonic() <= deadline
