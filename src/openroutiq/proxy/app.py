from __future__ import annotations

import asyncio
from contextvars import ContextVar
import inspect
import json
import logging
import math
import os
import re
import secrets
import time
from collections.abc import AsyncIterable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from openroutiq.adaptive.registry import AdaptiveRouter, AdaptiveStoreError
from openroutiq.router.failures import FailureType, classify_failure, exception_status_code
from openroutiq.router.core import OpenRoutiQError, NoEligibleModelError, Router


AUTO_MODELS = frozenset({"auto", "openroutiq/auto"})
_STREAM_END = object()
_LOGGER = logging.getLogger("openroutiq.proxy")
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_REQUEST_ID: ContextVar[str | None] = ContextVar("openroutiq_request_id", default=None)
_MAX_ROUTE_LABEL_CHARS = 512
_OUTPUT_TOKEN_FIELDS = frozenset({"max_output_tokens", "max_completion_tokens", "max_tokens"})
# This is intentionally an allowlist rather than a LiteLLM-derived denylist. New LiteLLM
# keyword arguments may control credentials or transports and must remain rejected until reviewed.
_PROVIDER_REQUEST_KEYS = {
    "chat_completions": frozenset(
        {
            "audio",
            "effort",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "logprobs",
            "max_completion_tokens",
            "max_tokens",
            "messages",
            "metadata",
            "min_p",
            "modalities",
            "model",
            "n",
            "parallel_tool_calls",
            "prediction",
            "presence_penalty",
            "prompt_cache_key",
            "reasoning_effort",
            "repetition_penalty",
            "response_format",
            "safety_identifier",
            "seed",
            "service_tier",
            "stop",
            "store",
            "stream",
            "stream_options",
            "temperature",
            "thinking",
            "tool_choice",
            "tools",
            "top_k",
            "top_logprobs",
            "top_p",
            "user",
            "verbosity",
            "web_search_options",
        }
    ),
    "responses": frozenset(
        {
            "background",
            "conversation",
            "include",
            "input",
            "instructions",
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "modalities",
            "model",
            "parallel_tool_calls",
            "previous_response_id",
            "prompt",
            "prompt_cache_key",
            "reasoning",
            "reasoning_effort",
            "response_format",
            "safety_identifier",
            "service_tier",
            "store",
            "stream",
            "stream_options",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
        }
    ),
    "messages": frozenset(
        {
            "container",
            "context_management",
            "effort",
            "inference_geo",
            "max_tokens",
            "messages",
            "metadata",
            "model",
            "output_config",
            "parallel_tool_calls",
            "service_tier",
            "stop_sequences",
            "stream",
            "system",
            "temperature",
            "thinking",
            "tool_choice",
            "tools",
            "top_k",
            "top_p",
        }
    ),
}
_PROVIDER_OUTPUT_TOKEN_FIELD = {
    "chat_completions": "max_tokens",
    "responses": "max_output_tokens",
    "messages": "max_tokens",
}
ROUTING_KEYS = frozenset(
    {
        "task",
        "weights",
        "constraints",
        "risk_policy",
        "input_tokens",
        "expected_output_tokens",
        "high_risk",
        "soft_budget",
        "strategy",
        "complexity",
        "reasoning_effort",
        "pinned_model",
        "explore",
    }
)


@dataclass(frozen=True)
class ProxyLimits:
    """In-process safety limits; an internet-facing deployment still needs edge limits."""

    max_request_bytes: int = 4 * 1024 * 1024
    max_concurrency: int = 128
    max_declared_tokens: int = 10_000_000
    queue_timeout_seconds: float = 5.0
    routing_timeout_seconds: float = 30.0
    provider_timeout_seconds: float = 600.0
    stream_idle_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name, integer_value in (
            ("max_request_bytes", self.max_request_bytes),
            ("max_concurrency", self.max_concurrency),
            ("max_declared_tokens", self.max_declared_tokens),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value < 1
            ):
                raise OpenRoutiQError(f"{name} must be an integer >= 1")
        for name, seconds_value in (
            ("queue_timeout_seconds", self.queue_timeout_seconds),
            ("routing_timeout_seconds", self.routing_timeout_seconds),
            ("provider_timeout_seconds", self.provider_timeout_seconds),
            ("stream_idle_timeout_seconds", self.stream_idle_timeout_seconds),
        ):
            if (
                isinstance(seconds_value, bool)
                or not isinstance(seconds_value, (int, float))
                or not float(seconds_value) > 0
            ):
                raise OpenRoutiQError(f"{name} must be a number > 0")


class _RequestBodyTooLarge(Exception):
    pass


class _RequestBodyLimitMiddleware:
    def __init__(self, app: Any, *, maximum: int) -> None:
        self.app = app
        self.maximum = maximum

    async def __call__(self, scope: Mapping[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {bytes(key).lower(): bytes(value) for key, value in scope.get("headers", ())}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                declared = int(raw_length)
            except ValueError:
                declared = -1
            if declared < 0:
                await _asgi_error(send, 400, "invalid content-length", FailureType.PROTOCOL_FAILURE)
                return
            if declared > self.maximum:
                await _asgi_error(
                    send,
                    413,
                    "request body exceeds configured limit",
                    FailureType.PROTOCOL_FAILURE,
                )
                return
        consumed = 0

        async def limited_receive() -> Any:
            nonlocal consumed
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.maximum:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await _asgi_error(
                send,
                413,
                "request body exceeds configured limit",
                FailureType.PROTOCOL_FAILURE,
            )


async def _asgi_error(
    send: Any,
    status: int,
    detail: str,
    failure_type: FailureType,
) -> None:
    body = json.dumps({"detail": detail}, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"x-openroutiq-error-type", failure_type.value.encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _routing_options(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("openroutiq", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise OpenRoutiQError("openroutiq must be an object")
    unknown = set(raw) - ROUTING_KEYS
    if unknown:
        raise OpenRoutiQError(f"unknown openroutiq options: {', '.join(sorted(unknown))}")
    for key in ("task", "pinned_model", "strategy", "reasoning_effort"):
        value = raw.get(key)
        if isinstance(value, str) and len(value) > _MAX_ROUTE_LABEL_CHARS:
            raise OpenRoutiQError(
                f"openroutiq.{key} must be at most {_MAX_ROUTE_LABEL_CHARS} characters"
            )
    return dict(raw)


def _validate_declared_tokens(payload: Mapping[str, Any], *, maximum: int) -> None:
    output_fields = ("max_output_tokens", "max_completion_tokens", "max_tokens")
    supplied_outputs: dict[str, int] = {}
    for field in output_fields:
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise OpenRoutiQError(f"{field} must be an integer >= 1")
        if value > maximum:
            raise OpenRoutiQError(f"{field} exceeds the configured declared-token limit")
        supplied_outputs[field] = value
    if len(set(supplied_outputs.values())) > 1:
        raise OpenRoutiQError("output-token cap fields must contain one identical value")

    raw_options = payload.get("openroutiq")
    if not isinstance(raw_options, Mapping):
        return
    for field, minimum in (("input_tokens", 0), ("expected_output_tokens", 1)):
        value = raw_options.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise OpenRoutiQError(f"openroutiq.{field} must be an integer >= {minimum}")
        if value > maximum:
            raise OpenRoutiQError(f"openroutiq.{field} exceeds the configured declared-token limit")
    routed_output = raw_options.get("expected_output_tokens")
    if supplied_outputs and routed_output is not None:
        provider_output = next(iter(supplied_outputs.values()))
        if routed_output != provider_output:
            raise OpenRoutiQError(
                "openroutiq.expected_output_tokens must match the provider output-token cap"
            )


def _route_payload(
    router: Router | AdaptiveRouter,
    payload: Mapping[str, Any],
    *,
    request_field: str,
    include_system: bool = False,
):
    request = payload.get(request_field)
    if request_field == "messages":
        if not isinstance(request, list) or not request:
            raise OpenRoutiQError("messages must be a non-empty list")
        routing_request: str | list[Any] = list(request)
    elif isinstance(request, str) or isinstance(request, list) and request:
        routing_request = request
    else:
        raise OpenRoutiQError("input must be text or a non-empty input list")
    if include_system and payload.get("system") is not None:
        if not isinstance(routing_request, list):
            raise OpenRoutiQError("system messages require a message-list request")
        routing_request = [
            {"role": "system", "content": payload["system"]},
            *routing_request,
        ]
    model = payload.get("model", "auto")
    if not isinstance(model, str) or not model.strip():
        raise OpenRoutiQError("model must be a non-empty string")
    if len(model) > _MAX_ROUTE_LABEL_CHARS:
        raise OpenRoutiQError(f"model must be at most {_MAX_ROUTE_LABEL_CHARS} characters")
    options = _routing_options(payload)
    declared_input_tokens = options.pop("input_tokens", None)
    if "explore" in options and not isinstance(router, AdaptiveRouter):
        raise OpenRoutiQError("openroutiq.explore requires AdaptiveRouter")
    if model not in AUTO_MODELS:
        matches = [
            profile.id for profile in router.profiles if model in {profile.id, profile.model}
        ]
        if len(matches) != 1:
            raise OpenRoutiQError(
                "explicit model must uniquely match a catalog id or provider model"
            )
        configured_pin = options.get("pinned_model")
        if configured_pin is not None and configured_pin != matches[0]:
            raise OpenRoutiQError("model conflicts with openroutiq.pinned_model")
        options["pinned_model"] = matches[0]
    expected = payload.get(
        "max_output_tokens",
        payload.get("max_completion_tokens", payload.get("max_tokens")),
    )
    if expected is not None and "expected_output_tokens" not in options:
        options["expected_output_tokens"] = expected
    response_format = payload.get("response_format")
    text_config = payload.get("text")
    if response_format is None and isinstance(text_config, Mapping):
        response_format = text_config.get("format")
    output_config = payload.get("output_config")
    if response_format is None and isinstance(output_config, Mapping):
        response_format = output_config.get("format")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise OpenRoutiQError("stream must be a boolean")
    decision = router.route(
        routing_request,
        tools=payload.get("tools"),
        tool_choice=payload.get("tool_choice"),
        parallel_tool_calls=payload.get("parallel_tool_calls"),
        response_format=response_format,
        output_modalities=payload.get("modalities"),
        stream=stream,
        minimum_input_tokens=declared_input_tokens,
        **options,
    )
    return routing_request, decision


def _provider_kwargs(
    payload: Mapping[str, Any],
    provider_model: str,
    *,
    provider_api: str,
    expected_output_tokens: int | None = None,
) -> dict[str, Any]:
    allowed = _PROVIDER_REQUEST_KEYS.get(provider_api)
    output_field = _PROVIDER_OUTPUT_TOKEN_FIELD.get(provider_api)
    if allowed is None or output_field is None:
        raise OpenRoutiQError(f"unsupported provider API: {provider_api}")
    unsupported = sorted(
        str(key)
        for key in payload
        if not isinstance(key, str) or key not in allowed and key != "openroutiq"
    )
    if unsupported:
        raise OpenRoutiQError(
            "unsupported provider request fields; provider transport controls are "
            "server-managed: " + ", ".join(unsupported)
        )
    excluded = {"openroutiq"}
    result = {key: value for key, value in payload.items() if key not in excluded}
    result["model"] = provider_model
    if expected_output_tokens is not None and not (_OUTPUT_TOKEN_FIELDS & set(result)):
        if (
            isinstance(expected_output_tokens, bool)
            or not isinstance(expected_output_tokens, int)
            or expected_output_tokens < 1
        ):
            raise OpenRoutiQError("routed output-token cap must be an integer >= 1")
        result[output_field] = expected_output_tokens
    return result


def _headers(decision) -> dict[str, str]:
    return {
        "x-openroutiq-model": decision.selected.model_id,
        "x-openroutiq-provider": decision.selected.provider,
        "x-openroutiq-score": f"{decision.selected.total_score:.4f}",
        "x-openroutiq-review-required": str(decision.review_required).lower(),
    }


def _reported_cost(value: Any) -> float | None:
    body = _jsonable(value)
    if not isinstance(body, Mapping):
        return None
    usage = body.get("usage")
    candidates: list[Any] = [body.get("cost")]
    if isinstance(usage, Mapping):
        candidates.extend(
            [
                usage.get("cost"),
                usage.get("total_cost"),
                usage.get("estimated_cost"),
            ]
        )
    for candidate in candidates:
        if (
            isinstance(candidate, (int, float))
            and not isinstance(candidate, bool)
            and candidate >= 0
            and math.isfinite(float(candidate))
        ):
            return float(candidate)
    return None


def _reported_tokens(value: Any) -> tuple[int | None, int | None]:
    body = _jsonable(value)
    if not isinstance(body, Mapping) or not isinstance(body.get("usage"), Mapping):
        return None, None
    usage = body["usage"]

    def count(*names: str) -> int | None:
        for name in names:
            candidate = usage.get(name)
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
                return candidate
        return None

    return (
        count("input_tokens", "prompt_tokens"),
        count("output_tokens", "completion_tokens"),
    )


def _observe_execution(
    router: Router | AdaptiveRouter,
    decision: Any,
    *,
    started: float,
    success: bool,
    result: Any = None,
    failure_type: FailureType | None = None,
    streaming: bool = False,
) -> None:
    adaptive = isinstance(router, AdaptiveRouter)
    observability = getattr(router, "observability", None)
    if not adaptive and observability is None:
        return
    duration_ms = (time.perf_counter() - started) * 1000
    try:
        input_tokens, output_tokens = _reported_tokens(result)
        actual_cost_usd = _reported_cost(result)
    except Exception:
        # Provider SDK serialization is untrusted telemetry input. Ignore it completely.
        input_tokens, output_tokens, actual_cost_usd = None, None, None
    if adaptive:
        try:
            assert isinstance(router, AdaptiveRouter)
            router.observe_execution(
                decision.selected.model_id,
                task=decision.task,
                latency_ms=duration_ms,
                actual_cost_usd=actual_cost_usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=success,
                failure_type=failure_type,
            )
        except Exception:
            # Learning telemetry must never turn a provider result into a proxy failure.
            pass
    if observability is not None:
        try:
            observability.record_execution(
                decision,
                duration_ms=duration_ms,
                actual_cost_usd=actual_cost_usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=success,
                failure_type=failure_type,
                streaming=streaming,
            )
        except Exception:
            # Export is a best-effort side channel and cannot affect the response.
            pass


async def _invoke(executor: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    if inspect.iscoroutinefunction(executor):
        return await executor(**dict(kwargs))
    result = await asyncio.to_thread(executor, **dict(kwargs))
    return await result if inspect.isawaitable(result) else result


def _sse_chunk(chunk: Any, *, typed: bool) -> bytes:
    value = _jsonable(chunk)
    payload = json.dumps(value, separators=(",", ":"))
    if typed and isinstance(value, Mapping) and isinstance(value.get("type"), str):
        return f"event: {value['type']}\ndata: {payload}\n\n".encode("utf-8")
    return f"data: {payload}\n\n".encode("utf-8")


async def _stream(
    result: Any,
    *,
    typed: bool,
    on_complete: Callable[[bool, FailureType | None], Any] | None = None,
    idle_timeout_seconds: float | None = None,
):
    async def complete(success: bool, failure_type: FailureType | None = None) -> None:
        if on_complete is None:
            return
        callback_result = on_complete(success, failure_type)
        if inspect.isawaitable(callback_result):
            await callback_result

    async def close_result() -> None:
        async_close = getattr(result, "aclose", None)
        if callable(async_close):
            try:
                await async_close()
            except Exception:
                return
            return
        close = getattr(result, "close", None)
        if callable(close):
            try:
                await asyncio.to_thread(close)
            except Exception:
                return

    def next_or_end(iterator: Any) -> Any:
        try:
            return next(iterator)
        except StopIteration:
            return _STREAM_END

    try:
        if isinstance(result, AsyncIterable):
            async_iterator = result.__aiter__()
            while True:
                try:
                    next_chunk = async_iterator.__anext__()
                    chunk = (
                        await next_chunk
                        if idle_timeout_seconds is None
                        else await asyncio.wait_for(next_chunk, idle_timeout_seconds)
                    )
                except StopAsyncIteration:
                    break
                yield _sse_chunk(chunk, typed=typed)
        elif isinstance(result, Iterable) and not isinstance(
            result, (str, bytes, bytearray, Mapping)
        ):
            sync_iterator = iter(result)
            while True:
                pending_chunk = asyncio.to_thread(next_or_end, sync_iterator)
                chunk = (
                    await pending_chunk
                    if idle_timeout_seconds is None
                    else await asyncio.wait_for(pending_chunk, idle_timeout_seconds)
                )
                if chunk is _STREAM_END:
                    break
                yield _sse_chunk(chunk, typed=typed)
        else:
            yield _sse_chunk(result, typed=typed)
        if not typed:
            yield b"data: [DONE]\n\n"
    except BaseException as exc:
        await complete(False, classify_failure(exc))
        raise
    else:
        await complete(True)
    finally:
        await close_result()


def create_app(
    catalog: str | Path | Router | AdaptiveRouter,
    *,
    executor: Callable[..., Any] | None = None,
    responses_executor: Callable[..., Any] | None = None,
    messages_executor: Callable[..., Any] | None = None,
    api_key_env: str = "OPENROUTIQ_PROXY_API_KEY",
    limits: ProxyLimits | None = None,
):
    """Create an OpenAI-compatible FastAPI app.

    The default executor is LiteLLM's async completion function. Provider keys are
    resolved by LiteLLM from the environment/config and are never persisted here.
    """
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise OpenRoutiQError("api_key_env must be a non-empty environment variable name")
    configured_proxy_key = os.environ.get(api_key_env)
    if configured_proxy_key == "":
        raise OpenRoutiQError(f"proxy bearer token in {api_key_env} must not be empty")
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise OpenRoutiQError("the proxy requires 'pip install openroutiq[proxy]'") from exc

    route_engine = (
        catalog if isinstance(catalog, (Router, AdaptiveRouter)) else Router.from_file(catalog)
    )
    if executor is None:
        try:
            from litellm import acompletion, aresponses, anthropic_messages
        except ImportError as exc:
            raise OpenRoutiQError("the proxy requires 'pip install openroutiq[proxy]'") from exc
        executor = acompletion
        responses_executor = responses_executor or aresponses
        messages_executor = messages_executor or anthropic_messages
    elif responses_executor is None:
        responses_executor = executor
    if messages_executor is None:
        messages_executor = executor

    app = FastAPI(
        title="OpenRoutiQ Proxy",
        version="1.0",
        docs_url=None,
        redoc_url=None,
    )
    proxy_limits = limits or ProxyLimits()
    semaphore = asyncio.Semaphore(proxy_limits.max_concurrency)
    app.add_middleware(_RequestBodyLimitMiddleware, maximum=proxy_limits.max_request_bytes)

    @app.exception_handler(_RequestBodyTooLarge)
    async def request_body_too_large(_request: Any, _exc: _RequestBodyTooLarge):
        return JSONResponse(
            status_code=413,
            content={"detail": "request body exceeds configured limit"},
            headers={"x-openroutiq-error-type": FailureType.PROTOCOL_FAILURE.value},
        )

    @app.middleware("http")
    async def request_identity(request: Any, call_next: Callable[..., Any]):
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if _SAFE_REQUEST_ID.fullmatch(supplied) else uuid4().hex
        token = _REQUEST_ID.set(request_id)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            _REQUEST_ID.reset(token)

    def authorize(authorization: str | None) -> None:
        expected = os.environ.get(api_key_env)
        if expected is None:
            return
        if not expected:
            raise HTTPException(status_code=503, detail="proxy authentication is misconfigured")
        supplied = "" if authorization is None else authorization
        if not secrets.compare_digest(supplied, f"Bearer {expected}"):
            raise HTTPException(status_code=401, detail="invalid proxy credentials")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        try:
            return {"status": "ready", "models": len(route_engine.profiles)}
        except AdaptiveStoreError as exc:
            raise HTTPException(status_code=503, detail="adaptive registry unavailable") from exc

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)):
        authorize(authorization)
        data = [
            {
                "id": profile.id,
                "object": "model",
                "owned_by": profile.provider,
                "provider_model": profile.model,
                "reasoning_level": profile.reasoning_level,
            }
            for profile in route_engine.profiles
            if profile.available
        ]
        data.insert(0, {"id": "auto", "object": "model", "owned_by": "openroutiq"})
        return {"object": "list", "data": data}

    async def completion(
        payload: Mapping[str, Any],
        authorization: str | None,
        *,
        provider_api: str,
        request_field: str,
        selected_executor: Callable[..., Any],
        include_system: bool = False,
        typed_stream: bool = False,
    ):
        authorize(authorization)
        decision = None
        execution_phase = "routing"
        started = time.perf_counter()
        slot_acquired = False
        slot_handed_to_stream = False
        slot_release_deferred = False
        result: Any = _STREAM_END
        provider_task: asyncio.Task[Any] | None = None

        def defer_slot_release(task: asyncio.Task[Any]) -> None:
            nonlocal slot_release_deferred
            if slot_release_deferred:
                return
            slot_release_deferred = True

            def release_when_finished(completed: asyncio.Task[Any]) -> None:
                try:
                    completed.result()
                except BaseException:
                    # The request already reported the timeout or disconnect. Consume
                    # a late exception solely to avoid an unobserved-task leak.
                    pass
                semaphore.release()

            task.add_done_callback(release_when_finished)

        try:
            # Reject client attempts to replace server-owned credentials, transports,
            # destinations, retry behavior, or HTTP clients before doing routing work.
            _provider_kwargs(
                payload,
                "validation-only",
                provider_api=provider_api,
            )
            _validate_declared_tokens(payload, maximum=proxy_limits.max_declared_tokens)
            try:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=proxy_limits.queue_timeout_seconds,
                )
                slot_acquired = True
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="proxy concurrency limit reached",
                    headers={"x-openroutiq-error-type": FailureType.PROVIDER_FAILURE.value},
                ) from exc
            # Routing can include local embedding inference and SQLite reads. Keep it
            # off the event loop and inside the same bounded concurrency envelope as
            # provider execution so oversized bursts cannot create unbounded workers.
            routing_task = asyncio.create_task(
                asyncio.to_thread(
                    _route_payload,
                    route_engine,
                    payload,
                    request_field=request_field,
                    include_system=include_system,
                )
            )
            try:
                _, decision = await asyncio.wait_for(
                    asyncio.shield(routing_task),
                    timeout=proxy_limits.routing_timeout_seconds,
                )
            except (TimeoutError, asyncio.CancelledError):
                # Local embedding/tokenizer/database code runs in a thread and cannot
                # be killed safely. Keep its slot until it exits.
                defer_slot_release(routing_task)
                raise
            kwargs = _provider_kwargs(
                payload,
                decision.selected.provider_model,
                provider_api=provider_api,
                expected_output_tokens=decision.expected_output_tokens,
            )
            execution_phase = "provider"
            provider_task = asyncio.create_task(_invoke(selected_executor, kwargs))
            if inspect.iscoroutinefunction(selected_executor):
                result = await asyncio.wait_for(
                    provider_task,
                    timeout=proxy_limits.provider_timeout_seconds,
                )
            else:
                # Cancelling asyncio.to_thread() does not stop the worker thread.
                # Shield it and keep the concurrency slot until the real work exits.
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(provider_task),
                        timeout=proxy_limits.provider_timeout_seconds,
                    )
                except (TimeoutError, asyncio.CancelledError):
                    defer_slot_release(provider_task)
                    raise
        except NoEligibleModelError as exc:
            failure_type = (
                FailureType.CAPABILITY_MISMATCH
                if exc.excluded
                and all(
                    item.failure_type == FailureType.CAPABILITY_MISMATCH for item in exc.excluded
                )
                else FailureType.ROUTING_FAILURE
            )
            raise HTTPException(
                status_code=422,
                detail=str(exc),
                headers={"x-openroutiq-error-type": failure_type.value},
            ) from exc
        except AdaptiveStoreError as exc:
            raise HTTPException(
                status_code=503,
                detail="adaptive registry unavailable",
                headers={"x-openroutiq-error-type": FailureType.ROUTING_FAILURE.value},
            ) from exc
        except OpenRoutiQError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
                headers={"x-openroutiq-error-type": FailureType.PROTOCOL_FAILURE.value},
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            provider_status = exception_status_code(exc)
            failure_type = classify_failure(exc, status_code=provider_status)
            if decision is not None:
                await asyncio.to_thread(
                    _observe_execution,
                    route_engine,
                    decision,
                    started=started,
                    success=False,
                    failure_type=failure_type,
                    streaming=bool(payload.get("stream")),
                )
            _LOGGER.warning(
                "%s execution failed",
                execution_phase,
                extra={
                    "request_id": _REQUEST_ID.get(),
                    "execution_phase": execution_phase,
                    "failure_type": failure_type.value,
                    "provider": None if decision is None else decision.selected.provider,
                    "model_id": None if decision is None else decision.selected.model_id,
                },
            )
            raise HTTPException(
                status_code=(
                    504
                    if failure_type == FailureType.TIMEOUT
                    else 429
                    if failure_type == FailureType.RATE_LIMIT
                    else 502
                ),
                detail=f"{execution_phase} execution failed: {type(exc).__name__}",
                headers={"x-openroutiq-error-type": failure_type.value},
            ) from exc
        finally:
            if (
                slot_acquired
                and not slot_handed_to_stream
                and not slot_release_deferred
                and result is _STREAM_END
            ):
                semaphore.release()
        response_headers = _headers(decision)
        if payload.get("stream"):
            slot_handed_to_stream = True
            completed = False

            async def observe_stream(
                success: bool,
                failure_type: FailureType | None,
            ) -> None:
                nonlocal completed
                if completed:
                    return
                completed = True
                try:
                    await asyncio.to_thread(
                        _observe_execution,
                        route_engine,
                        decision,
                        started=started,
                        success=success,
                        result=result if success else None,
                        failure_type=failure_type,
                        streaming=True,
                    )
                finally:
                    semaphore.release()

            return StreamingResponse(
                _stream(
                    result,
                    typed=typed_stream,
                    on_complete=observe_stream,
                    idle_timeout_seconds=proxy_limits.stream_idle_timeout_seconds,
                ),
                media_type="text/event-stream",
                headers=response_headers,
            )
        try:
            await asyncio.to_thread(
                _observe_execution,
                route_engine,
                decision,
                started=started,
                success=True,
                result=result,
                streaming=False,
            )
        finally:
            semaphore.release()
        body = _jsonable(result)
        if isinstance(body, Mapping):
            body = {**body, "openroutiq": decision.to_dict()}
        return JSONResponse(content=body, headers=response_headers)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ):
        return await completion(
            payload,
            authorization,
            provider_api="chat_completions",
            request_field="messages",
            selected_executor=executor,
        )

    @app.post("/v1/responses")
    async def responses(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ):
        return await completion(
            payload,
            authorization,
            provider_api="responses",
            request_field="input",
            selected_executor=responses_executor,
            typed_stream=True,
        )

    @app.post("/v1/messages")
    async def messages(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ):
        return await completion(
            payload,
            authorization,
            provider_api="messages",
            request_field="messages",
            selected_executor=messages_executor,
            include_system=True,
            typed_stream=True,
        )

    @app.post("/chat/completions", include_in_schema=False)
    async def chat_completions_legacy(
        payload: dict[str, Any],
        authorization: str | None = Header(default=None),
    ):
        return await completion(
            payload,
            authorization,
            provider_api="chat_completions",
            request_field="messages",
            selected_executor=executor,
        )

    return app
