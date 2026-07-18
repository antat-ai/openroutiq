from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any


class FailureType(str, Enum):
    """Stable, provider-neutral failure categories used by routing telemetry."""

    ROUTING_FAILURE = "ROUTING_FAILURE"
    MODEL_FAILURE = "MODEL_FAILURE"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    TOOL_FAILURE = "TOOL_FAILURE"
    PROTOCOL_FAILURE = "PROTOCOL_FAILURE"
    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


_ALIASES = {
    "routing": FailureType.ROUTING_FAILURE,
    "model": FailureType.MODEL_FAILURE,
    "provider": FailureType.PROVIDER_FAILURE,
    "capability": FailureType.CAPABILITY_MISMATCH,
    "capability_mismatch": FailureType.CAPABILITY_MISMATCH,
    "tool": FailureType.TOOL_FAILURE,
    "protocol": FailureType.PROTOCOL_FAILURE,
    "timeout": FailureType.TIMEOUT,
    "rate_limit": FailureType.RATE_LIMIT,
    "ratelimit": FailureType.RATE_LIMIT,
    "unknown": FailureType.UNKNOWN_FAILURE,
}


def normalize_failure_type(value: FailureType | str | None) -> FailureType | None:
    """Normalize a public failure label without silently inventing a category."""

    if value is None:
        return None
    if isinstance(value, FailureType):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("failure_type must be a non-empty string or None")
    normalized = value.strip()
    try:
        return FailureType(normalized.upper())
    except ValueError:
        alias = _ALIASES.get(normalized.casefold().replace("-", "_"))
        if alias is None:
            allowed = ", ".join(item.value for item in FailureType)
            raise ValueError(f"failure_type must be one of: {allowed}") from None
        return alias


def classify_failure(
    error: BaseException | str | None, *, status_code: int | None = None
) -> FailureType:
    """Classify provider/framework errors without retaining their sensitive message text."""

    if status_code == 429:
        return FailureType.RATE_LIMIT
    if status_code in {408, 504}:
        return FailureType.TIMEOUT
    if status_code is not None and 400 <= status_code < 500:
        return FailureType.PROTOCOL_FAILURE
    if status_code is not None and status_code >= 500:
        return FailureType.PROVIDER_FAILURE
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return FailureType.TIMEOUT

    name = "" if error is None else type(error).__name__.casefold()
    text = "" if error is None else str(error).casefold()
    combined = f"{name} {text}"
    if "rate limit" in combined or "ratelimit" in combined or "too many requests" in combined:
        return FailureType.RATE_LIMIT
    if "timeout" in combined or "timed out" in combined or "deadline" in combined:
        return FailureType.TIMEOUT
    if "tool" in combined and any(
        marker in combined for marker in ("error", "fail", "invalid", "mismatch")
    ):
        return FailureType.TOOL_FAILURE
    if "capability" in combined or "unsupported parameter" in combined:
        return FailureType.CAPABILITY_MISMATCH
    if any(marker in name for marker in ("validation", "badrequest", "protocol", "parse")):
        return FailureType.PROTOCOL_FAILURE
    if any(marker in name for marker in ("connection", "provider", "serviceunavailable")):
        return FailureType.PROVIDER_FAILURE
    if any(marker in name for marker in ("contentfilter", "model", "refusal")):
        return FailureType.MODEL_FAILURE
    return FailureType.UNKNOWN_FAILURE


def exception_status_code(error: BaseException) -> int | None:
    """Read common SDK status fields while keeping provider SDKs optional."""

    candidates: tuple[Any, ...] = (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None
