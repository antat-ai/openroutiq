"""Lifecycle and output helpers shared by runnable examples."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from openroutiq import Observability, RouteDecision, Router

from examples.common.observability import build_observability_from_environment


LOGGER = logging.getLogger("openroutiq.examples")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configured_catalog() -> Path:
    """Resolve a caller-controlled catalog path without changing process state."""

    configured = os.environ.get("OPENROUTIQ_CATALOG")
    path = Path(configured).expanduser() if configured else PROJECT_ROOT / "models.example.json"
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"OpenRoutiQ catalog does not exist: {resolved}")
    return resolved


@contextmanager
def managed_router(
    *,
    observability: Observability | None = None,
    review_margin: float = 3.0,
) -> Iterator[Router]:
    """Own the optional telemetry worker and yield one long-lived router."""

    owns_observability = observability is None
    configured_observability = (
        build_observability_from_environment() if owns_observability else observability
    )
    router = Router.from_file(
        configured_catalog(),
        observability=configured_observability,
        review_margin=review_margin,
    )
    try:
        yield router
    finally:
        if (
            owns_observability
            and configured_observability is not None
            and not configured_observability.shutdown(timeout_seconds=5)
        ):
            # Do not include prompts, outputs, identifiers, or exception text in this message.
            LOGGER.error("OpenRoutiQ observability did not shut down within five seconds")


def print_decision(decision: RouteDecision) -> None:
    """Print routing metadata only; production services should use structured application logs."""

    summary = {
        "model_id": decision.selected.model_id,
        "provider": decision.selected.provider,
        "reasoning_level": decision.selected.reasoning_level,
        "review_required": decision.review_required,
        "predicted_cost_usd": decision.selected.predicted_cost,
        "predicted_latency_ms": decision.selected.expected_latency_ms,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
