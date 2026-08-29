"""Onboard a private model as provisional, pin it, and record a verified outcome."""

from __future__ import annotations

import os
from pathlib import Path

from openroutiq import AdaptiveRouter, Observability

from examples.common.runtime import PROJECT_ROOT, print_decision


def build_router(observability: Observability | None = None) -> AdaptiveRouter:
    registry = Path(
        os.environ.get(
            "OPENROUTIQ_ADAPTIVE_REGISTRY",
            PROJECT_ROOT / ".openroutiq" / "adaptive-models.sqlite3",
        )
    )
    router = AdaptiveRouter.from_file(
        PROJECT_ROOT / "models.example.json",
        registry=registry,
        observability=observability,
    )
    router.encounter_opaque(
        model_id="customer/legal-v7:high",
        provider="customer-private",
        model="legal-v7",
        api_style="openai_compatible",
        base_url="https://models.customer.internal/v1",
        reasoning_level="high",
        max_context_tokens=64_000,
        capabilities=["text", "tools", "json_schema"],
        tasks=["legal_review"],
        latency_ms=800,
        input_price_per_million=0.40,
        output_price_per_million=0.80,
        local=True,
    )
    return router


def record_verified_outcome(
    router: AdaptiveRouter,
    *,
    request: str,
    model_id: str,
    quality_score: float,
    latency_ms: float,
    actual_cost_usd: float,
    success: bool,
) -> None:
    """Persist a trusted evaluator's numeric result, never model self-confidence."""

    router.record_evaluation(
        request,
        model_id,
        quality_score,
        task="legal_review",
        latency_ms=latency_ms,
        actual_cost_usd=actual_cost_usd,
        success=success,
    )


def main() -> None:
    router = build_router()
    request = "Review this synthetic agreement against the approved legal policy."
    decision = router.route(
        request,
        task="legal_review",
        pinned_model="customer/legal-v7:high",
        high_risk=True,
    )
    print_decision(decision)
    # After an independent evaluator checks the complete task, call
    # record_verified_outcome(router, request=request, model_id=decision.selected.model_id, ...).


if __name__ == "__main__":
    main()
