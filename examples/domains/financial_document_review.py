"""Route financial extraction with a cost ceiling and mandatory analyst approval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openroutiq import RouteContext, RouteDecision, Router

from examples.common.runtime import managed_router, print_decision


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "financial_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reporting_period": {"type": "string"},
                "revenue": {"type": "number"},
                "operating_expense": {"type": "number"},
                "currency": {"type": "string"},
            },
            "required": ["reporting_period", "revenue", "operating_expense", "currency"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class AnalystReviewRoute:
    decision: RouteDecision
    analyst_approval_required: bool = True


def route_financial_document(
    router: Router,
    document: str,
    *,
    allowed_providers: tuple[str, ...] = ("example",),
) -> AnalystReviewRoute:
    context = RouteContext(
        document,
        agent_role="financial-analysis",
        workflow_step="structured-extraction",
        side_effect_level="decision-support",
        budget_remaining=0.08,
        latency_deadline_ms=8_000,
    )
    decision = router.route(
        context,
        task="extraction",
        strategy="risk_aware",
        high_risk=True,
        response_format=EXTRACTION_SCHEMA,
        constraints={
            "allowed_providers": list(allowed_providers),
            "required_capabilities": ["text", "json"],
            "min_quality": 75,
            "max_predicted_cost": 0.08,
        },
        soft_budget=0.05,
    )
    return AnalystReviewRoute(decision=decision)


def main() -> None:
    synthetic_document = (
        "Synthetic statement for FY2026: revenue 125000 USD; operating expense 79000 USD."
    )
    with managed_router() as router:
        routed = route_financial_document(router, synthetic_document)
        print_decision(routed.decision)
        print(f"analyst_approval_required={routed.analyst_approval_required}")


if __name__ == "__main__":
    main()
