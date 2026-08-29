"""Route clinical-document assistance without authorizing autonomous care decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openroutiq import RouteContext, RouteDecision, Router

from examples.common.runtime import managed_router, print_decision


DOCUMENT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "clinical_document_fields",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "document_type": {"type": "string"},
                "follow_up_date": {"type": ["string", "null"]},
                "review_flags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["document_type", "follow_up_date", "review_flags"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class ClinicianReviewRoute:
    decision: RouteDecision
    clinician_approval_required: bool = True
    autonomous_care_decision_allowed: bool = False


def route_clinical_document(
    router: Router,
    synthetic_document: str,
    *,
    allowed_providers: tuple[str, ...] = ("example",),
) -> ClinicianReviewRoute:
    context = RouteContext(
        synthetic_document,
        agent_role="clinical-document-assistant",
        workflow_step="draft-extraction",
        side_effect_level="human-reviewed",
        latency_deadline_ms=10_000,
    )
    decision = router.route(
        context,
        task="extraction",
        strategy="quality",
        high_risk=True,
        response_format=DOCUMENT_SCHEMA,
        constraints={
            "allowed_providers": list(allowed_providers),
            "required_capabilities": ["text", "json"],
            "min_quality": 80,
        },
    )
    return ClinicianReviewRoute(decision=decision)


def main() -> None:
    document = "Synthetic discharge note: document type discharge; follow-up 2026-09-10."
    with managed_router() as router:
        routed = route_clinical_document(router, document)
        print_decision(routed.decision)
        print(f"clinician_approval_required={routed.clinician_approval_required}")


if __name__ == "__main__":
    main()
