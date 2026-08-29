"""Route a tool-using customer-support workflow with an explicit provider policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openroutiq import RouteContext, RouteDecision, Router

from examples.common.runtime import managed_router, print_decision


LOOKUP_ORDER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Read an order by its opaque support reference.",
        "parameters": {
            "type": "object",
            "properties": {"reference": {"type": "string"}},
            "required": ["reference"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class SupportRoutingPolicy:
    allowed_providers: tuple[str, ...] = ("example",)
    latency_deadline_ms: float = 2_000
    maximum_predicted_cost_usd: float = 0.03


def route_support_request(
    router: Router,
    messages: list[dict[str, Any]],
    *,
    policy: SupportRoutingPolicy = SupportRoutingPolicy(),
) -> RouteDecision:
    context = RouteContext(
        messages,
        agent_role="customer-support",
        workflow_step="order-lookup",
        side_effect_level="read-only",
        latency_deadline_ms=policy.latency_deadline_ms,
    )
    return router.route(
        context,
        task="tool_use",
        strategy="auto",
        tools=[LOOKUP_ORDER_TOOL],
        tool_choice="required",
        constraints={
            "allowed_providers": list(policy.allowed_providers),
            "required_capabilities": ["text", "tools"],
            "max_predicted_cost": policy.maximum_predicted_cost_usd,
            "max_latency_ms": policy.latency_deadline_ms,
        },
    )


def main() -> None:
    messages = [
        {
            "role": "user",
            "content": "Check synthetic order REF-1007 and summarize its shipping status.",
        }
    ]
    with managed_router() as router:
        print_decision(route_support_request(router, messages))


if __name__ == "__main__":
    main()
