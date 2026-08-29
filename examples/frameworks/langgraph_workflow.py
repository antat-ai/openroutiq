"""Use OpenRoutiQ as a routing node with an explicit human-review branch in LangGraph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypedDict

from openroutiq import RouteDecision, Router

from examples.common.runtime import managed_router


class WorkflowState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    decision: RouteDecision
    result: Any
    status: str


def build_workflow(
    router: Router,
    model_handlers: Mapping[str, Callable[[list[dict[str, Any]]], Any]],
) -> Any:
    """Compile a graph that never executes unapproved or review-gated routes."""

    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install langgraph to run this example") from exc

    trusted_handlers = dict(model_handlers)
    if not trusted_handlers:
        raise ValueError("model_handlers must contain at least one trusted model variant")

    def select_route(state: WorkflowState) -> WorkflowState:
        messages = state.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        decision = router.route(
            messages,
            task="general",
            strategy="auto",
            constraints={"candidate_ids": sorted(trusted_handlers)},
        )
        return {"decision": decision, "status": "routed"}

    def next_step(state: WorkflowState) -> str:
        return "human_review" if state["decision"].review_required else "dispatch"

    def require_human_review(_state: WorkflowState) -> WorkflowState:
        # Persist a review task in the application's own durable queue here. Do not put the
        # messages into observability attributes.
        return {"status": "human_review_required"}

    def dispatch(state: WorkflowState) -> WorkflowState:
        decision = state["decision"]
        handler = trusted_handlers.get(decision.selected.model_id)
        if handler is None:
            raise RuntimeError("the selected model variant has no trusted handler")
        return {"result": handler(state["messages"]), "status": "completed"}

    graph = StateGraph(WorkflowState)
    graph.add_node("route", select_route)
    graph.add_node("dispatch", dispatch)
    graph.add_node("human_review", require_human_review)
    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        next_step,
        {"dispatch": "dispatch", "human_review": "human_review"},
    )
    graph.add_edge("dispatch", END)
    graph.add_edge("human_review", END)
    return graph.compile()


def main() -> None:
    def offline_handler(_messages: list[dict[str, Any]]) -> dict[str, str]:
        return {"content": "synthetic offline response"}

    with managed_router(review_margin=0) as router:
        handlers = {profile.id: offline_handler for profile in router.profiles}
        workflow = build_workflow(router, handlers)
        result = workflow.invoke(
            {"messages": [{"role": "user", "content": "Summarize retry safety."}]}
        )
        print(result["status"])


if __name__ == "__main__":
    main()
