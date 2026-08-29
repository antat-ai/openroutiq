"""Select a preconfigured LangChain runnable with OpenRoutiQ."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

from openroutiq import Router

from examples.common.runtime import managed_router


class ChainRequest(TypedDict):
    messages: list[dict[str, Any]]


class RoutedChainResult(TypedDict):
    result: Any
    routing: dict[str, Any]
    selected_model_id: str


def build_routed_chain(router: Router, model_runnables: Mapping[str, Any]) -> Any:
    """Create a chain that dispatches only to an allowlisted model-variant runnable."""

    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as exc:
        raise RuntimeError("Install langchain-core to run this example") from exc

    trusted_runnables = dict(model_runnables)
    if not trusted_runnables:
        raise ValueError("model_runnables must contain at least one trusted model variant")

    def select_model(request: ChainRequest) -> dict[str, Any]:
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        decision = router.route(
            messages,
            task="general",
            strategy="auto",
            constraints={"candidate_ids": sorted(trusted_runnables)},
        )
        return {"messages": messages, "decision": decision}

    def invoke_selected(state: dict[str, Any]) -> RoutedChainResult:
        decision = state["decision"]
        runnable = trusted_runnables.get(decision.selected.model_id)
        if runnable is None:
            raise RuntimeError("the selected model variant has no trusted LangChain runnable")
        result = runnable.invoke(state["messages"])
        # Application state can contain request/result data. The OpenRoutiQ observer receives
        # only its fixed, numeric event schema and never receives this state object.
        return {
            "result": result,
            "routing": decision.to_dict(),
            "selected_model_id": decision.selected.model_id,
        }

    return RunnableLambda(select_model) | RunnableLambda(invoke_selected)


def main() -> None:
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as exc:
        raise SystemExit("Install langchain-core to run this example") from exc

    offline_model = RunnableLambda(lambda _messages: {"content": "synthetic offline response"})
    with managed_router(review_margin=0) as router:
        model_runnables = {profile.id: offline_model for profile in router.profiles}
        chain = build_routed_chain(router, model_runnables)
        result = chain.invoke(
            {"messages": [{"role": "user", "content": "Explain idempotency briefly."}]}
        )
        print(result["selected_model_id"])


if __name__ == "__main__":
    main()
