"""Route a synthetic coding request without calling a model provider."""

from __future__ import annotations

from examples.common.runtime import managed_router, print_decision


def main() -> None:
    messages = [
        {"role": "system", "content": "You review production Python."},
        {"role": "user", "content": "Find and fix the race condition in this worker."},
    ]
    with managed_router() as router:
        decision = router.route(
            messages,
            task="coding",
            strategy="auto",
            constraints={"required_capabilities": ["text"]},
        )
        print_decision(decision)


if __name__ == "__main__":
    main()
