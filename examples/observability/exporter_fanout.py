"""Send one privacy-filtered event stream to configured OTLP backends."""

from __future__ import annotations

from openroutiq import Router

from examples.common.observability import build_observability_from_environment
from examples.common.runtime import configured_catalog, print_decision


def main() -> None:
    observability = build_observability_from_environment()
    if observability is None:
        raise SystemExit(
            "Set OPENROUTIQ_OBSERVABILITY_BACKENDS to otlp, langsmith, langtrace, "
            "or a comma-separated fan-out list"
        )
    try:
        router = Router.from_file(configured_catalog(), observability=observability)
        decision = router.route(
            "Route this synthetic reliability-analysis request.",
            task="reasoning",
        )
        print_decision(decision)
        if not observability.flush(timeout_seconds=5):
            raise RuntimeError("observability flush did not finish within five seconds")
    finally:
        if not observability.shutdown(timeout_seconds=5):
            raise RuntimeError("observability shutdown did not finish within five seconds")


if __name__ == "__main__":
    main()
