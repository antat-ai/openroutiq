from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from collections.abc import Sequence

from openroutiq.adaptive import AdaptiveRouter
from openroutiq.quickstart import init_catalog
from openroutiq.router import OpenRoutiQError, Router, TASKS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openroutiq", description="Route AI requests by quality, latency, and cost"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create an editable starter catalog")
    init.add_argument("--catalog", default="models.json")
    init.add_argument(
        "--provider",
        choices=["all", "openai", "anthropic", "openrouter", "requesty", "litellm"],
        default="all",
    )
    init.add_argument("--force", action="store_true")
    serve = commands.add_parser("serve", help="run the OpenAI-compatible routing proxy")
    serve.add_argument("--catalog", default="models.json")
    serve.add_argument(
        "--adaptive-registry",
        help="optional local SQLite path for encounter-driven model learning",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument(
        "--api-key-env",
        default="OPENROUTIQ_PROXY_API_KEY",
        help="environment variable containing the proxy bearer token",
    )
    serve.add_argument(
        "--log-level", choices=["critical", "error", "warning", "info", "debug"], default="info"
    )
    serve.add_argument("--max-request-bytes", type=int, default=4 * 1024 * 1024)
    serve.add_argument("--max-concurrency", type=int, default=128)
    serve.add_argument("--max-declared-tokens", type=int, default=10_000_000)
    serve.add_argument("--queue-timeout", type=float, default=5.0)
    serve.add_argument("--routing-timeout", type=float, default=30.0)
    serve.add_argument("--provider-timeout", type=float, default=600.0)
    serve.add_argument("--stream-idle-timeout", type=float, default=120.0)
    route = commands.add_parser("route", help="select and explain a model")
    route.add_argument("prompt")
    route.add_argument("--catalog", default="models.json")
    route.add_argument(
        "--task",
        help=f"task label from the catalog (built-ins include: {', '.join(sorted(TASKS))})",
    )
    route.add_argument(
        "--strategy",
        choices=["auto", "balanced", "quality", "speed", "cost", "risk_aware"],
        default="auto",
    )
    route.add_argument("--quality", type=float)
    route.add_argument("--latency", type=float)
    route.add_argument("--cost", type=float)
    route.add_argument("--complexity", type=float)
    route.add_argument("--require-capability", action="append", default=[])
    route.add_argument("--allow-provider", action="append", default=[])
    route.add_argument("--block-provider", action="append", default=[])
    route.add_argument("--candidate", action="append", default=[])
    route.add_argument("--min-context", type=int, default=0)
    route.add_argument("--max-cost", type=float)
    route.add_argument("--max-latency", type=float)
    route.add_argument("--min-quality", type=float)
    route.add_argument("--reasoning-level", action="append", default=[])
    route.add_argument(
        "--reasoning-effort",
        choices=["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default="auto",
    )
    route.add_argument("--pin-model")
    route.add_argument("--requires-tools", action="store_true")
    route.add_argument("--parallel-tools", action="store_true")
    route.add_argument("--structured-output", action="store_true")
    route.add_argument("--output-modality", action="append", default=[])
    route.add_argument("--stream", action="store_true")
    route.add_argument("--soft-budget", type=float)
    route.add_argument("--input-tokens", type=int)
    route.add_argument("--output-tokens", type=int)
    route.add_argument("--local-only", action="store_true")
    route.add_argument("--risk", choices=["auto", "normal", "high"], default="auto")
    return parser


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            path = init_catalog(args.catalog, provider=args.provider, force=args.force)
            print(f"Created {path}. Replace its placeholder values before production use.")
            return 0
        if args.command == "serve":
            try:
                import uvicorn
            except ImportError as exc:
                raise OpenRoutiQError("the proxy requires 'pip install openroutiq[proxy]'") from exc
            from openroutiq.proxy import ProxyLimits, create_app

            if not 1 <= args.port <= 65535:
                raise OpenRoutiQError("port must be between 1 and 65535")
            if not args.api_key_env.strip():
                raise OpenRoutiQError("api-key-env must be a non-empty environment variable name")
            if not _is_loopback_host(args.host) and not os.environ.get(args.api_key_env):
                raise OpenRoutiQError(
                    f"non-loopback proxy host requires a bearer token in {args.api_key_env}"
                )
            route_engine = (
                AdaptiveRouter.from_file(
                    args.catalog,
                    registry=args.adaptive_registry,
                )
                if args.adaptive_registry
                else args.catalog
            )
            limits = ProxyLimits(
                max_request_bytes=args.max_request_bytes,
                max_concurrency=args.max_concurrency,
                max_declared_tokens=args.max_declared_tokens,
                queue_timeout_seconds=args.queue_timeout,
                routing_timeout_seconds=args.routing_timeout,
                provider_timeout_seconds=args.provider_timeout,
                stream_idle_timeout_seconds=args.stream_idle_timeout,
            )
            uvicorn.run(
                create_app(route_engine, api_key_env=args.api_key_env, limits=limits),
                host=args.host,
                port=args.port,
                log_level=args.log_level,
            )
            return 0
        router = Router.from_file(args.catalog)
        required_capabilities = list(args.require_capability)
        if args.requires_tools or args.parallel_tools:
            required_capabilities.append("tools")
        if args.parallel_tools:
            required_capabilities.append("parallel_tools")
        if args.structured_output:
            required_capabilities.append("json_schema")
        if args.stream:
            required_capabilities.append("streaming")
        required_capabilities.extend(
            f"{modality.strip().lower()}_output"
            for modality in args.output_modality
            if modality.strip().lower() != "text"
        )
        weights = None
        if any(value is not None for value in (args.quality, args.latency, args.cost)):
            weights = {
                "quality": 60 if args.quality is None else args.quality,
                "latency": 25 if args.latency is None else args.latency,
                "cost": 15 if args.cost is None else args.cost,
            }
        decision = router.route(
            args.prompt,
            task=args.task,
            weights=weights,
            constraints={
                "required_capabilities": sorted(set(required_capabilities)),
                "allowed_providers": args.allow_provider,
                "blocked_providers": args.block_provider,
                "candidate_ids": args.candidate,
                "min_context_tokens": args.min_context,
                "max_predicted_cost": args.max_cost,
                "max_latency_ms": args.max_latency,
                "min_quality": args.min_quality,
                "reasoning_levels": args.reasoning_level,
                "local_only": args.local_only,
            },
            input_tokens=args.input_tokens,
            expected_output_tokens=args.output_tokens,
            high_risk=None if args.risk == "auto" else args.risk == "high",
            soft_budget=args.soft_budget,
            strategy=args.strategy,
            complexity=args.complexity,
            reasoning_effort=None if args.reasoning_effort == "auto" else args.reasoning_effort,
            pinned_model=args.pin_model,
        )
    except OpenRoutiQError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
    return 0
