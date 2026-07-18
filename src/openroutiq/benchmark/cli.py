from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openroutiq.benchmark.core import (
    BenchmarkDataset,
    BenchmarkError,
    BenchmarkRun,
    BestSingleBenchmarkRouter,
    FixedModelBenchmarkRouter,
    OpenRoutiQBenchmarkRouter,
    OracleBenchmarkRouter,
    RandomBenchmarkRouter,
    run_benchmark,
)
from openroutiq.benchmark.adapters import (
    CommandBenchmarkRouter,
    NotDiamondBenchmarkRouter,
    OpenAICompatibleBenchmarkRouter,
    OpenAIExecutionBenchmarkRouter,
    RouteLLMBenchmarkRouter,
    SelectionFileBenchmarkRouter,
)
from openroutiq.benchmark.reporting import render_benchmark_report
from openroutiq.router.core import OpenRoutiQError, Router


_TEMPLATE_PACKAGE = "openroutiq.benchmark.templates"
_TEMPLATE_FILES = ("catalog.json", "recorded.json", "replay.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openroutiq-benchmark",
        description="Reproducible accuracy-cost benchmarks for model routers",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="create a runnable local benchmark example")
    initialize.add_argument(
        "directory",
        nargs="?",
        default=".openroutiq/benchmark-example",
        help="directory to create (default: .openroutiq/benchmark-example)",
    )
    initialize.add_argument(
        "--force",
        action="store_true",
        help="replace benchmark template files that already exist",
    )
    validate = commands.add_parser("validate", help="validate a dataset and benchmark config")
    validate.add_argument("config")
    estimate = commands.add_parser(
        "estimate", help="show cases, live calls, and worst-case model cost"
    )
    estimate.add_argument("config")
    run = commands.add_parser("run", help="execute a benchmark after explicit approval")
    run.add_argument("config")
    run.add_argument(
        "--confirm-benchmark",
        action="store_true",
        help="confirm that benchmark execution has been reviewed and approved",
    )
    run.add_argument(
        "--allow-live",
        action="store_true",
        help="allow configured network/paid router adapters",
    )
    run.add_argument(
        "--max-estimated-cost",
        type=float,
        help="maximum accepted worst-case model cost in USD for live inference",
    )
    run.add_argument(
        "--max-live-calls",
        type=int,
        help="maximum accepted counted model plus hosted-router calls",
    )
    run.add_argument(
        "--env-file",
        help=(
            "load only credentials required by configured live routers from this dotenv file; "
            "the file is read only after all approval and budget gates pass"
        ),
    )
    report = commands.add_parser("report", help="render an existing result JSON without rerunning")
    report.add_argument("results")
    report.add_argument("--output", default="benchmark-report.html")
    return parser


def _initialize_example(directory: str | Path, *, force: bool = False) -> tuple[Path, ...]:
    target = Path(directory).resolve()
    if target.exists() and not target.is_dir():
        raise BenchmarkError(f"benchmark example destination is not a directory: {target}")
    destinations = tuple(target / name for name in _TEMPLATE_FILES)
    existing = [path.name for path in destinations if path.exists()]
    if existing and not force:
        raise BenchmarkError(
            "benchmark example files already exist; choose another directory or pass --force: "
            + ", ".join(existing)
        )
    target.mkdir(parents=True, exist_ok=True)
    templates = importlib.resources.files(_TEMPLATE_PACKAGE)
    for name, destination in zip(_TEMPLATE_FILES, destinations, strict=True):
        destination.write_bytes(templates.joinpath(name).read_bytes())
    return destinations


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"cannot read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise BenchmarkError(f"{label} root must be an object")
    return dict(raw)


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{name} must be a non-empty path")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _router(spec: Mapping[str, Any], base: Path):
    kind = str(spec.get("type", "")).lower()
    name = str(spec.get("name", kind))
    if kind == "openroutiq":
        catalog = _path(base, spec.get("catalog"), "routers[].catalog")
        return OpenRoutiQBenchmarkRouter(
            Router.from_file(catalog),
            name=name,
            strategy=str(spec.get("strategy", "auto")),
            risk_policy=spec.get("risk_policy"),
        )
    if kind == "random":
        return RandomBenchmarkRouter(seed=int(spec.get("seed", 3407)), name=name)
    if kind == "best_single":
        model_id = spec.get("model_id")
        return BestSingleBenchmarkRouter(
            model_id=None if model_id is None else str(model_id),
            name=name,
        )
    if kind == "fixed_model":
        return FixedModelBenchmarkRouter(str(spec.get("model_id", "")), name=name)
    if kind == "oracle":
        return OracleBenchmarkRouter(name=name)
    if kind == "notdiamond":
        return NotDiamondBenchmarkRouter(
            name=name,
            api_key_env=str(spec.get("api_key_env", "NOTDIAMOND_API_KEY")),
            tradeoff=spec.get("tradeoff"),
        )
    if kind == "routellm":
        return RouteLLMBenchmarkRouter(
            name=name,
            strong_model_id=str(spec.get("strong_model_id", "")),
            weak_model_id=str(spec.get("weak_model_id", "")),
            router=str(spec.get("router", "mf")),
            threshold=float(spec.get("threshold", 0.5)),
            config=spec.get("config"),
        )
    if kind in {"openrouter_auto", "openai_compatible"}:
        openrouter = kind == "openrouter_auto"
        return OpenAICompatibleBenchmarkRouter(
            name=name,
            base_url=str(
                spec.get("base_url", "https://openrouter.ai/api/v1" if openrouter else "")
            ),
            router_model=str(spec.get("router_model", "openrouter/auto" if openrouter else "")),
            api_key_env=str(spec.get("api_key_env", "OPENROUTER_API_KEY")),
            timeout_seconds=float(spec.get("timeout_seconds", 120)),
            max_output_tokens=spec.get("max_output_tokens"),
            allowed_models_field=(
                "openrouter_plugins" if openrouter else spec.get("allowed_models_field")
            ),
            headers=spec.get("headers"),
            extra_body=spec.get("extra_body"),
        )
    if kind in {"openrouter_execute", "openai_execute"}:
        selector_spec = spec.get("selector")
        if not isinstance(selector_spec, Mapping):
            raise BenchmarkError(f"{kind} requires a selector object")
        openrouter = kind == "openrouter_execute"
        return OpenAIExecutionBenchmarkRouter(
            _router(selector_spec, base),
            name=name,
            base_url=str(
                spec.get("base_url", "https://openrouter.ai/api/v1" if openrouter else "")
            ),
            api_key_env=str(spec.get("api_key_env", "OPENROUTER_API_KEY")),
            timeout_seconds=float(spec.get("timeout_seconds", 120)),
            max_output_tokens=spec.get("max_output_tokens"),
            reasoning_style=str(
                spec.get("reasoning_style", "openrouter" if openrouter else "reasoning_effort")
            ),
            headers=spec.get("headers"),
            extra_body=spec.get("extra_body"),
        )
    if kind == "selection_file":
        return SelectionFileBenchmarkRouter(
            _path(base, spec.get("path"), "routers[].path"),
            name=name,
        )
    if kind == "command":
        command = spec.get("command")
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes, bytearray)):
            raise BenchmarkError("command router requires an argument-list command")
        cwd = spec.get("cwd")
        return CommandBenchmarkRouter(
            command,
            name=name,
            timeout_seconds=float(spec.get("timeout_seconds", 120)),
            live=bool(spec.get("live", False)),
            model_calls_per_case=spec.get("model_calls_per_case"),
            router_calls_per_case=spec.get("router_calls_per_case"),
            cwd=None if cwd is None else _path(base, cwd, "routers[].cwd"),
            pass_env=spec.get("pass_env", ()),
        )
    raise BenchmarkError(f"unknown benchmark router type: {kind or '<missing>'}")


def _load_config(path: str | Path):
    config_path = Path(path).resolve()
    config = _read_object(config_path, "benchmark config")
    dataset_path = _path(config_path.parent, config.get("dataset"), "dataset")
    dataset = BenchmarkDataset.from_file(dataset_path)
    specs = config.get("routers")
    if (
        not isinstance(specs, list)
        or not specs
        or any(not isinstance(item, Mapping) for item in specs)
    ):
        raise BenchmarkError("benchmark config routers must be a non-empty object list")
    routers = [_router(item, config_path.parent) for item in specs]
    output = _path(config_path.parent, config.get("output", "results"), "output")
    return config, dataset, routers, output


def _estimate(
    dataset: BenchmarkDataset,
    routers: Sequence[Any],
    *,
    repetitions: int,
) -> dict[str, Any]:
    model_calls_per_case = sum(
        int(getattr(router, "model_calls_per_case", 0)) for router in routers
    )
    hosted_router_calls_per_case = sum(
        int(getattr(router, "router_calls_per_case", 0)) for router in routers
    )
    worst_case_cost = 0.0
    for case in dataset.cases:
        input_tokens = case.input_tokens or 0
        output_tokens = case.expected_output_tokens or 0
        worst_case_cost += max(
            candidate.predicted_cost(input_tokens, output_tokens)
            for candidate in case.eligible_candidates
        )
    return {
        "dataset": dataset.name,
        "cases": len(dataset.cases),
        "routers": [router.name for router in routers],
        "live_routers": [router.name for router in routers if router.is_live],
        "repetitions": repetitions,
        "model_calls_per_case": model_calls_per_case,
        "hosted_router_calls_per_case": hosted_router_calls_per_case,
        "live_inference_calls": len(dataset.cases) * model_calls_per_case * repetitions,
        "hosted_router_selection_calls": (
            len(dataset.cases) * hosted_router_calls_per_case * repetitions
        ),
        "total_counted_live_calls": (
            len(dataset.cases) * (model_calls_per_case + hosted_router_calls_per_case) * repetitions
        ),
        "worst_case_model_cost_usd": (worst_case_cost * model_calls_per_case * repetitions),
        "note": "Hosted router-selection fees, retries, judge calls, taxes, and price changes are not included.",
    }


def _targets(config: Mapping[str, Any]) -> dict[str, float]:
    targets: dict[str, float] = {}
    cost_budget = config.get("cost_budget_per_request")
    if cost_budget is not None:
        if isinstance(cost_budget, bool) or not isinstance(cost_budget, (int, float)):
            raise BenchmarkError("cost_budget_per_request must be a non-negative number")
        if cost_budget < 0:
            raise BenchmarkError("cost_budget_per_request must be a non-negative number")
        targets["cost_budget_per_request"] = float(cost_budget)
    accuracy_target = config.get("accuracy_target")
    if accuracy_target is not None:
        if (
            isinstance(accuracy_target, bool)
            or not isinstance(accuracy_target, (int, float))
            or not 0 <= accuracy_target <= 1
        ):
            raise BenchmarkError("accuracy_target must be a number between 0 and 1")
        targets["accuracy_target"] = float(accuracy_target)
    return targets


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _required_credential_envs(routers: Sequence[Any]) -> set[str]:
    """Return credential names without reading any environment values."""

    required: set[str] = set()

    def visit(router: Any) -> None:
        name = getattr(router, "api_key_env", None)
        if isinstance(name, str) and name:
            required.add(name)
        selector = getattr(router, "selector", None)
        if selector is not None:
            visit(selector)

    for benchmark_router in routers:
        visit(benchmark_router)
    return required


def _dotenv_values(path: Path, allowed_names: set[str]) -> dict[str, str]:
    """Parse a deliberately small, non-expanding dotenv subset.

    Benchmark credentials do not need shell interpolation. Refusing multiline and
    malformed entries keeps parsing deterministic and prevents accidental logging or
    execution of file contents.
    """

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise BenchmarkError(f"cannot read env file {path}: {exc}") from exc
    values: dict[str, str] = {}
    for line_number, original in enumerate(lines, start=1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise BenchmarkError(f"invalid env file entry at {path}:{line_number}")
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME.fullmatch(name):
            raise BenchmarkError(f"invalid env name at {path}:{line_number}")
        if name not in allowed_names:
            continue
        value = raw_value.strip()
        if value.startswith(("'", '"')):
            quote = value[0]
            if len(value) < 2 or not value.endswith(quote):
                raise BenchmarkError(f"unterminated env value at {path}:{line_number}")
            value = value[1:-1]
            if quote == '"':
                try:
                    value = json.loads(f'"{value}"')
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(
                        f"invalid quoted env value at {path}:{line_number}"
                    ) from exc
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if not value:
            raise BenchmarkError(f"empty credential {name} in env file {path}")
        values[name] = value
    return values


def _load_required_env_file(path: Path, routers: Sequence[Any]) -> None:
    required = _required_credential_envs(routers)
    values = _dotenv_values(path, required)
    for name, value in values.items():
        os.environ.setdefault(name, value)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            paths = _initialize_example(args.directory, force=args.force)
            print(
                json.dumps(
                    {
                        "directory": str(paths[0].parent),
                        "files": [str(path) for path in paths],
                        "next": f"openroutiq-benchmark validate {paths[-1]}",
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "report":
            target = render_benchmark_report(BenchmarkRun.from_file(args.results), args.output)
            print(json.dumps({"report": str(target.resolve())}))
            return 0
        config, dataset, routers, output = _load_config(args.config)
        repetitions = config.get("repetitions", 1)
        if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
            raise BenchmarkError("repetitions must be an integer >= 1")
        targets = _targets(config)
        estimate = _estimate(dataset, routers, repetitions=repetitions)
        estimate["measurement_targets"] = targets
        if args.command == "validate":
            for benchmark_router in routers:
                if not benchmark_router.is_live:
                    benchmark_router.prepare(dataset)
            print(json.dumps({**estimate, "valid": True}, indent=2))
            return 0
        if args.command == "estimate":
            print(json.dumps(estimate, indent=2))
            return 0
        if not args.confirm_benchmark:
            raise BenchmarkError(
                "benchmark not run: obtain approval, review 'estimate', then pass --confirm-benchmark"
            )
        live = bool(estimate["live_routers"])
        if live and not args.allow_live:
            raise BenchmarkError("live benchmark not run: pass --allow-live after approval")
        estimated_cost = float(estimate["worst_case_model_cost_usd"])
        counted_live_calls = int(estimate["total_counted_live_calls"])
        if counted_live_calls:
            if args.max_live_calls is None:
                raise BenchmarkError(
                    "live calls require --max-live-calls after reviewing the estimate"
                )
            if args.max_live_calls < 0:
                raise BenchmarkError("--max-live-calls must be non-negative")
            if counted_live_calls > args.max_live_calls:
                raise BenchmarkError(
                    f"estimated {counted_live_calls} live calls exceed approved cap "
                    f"{args.max_live_calls}"
                )
        if estimate["live_inference_calls"]:
            if args.max_estimated_cost is None:
                raise BenchmarkError(
                    "live inference requires --max-estimated-cost after reviewing the estimate"
                )
            if estimated_cost > args.max_estimated_cost:
                raise BenchmarkError(
                    f"worst-case model cost ${estimated_cost:.6f} exceeds approved cap "
                    f"${args.max_estimated_cost:.6f}"
                )
        if live and args.env_file:
            _load_required_env_file(Path(args.env_file).resolve(), routers)
        run = run_benchmark(
            dataset,
            routers,
            allow_live=args.allow_live,
            repetitions=repetitions,
            metadata={
                "config": str(Path(args.config).resolve()),
                "config_fingerprint": hashlib.sha256(
                    Path(args.config).resolve().read_bytes()
                ).hexdigest(),
                "estimate": estimate,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "measurement_targets": targets,
            },
        )
        json_path, csv_path = run.write(output)
        report_path = render_benchmark_report(run, output / "benchmark-report.html")
        print(
            json.dumps(
                {
                    "results": str(json_path.resolve()),
                    "summary": str((output / "benchmark-summary.json").resolve()),
                    "observations": str(csv_path.resolve()),
                    "report": str(report_path.resolve()),
                },
                indent=2,
            )
        )
        return 0
    except (BenchmarkError, OpenRoutiQError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
