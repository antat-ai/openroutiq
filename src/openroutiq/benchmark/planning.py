from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from typing import Any

from openroutiq.benchmark.core import BenchmarkError
from openroutiq.benchmark.protocol import OPENROUTER_BENCHMARK_PROTOCOL


FLOW_PLAN_SCHEMA_VERSION = 1
_ZERO = Decimal("0")
_ONE = Decimal("1")


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{name} must be an object")
    return dict(value)


def _object_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise BenchmarkError(f"{name} must be an object list")
    return [dict(item) for item in value]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{name} must be non-empty text")
    return value.strip()


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkError(f"{name} must be an integer >= {minimum}")
    return value


def _decimal(value: Any, name: str, *, minimum: Decimal = _ZERO) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise BenchmarkError(f"{name} must be a decimal number >= {minimum}")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise BenchmarkError(f"{name} must be a decimal number >= {minimum}") from exc
    if not number.is_finite() or number < minimum:
        raise BenchmarkError(f"{name} must be a decimal number >= {minimum}")
    return number


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise BenchmarkError(f"{name} must be a string list")
    if len(set(value)) != len(value):
        raise BenchmarkError(f"{name} must not contain duplicates")
    return list(value)


def _positive_integer_mapping(value: Any, name: str) -> dict[str, int]:
    raw = _object(value, name)
    if not raw:
        raise BenchmarkError(f"{name} must not be empty")
    result: dict[str, int] = {}
    for key, item in raw.items():
        if not isinstance(key, str) or not key:
            raise BenchmarkError(f"{name} keys must be non-empty text")
        result[key] = _integer(item, f"{name}.{key}", minimum=1)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: Any, name: str) -> Path:
    path = Path(_text(value, name))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _max_price(pricing: Mapping[str, Any], key: str) -> Decimal:
    values: list[Decimal] = []
    if key in pricing and pricing[key] is not None:
        values.append(_decimal(pricing[key], f"pricing.{key}"))
    overrides = pricing.get("overrides", [])
    if overrides is not None:
        for index, override in enumerate(_object_list(overrides, "pricing.overrides")):
            if key in override and override[key] is not None:
                values.append(_decimal(override[key], f"pricing.overrides[{index}].{key}"))
    return max(values, default=_ZERO)


@dataclass(frozen=True)
class FrozenModelPrice:
    model_id: str
    context_length: int
    input_modalities: frozenset[str]
    supported_parameters: frozenset[str]
    reasoning_mandatory: bool
    supported_efforts: frozenset[str]
    prompt_per_token: Decimal
    completion_per_token: Decimal
    image_per_token: Decimal
    audio_per_token: Decimal
    request_per_call: Decimal
    web_search_per_operation: Decimal

    @property
    def is_paid(self) -> bool:
        return any(
            price > _ZERO
            for price in (
                self.prompt_per_token,
                self.completion_per_token,
                self.image_per_token,
                self.audio_per_token,
                self.request_per_call,
                self.web_search_per_operation,
            )
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> "FrozenModelPrice":
        model_id = _text(value.get("id"), f"price catalog data[{index}].id")
        architecture = _object(
            value.get("architecture", {}), f"price catalog model {model_id}.architecture"
        )
        pricing = _object(value.get("pricing", {}), f"price catalog model {model_id}.pricing")
        reasoning = _object(
            value.get("reasoning", {}) or {}, f"price catalog model {model_id}.reasoning"
        )
        supported_efforts = reasoning.get("supported_efforts", []) or []
        return cls(
            model_id=model_id,
            context_length=_integer(
                value.get("context_length"),
                f"price catalog model {model_id}.context_length",
                minimum=1,
            ),
            input_modalities=frozenset(
                _string_list(
                    architecture.get("input_modalities", []),
                    f"price catalog model {model_id}.architecture.input_modalities",
                )
            ),
            supported_parameters=frozenset(
                _string_list(
                    value.get("supported_parameters", []),
                    f"price catalog model {model_id}.supported_parameters",
                )
            ),
            reasoning_mandatory=bool(reasoning.get("mandatory", False)),
            supported_efforts=frozenset(
                _string_list(
                    supported_efforts,
                    f"price catalog model {model_id}.reasoning.supported_efforts",
                )
            ),
            prompt_per_token=_max_price(pricing, "prompt"),
            completion_per_token=max(
                _max_price(pricing, "completion"),
                _max_price(pricing, "internal_reasoning"),
            ),
            image_per_token=_max_price(pricing, "image"),
            audio_per_token=_max_price(pricing, "audio"),
            request_per_call=_max_price(pricing, "request"),
            web_search_per_operation=_max_price(pricing, "web_search"),
        )

    def supports(
        self,
        *,
        modalities: set[str],
        required_parameters: set[str],
        any_parameter_groups: Sequence[set[str]],
        context_tokens: int,
        reasoning_effort: str | None,
    ) -> bool:
        if not modalities <= self.input_modalities:
            return False
        if not required_parameters <= self.supported_parameters:
            return False
        if any(group and not (group & self.supported_parameters) for group in any_parameter_groups):
            return False
        if context_tokens > self.context_length:
            return False
        if reasoning_effort is not None:
            if reasoning_effort == "none" and self.reasoning_mandatory:
                return False
            if (
                reasoning_effort != "none"
                and self.supported_efforts
                and reasoning_effort not in self.supported_efforts
            ):
                return False
        return True

    def maximum_call_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        modalities: set[str],
        web_searches: int,
        media_reserve: Decimal,
    ) -> Decimal:
        input_rate = self.prompt_per_token
        if "image" in modalities:
            input_rate = max(input_rate, self.image_per_token)
        if "audio" in modalities:
            input_rate = max(input_rate, self.audio_per_token)
        return (
            input_rate * input_tokens
            + self.completion_per_token * output_tokens
            + self.request_per_call
            + self.web_search_per_operation * web_searches
            + media_reserve
        )


def _load_price_catalog(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    required_model_ids: set[str],
    *,
    manifest_key: str = "price_catalog",
) -> tuple[dict[str, FrozenModelPrice], dict[str, Any]]:
    label = manifest_key.replace("_", " ")
    spec = _object(manifest.get(manifest_key), manifest_key)
    path = _resolve(manifest_path.parent, spec.get("path"), f"{manifest_key}.path")
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"cannot read {label} {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"invalid {label} {path}: {exc}") from exc
    root = _object(raw, label)
    rows = _object_list(root.get("data"), f"{label} data")
    actual_hash = hashlib.sha256(raw_bytes).hexdigest()
    expected_hash = _text(spec.get("sha256"), f"{manifest_key}.sha256").lower()
    if actual_hash.lower() != expected_hash:
        raise BenchmarkError(
            f"{label} SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    models: dict[str, FrozenModelPrice] = {}
    for index, row in enumerate(rows):
        if row.get("id") not in required_model_ids:
            continue
        model = FrozenModelPrice.from_mapping(row, index)
        if model.model_id in models:
            raise BenchmarkError(f"duplicate {label} model {model.model_id}")
        models[model.model_id] = model
    return models, {
        "path": str(path),
        "sha256": actual_hash,
        "source_url": _text(spec.get("source_url"), f"{manifest_key}.source_url"),
        "retrieved_at": _text(spec.get("retrieved_at"), f"{manifest_key}.retrieved_at"),
        "models_in_catalog": len(rows),
        "models_loaded_for_plan": len(models),
    }


def _model_ids(spec: Mapping[str, Any], pools: Mapping[str, list[str]], name: str) -> list[str]:
    has_pool = "pool" in spec
    has_models = "models" in spec
    if has_pool == has_models:
        raise BenchmarkError(f"{name} must define exactly one of pool or models")
    if has_pool:
        pool = _text(spec.get("pool"), f"{name}.pool")
        if pool not in pools:
            raise BenchmarkError(f"{name} references unknown pool {pool}")
        return list(pools[pool])
    return _string_list(spec.get("models"), f"{name}.models")


def _money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000000001")), "f")


def _approval_cap(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01"), rounding=ROUND_CEILING), "f")


def build_flow_plan(manifest_path: str | Path) -> dict[str, Any]:
    """Validate a real-flow manifest and compute an offline worst-case run plan.

    This function never reads environment variables and never performs network calls.
    It only reads the manifest and its hash-pinned public price snapshot.
    """

    path = Path(manifest_path).resolve()
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"cannot read flow manifest {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"invalid flow manifest {path}: {exc}") from exc
    manifest = _object(raw, "flow manifest")
    if manifest.get("schema_version") != FLOW_PLAN_SCHEMA_VERSION:
        raise BenchmarkError(f"flow manifest schema_version must be {FLOW_PLAN_SCHEMA_VERSION}")
    suite_id = _text(manifest.get("suite_id"), "suite_id")
    seed = _integer(manifest.get("seed"), "seed")
    repetitions = _integer(manifest.get("repetitions", 1), "repetitions", minimum=1)
    reserve_fraction = _decimal(
        manifest.get("price_change_reserve_fraction", 0),
        "price_change_reserve_fraction",
    )
    reserve_multiplier = _ONE + reserve_fraction
    raw_split_counts = manifest.get("selection_split_counts")
    selection_split_counts = (
        None
        if raw_split_counts is None
        else _positive_integer_mapping(raw_split_counts, "selection_split_counts")
    )
    require_paid_models = manifest.get("require_paid_models", False)
    if not isinstance(require_paid_models, bool):
        raise BenchmarkError("require_paid_models must be true or false")
    hard_budget_value = manifest.get("hard_budget_limit_usd")
    hard_budget_limit = (
        None if hard_budget_value is None else _decimal(hard_budget_value, "hard_budget_limit_usd")
    )
    retry_policy = _object(manifest.get("retry_policy", {}), "retry_policy")
    retries = _integer(
        retry_policy.get("max_retries_per_call", 0),
        "retry_policy.max_retries_per_call",
    )
    if retries:
        raise BenchmarkError(
            "release flow plans must set max_retries_per_call to 0 so the call ceiling is exact"
        )

    pool_specs = _object(manifest.get("model_pools"), "model_pools")
    system_specs = _object_list(manifest.get("systems"), "systems")
    required_model_ids: set[str] = set()
    required_router_model_ids: set[str] = set()
    for pool_name, pool_value in pool_specs.items():
        required_model_ids.update(_string_list(pool_value, f"model_pools.{pool_name}"))
    for index, system in enumerate(system_specs):
        if "models" in system:
            required_model_ids.update(
                _string_list(system.get("models"), f"systems[{index}].models")
            )
        router_model = system.get("hosted_router_model")
        if router_model is not None:
            required_router_model_ids.add(
                _text(router_model, f"systems[{index}].hosted_router_model")
            )
    price_models, price_provenance = _load_price_catalog(path, manifest, required_model_ids)
    router_price_models: dict[str, FrozenModelPrice] = {}
    router_price_provenance: dict[str, Any] | None = None
    if required_router_model_ids:
        router_price_models, router_price_provenance = _load_price_catalog(
            path,
            manifest,
            required_router_model_ids,
            manifest_key="router_price_catalog",
        )
        missing_router_prices = required_router_model_ids - set(router_price_models)
        if missing_router_prices:
            raise BenchmarkError(
                "hosted router models missing from the frozen router price catalog: "
                + ", ".join(sorted(missing_router_prices))
            )
    if require_paid_models:
        free_models = sorted(
            model_id for model_id, model in price_models.items() if not model.is_paid
        )
        free_router_models = sorted(
            model_id for model_id, model in router_price_models.items() if not model.is_paid
        )
        if free_models or free_router_models:
            raise BenchmarkError(
                "require_paid_models forbids zero-price model variants: "
                + ", ".join([*free_models, *free_router_models])
            )
    pools: dict[str, list[str]] = {}
    for pool_name, pool_value in pool_specs.items():
        if not isinstance(pool_name, str) or not pool_name:
            raise BenchmarkError("model_pools keys must be non-empty text")
        models = _string_list(pool_value, f"model_pools.{pool_name}")
        missing = [model for model in models if model not in price_models]
        if missing:
            raise BenchmarkError(
                f"model_pools.{pool_name} has models missing from the frozen catalog: "
                + ", ".join(missing)
            )
        pools[pool_name] = models

    systems: dict[str, dict[str, Any]] = {}
    for index, system in enumerate(system_specs):
        system_id = _text(system.get("id"), f"systems[{index}].id")
        if system_id in systems:
            raise BenchmarkError(f"duplicate system id {system_id}")
        system["_models"] = _model_ids(system, pools, f"systems[{index}]")
        system["_hosted_router_calls"] = _integer(
            system.get("hosted_router_calls_per_case", 0),
            f"systems[{index}].hosted_router_calls_per_case",
        )
        router_model = system.get("hosted_router_model")
        explicit_router_cost = system.get("hosted_router_cost_per_call_usd")
        if router_model is not None and explicit_router_cost is not None:
            raise BenchmarkError(
                f"systems[{index}] cannot combine hosted_router_model with an explicit cost"
            )
        if system["_hosted_router_calls"]:
            if router_model is None:
                system["_hosted_router_model"] = None
                system["_hosted_router_input_tokens"] = 0
                system["_hosted_router_cost"] = _decimal(
                    explicit_router_cost,
                    f"systems[{index}].hosted_router_cost_per_call_usd",
                )
            else:
                router_model_id = _text(router_model, f"systems[{index}].hosted_router_model")
                router_input_tokens = _integer(
                    system.get("max_hosted_router_input_tokens_per_call"),
                    f"systems[{index}].max_hosted_router_input_tokens_per_call",
                    minimum=1,
                )
                frozen_router = router_price_models[router_model_id]
                if router_input_tokens > frozen_router.context_length:
                    raise BenchmarkError(
                        f"systems[{index}] hosted router input cap exceeds "
                        f"{router_model_id} context length"
                    )
                system["_hosted_router_model"] = router_model_id
                system["_hosted_router_input_tokens"] = router_input_tokens
                system["_hosted_router_cost"] = frozen_router.maximum_call_cost(
                    input_tokens=router_input_tokens,
                    output_tokens=0,
                    modalities={"text"},
                    web_searches=0,
                    media_reserve=_ZERO,
                )
        else:
            if router_model is not None or explicit_router_cost is not None:
                raise BenchmarkError(
                    f"systems[{index}] declares hosted router pricing but no hosted router calls"
                )
            system["_hosted_router_model"] = None
            system["_hosted_router_input_tokens"] = 0
            system["_hosted_router_cost"] = _ZERO
        effort = system.get("reasoning_effort")
        if effort is not None:
            effort = _text(effort, f"systems[{index}].reasoning_effort").lower()
        system["_reasoning_effort"] = effort
        model_cost_multiplier = _decimal(
            system.get("model_cost_reservation_multiplier", 1),
            f"systems[{index}].model_cost_reservation_multiplier",
        )
        if model_cost_multiplier < _ONE:
            raise BenchmarkError(f"systems[{index}].model_cost_reservation_multiplier must be >= 1")
        system["_model_cost_multiplier"] = model_cost_multiplier
        raw_case_splits = system.get("case_splits")
        if raw_case_splits is None:
            system["_case_splits"] = None
        else:
            if selection_split_counts is None:
                raise BenchmarkError(
                    f"systems[{index}].case_splits requires selection_split_counts"
                )
            case_splits = _string_list(raw_case_splits, f"systems[{index}].case_splits")
            unknown_splits = set(case_splits) - set(selection_split_counts)
            if unknown_splits:
                raise BenchmarkError(
                    f"systems[{index}].case_splits references unknown splits: "
                    + ", ".join(sorted(unknown_splits))
                )
            system["_case_splits"] = case_splits
        systems[system_id] = system

    track_specs = _object_list(manifest.get("tracks"), "tracks")
    if not track_specs:
        raise BenchmarkError("tracks must not be empty")
    track_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    total_case_runs = 0
    total_model_calls = 0
    total_router_calls = 0
    total_judge_calls = 0
    base_cost = _ZERO
    datasets: dict[str, dict[str, Any]] = {}

    for track_index, track in enumerate(track_specs):
        prefix = f"tracks[{track_index}]"
        track_id = _text(track.get("id"), f"{prefix}.id")
        if track_id in track_ids:
            raise BenchmarkError(f"duplicate track id {track_id}")
        track_ids.add(track_id)
        scenario = _text(track.get("scenario"), f"{prefix}.scenario")
        framework = _text(track.get("framework"), f"{prefix}.framework")
        dataset = _object(track.get("dataset"), f"{prefix}.dataset")
        dataset_id = _text(dataset.get("id"), f"{prefix}.dataset.id")
        dataset_revision = _text(dataset.get("revision"), f"{prefix}.dataset.revision")
        dataset_split = _text(dataset.get("split"), f"{prefix}.dataset.split")
        evaluator = _text(dataset.get("evaluator"), f"{prefix}.dataset.evaluator")
        cases = _integer(track.get("cases"), f"{prefix}.cases", minimum=1)
        if selection_split_counts is not None and sum(selection_split_counts.values()) != cases:
            raise BenchmarkError(f"{prefix}.cases must equal the sum of selection_split_counts")
        sample_size = _integer(
            dataset.get("sample_size"), f"{prefix}.dataset.sample_size", minimum=1
        )
        if sample_size < cases:
            raise BenchmarkError(
                f"{prefix}.dataset.sample_size must be greater than or equal to cases"
            )
        sample_method = _text(dataset.get("sample_method"), f"{prefix}.dataset.sample_method")
        datasets.setdefault(
            dataset_id,
            {
                "id": dataset_id,
                "revision": dataset_revision,
                "split": dataset_split,
                "sample_method": sample_method,
                "frozen_sample_size": sample_size,
                "maximum_run_cases": cases,
                "evaluator": evaluator,
            },
        )
        if datasets[dataset_id]["revision"] != dataset_revision:
            raise BenchmarkError(f"dataset {dataset_id} uses multiple revisions")
        datasets[dataset_id]["frozen_sample_size"] = max(
            int(datasets[dataset_id]["frozen_sample_size"]), sample_size
        )
        datasets[dataset_id]["maximum_run_cases"] = max(
            int(datasets[dataset_id]["maximum_run_cases"]), cases
        )

        system_ids = _string_list(track.get("systems"), f"{prefix}.systems")
        unknown_systems = [item for item in system_ids if item not in systems]
        if unknown_systems:
            raise BenchmarkError(
                f"{prefix}.systems references unknown systems: {', '.join(unknown_systems)}"
            )
        model_calls_per_case = _integer(
            track.get("max_model_calls_per_case"),
            f"{prefix}.max_model_calls_per_case",
            minimum=1,
        )
        judge_calls_per_case = _integer(
            track.get("judge_calls_per_case", 0),
            f"{prefix}.judge_calls_per_case",
        )
        if judge_calls_per_case:
            raise BenchmarkError(
                f"{prefix} uses judge calls; release suite requires deterministic official graders"
            )
        input_tokens = _integer(
            track.get("max_input_tokens_per_call"),
            f"{prefix}.max_input_tokens_per_call",
            minimum=1,
        )
        output_tokens = _integer(
            track.get("max_output_tokens_per_call"),
            f"{prefix}.max_output_tokens_per_call",
            minimum=1,
        )
        modalities = set(
            _string_list(
                track.get("required_modalities", ["text"]), f"{prefix}.required_modalities"
            )
        )
        required_parameters = set(
            _string_list(track.get("required_parameters", []), f"{prefix}.required_parameters")
        )
        raw_groups = track.get("required_any_parameter_groups", [])
        if not isinstance(raw_groups, list):
            raise BenchmarkError(f"{prefix}.required_any_parameter_groups must be a list")
        any_groups = [
            set(_string_list(group, f"{prefix}.required_any_parameter_groups[{index}]"))
            for index, group in enumerate(raw_groups)
        ]
        web_searches = _integer(
            track.get("max_web_searches_per_call", 0),
            f"{prefix}.max_web_searches_per_call",
        )
        media_reserve = _decimal(
            track.get("media_reserve_usd_per_call", 0),
            f"{prefix}.media_reserve_usd_per_call",
        )
        track_repetitions = _integer(
            track.get("repetitions", repetitions), f"{prefix}.repetitions", minimum=1
        )

        for system_id in system_ids:
            system = systems[system_id]
            effort = system["_reasoning_effort"]
            eligible = [
                price_models[model_id]
                for model_id in system["_models"]
                if price_models[model_id].supports(
                    modalities=modalities,
                    required_parameters=required_parameters,
                    any_parameter_groups=any_groups,
                    context_tokens=input_tokens + output_tokens,
                    reasoning_effort=effort,
                )
            ]
            if not eligible:
                raise BenchmarkError(
                    f"track {track_id} system {system_id} has no eligible model in its frozen pool"
                )
            base_maximum_call_cost = max(
                model.maximum_call_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    modalities=modalities,
                    web_searches=web_searches,
                    media_reserve=media_reserve,
                )
                for model in eligible
            )
            maximum_prompt_rate = max(model.prompt_per_token for model in eligible)
            maximum_completion_rate = max(model.completion_per_token for model in eligible)
            maximum_request_rate = max(model.request_per_call for model in eligible)
            model_cost_multiplier = system["_model_cost_multiplier"]
            maximum_call_cost = base_maximum_call_cost * model_cost_multiplier
            case_splits = system["_case_splits"]
            if case_splits is None:
                selected_cases = cases
            else:
                if selection_split_counts is None:
                    raise BenchmarkError("case split counts were not validated")
                selected_cases = sum(selection_split_counts[name] for name in case_splits)
            case_runs = selected_cases * track_repetitions
            model_calls = case_runs * model_calls_per_case
            router_calls = case_runs * system["_hosted_router_calls"]
            judge_calls = case_runs * judge_calls_per_case
            model_cost = maximum_call_cost * model_calls
            router_cost = system["_hosted_router_cost"] * router_calls
            row_cost = model_cost + router_cost
            total_case_runs += case_runs
            total_model_calls += model_calls
            total_router_calls += router_calls
            total_judge_calls += judge_calls
            base_cost += row_cost
            rows.append(
                {
                    "track": track_id,
                    "scenario": scenario,
                    "framework": framework,
                    "system": system_id,
                    "system_label": _text(system.get("label"), f"system {system_id}.label"),
                    "system_category": _text(
                        system.get("category"), f"system {system_id}.category"
                    ),
                    "cases": selected_cases,
                    "track_cases": cases,
                    "case_splits": case_splits,
                    "repetitions": track_repetitions,
                    "case_runs": case_runs,
                    "maximum_model_calls": model_calls,
                    "maximum_hosted_router_calls": router_calls,
                    "hosted_router_model": system["_hosted_router_model"],
                    "maximum_hosted_router_input_tokens_per_call": system[
                        "_hosted_router_input_tokens"
                    ],
                    "maximum_hosted_router_call_cost_usd": _money(system["_hosted_router_cost"]),
                    "maximum_hosted_router_call_cost_with_reserve_usd": _money(
                        system["_hosted_router_cost"] * reserve_multiplier
                    ),
                    "maximum_judge_calls": judge_calls,
                    "maximum_counted_live_calls": model_calls + router_calls + judge_calls,
                    "maximum_input_tokens_per_call": input_tokens,
                    "maximum_output_tokens_per_call": output_tokens,
                    "eligible_models": [model.model_id for model in eligible],
                    "base_maximum_model_call_cost_usd": _money(base_maximum_call_cost),
                    "model_cost_reservation_multiplier": float(model_cost_multiplier),
                    "maximum_model_call_cost_usd": _money(maximum_call_cost),
                    "maximum_model_call_cost_with_reserve_usd": _money(
                        maximum_call_cost * reserve_multiplier
                    ),
                    "provider_max_prompt_price_per_million_usd": _money(
                        maximum_prompt_rate * Decimal(1_000_000) * reserve_multiplier
                    ),
                    "provider_max_completion_price_per_million_usd": _money(
                        maximum_completion_rate * Decimal(1_000_000) * reserve_multiplier
                    ),
                    "provider_max_request_price_usd": _money(
                        maximum_request_rate * reserve_multiplier
                    ),
                    "maximum_cost_before_reserve_usd": _money(row_cost),
                    "dataset": dataset_id,
                    "dataset_revision": dataset_revision,
                    "dataset_split": dataset_split,
                    "frozen_sample_size": sample_size,
                    "case_selection": (
                        "frozen_snapshot_prefix"
                        if case_splits is None
                        else "frozen_snapshot_deterministic_split"
                    ),
                    "evaluator": evaluator,
                }
            )

    cost_with_reserve = base_cost * reserve_multiplier
    if hard_budget_limit is not None and cost_with_reserve > hard_budget_limit:
        raise BenchmarkError(
            "planned cost with reserve "
            f"${_money(cost_with_reserve)} exceeds hard_budget_limit_usd "
            f"${_money(hard_budget_limit)}"
        )
    counted_calls = total_model_calls + total_router_calls + total_judge_calls
    by_system: list[dict[str, Any]] = []
    for system_id in systems:
        selected = [row for row in rows if row["system"] == system_id]
        if not selected:
            continue
        system_cost = sum(
            (Decimal(row["maximum_cost_before_reserve_usd"]) for row in selected), _ZERO
        )
        by_system.append(
            {
                "system": system_id,
                "label": selected[0]["system_label"],
                "category": selected[0]["system_category"],
                "tracks": len(selected),
                "case_runs": sum(row["case_runs"] for row in selected),
                "maximum_counted_live_calls": sum(
                    row["maximum_counted_live_calls"] for row in selected
                ),
                "maximum_cost_before_reserve_usd": _money(system_cost),
                "maximum_cost_with_reserve_usd": _money(system_cost * (_ONE + reserve_fraction)),
            }
        )

    return {
        "schema_version": FLOW_PLAN_SCHEMA_VERSION,
        "suite_id": suite_id,
        "seed": seed,
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "offline_plan_only": True,
        "credential_files_read": 0,
        "network_calls": 0,
        "model_or_hosted_router_calls_made": 0,
        "case_selection": "frozen_snapshot_prefix",
        "selection_split_counts": selection_split_counts,
        "price_catalog": price_provenance,
        "router_price_catalog": router_price_provenance,
        "price_change_reserve_fraction": float(reserve_fraction),
        "require_paid_models": require_paid_models,
        "hard_budget_limit_usd": (None if hard_budget_limit is None else _money(hard_budget_limit)),
        "retry_policy": {"max_retries_per_call": retries},
        "request_protocol": dict(OPENROUTER_BENCHMARK_PROTOCOL),
        "datasets": list(datasets.values()),
        "summary": {
            "tracks": len(track_specs),
            "systems": len({row["system"] for row in rows}),
            "track_system_combinations": len(rows),
            "planned_case_runs": total_case_runs,
            "maximum_model_calls": total_model_calls,
            "maximum_hosted_router_calls": total_router_calls,
            "maximum_judge_calls": total_judge_calls,
            "maximum_counted_live_calls": counted_calls,
            "maximum_cost_before_reserve_usd": _money(base_cost),
            "maximum_cost_with_reserve_usd": _money(cost_with_reserve),
            "approval_cost_cap_usd": _approval_cap(cost_with_reserve),
            "approval_call_cap": counted_calls,
        },
        "by_system": by_system,
        "rows": rows,
    }


def write_flow_plan(plan: Mapping[str, Any], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return target
