from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from openroutiq.router.core import OpenRoutiQError, RiskPolicy, Router


BENCHMARK_SCHEMA_VERSION = 1


class BenchmarkError(OpenRoutiQError):
    """Raised when benchmark data, adapters, or results are invalid."""


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BenchmarkError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise BenchmarkError(f"{name} must be >= {minimum}")
    return result


def _non_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{name} must be non-empty text")
    return value.strip()


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"{name} must be an integer >= 0")
    return value


@dataclass(frozen=True)
class BenchmarkCandidate:
    id: str
    provider: str
    model: str
    reasoning_level: str = "none"
    accuracy: float | None = None
    cost: float | None = None
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    cached_input_price_per_million: float | None = None
    reasoning_price_per_million: float | None = None
    request_price: float = 0.0
    eligible: bool = True
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, location: str) -> BenchmarkCandidate:
        if not isinstance(raw, Mapping):
            raise BenchmarkError(f"{location} must be an object")
        accuracy_value = raw.get("accuracy")
        accuracy = (
            None
            if accuracy_value is None
            else _finite_number(accuracy_value, f"{location}.accuracy", minimum=0)
        )
        if accuracy is not None and accuracy > 1:
            raise BenchmarkError(f"{location}.accuracy must be <= 1")
        cost_value = raw.get("cost")
        cost = (
            None
            if cost_value is None
            else _finite_number(cost_value, f"{location}.cost", minimum=0)
        )
        eligible = raw.get("eligible", True)
        if not isinstance(eligible, bool):
            raise BenchmarkError(f"{location}.eligible must be a boolean")
        aliases = raw.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise BenchmarkError(f"{location}.aliases must be a list of non-empty strings")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise BenchmarkError(f"{location}.metadata must be an object")
        return cls(
            id=_non_empty_text(raw.get("id"), f"{location}.id"),
            provider=_non_empty_text(raw.get("provider", "unknown"), f"{location}.provider"),
            model=_non_empty_text(raw.get("model", raw.get("id")), f"{location}.model"),
            reasoning_level=_non_empty_text(
                raw.get("reasoning_level", "none"), f"{location}.reasoning_level"
            ).lower(),
            accuracy=accuracy,
            cost=cost,
            input_price_per_million=_finite_number(
                raw.get("input_price_per_million", 0),
                f"{location}.input_price_per_million",
                minimum=0,
            ),
            output_price_per_million=_finite_number(
                raw.get("output_price_per_million", 0),
                f"{location}.output_price_per_million",
                minimum=0,
            ),
            cached_input_price_per_million=(
                None
                if raw.get("cached_input_price_per_million") is None
                else _finite_number(
                    raw.get("cached_input_price_per_million"),
                    f"{location}.cached_input_price_per_million",
                    minimum=0,
                )
            ),
            reasoning_price_per_million=(
                None
                if raw.get("reasoning_price_per_million") is None
                else _finite_number(
                    raw.get("reasoning_price_per_million"),
                    f"{location}.reasoning_price_per_million",
                    minimum=0,
                )
            ),
            request_price=_finite_number(
                raw.get("request_price", 0), f"{location}.request_price", minimum=0
            ),
            eligible=eligible,
            aliases=tuple(alias.strip() for alias in aliases),
            metadata=dict(metadata),
        )

    def matches(self, model_id: str) -> bool:
        return model_id == self.id or model_id == self.model or model_id in self.aliases

    def predicted_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> float:
        if self.cost is not None:
            return self.cost
        usage = usage or {}
        prompt_details = usage.get("prompt_tokens_details", {})
        completion_details = usage.get("completion_tokens_details", {})
        if not isinstance(prompt_details, Mapping):
            prompt_details = {}
        if not isinstance(completion_details, Mapping):
            completion_details = {}
        cached_tokens = prompt_details.get("cached_tokens", 0)
        reasoning_tokens = completion_details.get("reasoning_tokens", 0)
        cached_tokens = cached_tokens if isinstance(cached_tokens, int) else 0
        reasoning_tokens = reasoning_tokens if isinstance(reasoning_tokens, int) else 0
        cached_tokens = max(0, min(input_tokens, cached_tokens))
        reasoning_tokens = max(0, min(output_tokens, reasoning_tokens))
        cached_price = (
            self.input_price_per_million
            if self.cached_input_price_per_million is None
            else self.cached_input_price_per_million
        )
        reasoning_price = (
            self.output_price_per_million
            if self.reasoning_price_per_million is None
            else self.reasoning_price_per_million
        )
        return (
            (input_tokens - cached_tokens) * self.input_price_per_million
            + cached_tokens * cached_price
            + (output_tokens - reasoning_tokens) * self.output_price_per_million
            + reasoning_tokens * reasoning_price
        ) / 1_000_000 + self.request_price


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    request: str | Sequence[Any]
    candidates: tuple[BenchmarkCandidate, ...]
    task: str | None = None
    constraints: Mapping[str, Any] = field(default_factory=dict)
    input_tokens: int | None = None
    expected_output_tokens: int | None = None
    evaluation: Mapping[str, Any] = field(default_factory=lambda: {"type": "recorded"})
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int) -> BenchmarkCase:
        location = f"cases[{index}]"
        if not isinstance(raw, Mapping):
            raise BenchmarkError(f"{location} must be an object")
        request = raw.get("request", raw.get("messages"))
        if not isinstance(request, str) and not (
            isinstance(request, Sequence)
            and not isinstance(request, (bytes, bytearray))
            and request
        ):
            raise BenchmarkError(f"{location}.request must be text or a non-empty message list")
        raw_candidates = raw.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise BenchmarkError(f"{location}.candidates must be a non-empty list")
        candidates = tuple(
            BenchmarkCandidate.from_mapping(candidate, location=f"{location}.candidates[{offset}]")
            for offset, candidate in enumerate(raw_candidates)
        )
        ids = [candidate.id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise BenchmarkError(f"{location}.candidates contains duplicate ids")
        if not any(candidate.eligible for candidate in candidates):
            raise BenchmarkError(f"{location} has no eligible candidates")
        constraints = raw.get("constraints", {})
        evaluation = raw.get("evaluation", {"type": "recorded"})
        metadata = raw.get("metadata", {})
        for value, name in (
            (constraints, "constraints"),
            (evaluation, "evaluation"),
            (metadata, "metadata"),
        ):
            if not isinstance(value, Mapping):
                raise BenchmarkError(f"{location}.{name} must be an object")
        task = raw.get("task")
        if task is not None:
            task = _non_empty_text(task, f"{location}.task")
        evaluation_type = _non_empty_text(
            evaluation.get("type", "recorded"), f"{location}.evaluation.type"
        ).lower()
        if evaluation_type == "recorded" and any(
            candidate.accuracy is None for candidate in candidates if candidate.eligible
        ):
            raise BenchmarkError(
                f"{location} recorded evaluation requires accuracy for every eligible candidate"
            )
        return cls(
            id=_non_empty_text(raw.get("id"), f"{location}.id"),
            request=request,
            candidates=candidates,
            task=task,
            constraints=dict(constraints),
            input_tokens=_optional_int(raw.get("input_tokens"), f"{location}.input_tokens"),
            expected_output_tokens=_optional_int(
                raw.get("expected_output_tokens"), f"{location}.expected_output_tokens"
            ),
            evaluation=dict(evaluation),
            metadata=dict(metadata),
        )

    @property
    def eligible_candidates(self) -> tuple[BenchmarkCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.eligible)

    def candidate_for(self, model_id: str) -> BenchmarkCandidate | None:
        matches = [candidate for candidate in self.candidates if candidate.matches(model_id)]
        if len(matches) > 1:
            raise BenchmarkError(f"case {self.id} model {model_id!r} matches multiple candidates")
        return matches[0] if matches else None


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    cases: tuple[BenchmarkCase, ...]
    description: str = ""
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = BENCHMARK_SCHEMA_VERSION
    fingerprint: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BenchmarkDataset:
        if not isinstance(raw, Mapping):
            raise BenchmarkError("benchmark dataset root must be an object")
        version = raw.get("schema_version", BENCHMARK_SCHEMA_VERSION)
        if version != BENCHMARK_SCHEMA_VERSION:
            raise BenchmarkError(
                f"unsupported benchmark schema_version {version}; expected {BENCHMARK_SCHEMA_VERSION}"
            )
        raw_cases = raw.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise BenchmarkError("benchmark dataset must contain a non-empty cases list")
        cases = tuple(
            BenchmarkCase.from_mapping(case, index=index) for index, case in enumerate(raw_cases)
        )
        case_ids = [case.id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise BenchmarkError("benchmark dataset contains duplicate case ids")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise BenchmarkError("benchmark metadata must be an object")
        return cls(
            name=_non_empty_text(raw.get("name"), "name"),
            cases=cases,
            description=str(raw.get("description", "")),
            source=str(raw.get("source", "")),
            metadata=dict(metadata),
            schema_version=version,
            fingerprint=hashlib.sha256(
                json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> BenchmarkDataset:
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise BenchmarkError(f"cannot read benchmark dataset {source}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"invalid JSON in benchmark dataset {source}: {exc}") from exc
        return cls.from_mapping(raw)


@dataclass(frozen=True)
class RouterSelection:
    model_id: str | None
    output: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BenchmarkRouter(Protocol):
    name: str
    is_live: bool

    def prepare(self, dataset: BenchmarkDataset) -> None: ...

    def select(self, case: BenchmarkCase) -> RouterSelection: ...


class OpenRoutiQBenchmarkRouter:
    is_live = False
    timing_scope = "routing"

    def __init__(
        self,
        router: Router,
        *,
        name: str = "openroutiq",
        strategy: str = "auto",
        risk_policy: RiskPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        self.router = router
        self.name = name
        self.strategy = strategy
        self.risk_policy = RiskPolicy.parse(risk_policy) if risk_policy is not None else None

    def prepare(self, dataset: BenchmarkDataset) -> None:
        del dataset

    def select(self, case: BenchmarkCase) -> RouterSelection:
        constraints = dict(case.constraints)
        case_ids = {candidate.id for candidate in case.eligible_candidates}
        configured_ids = set(constraints.get("candidate_ids", []))
        constraints["candidate_ids"] = sorted(
            case_ids & configured_ids if configured_ids else case_ids
        )
        decision = self.router.route(
            case.request,
            task=case.task,
            constraints=constraints,
            input_tokens=case.input_tokens,
            expected_output_tokens=case.expected_output_tokens,
            # Recorded matrices can contain aggregate token usage across an
            # agent trajectory. Candidate presence proves source eligibility;
            # aggregate billing tokens must not be mistaken for one context
            # window. Explicit case constraints still apply.
            required_context_tokens=(
                0 if str(case.evaluation.get("type", "recorded")).lower() == "recorded" else None
            ),
            strategy=self.strategy,
            risk_policy=self.risk_policy,
        )
        return RouterSelection(
            model_id=decision.selected.model_id,
            metadata={
                "review_required": decision.review_required,
                "score": decision.selected.total_score,
                # quality_score is the training-derived probability-like estimate
                # that can be compared with held-out accuracy. Model-profile
                # confidence measures evidence coverage and is not a correctness
                # probability, so keep it separate from calibration inputs.
                "predicted_accuracy": decision.selected.quality_score / 100,
                "catalog_confidence": decision.selected.confidence / 100,
                "cvar_loss": (
                    None
                    if decision.selected.forecast is None
                    else decision.selected.forecast.cvar_loss
                ),
            },
        )


class RandomBenchmarkRouter:
    is_live = False
    timing_scope = "routing"

    def __init__(self, *, seed: int = 3407, name: str = "random") -> None:
        self.seed = seed
        self.name = name

    def prepare(self, dataset: BenchmarkDataset) -> None:
        del dataset

    def select(self, case: BenchmarkCase) -> RouterSelection:
        digest = hashlib.sha256(f"{self.seed}:{case.id}".encode()).digest()
        rng = random.Random(digest)
        return RouterSelection(model_id=rng.choice(case.eligible_candidates).id)


class OracleBenchmarkRouter:
    is_live = False
    timing_scope = "routing"

    def __init__(self, *, name: str = "oracle") -> None:
        self.name = name

    def prepare(self, dataset: BenchmarkDataset) -> None:
        del dataset

    def select(self, case: BenchmarkCase) -> RouterSelection:
        candidates = case.eligible_candidates
        if any(candidate.accuracy is None for candidate in candidates):
            raise BenchmarkError("oracle requires recorded candidate accuracy")

        def oracle_key(candidate: BenchmarkCandidate) -> tuple[float, float, str]:
            if candidate.accuracy is None:
                raise BenchmarkError("oracle requires recorded candidate accuracy")
            return (
                -float(candidate.accuracy),
                candidate.predicted_cost(
                    case.input_tokens or 0,
                    case.expected_output_tokens or 0,
                ),
                candidate.id,
            )

        selected = min(
            candidates,
            key=oracle_key,
        )
        return RouterSelection(model_id=selected.id)


class BestSingleBenchmarkRouter:
    is_live = False
    timing_scope = "routing"

    def __init__(self, *, model_id: str | None = None, name: str = "best-single") -> None:
        self.name = name
        self.model_id = model_id

    def prepare(self, dataset: BenchmarkDataset) -> None:
        if self.model_id is not None:
            if not any(
                candidate.id == self.model_id
                for case in dataset.cases
                for candidate in case.eligible_candidates
            ):
                raise BenchmarkError(f"best-single model is absent: {self.model_id}")
            return
        scores: dict[str, list[float]] = defaultdict(list)
        for case in dataset.cases:
            for candidate in case.eligible_candidates:
                if candidate.accuracy is not None:
                    scores[candidate.id].append(candidate.accuracy)
        if not scores:
            raise BenchmarkError("best-single requires recorded candidate accuracy")
        self.model_id = min(
            scores,
            key=lambda model_id: (-statistics.fmean(scores[model_id]), model_id),
        )

    def select(self, case: BenchmarkCase) -> RouterSelection:
        del case
        return RouterSelection(model_id=self.model_id)


class FixedModelBenchmarkRouter:
    """Route every case to one predeclared model variant."""

    is_live = False
    timing_scope = "routing"

    def __init__(self, model_id: str, *, name: str | None = None) -> None:
        self.model_id = _non_empty_text(model_id, "model_id")
        self.name = name or f"Fixed: {self.model_id}"

    def prepare(self, dataset: BenchmarkDataset) -> None:
        if not any(
            candidate.id == self.model_id
            for case in dataset.cases
            for candidate in case.eligible_candidates
        ):
            raise BenchmarkError(f"fixed model is absent: {self.model_id}")

    def select(self, case: BenchmarkCase) -> RouterSelection:
        del case
        return RouterSelection(model_id=self.model_id)


@dataclass(frozen=True)
class OutcomeMatrixObservation:
    """One measured request/model cell in a fully paired selection benchmark."""

    request_id: str
    model_id: str
    success: bool
    quality_score: float
    cost_usd: float
    total_latency_ms: float
    segment: str = "unknown"
    ttft_ms: float | None = None
    tool_success: bool | None = None
    framework_success: bool | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        _non_empty_text(self.request_id, "outcome.request_id")
        _non_empty_text(self.model_id, "outcome.model_id")
        _non_empty_text(self.segment, "outcome.segment")
        if not isinstance(self.success, bool):
            raise BenchmarkError("outcome.success must be a boolean")
        quality = _finite_number(self.quality_score, "outcome.quality_score", minimum=0)
        if quality > 1:
            raise BenchmarkError("outcome.quality_score must be <= 1")
        _finite_number(self.cost_usd, "outcome.cost_usd", minimum=0)
        _finite_number(self.total_latency_ms, "outcome.total_latency_ms", minimum=0)
        if self.ttft_ms is not None:
            _finite_number(self.ttft_ms, "outcome.ttft_ms", minimum=0)
        for name, value in (
            ("tool_success", self.tool_success),
            ("framework_success", self.framework_success),
        ):
            if value is not None and not isinstance(value, bool):
                raise BenchmarkError(f"outcome.{name} must be a boolean or null")
        if self.error_type is not None:
            _non_empty_text(self.error_type, "outcome.error_type")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _matrix_slice(
    by_request: Mapping[str, Mapping[str, OutcomeMatrixObservation]],
    request_ids: Sequence[str],
    model_ids: Sequence[str],
) -> dict[str, Any]:
    model_accuracy = {
        model_id: statistics.fmean(
            by_request[request_id][model_id].quality_score for request_id in request_ids
        )
        for model_id in model_ids
    }
    model_success = {
        model_id: statistics.fmean(
            by_request[request_id][model_id].success for request_id in request_ids
        )
        for model_id in model_ids
    }
    best_accuracy = max(model_accuracy.values())
    best_success = max(model_success.values())
    oracle_accuracy = statistics.fmean(
        max(cell.quality_score for cell in by_request[request_id].values())
        for request_id in request_ids
    )
    oracle_success = statistics.fmean(
        any(cell.success for cell in by_request[request_id].values()) for request_id in request_ids
    )
    return {
        "cases": len(request_ids),
        "best_fixed_accuracy": best_accuracy,
        "best_fixed_model_ids": [
            model_id
            for model_id, accuracy in model_accuracy.items()
            if math.isclose(accuracy, best_accuracy, rel_tol=0, abs_tol=1e-12)
        ],
        "oracle_accuracy": oracle_accuracy,
        "oracle_headroom": oracle_accuracy - best_accuracy,
        "best_fixed_success_rate": best_success,
        "best_fixed_success_model_ids": [
            model_id
            for model_id, success in model_success.items()
            if math.isclose(success, best_success, rel_tol=0, abs_tol=1e-12)
        ],
        "oracle_success_rate": oracle_success,
        "oracle_success_headroom": oracle_success - best_success,
        "all_models_failed_cases": sum(
            not any(cell.success for cell in by_request[request_id].values())
            for request_id in request_ids
        ),
    }


def analyze_outcome_matrix(
    outcomes: Sequence[OutcomeMatrixObservation],
    *,
    router_accuracies: Mapping[str, float] | None = None,
    actionable_headroom: float = 0.05,
) -> dict[str, Any]:
    """Analyze oracle headroom and model complementarity on a complete matrix.

    The function deliberately rejects incomplete or duplicate matrices. A router upper
    bound is meaningful only when every candidate model was measured on every request.
    """

    cells = tuple(outcomes)
    if not cells:
        raise BenchmarkError("outcome matrix must contain observations")
    threshold = _finite_number(
        actionable_headroom,
        "actionable_headroom",
        minimum=0,
    )
    if threshold > 1:
        raise BenchmarkError("actionable_headroom must be <= 1")
    model_ids = sorted({cell.model_id for cell in cells})
    request_ids = sorted({cell.request_id for cell in cells})
    if len(model_ids) < 2:
        raise BenchmarkError("outcome matrix requires at least two models")

    by_request: dict[str, dict[str, OutcomeMatrixObservation]] = defaultdict(dict)
    for cell in cells:
        if cell.model_id in by_request[cell.request_id]:
            raise BenchmarkError(
                f"outcome matrix contains duplicate cell: {cell.request_id} / {cell.model_id}"
            )
        by_request[cell.request_id][cell.model_id] = cell
    missing = [
        f"{request_id} / {model_id}"
        for request_id in request_ids
        for model_id in model_ids
        if model_id not in by_request[request_id]
    ]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise BenchmarkError(
            f"outcome matrix is incomplete; missing {len(missing)} cell(s): {preview}{suffix}"
        )

    request_segments: dict[str, str] = {}
    for request_id, model_cells in by_request.items():
        request_segment_values = {cell.segment for cell in model_cells.values()}
        if len(request_segment_values) != 1:
            raise BenchmarkError(f"outcome matrix request has inconsistent segments: {request_id}")
        request_segments[request_id] = next(iter(request_segment_values))

    model_metrics: list[dict[str, Any]] = []
    for model_id in model_ids:
        measured = [by_request[request_id][model_id] for request_id in request_ids]
        unique_successes = sum(
            cell.success
            and all(
                not other.success
                for other_id, other in by_request[cell.request_id].items()
                if other_id != model_id
            )
            for cell in measured
        )
        strict_quality_wins = sum(
            cell.quality_score
            > max(
                other.quality_score
                for other_id, other in by_request[cell.request_id].items()
                if other_id != model_id
            )
            for cell in measured
        )
        ttft = [cell.ttft_ms for cell in measured if cell.ttft_ms is not None]
        tool = [cell.tool_success for cell in measured if cell.tool_success is not None]
        framework = [
            cell.framework_success for cell in measured if cell.framework_success is not None
        ]
        errors: dict[str, int] = defaultdict(int)
        for cell in measured:
            if cell.error_type is not None:
                errors[cell.error_type] += 1
        model_metrics.append(
            {
                "model_id": model_id,
                "cases": len(measured),
                "accuracy": statistics.fmean(cell.quality_score for cell in measured),
                "success_rate": statistics.fmean(cell.success for cell in measured),
                "mean_cost_usd": statistics.fmean(cell.cost_usd for cell in measured),
                "median_total_latency_ms": statistics.median(
                    cell.total_latency_ms for cell in measured
                ),
                "p95_total_latency_ms": _percentile(
                    [cell.total_latency_ms for cell in measured], 0.95
                ),
                "mean_ttft_ms": statistics.fmean(ttft) if ttft else None,
                "tool_success_rate": statistics.fmean(tool) if tool else None,
                "framework_success_rate": (statistics.fmean(framework) if framework else None),
                "unique_successes": unique_successes,
                "unique_success_rate": unique_successes / len(measured),
                "strict_quality_wins": strict_quality_wins,
                "strict_quality_win_rate": strict_quality_wins / len(measured),
                "error_types": dict(sorted(errors.items())),
            }
        )

    overall = _matrix_slice(by_request, request_ids, model_ids)
    successful_model_counts = [
        sum(cell.success for cell in by_request[request_id].values()) for request_id in request_ids
    ]
    oracle_selections = [
        min(
            by_request[request_id].values(),
            key=lambda cell: (
                -cell.quality_score,
                cell.cost_usd,
                cell.total_latency_ms,
                cell.model_id,
            ),
        )
        for request_id in request_ids
    ]
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(model_ids):
        for right in model_ids[left_index + 1 :]:
            both = sum(
                by_request[request_id][left].success and by_request[request_id][right].success
                for request_id in request_ids
            )
            left_only = sum(
                by_request[request_id][left].success and not by_request[request_id][right].success
                for request_id in request_ids
            )
            right_only = sum(
                by_request[request_id][right].success and not by_request[request_id][left].success
                for request_id in request_ids
            )
            neither = len(request_ids) - both - left_only - right_only
            union = both + left_only + right_only
            pairs.append(
                {
                    "left_model_id": left,
                    "right_model_id": right,
                    "cases": len(request_ids),
                    "both_succeeded": both,
                    "left_only_succeeded": left_only,
                    "right_only_succeeded": right_only,
                    "neither_succeeded": neither,
                    "disagreement_cases": left_only + right_only,
                    "disagreement_rate": (left_only + right_only) / len(request_ids),
                    "success_jaccard": both / union if union else None,
                }
            )

    segment_metrics = []
    for segment in sorted(set(request_segments.values())):
        segment_requests = [
            request_id for request_id in request_ids if request_segments[request_id] == segment
        ]
        segment_metrics.append(
            {"segment": segment, **_matrix_slice(by_request, segment_requests, model_ids)}
        )
    segment_selector_accuracy = sum(
        float(item["best_fixed_accuracy"]) * int(item["cases"]) for item in segment_metrics
    ) / len(request_ids)
    segment_selector_success = sum(
        float(item["best_fixed_success_rate"]) * int(item["cases"]) for item in segment_metrics
    ) / len(request_ids)
    segment_captured = segment_selector_accuracy - float(overall["best_fixed_accuracy"])
    oracle_headroom = float(overall["oracle_headroom"])

    routers = []
    for name, raw_accuracy in sorted((router_accuracies or {}).items()):
        accuracy = _finite_number(raw_accuracy, f"router_accuracies[{name!r}]", minimum=0)
        if accuracy > 1:
            raise BenchmarkError(f"router_accuracies[{name!r}] must be <= 1")
        captured = accuracy - float(overall["best_fixed_accuracy"])
        headroom = float(overall["oracle_headroom"])
        routers.append(
            {
                "router": _non_empty_text(name, "router name"),
                "accuracy": accuracy,
                "captured_headroom": captured,
                "routing_efficiency": captured / headroom if headroom > 1e-12 else None,
            }
        )

    return {
        "schema_version": 1,
        "matrix": {
            "cases": len(request_ids),
            "models": len(model_ids),
            "observations": len(cells),
            "complete": True,
            "model_ids": model_ids,
        },
        "model_metrics": model_metrics,
        "best_fixed_accuracy": overall["best_fixed_accuracy"],
        "best_fixed_model_ids": overall["best_fixed_model_ids"],
        "oracle_accuracy": overall["oracle_accuracy"],
        "oracle_headroom": overall["oracle_headroom"],
        "best_fixed_success_rate": overall["best_fixed_success_rate"],
        "best_fixed_success_model_ids": overall["best_fixed_success_model_ids"],
        "oracle_success_rate": overall["oracle_success_rate"],
        "oracle_success_headroom": overall["oracle_success_headroom"],
        "headroom_assessment": {
            "actionable_threshold": threshold,
            "status": (
                "router_opportunity"
                if float(overall["oracle_headroom"]) >= threshold
                else "portfolio_limited"
            ),
        },
        "oracle_policy": {
            "selection": "maximum observed quality, then lowest cost, latency, and model ID",
            "mean_cost_usd": statistics.fmean(cell.cost_usd for cell in oracle_selections),
            "median_total_latency_ms": statistics.median(
                cell.total_latency_ms for cell in oracle_selections
            ),
            "p95_total_latency_ms": _percentile(
                [cell.total_latency_ms for cell in oracle_selections], 0.95
            ),
        },
        "complementarity": {
            "all_models_succeeded": successful_model_counts.count(len(model_ids)),
            "some_but_not_all_succeeded": sum(
                0 < count < len(model_ids) for count in successful_model_counts
            ),
            "exactly_one_model_succeeded": successful_model_counts.count(1),
            "all_models_failed": successful_model_counts.count(0),
            "pairwise_overlap": pairs,
        },
        "router_metrics": routers,
        "diagnostic_segment_selector": {
            "method": "choose one hindsight-best fixed model per declared segment",
            "accuracy": segment_selector_accuracy,
            "captured_headroom": segment_captured,
            "routing_efficiency": (
                segment_captured / oracle_headroom if oracle_headroom > 1e-12 else None
            ),
            "residual_within_segment_oracle_headroom": (
                float(overall["oracle_accuracy"]) - segment_selector_accuracy
            ),
            "success_rate": segment_selector_success,
            "residual_within_segment_success_headroom": (
                float(overall["oracle_success_rate"]) - segment_selector_success
            ),
        },
        "segments": segment_metrics,
        "definitions": {
            "oracle_accuracy": "mean best observed quality score across models for each request",
            "oracle_headroom": "oracle accuracy minus best fixed-model accuracy",
            "captured_headroom": "router accuracy minus best fixed-model accuracy",
            "routing_efficiency": "captured headroom divided by oracle headroom",
            "diagnostic_segment_selector": "one hindsight-best fixed model per declared segment; an upper-bound diagnostic, not held-out router evidence",
            "unique_success": "model succeeded on a request where every other candidate failed",
        },
    }


@dataclass(frozen=True)
class BenchmarkObservation:
    router: str
    case_id: str
    task: str
    model_id: str | None
    provider: str | None
    reasoning_level: str | None
    accuracy: float
    cost: float
    routing_ms: float
    covered: bool
    constraint_violation: bool
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BenchmarkObservation:
        return cls(
            router=_non_empty_text(raw.get("router"), "observation.router"),
            case_id=_non_empty_text(raw.get("case_id"), "observation.case_id"),
            task=_non_empty_text(raw.get("task", "unknown"), "observation.task"),
            model_id=raw.get("model_id"),
            provider=raw.get("provider"),
            reasoning_level=raw.get("reasoning_level"),
            accuracy=_finite_number(raw.get("accuracy"), "observation.accuracy", minimum=0),
            cost=_finite_number(raw.get("cost"), "observation.cost", minimum=0),
            routing_ms=_finite_number(raw.get("routing_ms"), "observation.routing_ms", minimum=0),
            covered=bool(raw.get("covered")),
            constraint_violation=bool(raw.get("constraint_violation")),
            error=raw.get("error"),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "router": self.router,
            "case_id": self.case_id,
            "task": self.task,
            "model_id": self.model_id,
            "provider": self.provider,
            "reasoning_level": self.reasoning_level,
            "accuracy": self.accuracy,
            "cost": self.cost,
            "routing_ms": self.routing_ms,
            "covered": self.covered,
            "constraint_violation": self.constraint_violation,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkSummary:
    router: str
    cases: int
    accuracy: float
    total_cost: float
    mean_cost: float
    cost_per_correct: float | None
    accuracy_per_dollar: float | None
    coverage: float
    constraint_violation_rate: float
    failure_rate: float
    routing_p50_ms: float
    routing_p95_ms: float
    selection_stability: float
    calibration_error: float | None
    providers_used: int
    models_used: int
    pareto_optimal: bool = False
    pareto_distance: float = 0.0
    observations: int = 0
    accuracy_ci_low: float = 0.0
    accuracy_ci_high: float = 0.0
    mean_cost_ci_low: float = 0.0
    mean_cost_ci_high: float = 0.0
    timing_scope: str = "routing"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BenchmarkSummary:
        values = dict(raw)
        values.setdefault("observations", values.get("cases", 0))
        values.setdefault("accuracy_ci_low", values.get("accuracy", 0.0))
        values.setdefault("accuracy_ci_high", values.get("accuracy", 0.0))
        values.setdefault("mean_cost_ci_low", values.get("mean_cost", 0.0))
        values.setdefault("mean_cost_ci_high", values.get("mean_cost", 0.0))
        values.setdefault("timing_scope", "routing")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class BenchmarkRun:
    dataset: str
    created_at: str
    summaries: tuple[BenchmarkSummary, ...]
    observations: tuple[BenchmarkObservation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = BENCHMARK_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BenchmarkRun:
        if raw.get("schema_version", BENCHMARK_SCHEMA_VERSION) != BENCHMARK_SCHEMA_VERSION:
            raise BenchmarkError("unsupported benchmark result schema_version")
        summaries = raw.get("summaries")
        observations = raw.get("observations")
        if not isinstance(summaries, list) or not isinstance(observations, list):
            raise BenchmarkError("benchmark results require summaries and observations lists")
        return cls(
            dataset=_non_empty_text(raw.get("dataset"), "dataset"),
            created_at=_non_empty_text(raw.get("created_at"), "created_at"),
            summaries=tuple(BenchmarkSummary.from_mapping(item) for item in summaries),
            observations=tuple(BenchmarkObservation.from_mapping(item) for item in observations),
            metadata=dict(raw.get("metadata", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> BenchmarkRun:
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise BenchmarkError(f"cannot read benchmark results {source}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"invalid JSON in benchmark results {source}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise BenchmarkError("benchmark result root must be an object")
        return cls.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary_dict(),
            "observations": [observation.to_dict() for observation in self.observations],
        }

    def summary_dict(self) -> dict[str, Any]:
        """Return the compact, observation-free portion of a benchmark run."""
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "summaries": [summary.to_dict() for summary in self.summaries],
        }

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "benchmark-results.json"
        summary_path = target / "benchmark-summary.json"
        csv_path = target / "benchmark-observations.csv"
        json_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        summary_path.write_text(json.dumps(self.summary_dict(), indent=2) + "\n", encoding="utf-8")
        fields = [
            "router",
            "case_id",
            "task",
            "model_id",
            "provider",
            "reasoning_level",
            "accuracy",
            "cost",
            "routing_ms",
            "covered",
            "constraint_violation",
            "error",
            "predicted_accuracy",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for observation in self.observations:
                flat = observation.to_dict()
                predicted_accuracy = observation.metadata.get(
                    "predicted_accuracy", observation.metadata.get("confidence")
                )
                writer.writerow(
                    {
                        **{key: flat[key] for key in fields if key != "predicted_accuracy"},
                        "predicted_accuracy": (
                            predicted_accuracy
                            if isinstance(predicted_accuracy, (int, float))
                            and not isinstance(predicted_accuracy, bool)
                            else ""
                        ),
                    }
                )
        return json_path, csv_path


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _mean_ci(
    values: Sequence[float], *, minimum: float = 0.0, maximum: float | None = None
) -> tuple[float, float]:
    """Approximate 95% confidence interval around a case-level mean."""
    if not values:
        return minimum, minimum
    mean = statistics.fmean(values)
    margin = 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    low = max(minimum, mean - margin)
    high = mean + margin if maximum is None else min(maximum, mean + margin)
    return low, high


def _expected_calibration_error(
    pairs: Sequence[tuple[float, float]], bins: int = 10
) -> float | None:
    """Return equal-width expected calibration error for prediction/outcome pairs."""
    if not pairs:
        return None
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for predicted, observed in pairs:
        grouped[min(bins - 1, int(predicted * bins))].append((predicted, observed))
    return sum(
        len(items)
        / len(pairs)
        * abs(
            statistics.fmean(predicted for predicted, _ in items)
            - statistics.fmean(observed for _, observed in items)
        )
        for items in grouped.values()
    )


def _score_output(case: BenchmarkCase, output: str | None) -> float:
    evaluation = case.evaluation
    kind = str(evaluation.get("type", "recorded")).lower()
    if kind == "recorded":
        raise BenchmarkError("recorded accuracy is resolved from the selected candidate")
    if output is None:
        return 0.0
    expected = evaluation.get("expected")
    if kind == "exact_match":
        if not isinstance(expected, str):
            raise BenchmarkError(f"case {case.id} exact_match requires evaluation.expected")

        def normalizer(value: str) -> str:
            return " ".join(value.casefold().split())

        return float(normalizer(output) == normalizer(expected))
    if kind == "contains":
        if not isinstance(expected, str):
            raise BenchmarkError(f"case {case.id} contains requires evaluation.expected")
        return float(expected.casefold() in output.casefold())
    if kind == "regex":
        pattern = evaluation.get("pattern")
        if not isinstance(pattern, str):
            raise BenchmarkError(f"case {case.id} regex requires evaluation.pattern")
        return float(re.search(pattern, output, flags=re.IGNORECASE | re.MULTILINE) is not None)
    if kind == "numeric":
        tolerance = _finite_number(
            evaluation.get("tolerance", 0), "evaluation.tolerance", minimum=0
        )
        target = _finite_number(expected, "evaluation.expected")
        match = re.search(r"[-+]?\d+(?:\.\d+)?", output.replace(",", ""))
        return float(match is not None and abs(float(match.group()) - target) <= tolerance)
    if kind == "json_equals":
        try:
            return float(json.loads(output) == expected)
        except json.JSONDecodeError:
            return 0.0
    raise BenchmarkError(f"case {case.id} has unsupported evaluation type: {kind}")


def _observation(
    case: BenchmarkCase, router_name: str, selection: RouterSelection, elapsed: float
) -> BenchmarkObservation:
    observation_metadata = {
        "case_metadata": dict(case.metadata),
        "input_tokens": case.input_tokens,
        "expected_output_tokens": case.expected_output_tokens,
        **dict(selection.metadata),
    }
    candidate = case.candidate_for(selection.model_id) if selection.model_id is not None else None
    covered = candidate is not None
    violation = candidate is not None and not candidate.eligible
    if candidate is None:
        return BenchmarkObservation(
            router=router_name,
            case_id=case.id,
            task=case.task or str(case.metadata.get("task", "unknown")),
            model_id=selection.model_id,
            provider=None,
            reasoning_level=None,
            accuracy=0.0,
            cost=selection.cost or 0.0,
            routing_ms=elapsed,
            covered=False,
            constraint_violation=False,
            error="router returned no matching candidate",
            metadata=observation_metadata,
        )
    input_tokens = (
        selection.prompt_tokens if selection.prompt_tokens is not None else case.input_tokens or 0
    )
    output_tokens = (
        selection.completion_tokens
        if selection.completion_tokens is not None
        else case.expected_output_tokens or 0
    )
    recorded = str(case.evaluation.get("type", "recorded")).lower() == "recorded"
    if violation:
        accuracy = 0.0
    elif recorded:
        if candidate.accuracy is None:
            raise BenchmarkError(
                f"recorded case {case.id} candidate {candidate.id} has no accuracy"
            )
        accuracy = float(candidate.accuracy)
    else:
        accuracy = _score_output(case, selection.output)
    return BenchmarkObservation(
        router=router_name,
        case_id=case.id,
        task=case.task or str(case.metadata.get("task", "unknown")),
        model_id=candidate.id,
        provider=candidate.provider,
        reasoning_level=candidate.reasoning_level,
        accuracy=accuracy,
        cost=(
            candidate.predicted_cost(input_tokens, output_tokens)
            if recorded
            else selection.cost
            if selection.cost is not None
            else candidate.predicted_cost(
                input_tokens,
                output_tokens,
                usage=(
                    selection.metadata.get("usage")
                    if isinstance(selection.metadata.get("usage"), Mapping)
                    else None
                ),
            )
        ),
        routing_ms=elapsed,
        covered=covered,
        constraint_violation=violation,
        metadata=observation_metadata,
    )


def summarize(observations: Iterable[BenchmarkObservation]) -> tuple[BenchmarkSummary, ...]:
    grouped: dict[str, list[BenchmarkObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.router].append(observation)
    provisional: list[BenchmarkSummary] = []
    for router_name, rows in sorted(grouped.items()):
        rows_by_case: dict[str, list[BenchmarkObservation]] = defaultdict(list)
        for row in rows:
            rows_by_case[row.case_id].append(row)
        cases = len(rows_by_case)
        observation_count = len(rows)
        case_accuracy = [
            statistics.fmean(row.accuracy for row in case_rows)
            for case_rows in rows_by_case.values()
        ]
        case_cost = [
            statistics.fmean(row.cost for row in case_rows) for case_rows in rows_by_case.values()
        ]
        accuracy_sum = sum(row.accuracy for row in rows)
        total_cost = sum(row.cost for row in rows)
        accuracy = statistics.fmean(case_accuracy)
        mean_cost = statistics.fmean(case_cost)
        accuracy_ci = _mean_ci(case_accuracy, maximum=1.0)
        cost_ci = _mean_ci(case_cost)
        failures = sum(row.error is not None for row in rows)
        selections_by_case: dict[str, list[str | None]] = defaultdict(list)
        for row in rows:
            selections_by_case[row.case_id].append(row.model_id)
        stability = statistics.fmean(
            max(values.count(model_id) for model_id in set(values)) / len(values)
            for values in selections_by_case.values()
        )
        calibration_pairs: list[tuple[float, float]] = []
        for row in rows:
            predicted_accuracy = row.metadata.get(
                "predicted_accuracy", row.metadata.get("confidence")
            )
            if (
                isinstance(predicted_accuracy, (int, float))
                and not isinstance(predicted_accuracy, bool)
                and 0 <= float(predicted_accuracy) <= 1
            ):
                calibration_pairs.append((float(predicted_accuracy), row.accuracy))
        provisional.append(
            BenchmarkSummary(
                router=router_name,
                cases=cases,
                accuracy=accuracy,
                total_cost=total_cost,
                mean_cost=mean_cost,
                cost_per_correct=total_cost / accuracy_sum if accuracy_sum > 0 else None,
                accuracy_per_dollar=accuracy_sum / total_cost if total_cost > 0 else None,
                coverage=sum(row.covered for row in rows) / observation_count,
                constraint_violation_rate=(
                    sum(row.constraint_violation for row in rows) / observation_count
                ),
                failure_rate=failures / observation_count,
                routing_p50_ms=_percentile([row.routing_ms for row in rows], 0.5),
                routing_p95_ms=_percentile([row.routing_ms for row in rows], 0.95),
                selection_stability=stability,
                calibration_error=_expected_calibration_error(calibration_pairs),
                providers_used=len({row.provider for row in rows if row.provider is not None}),
                models_used=len({row.model_id for row in rows if row.model_id is not None}),
                observations=observation_count,
                accuracy_ci_low=accuracy_ci[0],
                accuracy_ci_high=accuracy_ci[1],
                mean_cost_ci_low=cost_ci[0],
                mean_cost_ci_high=cost_ci[1],
                timing_scope=(
                    next(iter({str(row.metadata.get("timing_scope", "routing")) for row in rows}))
                    if len({str(row.metadata.get("timing_scope", "routing")) for row in rows}) == 1
                    else "mixed"
                ),
            )
        )
    accuracy_values = [item.accuracy for item in provisional]
    cost_values = [item.mean_cost for item in provisional]
    accuracy_span = max(accuracy_values, default=1) - min(accuracy_values, default=0)
    cost_span = max(cost_values, default=1) - min(cost_values, default=0)
    final: list[BenchmarkSummary] = []
    for item in provisional:
        dominated = any(
            other.router != item.router
            and other.accuracy >= item.accuracy
            and other.mean_cost <= item.mean_cost
            and (other.accuracy > item.accuracy or other.mean_cost < item.mean_cost)
            for other in provisional
        )
        frontier = [
            other
            for other in provisional
            if not any(
                candidate.router != other.router
                and candidate.accuracy >= other.accuracy
                and candidate.mean_cost <= other.mean_cost
                and (candidate.accuracy > other.accuracy or candidate.mean_cost < other.mean_cost)
                for candidate in provisional
            )
        ]
        distance = min(
            math.sqrt(
                ((other.accuracy - item.accuracy) / (accuracy_span or 1)) ** 2
                + ((item.mean_cost - other.mean_cost) / (cost_span or 1)) ** 2
            )
            for other in frontier
        )
        final.append(
            BenchmarkSummary(
                **{
                    **item.__dict__,
                    "pareto_optimal": not dominated,
                    "pareto_distance": distance,
                }
            )
        )
    return tuple(final)


def run_benchmark(
    dataset: BenchmarkDataset,
    routers: Sequence[BenchmarkRouter],
    *,
    allow_live: bool = False,
    repetitions: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> BenchmarkRun:
    """Run a benchmark. Callers must explicitly opt in before any live router is invoked."""
    if not routers:
        raise BenchmarkError("at least one benchmark router is required")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise BenchmarkError("repetitions must be an integer >= 1")
    names = [router.name for router in routers]
    if len(names) != len(set(names)):
        raise BenchmarkError("benchmark router names must be unique")
    live = [router.name for router in routers if router.is_live]
    if live and not allow_live:
        raise BenchmarkError(
            "live benchmark execution is disabled; explicit approval is required for: "
            + ", ".join(live)
        )
    observations: list[BenchmarkObservation] = []
    preparation_ms: dict[str, float] = {}
    for benchmark_router in routers:
        prepare_started = time.perf_counter()
        benchmark_router.prepare(dataset)
        preparation_ms[benchmark_router.name] = (time.perf_counter() - prepare_started) * 1000
        for case in dataset.cases:
            for repeat_index in range(repetitions):
                started = time.perf_counter()
                try:
                    selection = benchmark_router.select(case)
                    selection = RouterSelection(
                        **{
                            **selection.__dict__,
                            "metadata": {
                                **dict(selection.metadata),
                                "repeat_index": repeat_index,
                                "timing_scope": getattr(
                                    benchmark_router, "timing_scope", "routing"
                                ),
                            },
                        }
                    )
                    elapsed = (time.perf_counter() - started) * 1000
                    observations.append(
                        _observation(case, benchmark_router.name, selection, elapsed)
                    )
                except Exception as exc:
                    elapsed = (time.perf_counter() - started) * 1000
                    observations.append(
                        BenchmarkObservation(
                            router=benchmark_router.name,
                            case_id=case.id,
                            task=case.task or str(case.metadata.get("task", "unknown")),
                            model_id=None,
                            provider=None,
                            reasoning_level=None,
                            accuracy=0.0,
                            cost=0.0,
                            routing_ms=elapsed,
                            covered=False,
                            constraint_violation=False,
                            error=f"{type(exc).__name__}: {exc}",
                            metadata={
                                "repeat_index": repeat_index,
                                "timing_scope": getattr(
                                    benchmark_router, "timing_scope", "routing"
                                ),
                            },
                        )
                    )
    from datetime import UTC, datetime

    return BenchmarkRun(
        dataset=dataset.name,
        created_at=datetime.now(UTC).isoformat(),
        summaries=summarize(observations),
        observations=tuple(observations),
        metadata={
            "dataset_fingerprint": dataset.fingerprint,
            "preparation_ms": preparation_ms,
            **dict(metadata or {}),
        },
    )
