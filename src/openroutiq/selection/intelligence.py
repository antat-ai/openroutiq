from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any

from openroutiq.router.core import (
    CandidateScore,
    OpenRoutiQError,
    ModelProfile,
    RouteContext,
    RouteDecision,
    Router,
    analyze_context,
)


SELECTION_STATE_SCHEMA_VERSION = 1
_TOKEN = re.compile(r"[a-z0-9_./:+-]{2,}")
_MAXIMUM_STATE_BYTES = 64 * 1024 * 1024


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenRoutiQError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise OpenRoutiQError(f"{name} must be between 0 and 1")
    return result


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenRoutiQError(f"{name} must be a number")
    result = float(value)
    minimum_ok = result >= 0 if allow_zero else result > 0
    if not math.isfinite(result) or not minimum_ok:
        comparator = ">= 0" if allow_zero else "> 0"
        raise OpenRoutiQError(f"{name} must be finite and {comparator}")
    return result


def _logit(probability: float) -> float:
    bounded = min(1 - 1e-5, max(1e-5, probability))
    return math.log(bounded / (1 - bounded))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-min(value, 60))
        return 1 / (1 + exponent)
    exponent = math.exp(max(value, -60))
    return exponent / (1 + exponent)


@dataclass(frozen=True)
class SelectionPolicy:
    """Utility and exploration policy layered over calibrated component predictions."""

    success_weight: float = 0.55
    quality_weight: float = 0.45
    cost_weight: float = 0.05
    latency_weight: float = 0.03
    risk_aversion: float = 0.05
    exploration_rate: float = 0.0
    minimum_learning_samples: int = 20

    def __post_init__(self) -> None:
        for name in (
            "success_weight",
            "quality_weight",
            "cost_weight",
            "latency_weight",
            "risk_aversion",
        ):
            _positive(getattr(self, name), name, allow_zero=True)
        if self.success_weight + self.quality_weight <= 0:
            raise OpenRoutiQError("success_weight and quality_weight cannot both be zero")
        _probability(self.exploration_rate, "exploration_rate")
        if (
            isinstance(self.minimum_learning_samples, bool)
            or not isinstance(self.minimum_learning_samples, int)
            or self.minimum_learning_samples < 1
        ):
            raise OpenRoutiQError("minimum_learning_samples must be an integer >= 1")


@dataclass(frozen=True)
class SelectionTrainingObservation:
    request: str | Sequence[Any] | RouteContext
    model_id: str
    success: bool
    quality_score: float
    task: str = "general"
    actual_cost_usd: float | None = None
    latency_ms: float | None = None
    selection_probability: float = 1.0
    tools: Sequence[Any] | None = None
    response_format: Mapping[str, Any] | None = None
    stream: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise OpenRoutiQError("model_id must be non-empty text")
        if not isinstance(self.task, str) or not self.task.strip():
            raise OpenRoutiQError("task must be non-empty text")
        if not isinstance(self.success, bool):
            raise OpenRoutiQError("success must be a boolean")
        _positive(self.quality_score, "quality_score", allow_zero=True)
        if self.quality_score > 100:
            raise OpenRoutiQError("quality_score must be <= 100")
        _probability(self.selection_probability, "selection_probability")
        if self.selection_probability == 0:
            raise OpenRoutiQError("selection_probability must be greater than zero")
        if self.actual_cost_usd is not None:
            _positive(self.actual_cost_usd, "actual_cost_usd", allow_zero=True)
        if self.latency_ms is not None:
            _positive(self.latency_ms, "latency_ms", allow_zero=True)


@dataclass(frozen=True)
class SelectionPrediction:
    model_id: str
    success_probability: float
    raw_success_probability: float
    expected_quality: float
    expected_cost_usd: float
    expected_latency_ms: float
    utility: float
    samples: float
    calibration_samples: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "success_probability": round(self.success_probability, 6),
            "raw_success_probability": round(self.raw_success_probability, 6),
            "expected_quality": round(self.expected_quality, 4),
            "expected_cost_usd": round(self.expected_cost_usd, 8),
            "expected_latency_ms": round(self.expected_latency_ms, 2),
            "utility": round(self.utility, 6),
            "samples": round(self.samples, 4),
            "calibration_samples": round(self.calibration_samples, 4),
        }


@dataclass(frozen=True)
class SelectionChoice:
    model_id: str
    selection_probability: float
    explored: bool
    predictions: tuple[SelectionPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "selection_probability": round(self.selection_probability, 8),
            "explored": self.explored,
            "predictions": [item.to_dict() for item in self.predictions],
        }


@dataclass
class _Metric:
    weight: float = 0.0
    mean: float = 0.0

    def add(self, value: float, weight: float) -> None:
        total = self.weight + weight
        self.mean += weight * (value - self.mean) / total
        self.weight = total


@dataclass
class _ModelState:
    success_weights: dict[int, float] = field(default_factory=dict)
    quality_weights: dict[int, float] = field(default_factory=dict)
    success_bias: float = 0.0
    quality_bias: float = 0.0
    samples: float = 0.0
    updates: int = 0
    calibration: dict[str, list[float]] = field(default_factory=dict)
    costs: dict[str, _Metric] = field(default_factory=dict)
    latencies: dict[str, _Metric] = field(default_factory=dict)


class SelectionIntelligence:
    """Dependency-free calibrated request-by-model predictor with online updates.

    The persisted state contains only hashed feature weights and aggregate numeric telemetry;
    it never stores request text, tool payloads, model outputs, or ground truth.
    """

    def __init__(
        self,
        profiles: Sequence[ModelProfile | Mapping[str, Any]],
        *,
        policy: SelectionPolicy | None = None,
        feature_dimension: int = 16_384,
        learning_rate: float = 0.08,
        l2: float = 1e-5,
        calibration_prior: float = 10.0,
        maximum_importance_weight: float = 20.0,
        random_seed: int = 3407,
    ) -> None:
        parsed = tuple(
            item if isinstance(item, ModelProfile) else ModelProfile.from_mapping(item)
            for item in profiles
        )
        if not parsed:
            raise OpenRoutiQError("selection intelligence requires at least one model")
        ids = [item.id for item in parsed]
        if len(ids) != len(set(ids)):
            raise OpenRoutiQError("selection intelligence model ids must be unique")
        if (
            isinstance(feature_dimension, bool)
            or not isinstance(feature_dimension, int)
            or feature_dimension < 256
        ):
            raise OpenRoutiQError("feature_dimension must be an integer >= 256")
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            raise OpenRoutiQError("random_seed must be an integer")
        self.profiles = parsed
        self._profiles = {item.id: item for item in parsed}
        self.policy = policy or SelectionPolicy()
        self.feature_dimension = feature_dimension
        self.learning_rate = _positive(learning_rate, "learning_rate")
        self.l2 = _positive(l2, "l2", allow_zero=True)
        self.calibration_prior = _positive(calibration_prior, "calibration_prior")
        self.maximum_importance_weight = _positive(
            maximum_importance_weight, "maximum_importance_weight"
        )
        self.random_seed = random_seed
        self._hash_key = hashlib.sha256(str(random_seed).encode("ascii")).digest()[:16]
        self._states = {item.id: _ModelState() for item in parsed}
        self._random = random.Random(random_seed)
        self._lock = RLock()

    def _hash(self, feature: str) -> int:
        digest = hashlib.blake2b(
            feature.encode("utf-8", errors="ignore"),
            digest_size=8,
            key=self._hash_key,
        ).digest()
        return int.from_bytes(digest, "little") % self.feature_dimension

    def _features(
        self,
        request: str | Sequence[Any] | RouteContext,
        *,
        task: str,
        tools: Sequence[Any] | None,
        response_format: Mapping[str, Any] | None,
        stream: bool,
    ) -> dict[int, float]:
        context = request if isinstance(request, RouteContext) else RouteContext(request)
        analysis = analyze_context(context)
        raw = context.request
        text = raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True, default=str)
        text = text[:100_000].casefold()
        tokens = _TOKEN.findall(text)[:4096]
        names: list[str] = [
            "bias",
            f"task:{task.casefold()}",
            f"inferred:{analysis.task}",
            f"complexity:{int(analysis.complexity // 10)}",
            f"length:{min(20, int(math.log2(max(1, len(text)))))}",
            f"messages:{min(20, analysis.message_count)}",
        ]
        names.extend(f"capability:{item}" for item in sorted(analysis.required_capabilities))
        names.extend(f"token:{token}" for token in tokens)
        names.extend(f"bigram:{left}:{right}" for left, right in zip(tokens, tokens[1:257]))
        if tools:
            names.append("request:tools")
            tool_text = json.dumps(list(tools), sort_keys=True, default=str)[:20_000].casefold()
            names.extend(f"tool:{token}" for token in _TOKEN.findall(tool_text)[:256])
        if response_format is not None:
            names.append("request:structured-output")
        if stream:
            names.append("request:stream")
        features: dict[int, float] = {}
        for name in names:
            index = self._hash(name)
            features[index] = min(3.0, features.get(index, 0.0) + 1.0)
        return features

    @staticmethod
    def _dot(weights: Mapping[int, float], features: Mapping[int, float]) -> float:
        return sum(weights.get(index, 0.0) * value for index, value in features.items())

    def _prior(self, profile: ModelProfile, task: str) -> float:
        quality = profile.quality_for(task)
        return min(0.99, max(0.01, (50.0 if quality is None else quality) / 100))

    def _component_prediction(
        self,
        model_id: str,
        features: Mapping[int, float],
        task: str,
    ) -> tuple[float, float, float, float]:
        profile = self._profiles[model_id]
        state = self._states[model_id]
        prior = self._prior(profile, task)
        raw_success = _sigmoid(
            _logit(prior) + state.success_bias + self._dot(state.success_weights, features)
        )
        expected_quality = 100 * _sigmoid(
            _logit(prior) + state.quality_bias + self._dot(state.quality_weights, features)
        )
        bin_index = min(9, int(raw_success * 10))
        calibration_key = f"{task}:{bin_index}"
        successes, total = state.calibration.get(calibration_key, [0.0, 0.0])
        calibrated = (successes + self.calibration_prior * raw_success) / (
            total + self.calibration_prior
        )
        return raw_success, calibrated, expected_quality, total

    def predict(
        self,
        request: str | Sequence[Any] | RouteContext,
        *,
        candidate_ids: Sequence[str] | None = None,
        task: str = "general",
        tools: Sequence[Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> tuple[SelectionPrediction, ...]:
        candidates = tuple(candidate_ids or self._profiles)
        if not candidates or any(item not in self._profiles for item in candidates):
            raise OpenRoutiQError("candidate_ids must contain known model ids")
        if len(candidates) != len(set(candidates)):
            raise OpenRoutiQError("candidate_ids cannot contain duplicates")
        features = self._features(
            request,
            task=task,
            tools=tools,
            response_format=response_format,
            stream=stream,
        )
        with self._lock:
            components: list[dict[str, float | str]] = []
            for model_id in candidates:
                profile = self._profiles[model_id]
                state = self._states[model_id]
                raw, calibrated, quality, calibration_samples = self._component_prediction(
                    model_id, features, task
                )
                cost_metric = state.costs.get(task) or state.costs.get("general")
                latency_metric = state.latencies.get(task) or state.latencies.get("general")
                components.append(
                    {
                        "model_id": model_id,
                        "raw": raw,
                        "success": calibrated,
                        "quality": quality,
                        "cost": (
                            profile.input_price_per_million / 1_000_000
                            if cost_metric is None
                            else cost_metric.mean
                        ),
                        "latency": (
                            profile.latency_ms if latency_metric is None else latency_metric.mean
                        ),
                        "samples": state.samples,
                        "calibration_samples": calibration_samples,
                    }
                )
        maximum_cost = max(float(item["cost"]) for item in components) or 1.0
        maximum_latency = max(float(item["latency"]) for item in components) or 1.0
        predictions: list[SelectionPrediction] = []
        for item in components:
            success = float(item["success"])
            quality = float(item["quality"])
            cost = float(item["cost"])
            latency = float(item["latency"])
            utility = (
                self.policy.success_weight * success
                + self.policy.quality_weight * quality / 100
                - self.policy.cost_weight * cost / maximum_cost
                - self.policy.latency_weight * latency / maximum_latency
                - self.policy.risk_aversion * (1 - success)
            )
            predictions.append(
                SelectionPrediction(
                    model_id=str(item["model_id"]),
                    success_probability=success,
                    raw_success_probability=float(item["raw"]),
                    expected_quality=quality,
                    expected_cost_usd=cost,
                    expected_latency_ms=latency,
                    utility=utility,
                    samples=float(item["samples"]),
                    calibration_samples=float(item["calibration_samples"]),
                )
            )
        return tuple(sorted(predictions, key=lambda item: (-item.utility, item.model_id)))

    def select(
        self,
        request: str | Sequence[Any] | RouteContext,
        *,
        candidate_ids: Sequence[str] | None = None,
        task: str = "general",
        tools: Sequence[Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        stream: bool = False,
        explore: bool = False,
    ) -> SelectionChoice:
        predictions = self.predict(
            request,
            candidate_ids=candidate_ids,
            task=task,
            tools=tools,
            response_format=response_format,
            stream=stream,
        )
        rate = self.policy.exploration_rate if explore else 0.0
        exploration_weights = [
            1 / math.sqrt(1 + prediction.calibration_samples) for prediction in predictions
        ]
        total_exploration_weight = sum(exploration_weights)
        with self._lock:
            explored = rate > 0 and self._random.random() < rate
            selected = (
                self._random.choices(predictions, weights=exploration_weights, k=1)[0]
                if explored
                else predictions[0]
            )
        selected_index = next(
            index for index, prediction in enumerate(predictions) if prediction is selected
        )
        probability = rate * exploration_weights[selected_index] / total_exploration_weight
        if selected.model_id == predictions[0].model_id:
            probability += 1 - rate
        return SelectionChoice(
            model_id=selected.model_id,
            selection_probability=probability,
            explored=explored,
            predictions=predictions,
        )

    def _update_weights(
        self,
        weights: dict[int, float],
        features: Mapping[int, float],
        *,
        prediction: float,
        target: float,
        learning_rate: float,
        importance: float,
    ) -> float:
        gradient = (prediction - target) * importance
        for index, value in features.items():
            previous = weights.get(index, 0.0)
            updated = previous - learning_rate * (gradient * value + self.l2 * previous)
            if abs(updated) < 1e-12:
                weights.pop(index, None)
            else:
                weights[index] = updated
        return gradient

    def observe(self, observation: SelectionTrainingObservation) -> None:
        if observation.model_id not in self._profiles:
            raise OpenRoutiQError(f"unknown selection model id: {observation.model_id}")
        features = self._features(
            observation.request,
            task=observation.task,
            tools=observation.tools,
            response_format=observation.response_format,
            stream=observation.stream,
        )
        importance = min(
            self.maximum_importance_weight,
            1 / observation.selection_probability,
        )
        with self._lock:
            state = self._states[observation.model_id]
            raw_success, _, expected_quality, _ = self._component_prediction(
                observation.model_id, features, observation.task
            )
            bin_index = min(9, int(raw_success * 10))
            calibration = state.calibration.setdefault(
                f"{observation.task}:{bin_index}", [0.0, 0.0]
            )
            calibration[0] += importance * int(observation.success)
            calibration[1] += importance
            learning_rate = self.learning_rate / math.sqrt(1 + state.updates / 50)
            success_gradient = self._update_weights(
                state.success_weights,
                features,
                prediction=raw_success,
                target=float(observation.success),
                learning_rate=learning_rate,
                importance=importance,
            )
            quality_prediction = expected_quality / 100
            quality_gradient = self._update_weights(
                state.quality_weights,
                features,
                prediction=quality_prediction,
                target=observation.quality_score / 100,
                learning_rate=learning_rate,
                importance=importance,
            )
            state.success_bias -= learning_rate * (success_gradient + self.l2 * state.success_bias)
            state.quality_bias -= learning_rate * (quality_gradient + self.l2 * state.quality_bias)
            state.samples += importance
            state.updates += 1
            if observation.actual_cost_usd is not None:
                state.costs.setdefault(observation.task, _Metric()).add(
                    observation.actual_cost_usd, importance
                )
            if observation.latency_ms is not None:
                state.latencies.setdefault(observation.task, _Metric()).add(
                    observation.latency_ms, importance
                )

    def fit(
        self,
        observations: Sequence[SelectionTrainingObservation],
        *,
        epochs: int = 3,
        shuffle: bool = True,
    ) -> SelectionIntelligence:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise OpenRoutiQError("epochs must be an integer >= 1")
        rows = list(observations)
        if not rows:
            raise OpenRoutiQError("fit requires at least one observation")
        order_random = random.Random(self.random_seed)
        for _ in range(epochs):
            if shuffle:
                order_random.shuffle(rows)
            for row in rows:
                self.observe(row)
        return self

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            states = {
                model_id: {
                    "success_weights": {
                        str(index): value for index, value in sorted(state.success_weights.items())
                    },
                    "quality_weights": {
                        str(index): value for index, value in sorted(state.quality_weights.items())
                    },
                    "success_bias": state.success_bias,
                    "quality_bias": state.quality_bias,
                    "samples": state.samples,
                    "updates": state.updates,
                    "calibration": dict(sorted(state.calibration.items())),
                    "costs": {
                        task: {"weight": metric.weight, "mean": metric.mean}
                        for task, metric in sorted(state.costs.items())
                    },
                    "latencies": {
                        task: {"weight": metric.weight, "mean": metric.mean}
                        for task, metric in sorted(state.latencies.items())
                    },
                }
                for model_id, state in sorted(self._states.items())
            }
        return {
            "schema_version": SELECTION_STATE_SCHEMA_VERSION,
            "model_ids": sorted(self._profiles),
            "feature_dimension": self.feature_dimension,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "calibration_prior": self.calibration_prior,
            "maximum_importance_weight": self.maximum_importance_weight,
            "random_seed": self.random_seed,
            "policy": self.policy.__dict__,
            "states": states,
            "privacy": "hashed feature weights and aggregate numeric telemetry only",
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "nt":
                os.chmod(temporary, 0o600)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        profiles: Sequence[ModelProfile | Mapping[str, Any]],
    ) -> SelectionIntelligence:
        source = Path(path).expanduser().resolve()
        try:
            if source.stat().st_size > _MAXIMUM_STATE_BYTES:
                raise OpenRoutiQError("selection state exceeds the 64 MiB safety limit")
            raw = json.loads(source.read_text(encoding="utf-8"))
        except OpenRoutiQError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise OpenRoutiQError(f"cannot load selection state {source}: {exc}") from exc
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema_version") != SELECTION_STATE_SCHEMA_VERSION
        ):
            raise OpenRoutiQError("unsupported selection state schema")
        policy_raw = raw.get("policy")
        if not isinstance(policy_raw, Mapping):
            raise OpenRoutiQError("selection state policy must be an object")
        try:
            instance = cls(
                profiles,
                policy=SelectionPolicy(**dict(policy_raw)),
                feature_dimension=raw["feature_dimension"],
                learning_rate=raw["learning_rate"],
                l2=raw["l2"],
                calibration_prior=raw["calibration_prior"],
                maximum_importance_weight=raw["maximum_importance_weight"],
                random_seed=raw["random_seed"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OpenRoutiQError(f"selection state configuration is invalid: {exc}") from exc
        if sorted(instance._profiles) != raw.get("model_ids"):
            raise OpenRoutiQError("selection state model ids do not match the supplied catalog")
        states_raw = raw.get("states")
        if not isinstance(states_raw, Mapping):
            raise OpenRoutiQError("selection state states must be an object")
        if set(states_raw) != set(instance._states):
            raise OpenRoutiQError("selection state entries do not match the supplied catalog")

        def finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OpenRoutiQError(f"{name} must be numeric")
            parsed = float(value)
            if not math.isfinite(parsed) or (nonnegative and parsed < 0):
                suffix = " and non-negative" if nonnegative else ""
                raise OpenRoutiQError(f"{name} must be finite{suffix}")
            return parsed

        def parse_weights(value: Any, name: str) -> dict[int, float]:
            if not isinstance(value, Mapping):
                raise OpenRoutiQError(f"{name} must be an object")
            parsed: dict[int, float] = {}
            for raw_index, raw_value in value.items():
                try:
                    index = int(raw_index)
                except (TypeError, ValueError) as exc:
                    raise OpenRoutiQError(f"{name} has an invalid feature index") from exc
                if str(index) != str(raw_index) or not 0 <= index < instance.feature_dimension:
                    raise OpenRoutiQError(f"{name} feature index is out of range")
                parsed[index] = finite_number(raw_value, f"{name}[{index}]")
            return parsed

        for model_id, state_raw in states_raw.items():
            if model_id not in instance._states or not isinstance(state_raw, Mapping):
                raise OpenRoutiQError("selection state contains an invalid model")
            state = instance._states[str(model_id)]
            state.success_weights = parse_weights(
                state_raw.get("success_weights", {}), f"states.{model_id}.success_weights"
            )
            state.quality_weights = parse_weights(
                state_raw.get("quality_weights", {}), f"states.{model_id}.quality_weights"
            )
            state.success_bias = finite_number(
                state_raw.get("success_bias", 0), f"states.{model_id}.success_bias"
            )
            state.quality_bias = finite_number(
                state_raw.get("quality_bias", 0), f"states.{model_id}.quality_bias"
            )
            state.samples = finite_number(
                state_raw.get("samples", 0), f"states.{model_id}.samples", nonnegative=True
            )
            updates = state_raw.get("updates", 0)
            if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
                raise OpenRoutiQError(f"states.{model_id}.updates must be an integer >= 0")
            state.updates = updates
            calibration_raw = state_raw.get("calibration", {})
            if not isinstance(calibration_raw, Mapping):
                raise OpenRoutiQError(f"states.{model_id}.calibration must be an object")
            state.calibration = {}
            for key, value in calibration_raw.items():
                if not isinstance(key, str) or len(key) > 256:
                    raise OpenRoutiQError("selection calibration key is invalid")
                if not isinstance(value, list) or len(value) != 2:
                    raise OpenRoutiQError("selection calibration entry must have two values")
                successes = finite_number(
                    value[0],
                    f"states.{model_id}.calibration.{key}.successes",
                    nonnegative=True,
                )
                total = finite_number(
                    value[1], f"states.{model_id}.calibration.{key}.total", nonnegative=True
                )
                if successes > total:
                    raise OpenRoutiQError("selection calibration successes exceed total")
                state.calibration[key] = [successes, total]
            for field_name in ("costs", "latencies"):
                metrics_raw = state_raw.get(field_name, {})
                if not isinstance(metrics_raw, Mapping):
                    raise OpenRoutiQError(f"states.{model_id}.{field_name} must be an object")
                parsed_metrics: dict[str, _Metric] = {}
                for task, value in metrics_raw.items():
                    if (
                        not isinstance(task, str)
                        or not task
                        or len(task) > 256
                        or not isinstance(value, Mapping)
                    ):
                        raise OpenRoutiQError("selection metric entry is invalid")
                    parsed_metrics[task] = _Metric(
                        finite_number(
                            value.get("weight"),
                            f"states.{model_id}.{field_name}.{task}.weight",
                            nonnegative=True,
                        ),
                        finite_number(
                            value.get("mean"),
                            f"states.{model_id}.{field_name}.{task}.mean",
                            nonnegative=True,
                        ),
                    )
                setattr(state, field_name, parsed_metrics)
        return instance


class SelfLearningRouter:
    """Capability-gated Router wrapper driven by evaluated outcomes, never API success alone."""

    def __init__(
        self,
        router: Router,
        intelligence: SelectionIntelligence | None = None,
    ) -> None:
        if not isinstance(router, Router):
            raise OpenRoutiQError("SelfLearningRouter requires a Router")
        self.router = router
        self.intelligence = intelligence or SelectionIntelligence(router.profiles)
        if {item.id for item in router.profiles} != {
            item.id for item in self.intelligence.profiles
        }:
            raise OpenRoutiQError("router and selection intelligence model ids must match")

    @property
    def profiles(self) -> tuple[ModelProfile, ...]:
        return self.router.profiles

    def route(
        self,
        request: str | Sequence[Any] | RouteContext,
        *,
        explore: bool = False,
        **options: Any,
    ) -> RouteDecision:
        if options.get("pinned_model") is not None:
            return self.router.route(request, **options)
        base = self.router.route(request, **options)
        tools = options.get("tools")
        response_format = options.get("response_format")
        stream = bool(options.get("stream", False))
        choice = self.intelligence.select(
            request,
            candidate_ids=[item.model_id for item in base.ranked],
            task=base.task,
            tools=tools,
            response_format=response_format,
            stream=stream,
            explore=explore,
        )
        predictions = {item.model_id: item for item in choice.predictions}
        learned_ranked: list[CandidateScore] = []
        for candidate in base.ranked:
            prediction = predictions[candidate.model_id]
            confidence = (
                100
                * prediction.samples
                / (prediction.samples + self.intelligence.policy.minimum_learning_samples)
            )
            learned_ranked.append(
                replace(
                    candidate,
                    total_score=100 * prediction.utility,
                    quality_score=prediction.expected_quality,
                    confidence=max(
                        candidate.confidence if not prediction.samples else 0, confidence
                    ),
                )
            )
        learned_ranked.sort(key=lambda item: (-item.total_score, item.model_id))
        selected = next(item for item in learned_ranked if item.model_id == choice.model_id)
        prediction = predictions[selected.model_id]
        review_reasons = list(base.review_reasons)
        if prediction.samples < self.intelligence.policy.minimum_learning_samples:
            review_reasons.append(
                "selection predictor has fewer than the minimum evaluated samples"
            )
        if choice.explored:
            review_reasons.append("bounded exploration selected a non-greedy candidate")
        return replace(
            base,
            selected=selected,
            ranked=tuple(learned_ranked),
            strategy="self_learning",
            selection_probability=choice.selection_probability,
            review_required=bool(review_reasons),
            review_reasons=tuple(dict.fromkeys(review_reasons)),
        )

    def record_evaluation(
        self,
        request: str | Sequence[Any] | RouteContext,
        decision: RouteDecision,
        *,
        quality_score: float,
        success: bool,
        actual_cost_usd: float | None = None,
        latency_ms: float | None = None,
        tools: Sequence[Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> None:
        self.intelligence.observe(
            SelectionTrainingObservation(
                request=request,
                model_id=decision.selected.model_id,
                success=success,
                quality_score=quality_score,
                task=decision.task,
                actual_cost_usd=actual_cost_usd,
                latency_ms=latency_ms,
                selection_probability=decision.selection_probability,
                tools=tools,
                response_format=response_format,
                stream=stream,
            )
        )
