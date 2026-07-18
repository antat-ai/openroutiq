from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openroutiq.benchmark.core import (
    BenchmarkError,
    OutcomeMatrixObservation,
    analyze_outcome_matrix,
)
from openroutiq.router.core import ModelProfile
from openroutiq.selection.intelligence import (
    SelectionIntelligence,
    SelectionPolicy,
    SelectionTrainingObservation,
)


_CONTEXT_TOKEN = re.compile(r"[a-z0-9_+#.-]{2,}", re.IGNORECASE)
_LINUCB_DIMENSION = 64


@dataclass(frozen=True)
class SelectionBenchmarkRequest:
    """Private request payload plus leakage-controlled evaluation metadata."""

    request_id: str
    request: str | Sequence[Any]
    split: str
    sequence: int
    task: str = "general"
    segment: str = "unknown"
    group_id: str | None = None
    tools: Sequence[Any] | None = None
    response_format: Mapping[str, Any] | None = None
    stream: bool = False

    def __post_init__(self) -> None:
        if not self.request_id or not isinstance(self.request_id, str):
            raise BenchmarkError("selection request_id must be non-empty text")
        if self.split not in {"train", "validation", "test"}:
            raise BenchmarkError("selection split must be train, validation, or test")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise BenchmarkError("selection sequence must be an integer")
        for name, value in (("task", self.task), ("segment", self.segment)):
            if not isinstance(value, str) or not value.strip():
                raise BenchmarkError(f"selection {name} must be non-empty text")
        if self.group_id is not None and (
            not isinstance(self.group_id, str) or not self.group_id.strip()
        ):
            raise BenchmarkError("selection group_id must be non-empty text or null")


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise BenchmarkError("percentile requires values")
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _paired_interval(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int,
    repetitions: int = 5_000,
    confidence: float = 0.95,
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise BenchmarkError("paired interval requires equal non-empty vectors")
    if not 0 < confidence < 1:
        raise BenchmarkError("paired interval confidence must be between zero and one")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    generator = random.Random(seed)
    draws = [
        statistics.fmean(differences[generator.randrange(len(differences))] for _ in differences)
        for _ in range(repetitions)
    ]
    tail = (1 - confidence) / 2
    return {
        "delta": statistics.fmean(differences),
        "confidence": confidence,
        "ci_low": _percentile(draws, tail),
        "ci_high": _percentile(draws, 1 - tail),
    }


def _validate(
    requests: Sequence[SelectionBenchmarkRequest],
    outcomes: Sequence[OutcomeMatrixObservation],
    profiles: Sequence[ModelProfile | Mapping[str, Any]],
) -> tuple[
    tuple[ModelProfile, ...],
    dict[str, SelectionBenchmarkRequest],
    dict[str, dict[str, OutcomeMatrixObservation]],
]:
    parsed_profiles = tuple(
        item if isinstance(item, ModelProfile) else ModelProfile.from_mapping(item)
        for item in profiles
    )
    if len(parsed_profiles) < 2:
        raise BenchmarkError("selection evaluation requires at least two profiles")
    by_request_meta: dict[str, SelectionBenchmarkRequest] = {}
    sequences: set[int] = set()
    group_splits: dict[str, str] = {}
    for item in requests:
        if item.request_id in by_request_meta:
            raise BenchmarkError(f"duplicate selection request: {item.request_id}")
        if item.sequence in sequences:
            raise BenchmarkError(f"duplicate selection sequence: {item.sequence}")
        by_request_meta[item.request_id] = item
        sequences.add(item.sequence)
        if item.group_id is not None:
            previous = group_splits.setdefault(item.group_id, item.split)
            if previous != item.split:
                raise BenchmarkError(
                    f"group {item.group_id} leaks across {previous} and {item.split}"
                )
    if {item.split for item in requests} != {"train", "validation", "test"}:
        raise BenchmarkError("selection evaluation requires all three splits")
    model_ids = {item.id for item in parsed_profiles}
    by_outcome: dict[str, dict[str, OutcomeMatrixObservation]] = defaultdict(dict)
    for cell in outcomes:
        if cell.request_id not in by_request_meta:
            raise BenchmarkError(f"outcome has no private request metadata: {cell.request_id}")
        if cell.model_id not in model_ids:
            raise BenchmarkError(f"outcome has no model profile: {cell.model_id}")
        if cell.model_id in by_outcome[cell.request_id]:
            raise BenchmarkError(f"duplicate outcome cell: {cell.request_id} / {cell.model_id}")
        if cell.segment != by_request_meta[cell.request_id].segment:
            raise BenchmarkError(f"segment mismatch for request {cell.request_id}")
        by_outcome[cell.request_id][cell.model_id] = cell
    missing = [
        f"{request_id}/{model_id}"
        for request_id in by_request_meta
        for model_id in model_ids
        if model_id not in by_outcome[request_id]
    ]
    if missing:
        raise BenchmarkError(
            f"selection evaluation matrix is incomplete; missing {len(missing)} cell(s)"
        )
    return parsed_profiles, by_request_meta, dict(by_outcome)


def _training_observations(
    request_ids: Sequence[str],
    requests: Mapping[str, SelectionBenchmarkRequest],
    outcomes: Mapping[str, Mapping[str, OutcomeMatrixObservation]],
) -> list[SelectionTrainingObservation]:
    rows: list[SelectionTrainingObservation] = []
    for request_id in request_ids:
        metadata = requests[request_id]
        for cell in outcomes[request_id].values():
            rows.append(
                SelectionTrainingObservation(
                    request=metadata.request,
                    model_id=cell.model_id,
                    success=cell.success,
                    quality_score=100 * cell.quality_score,
                    task=metadata.task,
                    actual_cost_usd=cell.cost_usd,
                    latency_ms=cell.total_latency_ms,
                    tools=metadata.tools,
                    response_format=metadata.response_format,
                    stream=metadata.stream,
                )
            )
    return rows


def _result(
    name: str,
    selections: Sequence[tuple[str, str | None, float | None]],
    outcomes: Mapping[str, Mapping[str, OutcomeMatrixObservation]],
    *,
    failure_cells: Mapping[str, OutcomeMatrixObservation] | None = None,
) -> dict[str, Any]:
    cells: list[OutcomeMatrixObservation] = []
    for request_id, model_id, _ in selections:
        if model_id is None:
            if failure_cells is None or request_id not in failure_cells:
                raise BenchmarkError(
                    f"router {name} has no outcome for failed request {request_id}"
                )
            cells.append(failure_cells[request_id])
        else:
            cells.append(outcomes[request_id][model_id])
    probabilities = [
        (probability, cell.success)
        for (_, _, probability), cell in zip(selections, cells, strict=True)
        if probability is not None
    ]
    reliability: list[dict[str, float | int]] = []
    if probabilities:
        for bin_index in range(10):
            lower = bin_index / 10
            upper = (bin_index + 1) / 10
            rows = [
                (float(probability), int(success))
                for probability, success in probabilities
                if lower <= float(probability) < upper or bin_index == 9 and float(probability) == 1
            ]
            if rows:
                reliability.append(
                    {
                        "bin_low": lower,
                        "bin_high": upper,
                        "count": len(rows),
                        "mean_prediction": statistics.fmean(item[0] for item in rows),
                        "observed_success": statistics.fmean(item[1] for item in rows),
                    }
                )
    brier_score = (
        statistics.fmean(
            (float(probability) - int(success)) ** 2 for probability, success in probabilities
        )
        if probabilities
        else None
    )
    log_loss = (
        statistics.fmean(
            -(
                int(success) * math.log(min(1 - 1e-12, max(1e-12, float(probability))))
                + (1 - int(success)) * math.log(min(1 - 1e-12, max(1e-12, 1 - float(probability))))
            )
            for probability, success in probabilities
        )
        if probabilities
        else None
    )
    ece = (
        sum(
            int(row["count"])
            / len(probabilities)
            * abs(float(row["mean_prediction"]) - float(row["observed_success"]))
            for row in reliability
        )
        if probabilities
        else None
    )
    return {
        "router": name,
        "cases": len(cells),
        "accuracy": statistics.fmean(cell.quality_score for cell in cells),
        "success_rate": statistics.fmean(cell.success for cell in cells),
        "mean_cost_usd": statistics.fmean(cell.cost_usd for cell in cells),
        "p95_latency_ms": _percentile([cell.total_latency_ms for cell in cells], 0.95),
        "total_cost_usd": sum(cell.cost_usd for cell in cells),
        "quality_vector": [cell.quality_score for cell in cells],
        "success_vector": [int(cell.success) for cell in cells],
        "brier_score": brier_score,
        "log_loss": log_loss,
        "expected_calibration_error": ece,
        "reliability": reliability,
        "evaluation_transcript_sha256": _evaluation_transcript_sha256(selections, cells),
        "selections": [
            {"request_id": request_id, "model_id": model_id}
            for request_id, model_id, _ in selections
        ],
    }


def _selection_sha256(selections: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(selections),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evaluation_transcript_sha256(
    selections: Sequence[tuple[str, str | None, float | None]],
    cells: Sequence[OutcomeMatrixObservation],
) -> str:
    transcript = [
        {
            "request_id": request_id,
            "model_id": model_id,
            "selection_probability": probability,
            "quality_score": cell.quality_score,
            "success": cell.success,
            "cost_usd": cell.cost_usd,
            "total_latency_ms": cell.total_latency_ms,
        }
        for (request_id, model_id, probability), cell in zip(selections, cells, strict=True)
    ]
    payload = json.dumps(
        transcript,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _segment_rules(
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    requests: Mapping[str, SelectionBenchmarkRequest],
    outcomes: Mapping[str, Mapping[str, OutcomeMatrixObservation]],
    model_ids: Sequence[str],
) -> list[tuple[str, str, float | None]]:
    global_means = {
        model_id: statistics.fmean(outcomes[item][model_id].quality_score for item in train_ids)
        for model_id in model_ids
    }
    by_segment: dict[str, dict[str, float]] = {}
    for segment in sorted({requests[item].segment for item in train_ids}):
        rows = [item for item in train_ids if requests[item].segment == segment]
        by_segment[segment] = {
            model_id: statistics.fmean(outcomes[item][model_id].quality_score for item in rows)
            for model_id in model_ids
        }
    selections: list[tuple[str, str, float | None]] = []
    for request_id in test_ids:
        scores = by_segment.get(requests[request_id].segment, global_means)
        model_id = min(model_ids, key=lambda item: (-scores[item], item))
        selections.append((request_id, model_id, None))
    return selections


def _adaptive_baseline(
    kind: str,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    requests: Mapping[str, SelectionBenchmarkRequest],
    outcomes: Mapping[str, Mapping[str, OutcomeMatrixObservation]],
    model_ids: Sequence[str],
    *,
    seed: int,
) -> list[tuple[str, str, float | None]]:
    rewards: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    successes: dict[tuple[str, str], float] = defaultdict(lambda: 1.0)
    failures: dict[tuple[str, str], float] = defaultdict(lambda: 1.0)
    for request_id in train_ids:
        segment = requests[request_id].segment
        for model_id in model_ids:
            cell = outcomes[request_id][model_id]
            key = (segment, model_id)
            rewards[key] += cell.quality_score
            counts[key] += 1
            successes[key] += int(cell.success)
            failures[key] += int(not cell.success)
    generator = random.Random(seed)
    selections: list[tuple[str, str, float | None]] = []
    for step, request_id in enumerate(test_ids, 1):
        segment = requests[request_id].segment
        if kind == "ucb1":

            def value(model_id: str) -> float:
                key = (segment, model_id)
                if not counts[key]:
                    return math.inf
                return rewards[key] / counts[key] + math.sqrt(
                    2 * math.log(step + sum(counts.values()) + 1) / counts[key]
                )

            selected = min(model_ids, key=lambda item: (-value(item), item))
            probability = None
        elif kind == "thompson":
            draws = {
                model_id: generator.betavariate(
                    successes[(segment, model_id)], failures[(segment, model_id)]
                )
                for model_id in model_ids
            }
            selected = min(model_ids, key=lambda item: (-draws[item], item))
            probability = None
        else:
            raise BenchmarkError(f"unknown adaptive baseline: {kind}")
        selections.append((request_id, selected, probability))
        cell = outcomes[request_id][selected]
        key = (segment, selected)
        rewards[key] += cell.quality_score
        counts[key] += 1
        successes[key] += int(cell.success)
        failures[key] += int(not cell.success)
    return selections


def _linucb_features(metadata: SelectionBenchmarkRequest) -> list[float]:
    """Build a fixed, dependency-free contextual feature vector for LinUCB."""

    if isinstance(metadata.request, str):
        text = metadata.request
    else:
        text = json.dumps(metadata.request, sort_keys=True, default=str)
    values = [0.0] * _LINUCB_DIMENSION
    values[0] = 1.0
    names = [
        f"task:{metadata.task.casefold()}",
        f"segment:{metadata.segment.casefold()}",
        f"length:{min(15, int(math.log2(max(1, len(text)))))}",
        f"tools:{int(bool(metadata.tools))}",
        f"structured:{int(metadata.response_format is not None)}",
        f"stream:{int(metadata.stream)}",
    ]
    names.extend(f"token:{item.casefold()}" for item in _CONTEXT_TOKEN.findall(text)[:512])
    for name in names:
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        index = 1 + raw % (_LINUCB_DIMENSION - 1)
        sign = 1.0 if raw >> 63 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _linucb_baseline(
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    requests: Mapping[str, SelectionBenchmarkRequest],
    outcomes: Mapping[str, Mapping[str, OutcomeMatrixObservation]],
    model_ids: Sequence[str],
    *,
    alpha: float = 0.75,
) -> list[tuple[str, str, float | None]]:
    """Disjoint LinUCB with full-information history and bandit-only test updates."""

    inverses = {
        model_id: [
            [1.0 if row == column else 0.0 for column in range(_LINUCB_DIMENSION)]
            for row in range(_LINUCB_DIMENSION)
        ]
        for model_id in model_ids
    }
    rewards = {model_id: [0.0] * _LINUCB_DIMENSION for model_id in model_ids}

    def update(model_id: str, features: Sequence[float], reward: float) -> None:
        inverse = inverses[model_id]
        projected = _matvec(inverse, features)
        denominator = 1.0 + _dot(features, projected)
        if denominator <= 0 or not math.isfinite(denominator):
            raise BenchmarkError("LinUCB covariance update became invalid")
        for row in range(_LINUCB_DIMENSION):
            for column in range(_LINUCB_DIMENSION):
                inverse[row][column] -= projected[row] * projected[column] / denominator
        for index, value in enumerate(features):
            rewards[model_id][index] += reward * value

    for request_id in train_ids:
        features = _linucb_features(requests[request_id])
        for model_id in model_ids:
            update(model_id, features, outcomes[request_id][model_id].quality_score)

    selections: list[tuple[str, str, float | None]] = []
    for request_id in test_ids:
        features = _linucb_features(requests[request_id])
        scores: dict[str, float] = {}
        for model_id in model_ids:
            inverse = inverses[model_id]
            projected = _matvec(inverse, features)
            theta = _matvec(inverse, rewards[model_id])
            uncertainty = math.sqrt(max(0.0, _dot(features, projected)))
            scores[model_id] = _dot(theta, features) + alpha * uncertainty
        selected = min(model_ids, key=lambda item: (-scores[item], item))
        selections.append((request_id, selected, None))
        update(selected, features, outcomes[request_id][selected].quality_score)
    return selections


def evaluate_selection_learning(
    requests: Sequence[SelectionBenchmarkRequest],
    outcomes: Sequence[OutcomeMatrixObservation],
    profiles: Sequence[ModelProfile | Mapping[str, Any]],
    *,
    policy: SelectionPolicy | None = None,
    external_router_selections: Mapping[str, Mapping[str, str | None]] | None = None,
    external_router_failures: Mapping[str, Mapping[str, OutcomeMatrixObservation]] | None = None,
    base_comparators: Sequence[str] = ("best-fixed", "segment-rules"),
    online_comparators: Sequence[str] = (
        "openroutiq-frozen",
        "ucb1",
        "thompson",
        "linucb",
    ),
    minimum_oracle_headroom: float = 0.05,
    seed: int = 3407,
) -> dict[str, Any]:
    """Train on past data and evaluate frozen plus prequential selectors on test traffic."""

    parsed_profiles, request_map, outcome_map = _validate(requests, outcomes, profiles)
    model_ids = tuple(sorted(item.id for item in parsed_profiles))
    ordered = sorted(requests, key=lambda item: (item.sequence, item.request_id))
    train_ids = [item.request_id for item in ordered if item.split == "train"]
    validation_ids = [item.request_id for item in ordered if item.split == "validation"]
    test_ids = [item.request_id for item in ordered if item.split == "test"]
    if min(len(train_ids), len(validation_ids), len(test_ids)) < 2:
        raise BenchmarkError("each selection split requires at least two requests")

    training_rows = _training_observations(train_ids, request_map, outcome_map)
    selected_policy = policy or SelectionPolicy()

    validation_trials: list[dict[str, Any]] = []
    for epochs in (1, 2, 3, 5):
        candidate = SelectionIntelligence(
            parsed_profiles,
            policy=selected_policy,
            random_seed=seed,
        ).fit(training_rows, epochs=epochs)
        selected_cells: list[OutcomeMatrixObservation] = []
        brier_values: list[float] = []
        for request_id in validation_ids:
            metadata = request_map[request_id]
            choice = candidate.select(
                metadata.request,
                task=metadata.task,
                tools=metadata.tools,
                response_format=metadata.response_format,
                stream=metadata.stream,
            )
            selected_cells.append(outcome_map[request_id][choice.model_id])
            predicted = next(
                item.success_probability
                for item in choice.predictions
                if item.model_id == choice.model_id
            )
            brier_values.append(
                (predicted - int(outcome_map[request_id][choice.model_id].success)) ** 2
            )
        validation_trials.append(
            {
                "epochs": epochs,
                "accuracy": statistics.fmean(cell.quality_score for cell in selected_cells),
                "brier_score": statistics.fmean(brier_values),
                "mean_cost_usd": statistics.fmean(cell.cost_usd for cell in selected_cells),
            }
        )
    chosen_trial = min(
        validation_trials,
        key=lambda item: (
            -float(item["accuracy"]),
            float(item["brier_score"]),
            float(item["mean_cost_usd"]),
            int(item["epochs"]),
        ),
    )
    selected_epochs = int(chosen_trial["epochs"])

    def trained() -> SelectionIntelligence:
        return SelectionIntelligence(
            parsed_profiles,
            policy=selected_policy,
            random_seed=seed,
        ).fit(training_rows, epochs=selected_epochs)

    frozen = trained()
    frozen_selections: list[tuple[str, str, float | None]] = []
    for request_id in test_ids:
        metadata = request_map[request_id]
        choice = frozen.select(
            metadata.request,
            task=metadata.task,
            tools=metadata.tools,
            response_format=metadata.response_format,
            stream=metadata.stream,
        )
        probability = next(
            item.success_probability
            for item in choice.predictions
            if item.model_id == choice.model_id
        )
        frozen_selections.append((request_id, choice.model_id, probability))

    bandit = trained()
    bandit_selections: list[tuple[str, str, float | None]] = []
    for request_id in test_ids:
        metadata = request_map[request_id]
        choice = bandit.select(
            metadata.request,
            task=metadata.task,
            tools=metadata.tools,
            response_format=metadata.response_format,
            stream=metadata.stream,
            explore=selected_policy.exploration_rate > 0,
        )
        probability = next(
            item.success_probability
            for item in choice.predictions
            if item.model_id == choice.model_id
        )
        bandit_selections.append((request_id, choice.model_id, probability))
        cell = outcome_map[request_id][choice.model_id]
        bandit.observe(
            SelectionTrainingObservation(
                request=metadata.request,
                model_id=choice.model_id,
                success=cell.success,
                quality_score=100 * cell.quality_score,
                task=metadata.task,
                actual_cost_usd=cell.cost_usd,
                latency_ms=cell.total_latency_ms,
                selection_probability=choice.selection_probability,
                tools=metadata.tools,
                response_format=metadata.response_format,
                stream=metadata.stream,
            )
        )

    shadow = trained()
    shadow_selections: list[tuple[str, str, float | None]] = []
    for request_id in test_ids:
        metadata = request_map[request_id]
        choice = shadow.select(
            metadata.request,
            task=metadata.task,
            tools=metadata.tools,
            response_format=metadata.response_format,
            stream=metadata.stream,
        )
        probability = next(
            item.success_probability
            for item in choice.predictions
            if item.model_id == choice.model_id
        )
        shadow_selections.append((request_id, choice.model_id, probability))
        for cell in outcome_map[request_id].values():
            shadow.observe(
                SelectionTrainingObservation(
                    request=metadata.request,
                    model_id=cell.model_id,
                    success=cell.success,
                    quality_score=100 * cell.quality_score,
                    task=metadata.task,
                    actual_cost_usd=cell.cost_usd,
                    latency_ms=cell.total_latency_ms,
                    tools=metadata.tools,
                    response_format=metadata.response_format,
                    stream=metadata.stream,
                )
            )

    results: dict[str, dict[str, Any]] = {}
    results["openroutiq-frozen"] = _result("openroutiq-frozen", frozen_selections, outcome_map)
    results["openroutiq-self-learning-bandit"] = _result(
        "openroutiq-self-learning-bandit", bandit_selections, outcome_map
    )
    results["openroutiq-self-learning-shadow"] = _result(
        "openroutiq-self-learning-shadow", shadow_selections, outcome_map
    )
    segment_selections = _segment_rules(train_ids, test_ids, request_map, outcome_map, model_ids)
    results["segment-rules"] = _result("segment-rules", segment_selections, outcome_map)
    for name in ("ucb1", "thompson"):
        selections = _adaptive_baseline(
            name,
            train_ids,
            test_ids,
            request_map,
            outcome_map,
            model_ids,
            seed=seed,
        )
        results[name] = _result(name, selections, outcome_map)
    results["linucb"] = _result(
        "linucb",
        _linucb_baseline(train_ids, test_ids, request_map, outcome_map, model_ids),
        outcome_map,
    )

    fixed_accuracy = {
        model_id: statistics.fmean(outcome_map[item][model_id].quality_score for item in test_ids)
        for model_id in model_ids
    }
    best_fixed_model = min(
        model_ids,
        key=lambda item: (-fixed_accuracy[item], item),
    )
    results["best-fixed"] = _result(
        "best-fixed",
        [(request_id, best_fixed_model, None) for request_id in test_ids],
        outcome_map,
    )

    for name, external_selections in sorted((external_router_selections or {}).items()):
        if set(external_selections) != set(test_ids):
            raise BenchmarkError(f"external router {name} must select every test request")
        unknown = {item for item in external_selections.values() if item is not None} - set(
            model_ids
        )
        if unknown:
            raise BenchmarkError(
                f"external router {name} selected unknown models: {', '.join(sorted(unknown))}"
            )
        failures = (external_router_failures or {}).get(name, {})
        expected_failures = {
            request_id
            for request_id, selected_model in external_selections.items()
            if selected_model is None
        }
        if set(failures) != expected_failures:
            raise BenchmarkError(
                f"external router {name} failure outcomes do not match failed selections"
            )
        results[name] = _result(
            name,
            [(request_id, external_selections[request_id], None) for request_id in test_ids],
            outcome_map,
            failure_cells=failures,
        )

    oracle_vector = [
        max(cell.quality_score for cell in outcome_map[request_id].values())
        for request_id in test_ids
    ]
    for result in results.values():
        result["oracle_regret"] = statistics.fmean(
            oracle - selected
            for oracle, selected in zip(oracle_vector, result["quality_vector"], strict=True)
        )

    comparisons: dict[str, Any] = {}
    base = results["openroutiq-frozen"]
    online = results["openroutiq-self-learning-bandit"]
    base_confidence = 1 - 0.05 / max(1, len(base_comparators))
    online_confidence = 1 - 0.05 / max(1, len(online_comparators) + 1)
    for comparator in sorted(set(base_comparators) | set(online_comparators)):
        if comparator not in results:
            raise BenchmarkError(f"declared comparator was not measured: {comparator}")
        comparisons[f"base_vs_{comparator}"] = _paired_interval(
            base["quality_vector"],
            results[comparator]["quality_vector"],
            seed=seed,
            confidence=base_confidence,
        )
        comparisons[f"online_vs_{comparator}"] = _paired_interval(
            online["quality_vector"],
            results[comparator]["quality_vector"],
            seed=seed + 1,
            confidence=online_confidence,
        )

    base_pass = all(
        comparisons[f"base_vs_{name}"]["delta"] > 0 and comparisons[f"base_vs_{name}"]["ci_low"] > 0
        for name in base_comparators
    )
    online_pass = all(
        comparisons[f"online_vs_{name}"]["delta"] > 0
        and comparisons[f"online_vs_{name}"]["ci_low"] > 0
        for name in online_comparators
    )
    online_vs_frozen = comparisons["online_vs_openroutiq-frozen"]
    midpoint = len(test_ids) // 2
    late_window_vs_frozen = _paired_interval(
        online["quality_vector"][midpoint:],
        base["quality_vector"][midpoint:],
        seed=seed + 2,
        confidence=online_confidence,
    )
    early_count = max(1, len(test_ids) // 4)
    early_window_delta = statistics.fmean(
        selected - frozen_selected
        for selected, frozen_selected in zip(
            online["quality_vector"][:early_count],
            base["quality_vector"][:early_count],
            strict=True,
        )
    )
    matrix_analysis = analyze_outcome_matrix(
        [cell for request_id in test_ids for cell in outcome_map[request_id].values()],
        router_accuracies={name: item["accuracy"] for name, item in results.items()},
        actionable_headroom=minimum_oracle_headroom,
    )
    safe_results = []
    for name, result in sorted(results.items()):
        safe_result = {
            key: value
            for key, value in result.items()
            if key not in {"quality_vector", "success_vector", "selections"}
        }
        safe_result["selection_sha256"] = _selection_sha256(result["selections"])
        safe_results.append(safe_result)
    curve_names = (
        "openroutiq-frozen",
        "openroutiq-self-learning-bandit",
        "ucb1",
        "thompson",
        "linucb",
    )
    learning_curve: dict[str, list[dict[str, Any]]] = {}
    window_count = min(5, len(test_ids))
    for name in curve_names:
        result = results[name]
        windows: list[dict[str, Any]] = []
        for window_index in range(window_count):
            start = window_index * len(test_ids) // window_count
            end = (window_index + 1) * len(test_ids) // window_count
            values = result["quality_vector"][start:end]
            oracle_values = oracle_vector[start:end]
            cumulative = result["quality_vector"][:end]
            windows.append(
                {
                    "window": window_index + 1,
                    "start_case": start + 1,
                    "end_case": end,
                    "cases": end - start,
                    "accuracy": statistics.fmean(values),
                    "cumulative_accuracy": statistics.fmean(cumulative),
                    "oracle_regret": statistics.fmean(
                        oracle - selected
                        for oracle, selected in zip(oracle_values, values, strict=True)
                    ),
                }
            )
        learning_curve[name] = windows
    late_window_pass = late_window_vs_frozen["delta"] > 0 and late_window_vs_frozen["ci_low"] > 0
    early_safety_pass = early_window_delta >= -0.05
    return {
        "schema_version": 1,
        "method": "group-leakage-checked train/validation/test with prequential test updates",
        "split": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
            "test_sequence": "strictly increasing declared sequence; prediction precedes update",
        },
        "validation_selection": {
            "criterion": "highest accuracy, then lowest Brier score, cost, and epoch count",
            "selected_epochs": selected_epochs,
            "trials": validation_trials,
        },
        "models": list(model_ids),
        "results": safe_results,
        "learning_curve": {
            "ordering": "predeclared test sequence; prediction precedes selected-arm update",
            "windows": learning_curve,
        },
        "comparisons": comparisons,
        "oracle": matrix_analysis,
        "gates": {
            "oracle_headroom": {
                "minimum": minimum_oracle_headroom,
                "observed": matrix_analysis["oracle_headroom"],
                "passed": matrix_analysis["oracle_headroom"] >= minimum_oracle_headroom,
            },
            "base_beats_all_declared": {
                "comparators": list(base_comparators),
                "criterion": "positive paired mean delta and Bonferroni family-wise 5% bootstrap lower bound > 0",
                "per_comparison_confidence": base_confidence,
                "passed": base_pass,
            },
            "self_learning_beats_frozen_and_declared": {
                "evaluated_system": "openroutiq-self-learning-bandit",
                "comparators": list(online_comparators),
                "criterion": "positive paired mean delta and Bonferroni family-wise 5% bootstrap lower bound > 0",
                "per_comparison_confidence": online_confidence,
                "online_vs_frozen": online_vs_frozen,
                "lower_oracle_regret": online["oracle_regret"] < base["oracle_regret"],
                "passed": online_pass and online["oracle_regret"] < base["oracle_regret"],
            },
            "self_learning_improves_over_time": {
                "late_window": "second half of the test stream",
                "late_window_vs_frozen": late_window_vs_frozen,
                "early_window": "first quarter of the test stream",
                "early_window_delta_vs_frozen": early_window_delta,
                "maximum_allowed_early_regression": 0.05,
                "late_window_passed": late_window_pass,
                "early_safety_passed": early_safety_pass,
                "passed": late_window_pass and early_safety_pass,
            },
            "release_claim_ready": base_pass
            and online_pass
            and online["oracle_regret"] < base["oracle_regret"]
            and late_window_pass
            and early_safety_pass
            and matrix_analysis["oracle_headroom"] >= minimum_oracle_headroom,
        },
        "limitations": [
            "the release gate uses bandit self-learning and only the selected model outcome",
            "shadow self-learning consumes every candidate outcome and is an upper-bound diagnostic, not production-cost evidence",
            "online updates are not equivalent to safe automatic promotion; champion/challenger validation remains a separate release control",
            "no universal superiority claim is valid beyond named routers, models, requests, budgets, and evaluators",
        ],
    }
