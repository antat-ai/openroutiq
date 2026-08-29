from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping
from urllib.parse import parse_qsl, urlsplit

from openroutiq.router.capabilities import CapabilityGate, CapabilityRequirements
from openroutiq.router.failures import FailureType

if TYPE_CHECKING:
    from openroutiq.observability.dispatcher import Observability

TASKS = frozenset(
    {
        "general",
        "coding",
        "reasoning",
        "writing",
        "research",
        "extraction",
        "vision",
        "tool_use",
    }
)

API_STYLES = frozenset(
    {
        "openai_responses",
        "openai_chat",
        "anthropic_messages",
        "openrouter",
        "requesty",
        "openai_compatible",
        "litellm",
    }
)

REASONING_LEVELS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
REASONING_MODES = frozenset({"none", "effort", "adaptive", "budget"})
_CREDENTIAL_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|auth[_-]?token|"
    r"bearer[_-]?token|refresh[_-]?token|credential(?:s)?|^token$)$",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE = re.compile(
    r"^(?:bearer\s+|basic\s+|sk-[a-z0-9_-]{8,})",
    re.IGNORECASE,
)


class OpenRoutiQError(ValueError):
    """Base error for invalid catalogs and routing requests."""


class CatalogError(OpenRoutiQError):
    """Raised when model catalog data is invalid."""


class NoEligibleModelError(OpenRoutiQError):
    def __init__(self, excluded: Iterable[ExcludedModel]):
        self.excluded = tuple(excluded)
        detail = "; ".join(f"{item.model_id}: {', '.join(item.reasons)}" for item in self.excluded)
        super().__init__(f"No eligible model. {detail}" if detail else "No eligible model.")


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CatalogError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise CatalogError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise CatalogError(f"{name} must be <= {maximum}")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CatalogError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{name} must be a non-empty string")
    return value.strip()


def _credential_like_path(value: Any, *, path: str) -> str | None:
    """Return the first credential-like path without exposing its value."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            location = f"{path}.{name}"
            if _CREDENTIAL_KEY.search(name):
                return location
            found = _credential_like_path(item, path=location)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _credential_like_path(item, path=f"{path}[{index}]")
            if found is not None:
                return found
    elif isinstance(value, str) and _CREDENTIAL_VALUE.search(value.strip()):
        return path
    return None


def _string_set(value: Any, name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise CatalogError(f"{name} must be a list of non-empty strings")
    return frozenset(item.strip() for item in value)


def _reasoning_level(value: Any, name: str, *, allow_auto: bool = False) -> str:
    if not isinstance(value, str):
        raise CatalogError(f"{name} must be a reasoning level")
    level = {"min": "minimal"}.get(value.strip().lower(), value.strip().lower())
    allowed = REASONING_LEVELS | ({"auto"} if allow_auto else set())
    if level not in allowed:
        raise CatalogError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return level


def _default_api_style(provider: str) -> str:
    return {
        "openai": "openai_responses",
        "anthropic": "anthropic_messages",
        "openrouter": "openrouter",
        "requesty": "requesty",
        "litellm": "litellm",
    }.get(provider.lower(), "openai_compatible")


def _reasoning_set(value: Any, name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        raise CatalogError(f"{name} must be a list of reasoning levels")
    return frozenset(_reasoning_level(item, name) for item in value)


def _text_features(text: str) -> Counter[str]:
    # ponytail: bounded lexical features keep local inference predictable on oversized prompts.
    words = [word[:64] for word in re.findall(r"\w+|[+#.-]+", text.casefold())[:4096]]
    features: Counter[str] = Counter(f"w:{word}" for word in words)
    features.update(f"b:{left} {right}" for left, right in zip(words, words[1:]))
    for word in words:
        if len(word) < 4:
            continue
        padded = f"^{word}$"
        features.update(
            f"c:{padded[index : index + size]}"
            for size in (3, 4)
            for index in range(len(padded) - size + 1)
        )
    return features


@dataclass(frozen=True)
class TaskPrediction:
    task: str
    confidence: float
    scores: tuple[tuple[str, float], ...]


class TaskClassifier:
    """Small local TF-IDF classifier trained only from catalog examples."""

    def __init__(self, examples: Mapping[str, Sequence[str]]) -> None:
        if not isinstance(examples, Mapping) or len(examples) < 2:
            raise CatalogError("task_examples must contain at least two task labels")
        if any(not isinstance(label, str) or not label.strip() for label in examples):
            raise CatalogError("task_examples labels must be non-empty strings")
        self.labels = tuple(sorted(examples))
        documents: list[tuple[str, Counter[str]]] = []
        for label, samples in examples.items():
            if (
                isinstance(samples, (str, bytes, bytearray))
                or not isinstance(samples, Sequence)
                or not samples
            ):
                raise CatalogError(f"task_examples.{label} must be a non-empty list of prompts")
            for index, sample in enumerate(samples):
                if not isinstance(sample, str) or not sample.strip():
                    raise CatalogError(f"task_examples.{label}[{index}] must be non-empty text")
                documents.append((label, _text_features(sample)))

        document_frequency: Counter[str] = Counter()
        for _, features in documents:
            document_frequency.update(features.keys())
        self._idf = {
            name: math.log((1 + len(documents)) / (1 + count)) + 1
            for name, count in document_frequency.items()
        }
        centroids: dict[str, Counter[str]] = {label: Counter() for label in self.labels}
        document_counts: Counter[str] = Counter()
        for label, features in documents:
            centroids[label].update(self._vector(features))
            document_counts[label] += 1
        self._centroids = {
            label: self._normalize(
                {name: value / document_counts[label] for name, value in centroid.items()}
            )
            for label, centroid in centroids.items()
        }

    @staticmethod
    def _normalize(vector: Mapping[str, float]) -> dict[str, float]:
        length = math.sqrt(sum(value * value for value in vector.values()))
        return {name: value / length for name, value in vector.items()} if length else {}

    def _vector(self, features: Mapping[str, int]) -> dict[str, float]:
        return self._normalize(
            {
                name: (1 + math.log(count)) * self._idf[name]
                for name, count in features.items()
                if name in self._idf
            }
        )

    def predict(self, prompt: str) -> TaskPrediction:
        if not isinstance(prompt, str) or not prompt.strip():
            raise OpenRoutiQError("prompt must be a non-empty string")
        features = _text_features(prompt)
        vector = self._vector(features)
        similarities = {
            label: sum(
                value * self._centroids[label].get(name, 0.0) for name, value in vector.items()
            )
            for label in self.labels
        }
        sharpened = {label: similarity * similarity for label, similarity in similarities.items()}
        total = sum(sharpened.values())
        if total == 0:
            task = "general" if "general" in self.labels else self.labels[0]
            return TaskPrediction(task, 0.0, ((task, 1.0),))
        scores = {label: value / total for label, value in sharpened.items()}
        task = max(self.labels, key=lambda label: (scores[label], label == "general", label))
        ranked = sorted(similarities.values(), reverse=True)
        dominance = ranked[0] / max(ranked[0] + ranked[1], 1e-12)
        known = sum(count for name, count in features.items() if name in self._idf)
        coverage = known / max(1, sum(features.values()))
        confidence = 100 * dominance * min(1.0, coverage * 2)
        return TaskPrediction(task, confidence, tuple(sorted(scores.items())))


def _normalize_embedding(values: Iterable[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise OpenRoutiQError("embedder must return a numeric vector")
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise OpenRoutiQError("embedder must return a numeric vector") from exc
    if not vector or any(not math.isfinite(value) for value in vector):
        raise OpenRoutiQError("embedder must return a non-empty finite vector")
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        raise OpenRoutiQError("embedder must not return a zero vector")
    return tuple(value / length for value in vector)


def local_sentence_embedder(
    model_path: str | Path,
    *,
    device: str | None = None,
) -> Callable[[str], Iterable[float]]:
    """Load a Sentence Transformers model from disk without network access."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise OpenRoutiQError(f"embedding model directory does not exist: {path}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise OpenRoutiQError(
            "local_sentence_embedder requires 'pip install openroutiq[embeddings]'"
        ) from exc
    model = SentenceTransformer(
        str(path),
        device=device,
        local_files_only=True,
        trust_remote_code=False,
    )

    def embed(text: str) -> Iterable[float]:
        return model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    return embed


@dataclass(frozen=True)
class OutcomeScenario:
    """One jointly observed outcome used for empirical risk estimation."""

    quality_score: float
    latency_ms: float | None = None
    cost_usd: float | None = None
    success: bool | None = None
    failure_class: str | None = None
    weight: float = 1.0
    similarity: float = 1.0

    def __post_init__(self) -> None:
        _request_number(self.quality_score, "quality_score", minimum=0, maximum=100)
        if self.latency_ms is not None:
            _request_number(self.latency_ms, "latency_ms", minimum=0)
        if self.cost_usd is not None:
            _request_number(self.cost_usd, "cost_usd", minimum=0)
        if self.success is not None and not isinstance(self.success, bool):
            raise OpenRoutiQError("success must be a boolean or None")
        if self.failure_class is not None and (
            not isinstance(self.failure_class, str) or not self.failure_class.strip()
        ):
            raise OpenRoutiQError("failure_class must be non-empty text or None")
        _request_number(self.weight, "weight", minimum=0)
        if self.weight == 0:
            raise OpenRoutiQError("weight must be greater than zero")
        _request_number(self.similarity, "similarity", minimum=-1, maximum=1)


@dataclass(frozen=True)
class OutcomeEstimate:
    quality_score: float
    latency_ms: float | None
    similarity: float
    samples: int
    quality_stddev: float = 0.0
    quality_lower_bound: float | None = None
    success_probability: float | None = None
    failure_probabilities: tuple[tuple[str, float], ...] = ()
    average_cost_usd: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    cost_p50_usd: float | None = None
    cost_p95_usd: float | None = None
    effective_samples: float = 0.0
    scenarios: tuple[OutcomeScenario, ...] = ()


_CHANCE_EVENTS = frozenset({"success", "quality_at_least", "latency_at_most", "cost_at_most"})


@dataclass(frozen=True)
class ChanceConstraint:
    """Require an event to occur with at least the requested probability."""

    event: str
    minimum_probability: float
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event, str):
            raise OpenRoutiQError("chance constraint event must be text")
        event = self.event.strip().lower()
        if event not in _CHANCE_EVENTS:
            raise OpenRoutiQError(
                "chance constraint event must be success, quality_at_least, "
                "latency_at_most, or cost_at_most"
            )
        object.__setattr__(self, "event", event)
        _request_number(
            self.minimum_probability,
            "minimum_probability",
            minimum=0,
            maximum=1,
        )
        if event == "success":
            if self.threshold is not None:
                raise OpenRoutiQError("success chance constraints do not take a threshold")
        elif self.threshold is None:
            raise OpenRoutiQError(f"{event} chance constraints require a threshold")
        else:
            maximum = 100 if event == "quality_at_least" else None
            _request_number(self.threshold, "threshold", minimum=0, maximum=maximum)

    @classmethod
    def parse(cls, value: ChanceConstraint | Mapping[str, Any]) -> ChanceConstraint:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise OpenRoutiQError("each chance constraint must be an object")
        unknown = set(value) - {"event", "minimum_probability", "threshold"}
        if unknown:
            raise OpenRoutiQError(f"unknown chance constraint fields: {', '.join(sorted(unknown))}")
        event = value.get("event")
        if not isinstance(event, str):
            raise OpenRoutiQError("chance constraint event must be text")
        minimum_probability = _request_number(
            value.get("minimum_probability"),
            "minimum_probability",
            minimum=0,
            maximum=1,
        )
        return cls(
            event=event,
            minimum_probability=minimum_probability,
            threshold=value.get("threshold"),
        )

    @property
    def label(self) -> str:
        if self.event == "success":
            return "success"
        assert self.threshold is not None
        return f"{self.event}:{float(self.threshold):g}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "minimum_probability": self.minimum_probability,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class RiskPolicy:
    """Tail-risk and probabilistic-constraint policy for a routing decision."""

    constraints: tuple[ChanceConstraint, ...] = ()
    risk_aversion: float = 0.5
    cvar_alpha: float = 0.95
    constraint_penalty: float = 100.0
    minimum_samples: int = 0
    require_observed_probabilities: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.constraints, tuple):
            object.__setattr__(self, "constraints", tuple(self.constraints))
        parsed = tuple(ChanceConstraint.parse(item) for item in self.constraints)
        labels = [item.label for item in parsed]
        if len(labels) != len(set(labels)):
            raise OpenRoutiQError("risk policy contains duplicate chance constraints")
        object.__setattr__(self, "constraints", parsed)
        _request_number(self.risk_aversion, "risk_aversion", minimum=0, maximum=1)
        alpha = _request_number(self.cvar_alpha, "cvar_alpha", minimum=0, maximum=1)
        if alpha >= 1:
            raise OpenRoutiQError("cvar_alpha must be less than 1")
        _request_number(self.constraint_penalty, "constraint_penalty", minimum=0, maximum=100)
        _request_integer(self.minimum_samples, "minimum_samples")
        if not isinstance(self.require_observed_probabilities, bool):
            raise OpenRoutiQError("require_observed_probabilities must be a boolean")

    @classmethod
    def parse(cls, value: RiskPolicy | Mapping[str, Any] | None) -> RiskPolicy:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise OpenRoutiQError("risk_policy must be a RiskPolicy object or mapping")
        unknown = set(value) - {
            "constraints",
            "risk_aversion",
            "cvar_alpha",
            "constraint_penalty",
            "minimum_samples",
            "require_observed_probabilities",
        }
        if unknown:
            raise OpenRoutiQError(f"unknown risk policy fields: {', '.join(sorted(unknown))}")
        raw_constraints = value.get("constraints", ())
        if isinstance(raw_constraints, (str, bytes, bytearray)) or not isinstance(
            raw_constraints, Sequence
        ):
            raise OpenRoutiQError("risk_policy.constraints must be a sequence")
        return cls(
            constraints=tuple(ChanceConstraint.parse(item) for item in raw_constraints),
            risk_aversion=value.get("risk_aversion", 0.5),
            cvar_alpha=value.get("cvar_alpha", 0.95),
            constraint_penalty=value.get("constraint_penalty", 100.0),
            minimum_samples=value.get("minimum_samples", 0),
            require_observed_probabilities=value.get("require_observed_probabilities", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraints": [item.to_dict() for item in self.constraints],
            "risk_aversion": self.risk_aversion,
            "cvar_alpha": self.cvar_alpha,
            "constraint_penalty": self.constraint_penalty,
            "minimum_samples": self.minimum_samples,
            "require_observed_probabilities": self.require_observed_probabilities,
        }


@dataclass(frozen=True)
class OutcomeForecast:
    """Distributional summary for one candidate under the current request context."""

    expected_quality: float
    quality_stddev: float
    quality_lower_bound: float
    success_probability: float | None
    failure_probabilities: tuple[tuple[str, float], ...]
    expected_latency_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    expected_cost_usd: float
    cost_p50_usd: float
    cost_p95_usd: float
    cost_p99_usd: float
    evidence_samples: int
    effective_samples: float
    observed_metrics: frozenset[str]
    event_probabilities: tuple[tuple[str, float | None], ...] = ()
    joint_constraint_probability: float | None = None
    cvar_loss: float = 0.0
    cvar_alpha: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_quality": round(self.expected_quality, 4),
            "quality_stddev": round(self.quality_stddev, 4),
            "quality_lower_bound": round(self.quality_lower_bound, 4),
            "success_probability": (
                None if self.success_probability is None else round(self.success_probability, 6)
            ),
            "failure_probabilities": {
                name: round(value, 6) for name, value in self.failure_probabilities
            },
            "latency_ms": {
                "mean": round(self.expected_latency_ms, 2),
                "p50": round(self.latency_p50_ms, 2),
                "p95": round(self.latency_p95_ms, 2),
                "p99": round(self.latency_p99_ms, 2),
            },
            "cost_usd": {
                "mean": round(self.expected_cost_usd, 8),
                "p50": round(self.cost_p50_usd, 8),
                "p95": round(self.cost_p95_usd, 8),
                "p99": round(self.cost_p99_usd, 8),
            },
            "evidence_samples": self.evidence_samples,
            "effective_samples": round(self.effective_samples, 4),
            "observed_metrics": sorted(self.observed_metrics),
            "event_probabilities": {
                name: None if value is None else round(value, 6)
                for name, value in self.event_probabilities
            },
            "joint_constraint_probability": (
                None
                if self.joint_constraint_probability is None
                else round(self.joint_constraint_probability, 6)
            ),
            "cvar_loss": round(self.cvar_loss, 4),
            "cvar_alpha": self.cvar_alpha,
        }


@dataclass
class _EligibleCandidate:
    profile: ModelProfile
    quality: float
    latency: float
    cost: float
    catalog_quality: float | None
    context_quality: float | None
    similarity: float
    samples: int
    context_strength: float
    estimate: OutcomeEstimate | None
    forecast: OutcomeForecast | None = None
    scenarios: tuple[OutcomeScenario, ...] = ()


class OutcomeStore:
    """SQLite persistence for prompt-free evaluation embeddings and outcomes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection:
                connection.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS route_outcomes (
                        id INTEGER PRIMARY KEY,
                        embedding_id TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        quality_score REAL NOT NULL CHECK (quality_score BETWEEN 0 AND 100),
                        latency_ms REAL CHECK (latency_ms IS NULL OR latency_ms >= 0),
                        actual_cost_usd REAL CHECK (
                            actual_cost_usd IS NULL OR actual_cost_usd >= 0
                        ),
                        success INTEGER CHECK (success IS NULL OR success IN (0, 1)),
                        input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
                        output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
                        failure_class TEXT,
                        selection_probability REAL CHECK (
                            selection_probability IS NULL OR
                            (selection_probability > 0 AND selection_probability <= 1)
                        ),
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS route_outcomes_lookup
                    ON route_outcomes (embedding_id, model_id);
                    CREATE INDEX IF NOT EXISTS route_outcomes_lookup_recent
                    ON route_outcomes (embedding_id, model_id, id DESC);
                    CREATE INDEX IF NOT EXISTS route_outcomes_created
                    ON route_outcomes (created_at);
                    """
                )
                columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(route_outcomes)").fetchall()
                }
                migrations = {
                    "actual_cost_usd": "REAL CHECK (actual_cost_usd IS NULL OR actual_cost_usd >= 0)",
                    "success": "INTEGER CHECK (success IS NULL OR success IN (0, 1))",
                    "input_tokens": "INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0)",
                    "output_tokens": "INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0)",
                    "failure_class": "TEXT",
                    "selection_probability": (
                        "REAL CHECK (selection_probability IS NULL OR "
                        "(selection_probability > 0 AND selection_probability <= 1))"
                    ),
                }
                for name, definition in migrations.items():
                    if name not in columns:
                        connection.execute(
                            f"ALTER TABLE route_outcomes ADD COLUMN {name} {definition}"
                        )
                connection.commit()
            if os.name != "nt":
                os.chmod(self.path, 0o600)
        except (OSError, sqlite3.Error) as exc:
            raise OpenRoutiQError(f"cannot initialize outcome store {self.path}: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def add(
        self,
        *,
        embedding_id: str,
        model_id: str,
        embedding: Iterable[float],
        quality_score: float,
        latency_ms: float | None,
        metadata: Mapping[str, Any],
        actual_cost_usd: float | None = None,
        success: bool | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        failure_class: str | None = None,
        selection_probability: float | None = None,
    ) -> int:
        embedding_id = _text(embedding_id, "embedding_id")
        model_id = _text(model_id, "model_id")
        vector = _normalize_embedding(embedding)
        score = _request_number(quality_score, "quality_score", minimum=0, maximum=100)
        latency = (
            None if latency_ms is None else _request_number(latency_ms, "latency_ms", minimum=0)
        )
        actual_cost = (
            None
            if actual_cost_usd is None
            else _request_number(actual_cost_usd, "actual_cost_usd", minimum=0)
        )
        if success is not None and not isinstance(success, bool):
            raise OpenRoutiQError("success must be a boolean or None")
        input_count = (
            None if input_tokens is None else _request_integer(input_tokens, "input_tokens")
        )
        output_count = (
            None if output_tokens is None else _request_integer(output_tokens, "output_tokens")
        )
        failure = None if failure_class is None else _text(failure_class, "failure_class")
        propensity = (
            None
            if selection_probability is None
            else _request_number(
                selection_probability,
                "selection_probability",
                minimum=0,
                maximum=1,
            )
        )
        if propensity == 0:
            raise OpenRoutiQError("selection_probability must be greater than zero")
        try:
            metadata_json = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise OpenRoutiQError("outcome metadata must be JSON serializable") from exc
        with self._lock:
            try:
                with closing(self._connect()) as connection:
                    cursor = connection.execute(
                        """
                        INSERT INTO route_outcomes
                            (embedding_id, model_id, embedding_json, quality_score,
                             latency_ms, actual_cost_usd, success, input_tokens,
                             output_tokens, failure_class, selection_probability, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            embedding_id,
                            model_id,
                            json.dumps(vector, separators=(",", ":")),
                            score,
                            latency,
                            actual_cost,
                            None if success is None else int(success),
                            input_count,
                            output_count,
                            failure,
                            propensity,
                            metadata_json,
                        ),
                    )
                    connection.commit()
                    if cursor.lastrowid is None:
                        raise OpenRoutiQError("outcome store did not return an inserted row id")
                    return int(cursor.lastrowid)
            except sqlite3.Error as exc:
                raise OpenRoutiQError(f"cannot write outcome store {self.path}: {exc}") from exc

    def estimates(
        self,
        *,
        embedding_id: str,
        embedding: Iterable[float],
        model_ids: Iterable[str],
        neighbors: int,
        minimum_similarity: float,
        max_rows: int = 100_000,
    ) -> tuple[dict[str, OutcomeEstimate], float, bool]:
        embedding_id = _text(embedding_id, "embedding_id")
        query = _normalize_embedding(embedding)
        candidates = tuple(sorted({_text(model_id, "model_id") for model_id in model_ids}))
        neighbors = _integer(neighbors, "neighbors", minimum=1)
        minimum_similarity = _number(
            minimum_similarity,
            "minimum_similarity",
            minimum=-1,
            maximum=1,
        )
        max_rows = _integer(max_rows, "max_rows", minimum=1)
        if max_rows > 100_000:
            raise OpenRoutiQError("max_rows must be <= 100000")
        if not candidates:
            return {}, 0.0, False
        placeholders = ", ".join("?" for _ in candidates)
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    f"""
                    SELECT id, model_id, embedding_json, quality_score, latency_ms,
                           actual_cost_usd, success, failure_class, selection_probability
                    FROM route_outcomes
                    WHERE embedding_id = ? AND model_id IN ({placeholders})
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (embedding_id, *candidates, max_rows),
                ).fetchall()
        except sqlite3.Error as exc:
            raise OpenRoutiQError(f"cannot read outcome store {self.path}: {exc}") from exc

        grouped: dict[str, list[OutcomeScenario]] = {}
        maximum_similarity = 0.0
        has_data = False
        # Performance note: a linear scan is enough until outcome rows require a vector index.
        for (
            row_id,
            model_id,
            raw_embedding,
            quality,
            latency,
            actual_cost,
            success,
            failure_class,
            selection_probability,
        ) in rows:
            has_data = True
            try:
                stored = _normalize_embedding(json.loads(raw_embedding))
            except (json.JSONDecodeError, TypeError, OpenRoutiQError) as exc:
                raise OpenRoutiQError(f"invalid embedding in outcome row {row_id}") from exc
            if len(stored) != len(query):
                raise OpenRoutiQError(
                    f"embedding dimension mismatch in outcome row {row_id}; change embedding_id"
                )
            similarity = min(
                1.0,
                max(
                    -1.0,
                    sum(left * right for left, right in zip(query, stored, strict=True)),
                ),
            )
            maximum_similarity = max(maximum_similarity, similarity)
            if similarity >= minimum_similarity:
                grouped.setdefault(model_id, []).append(
                    OutcomeScenario(
                        quality_score=float(quality),
                        latency_ms=None if latency is None else float(latency),
                        cost_usd=None if actual_cost is None else float(actual_cost),
                        success=None if success is None else bool(success),
                        failure_class=None if failure_class is None else str(failure_class),
                        weight=max(similarity, 1e-9)
                        * (
                            min(100.0, 1.0 / float(selection_probability))
                            if selection_probability is not None
                            else 1.0
                        ),
                        similarity=similarity,
                    )
                )

        estimates: dict[str, OutcomeEstimate] = {}
        for model_id, matches in grouped.items():
            nearest = tuple(
                sorted(matches, key=lambda item: item.similarity, reverse=True)[:neighbors]
            )
            weights = [item.weight for item in nearest]
            quality_values = [(item.quality_score, item.weight) for item in nearest]
            quality = _weighted_mean(quality_values)
            quality_stddev = _weighted_stddev(quality_values, quality)
            effective_samples = _effective_sample_size(weights)
            quality_lower_bound = max(
                0.0,
                quality - 1.96 * quality_stddev / math.sqrt(max(effective_samples, 1.0)),
            )
            timed = [
                (float(item.latency_ms), item.weight)
                for item in nearest
                if item.latency_ms is not None
            ]
            costed = [
                (float(item.cost_usd), item.weight) for item in nearest if item.cost_usd is not None
            ]
            succeeded = [
                (bool(item.success), item.weight) for item in nearest if item.success is not None
            ]
            predicted_latency = _weighted_mean(timed) if timed else None
            average_cost = _weighted_mean(costed) if costed else None
            success_probability = (
                (1.0 + sum(weight for value, weight in succeeded if value))
                / (2.0 + sum(weight for _, weight in succeeded))
                if succeeded
                else None
            )
            known_outcome_weight = sum(item.weight for item in nearest if item.success is not None)
            failure_weights: dict[str, float] = {}
            if known_outcome_weight > 0:
                for item in nearest:
                    if item.success is not False:
                        continue
                    name = item.failure_class or "unclassified"
                    failure_weights[name] = failure_weights.get(name, 0.0) + item.weight
            failure_probabilities = tuple(
                sorted(
                    (name, weight / known_outcome_weight)
                    for name, weight in failure_weights.items()
                )
            )
            estimates[model_id] = OutcomeEstimate(
                quality_score=quality,
                latency_ms=predicted_latency,
                similarity=nearest[0].similarity,
                samples=len(nearest),
                quality_stddev=quality_stddev,
                quality_lower_bound=quality_lower_bound,
                success_probability=success_probability,
                failure_probabilities=failure_probabilities,
                average_cost_usd=average_cost,
                latency_p50_ms=_weighted_percentile(timed, 0.5) if timed else None,
                latency_p95_ms=_weighted_percentile(timed, 0.95) if timed else None,
                cost_p50_usd=_weighted_percentile(costed, 0.5) if costed else None,
                cost_p95_usd=_weighted_percentile(costed, 0.95) if costed else None,
                effective_samples=effective_samples,
                scenarios=nearest,
            )
        return estimates, maximum_similarity, has_data

    def prune(self, *, created_before: str) -> int:
        """Delete outcome rows older than an ISO-8601/SQLite timestamp boundary."""
        boundary = _text(created_before, "created_before")
        with self._lock:
            try:
                with closing(self._connect()) as connection:
                    cursor = connection.execute(
                        "DELETE FROM route_outcomes WHERE created_at < ?", (boundary,)
                    )
                    connection.commit()
                    return int(cursor.rowcount)
            except sqlite3.Error as exc:
                raise OpenRoutiQError(f"cannot prune outcome store {self.path}: {exc}") from exc


@dataclass(frozen=True)
class RouteContext:
    request: str | Sequence[Any]
    agent_role: str | None = None
    workflow_step: str | None = None
    side_effect_level: str | None = None
    budget_remaining: float | None = None
    latency_deadline_ms: float | None = None
    pinned_model: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("agent_role", self.agent_role),
            ("workflow_step", self.workflow_step),
            ("side_effect_level", self.side_effect_level),
            ("pinned_model", self.pinned_model),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise OpenRoutiQError(f"{name} must be non-empty text or None")
        if self.budget_remaining is not None:
            _request_number(self.budget_remaining, "budget_remaining", minimum=0)
        if self.latency_deadline_ms is not None:
            _request_number(self.latency_deadline_ms, "latency_deadline_ms", minimum=0)


def classify_task(prompt: str, task_classifier: TaskClassifier | None = None) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise OpenRoutiQError("prompt must be a non-empty string")
    return task_classifier.predict(prompt).task if task_classifier is not None else "general"


@dataclass(frozen=True)
class ContextAnalysis:
    task: str
    complexity: float
    confidence: float
    required_capabilities: frozenset[str]
    high_risk: bool
    estimated_input_tokens: int
    message_count: int
    signals: tuple[str, ...]
    task_scores: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "complexity": round(self.complexity, 2),
            "confidence": round(self.confidence, 2),
            "required_capabilities": sorted(self.required_capabilities),
            "high_risk": self.high_risk,
            "estimated_input_tokens": self.estimated_input_tokens,
            "message_count": self.message_count,
            "signals": list(self.signals),
            "task_scores": {task: round(score, 4) for task, score in self.task_scores},
        }


def analyze_context(
    request: str | Sequence[Any] | RouteContext,
    *,
    task_classifier: TaskClassifier | None = None,
) -> ContextAnalysis:
    context = request if isinstance(request, RouteContext) else RouteContext(request)
    text, message_count, inferred_capabilities, image_count = _request_context(context.request)
    semantic_text = _semantic_context_text(context, text)
    prediction = (
        task_classifier.predict(semantic_text)
        if task_classifier is not None
        else TaskPrediction("general", 0.0, (("general", 1.0),))
    )
    signals = ["local-task-classifier" if task_classifier is not None else "task-untrained"]
    for name, value in (
        ("agent-role", context.agent_role),
        ("workflow-step", context.workflow_step),
        ("side-effect", context.side_effect_level),
    ):
        if value is not None:
            signals.append(f"{name}:{value}")
    complexity = 10.0 + min(30.0, len(text) / 160) + min(15.0, max(0, message_count - 1) * 2.5)
    if "```" in text:
        complexity += 15
        signals.append("fenced-content")
    if "tools" in inferred_capabilities:
        complexity += 12
        signals.append("tools")
    if image_count:
        complexity += min(25, image_count * 15)
        signals.append("vision")

    # Estimation note: character/image estimates avoid a tokenizer dependency;
    # callers can pass exact token counts.
    estimated_tokens = max(1, math.ceil(len(text) / 4) + image_count * 1000)
    return ContextAnalysis(
        task=prediction.task,
        complexity=min(100.0, complexity),
        confidence=prediction.confidence,
        required_capabilities=frozenset(inferred_capabilities),
        high_risk=False,
        estimated_input_tokens=estimated_tokens,
        message_count=message_count,
        signals=tuple(signals),
        task_scores=prediction.scores,
    )


@dataclass(frozen=True)
class Weights:
    quality: float = 60.0
    latency: float = 25.0
    cost: float = 15.0

    def __post_init__(self) -> None:
        for name, value in (
            ("quality", self.quality),
            ("latency", self.latency),
            ("cost", self.cost),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 100
            ):
                raise OpenRoutiQError(f"{name} weight must be between 0 and 100")
        if self.quality + self.latency + self.cost == 0:
            raise OpenRoutiQError("at least one weight must be greater than zero")

    @classmethod
    def parse(cls, value: Weights | Mapping[str, Any] | None) -> Weights:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise OpenRoutiQError("weights must be a Weights object or mapping")
        unknown = set(value) - {"quality", "latency", "cost"}
        if unknown:
            raise OpenRoutiQError(f"unknown weights: {', '.join(sorted(unknown))}")
        return cls(**value)

    def to_dict(self) -> dict[str, float]:
        return {
            "quality": float(self.quality),
            "latency": float(self.latency),
            "cost": float(self.cost),
        }


@dataclass(frozen=True)
class Constraints:
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    min_context_tokens: int = 0
    allowed_providers: frozenset[str] = field(default_factory=frozenset)
    blocked_providers: frozenset[str] = field(default_factory=frozenset)
    local_only: bool = False
    max_predicted_cost: float | None = None
    max_latency_ms: float | None = None
    min_quality: float | None = None
    candidate_ids: frozenset[str] = field(default_factory=frozenset)
    reasoning_levels: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def parse(cls, value: Constraints | Mapping[str, Any] | None) -> Constraints:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise OpenRoutiQError("constraints must be a Constraints object or mapping")
        unknown = set(value) - {
            "required_capabilities",
            "min_context_tokens",
            "allowed_providers",
            "blocked_providers",
            "local_only",
            "max_predicted_cost",
            "max_latency_ms",
            "min_quality",
            "candidate_ids",
            "reasoning_levels",
        }
        if unknown:
            raise OpenRoutiQError(f"unknown constraints: {', '.join(sorted(unknown))}")
        try:
            min_context = _integer(value.get("min_context_tokens", 0), "min_context_tokens")
            max_cost_value = value.get("max_predicted_cost")
            max_cost = (
                None
                if max_cost_value is None
                else _number(max_cost_value, "max_predicted_cost", minimum=0)
            )
            max_latency_value = value.get("max_latency_ms")
            max_latency = (
                None
                if max_latency_value is None
                else _number(max_latency_value, "max_latency_ms", minimum=0)
            )
            min_quality_value = value.get("min_quality")
            min_quality = (
                None
                if min_quality_value is None
                else _number(min_quality_value, "min_quality", minimum=0, maximum=100)
            )
            local_only = value.get("local_only", False)
            if not isinstance(local_only, bool):
                raise CatalogError("local_only must be a boolean")
            return cls(
                required_capabilities=_string_set(
                    value.get("required_capabilities"), "required_capabilities"
                ),
                min_context_tokens=min_context,
                allowed_providers=_string_set(value.get("allowed_providers"), "allowed_providers"),
                blocked_providers=_string_set(value.get("blocked_providers"), "blocked_providers"),
                local_only=local_only,
                max_predicted_cost=max_cost,
                max_latency_ms=max_latency,
                min_quality=min_quality,
                candidate_ids=_string_set(value.get("candidate_ids"), "candidate_ids"),
                reasoning_levels=_reasoning_set(value.get("reasoning_levels"), "reasoning_levels"),
            )
        except CatalogError as exc:
            raise OpenRoutiQError(str(exc)) from exc


@dataclass(frozen=True)
class ModelProfile:
    id: str
    provider: str
    model: str
    quality: Mapping[str, float]
    latency_ms: float
    input_price_per_million: float
    output_price_per_million: float
    max_context_tokens: int
    capabilities: frozenset[str]
    supported_parameters: frozenset[str] = field(default_factory=frozenset)
    available: bool = True
    confidence: float = 75.0
    reasoning_level: str | None = None
    api_style: str = "openai_compatible"
    base_url: str | None = None
    reasoning_mode: str = "none"
    reasoning_budget_tokens: int | None = None
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    pricing: Mapping[str, float] = field(default_factory=dict)
    local: bool = False
    tags: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ModelProfile:
        if not isinstance(raw, Mapping):
            raise CatalogError("each model profile must be an object")
        quality_raw = raw.get("quality")
        if not isinstance(quality_raw, Mapping) or not quality_raw:
            raise CatalogError("quality must be a non-empty task-to-score object")
        quality: dict[str, float] = {}
        for task, score in quality_raw.items():
            task_name = _text(task, "quality task")
            quality[task_name] = _number(score, f"quality.{task_name}", minimum=0, maximum=100)

        available = raw.get("available", True)
        local = raw.get("local", False)
        if not isinstance(available, bool) or not isinstance(local, bool):
            raise CatalogError("available and local must be booleans")
        provider = _text(raw.get("provider"), "provider")
        reasoning = raw.get("reasoning_level")
        if reasoning is not None:
            reasoning = _reasoning_level(reasoning, "reasoning_level")
        api_style = raw.get("api_style", _default_api_style(provider))
        api_style = _text(api_style, "api_style").lower()
        if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", api_style) is None:
            raise CatalogError(
                "api_style must use 1-64 lowercase letters, digits, dots, dashes, or underscores"
            )
        reasoning_mode = raw.get("reasoning_mode", "effort" if reasoning is not None else "none")
        reasoning_mode = _text(reasoning_mode, "reasoning_mode").lower()
        if reasoning_mode not in REASONING_MODES:
            raise CatalogError(
                f"reasoning_mode must be one of: {', '.join(sorted(REASONING_MODES))}"
            )
        if reasoning is not None and reasoning_mode == "none":
            raise CatalogError("reasoning_mode cannot be none when reasoning_level is set")
        budget_raw = raw.get("reasoning_budget_tokens")
        budget = (
            None
            if budget_raw is None
            else _integer(budget_raw, "reasoning_budget_tokens", minimum=1)
        )
        if reasoning_mode == "budget" and budget is None:
            raise CatalogError("reasoning_budget_tokens is required for budget reasoning")
        base_url = raw.get("base_url")
        if base_url is not None:
            base_url = _text(base_url, "base_url")
            parsed_url = urlsplit(base_url)
            secret_query_key = next(
                (key for key, _ in parse_qsl(parsed_url.query) if _CREDENTIAL_KEY.search(key)),
                None,
            )
            if parsed_url.username or parsed_url.password or secret_query_key is not None:
                raise CatalogError("base_url cannot embed credentials")
        provider_options = raw.get("provider_options", {})
        if not isinstance(provider_options, Mapping) or any(
            not isinstance(key, str) or not key for key in provider_options
        ):
            raise CatalogError("provider_options must be an object with string keys")
        secret_path = _credential_like_path(provider_options, path="provider_options")
        if secret_path is not None:
            raise CatalogError(
                f"provider_options cannot contain credential-like field {secret_path}; "
                "use environment variables or the provider SDK's secret store"
            )
        pricing_raw = raw.get("pricing", {})
        if not isinstance(pricing_raw, Mapping) or any(
            not isinstance(key, str) or not key.strip() for key in pricing_raw
        ):
            raise CatalogError("pricing must be an object with string keys")
        pricing = {
            key.strip(): _number(value, f"pricing.{key}", minimum=0)
            for key, value in pricing_raw.items()
        }

        return cls(
            id=_text(raw.get("id"), "id"),
            provider=provider,
            model=_text(raw.get("model"), "model"),
            reasoning_level=reasoning,
            api_style=api_style,
            base_url=base_url,
            reasoning_mode=reasoning_mode,
            reasoning_budget_tokens=budget,
            provider_options=dict(provider_options),
            pricing=pricing,
            quality=quality,
            latency_ms=_number(raw.get("latency_ms"), "latency_ms", minimum=0),
            input_price_per_million=_number(
                raw.get("input_price_per_million"), "input_price_per_million", minimum=0
            ),
            output_price_per_million=_number(
                raw.get("output_price_per_million"), "output_price_per_million", minimum=0
            ),
            max_context_tokens=_integer(
                raw.get("max_context_tokens"), "max_context_tokens", minimum=1
            ),
            capabilities=_string_set(raw.get("capabilities"), "capabilities"),
            supported_parameters=_string_set(
                raw.get("supported_parameters"), "supported_parameters"
            ),
            available=available,
            confidence=_number(raw.get("confidence", 75), "confidence", minimum=0, maximum=100),
            local=local,
            tags=_string_set(raw.get("tags"), "tags"),
        )

    def quality_for(self, task: str) -> float | None:
        return self.quality.get(task, self.quality.get("general"))


@dataclass(frozen=True)
class CandidateScore:
    model_id: str
    provider: str
    provider_model: str
    reasoning_level: str | None
    api_style: str
    base_url: str | None
    reasoning_mode: str
    reasoning_budget_tokens: int | None
    provider_options: Mapping[str, Any]
    total_score: float
    quality_score: float
    latency_score: float
    cost_score: float
    expected_latency_ms: float
    predicted_cost: float
    confidence: float
    catalog_quality_score: float | None = None
    context_quality_score: float | None = None
    context_similarity: float = 0.0
    context_samples: int = 0
    context_weight: float = 0.0
    expected_score: float | None = None
    tail_score: float | None = None
    forecast: OutcomeForecast | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "provider": self.provider,
            "model": self.provider_model,
            "reasoning_level": self.reasoning_level,
            "api_style": self.api_style,
            "base_url": self.base_url,
            "total_score": round(self.total_score, 4),
            "quality_score": round(self.quality_score, 4),
            "latency_score": round(self.latency_score, 4),
            "cost_score": round(self.cost_score, 4),
            "expected_latency_ms": round(self.expected_latency_ms, 2),
            "predicted_cost": round(self.predicted_cost, 8),
            "confidence": round(self.confidence, 2),
            "catalog_quality_score": (
                None if self.catalog_quality_score is None else round(self.catalog_quality_score, 4)
            ),
            "context_quality_score": (
                None if self.context_quality_score is None else round(self.context_quality_score, 4)
            ),
            "context_similarity": round(self.context_similarity, 4),
            "context_samples": self.context_samples,
            "context_weight": round(self.context_weight, 4),
            "expected_score": (
                None if self.expected_score is None else round(self.expected_score, 4)
            ),
            "tail_score": None if self.tail_score is None else round(self.tail_score, 4),
            "forecast": None if self.forecast is None else self.forecast.to_dict(),
        }


@dataclass(frozen=True)
class ExcludedModel:
    model_id: str
    reasons: tuple[str, ...]
    failure_type: FailureType = FailureType.ROUTING_FAILURE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "reasons": list(self.reasons),
            "failure_type": self.failure_type.value,
        }


@dataclass(frozen=True)
class RouteDecision:
    selected: CandidateScore
    task: str
    strategy: str
    weights: Weights
    analysis: ContextAnalysis
    review_required: bool
    review_reasons: tuple[str, ...]
    ranked: tuple[CandidateScore, ...]
    excluded: tuple[ExcludedModel, ...]
    estimated_input_tokens: int
    expected_output_tokens: int
    out_of_domain: bool = False
    context_similarity: float = 0.0
    risk_policy: RiskPolicy | None = None
    selection_probability: float = 1.0

    def prepare(self, request: str | Sequence[Any] | RouteContext, **options: Any):
        """Build provider-native request kwargs without changing the original history."""
        from openroutiq.providers.requests import prepare_request

        raw_request = request.request if isinstance(request, RouteContext) else request
        return prepare_request(self, raw_request, **options)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "task": self.task,
            "strategy": self.strategy,
            "weights": self.weights.to_dict(),
            "analysis": self.analysis.to_dict(),
            "review_required": self.review_required,
            "review_reasons": list(self.review_reasons),
            "estimated_input_tokens": self.estimated_input_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "out_of_domain": self.out_of_domain,
            "context_similarity": round(self.context_similarity, 4),
            "risk_policy": None if self.risk_policy is None else self.risk_policy.to_dict(),
            "selection_probability": round(self.selection_probability, 8),
            "ranked": [item.to_dict() for item in self.ranked],
            "excluded": [item.to_dict() for item in self.excluded],
        }


class Router:
    def __init__(
        self,
        profiles: Iterable[ModelProfile | Mapping[str, Any]],
        *,
        weights: Weights | Mapping[str, Any] | None = None,
        review_margin: float = 5.0,
        min_confidence: float = 60.0,
        default_output_tokens: int = 512,
        latency_alpha: float = 0.3,
        strategy: str = "auto",
        context_analyzer: Callable[
            [str | Sequence[Any] | RouteContext, ContextAnalysis], ContextAnalysis
        ]
        | None = None,
        analysis_threshold: float = 70.0,
        task_examples: Mapping[str, Sequence[str]] | None = None,
        embedder: Callable[[str], Iterable[float]] | None = None,
        outcome_store: str | Path | None = None,
        embedding_id: str = "default",
        outcome_neighbors: int = 8,
        outcome_similarity_threshold: float = 0.5,
        outcome_prior_samples: int = 3,
        out_of_domain_model: str | None = None,
        outcome_max_rows: int = 100_000,
        outcome_prior: Callable[[str, str], OutcomeEstimate | None] | None = None,
        token_counter: Callable[[Any, Sequence[Any] | None], int] | None = None,
        cost_estimator: Callable[[ModelProfile, int, int], float] | None = None,
        capability_gate: CapabilityGate | None = None,
        observability: Observability | None = None,
    ) -> None:
        parsed = tuple(
            item if isinstance(item, ModelProfile) else ModelProfile.from_mapping(item)
            for item in profiles
        )
        if not parsed:
            raise CatalogError("catalog must contain at least one model")
        ids = [item.id for item in parsed]
        duplicates = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
        if duplicates:
            raise CatalogError(f"duplicate model ids: {', '.join(duplicates)}")
        self._profiles = parsed
        self._profiles_by_id = {item.id: item for item in parsed}
        self.weights = Weights.parse(weights)
        self.review_margin = _request_number(review_margin, "review_margin", minimum=0)
        self.min_confidence = _request_number(
            min_confidence, "min_confidence", minimum=0, maximum=100
        )
        self.default_output_tokens = _request_integer(
            default_output_tokens, "default_output_tokens"
        )
        self.latency_alpha = _request_number(latency_alpha, "latency_alpha", minimum=0, maximum=1)
        if self.latency_alpha == 0:
            raise OpenRoutiQError("latency_alpha must be greater than zero")
        _strategy_weights(strategy, 50, self.weights)
        self.strategy = strategy
        self.context_analyzer = context_analyzer
        self.task_classifier = TaskClassifier(task_examples) if task_examples is not None else None
        self.task_labels = (
            frozenset(task for profile in parsed for task in profile.quality)
            | frozenset(self.task_classifier.labels if self.task_classifier is not None else ())
            | TASKS
        )
        self.analysis_threshold = _request_number(
            analysis_threshold, "analysis_threshold", minimum=0, maximum=100
        )
        self._observed_latency: dict[str, float] = {}
        self._observed_quality: dict[tuple[str, str], float] = {}
        self._telemetry_lock = Lock()
        if (embedder is None) != (outcome_store is None):
            raise OpenRoutiQError("embedder and outcome_store must be configured together")
        if embedder is not None and not callable(embedder):
            raise OpenRoutiQError("embedder must be callable")
        self.embedder = embedder
        self.embedding_id = _text(embedding_id, "embedding_id")
        self.outcome_neighbors = _request_integer(outcome_neighbors, "outcome_neighbors")
        if self.outcome_neighbors < 1:
            raise OpenRoutiQError("outcome_neighbors must be at least 1")
        self.outcome_similarity_threshold = _request_number(
            outcome_similarity_threshold,
            "outcome_similarity_threshold",
            minimum=0,
            maximum=1,
        )
        if self.outcome_similarity_threshold >= 1:
            raise OpenRoutiQError("outcome_similarity_threshold must be less than 1")
        self.outcome_prior_samples = _request_integer(
            outcome_prior_samples, "outcome_prior_samples"
        )
        if self.outcome_prior_samples < 1:
            raise OpenRoutiQError("outcome_prior_samples must be at least 1")
        self.outcome_max_rows = _request_integer(outcome_max_rows, "outcome_max_rows")
        if self.outcome_max_rows < 1:
            raise OpenRoutiQError("outcome_max_rows must be at least 1")
        if out_of_domain_model is not None:
            out_of_domain_model = _text(out_of_domain_model, "out_of_domain_model")
            if out_of_domain_model not in self._profiles_by_id:
                raise OpenRoutiQError(f"unknown out_of_domain_model: {out_of_domain_model}")
            if outcome_store is None:
                raise OpenRoutiQError("out_of_domain_model requires an outcome_store")
        self.out_of_domain_model = out_of_domain_model
        self.outcome_store = OutcomeStore(outcome_store) if outcome_store is not None else None
        if outcome_prior is not None and not callable(outcome_prior):
            raise OpenRoutiQError("outcome_prior must be callable or None")
        self.outcome_prior = outcome_prior
        self._embedding_lock = Lock()
        if token_counter is not None and not callable(token_counter):
            raise OpenRoutiQError("token_counter must be callable")
        if cost_estimator is not None and not callable(cost_estimator):
            raise OpenRoutiQError("cost_estimator must be callable")
        self.token_counter = token_counter
        self.cost_estimator = cost_estimator
        if capability_gate is not None and not isinstance(capability_gate, CapabilityGate):
            raise OpenRoutiQError("capability_gate must be a CapabilityGate or None")
        self.capability_gate = capability_gate or CapabilityGate()
        if observability is not None:
            from openroutiq.observability.dispatcher import Observability

            if not isinstance(observability, Observability):
                raise OpenRoutiQError("observability must be an Observability object or None")
        self.observability = observability

    @classmethod
    def from_file(cls, path: str | Path, **options: Any) -> Router:
        catalog_path = Path(path)
        try:
            raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise CatalogError(f"cannot read catalog {catalog_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CatalogError(f"invalid JSON in catalog {catalog_path}: {exc}") from exc
        models = raw.get("models") if isinstance(raw, Mapping) else raw
        if not isinstance(models, list):
            raise CatalogError("catalog root must be a model list or an object with a models list")
        if isinstance(raw, Mapping) and "task_examples" not in options:
            options["task_examples"] = raw.get("task_examples")
        return cls(models, **options)

    @property
    def profiles(self) -> tuple[ModelProfile, ...]:
        return self._profiles

    def record_latency(self, model_id: str, latency_ms: float) -> float:
        if model_id not in self._profiles_by_id:
            raise OpenRoutiQError(f"unknown model id: {model_id}")
        observed = _request_number(latency_ms, "latency_ms", minimum=0)
        # ponytail: process-local EWMA; use a shared store only when multi-process telemetry is required.
        with self._telemetry_lock:
            baseline = self._observed_latency.get(
                model_id, self._profiles_by_id[model_id].latency_ms
            )
            updated = self.latency_alpha * observed + (1 - self.latency_alpha) * baseline
            self._observed_latency[model_id] = updated
        return updated

    def record_outcome(
        self, model_id: str, task: str, quality_score: float, *, alpha: float = 0.2
    ) -> float:
        if model_id not in self._profiles_by_id:
            raise OpenRoutiQError(f"unknown model id: {model_id}")
        if task not in self.task_labels:
            raise OpenRoutiQError(f"unknown task: {task}")
        score = _request_number(quality_score, "quality_score", minimum=0, maximum=100)
        weight = _request_number(alpha, "alpha", minimum=0, maximum=1)
        if weight == 0:
            raise OpenRoutiQError("alpha must be greater than zero")
        key = (model_id, task)
        with self._telemetry_lock:
            baseline = self._observed_quality.get(
                key, self._profiles_by_id[model_id].quality_for(task)
            )
            if baseline is None:
                baseline = score
            updated = weight * score + (1 - weight) * baseline
            self._observed_quality[key] = updated
        return updated

    def _embed(self, text: str) -> tuple[float, ...]:
        if self.embedder is None:
            raise OpenRoutiQError("contextual outcomes require an embedder and outcome_store")
        try:
            # ponytail: one lock keeps third-party local encoders safe; batch if throughput matters.
            with self._embedding_lock:
                return _normalize_embedding(self.embedder(text))
        except OpenRoutiQError:
            raise
        except Exception as exc:
            raise OpenRoutiQError(f"local embedder failed: {type(exc).__name__}") from exc

    def record_evaluation(
        self,
        request: str | Sequence[Any] | RouteContext,
        model_id: str,
        quality_score: float,
        *,
        task: str | None = None,
        latency_ms: float | None = None,
        actual_cost_usd: float | None = None,
        success: bool | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        failure_class: str | None = None,
        selection_probability: float | None = None,
        tools: Sequence[Any] | None = None,
        parallel_tool_calls: bool | None = None,
        response_format: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> int:
        if model_id not in self._profiles_by_id:
            raise OpenRoutiQError(f"unknown model id: {model_id}")
        if task is not None and (not isinstance(task, str) or not task.strip()):
            raise OpenRoutiQError("task must be non-empty text or None")
        if self.outcome_store is None:
            raise OpenRoutiQError("record_evaluation requires an embedder and outcome_store")
        score = _request_number(quality_score, "quality_score", minimum=0, maximum=100)
        latency = (
            None if latency_ms is None else _request_number(latency_ms, "latency_ms", minimum=0)
        )
        context = request if isinstance(request, RouteContext) else RouteContext(request)
        text = _learning_text(
            context,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            response_format=response_format,
            stream=stream,
        )
        row_id = self.outcome_store.add(
            embedding_id=self.embedding_id,
            model_id=model_id,
            embedding=self._embed(text),
            quality_score=score,
            latency_ms=latency,
            actual_cost_usd=actual_cost_usd,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            failure_class=failure_class,
            selection_probability=selection_probability,
            metadata=_outcome_metadata(
                context,
                tools=tools,
                parallel_tool_calls=parallel_tool_calls,
                response_format=response_format,
                stream=stream,
            ),
        )
        if latency is not None:
            self.record_latency(model_id, latency)
        if self.observability is not None:
            try:
                self.observability.record_evaluation(
                    model_id=model_id,
                    provider=self._profiles_by_id[model_id].provider,
                    task=task.strip() if task is not None else None,
                    quality_score=score,
                    latency_ms=latency,
                    actual_cost_usd=actual_cost_usd,
                    success=success,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            except Exception:
                # Export must never change a persisted evaluation or learning result.
                pass
        return row_id

    def route(
        self,
        prompt: str | Sequence[Any] | RouteContext,
        *,
        task: str | None = None,
        weights: Weights | Mapping[str, Any] | None = None,
        constraints: Constraints | Mapping[str, Any] | None = None,
        risk_policy: RiskPolicy | Mapping[str, Any] | None = None,
        input_tokens: int | None = None,
        minimum_input_tokens: int | None = None,
        expected_output_tokens: int | None = None,
        required_context_tokens: int | None = None,
        high_risk: bool | None = None,
        soft_budget: float | None = None,
        strategy: str | None = None,
        complexity: float | None = None,
        tools: Sequence[Any] | None = None,
        tool_choice: Any = None,
        parallel_tool_calls: bool | None = None,
        response_format: Mapping[str, Any] | None = None,
        output_modalities: Sequence[str] | None = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        pinned_model: str | None = None,
    ) -> RouteDecision:
        observability_started = perf_counter() if self.observability is not None else None
        context = prompt if isinstance(prompt, RouteContext) else RouteContext(prompt)
        if pinned_model is None:
            pinned_model = context.pinned_model
        analysis = analyze_context(context, task_classifier=self.task_classifier)
        if tools is not None and (
            not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray))
        ):
            raise OpenRoutiQError("tools must be a sequence")
        if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
            raise OpenRoutiQError("parallel_tool_calls must be a boolean or None")
        if not isinstance(stream, bool):
            raise OpenRoutiQError("stream must be a boolean")
        if response_format is not None and not isinstance(response_format, Mapping):
            raise OpenRoutiQError("response_format must be an object or None")
        if output_modalities is not None and (
            isinstance(output_modalities, (str, bytes, bytearray))
            or not isinstance(output_modalities, Sequence)
            or any(not isinstance(item, str) or not item.strip() for item in output_modalities)
        ):
            raise OpenRoutiQError("output_modalities must be a sequence of non-empty strings")
        tool_count, tool_tokens = _tool_summary(tools)
        if not tool_count and tool_choice not in (None, "none"):
            raise OpenRoutiQError("tool_choice requires tools")
        tools_enabled = bool(tool_count) and tool_choice != "none"
        if parallel_tool_calls and not tools_enabled:
            raise OpenRoutiQError("parallel_tool_calls requires enabled tools")

        capabilities = set(analysis.required_capabilities)
        signals = list(analysis.signals)
        inferred_complexity = analysis.complexity
        inferred_task = analysis.task
        inferred_confidence = analysis.confidence
        if tools_enabled:
            capabilities.add("tools")
            inferred_complexity += min(18, 4 + tool_count * 2)
            signals.append(f"tool-definitions:{tool_count}")
        if parallel_tool_calls:
            capabilities.add("parallel_tools")
            inferred_complexity += 8
            signals.append("parallel-tools")
        if response_format is not None:
            capabilities.add("json_schema")
            inferred_complexity += 5
            signals.append("structured-output")
        for modality in output_modalities or ():
            normalized_modality = modality.strip().lower()
            if normalized_modality != "text":
                capabilities.add(f"{normalized_modality}_output")
                signals.append(f"output:{normalized_modality}")
        if stream:
            capabilities.add("streaming")
            signals.append("streaming")
        analysis = ContextAnalysis(
            task=inferred_task,
            complexity=min(100.0, inferred_complexity),
            confidence=inferred_confidence,
            required_capabilities=frozenset(capabilities),
            high_risk=analysis.high_risk,
            estimated_input_tokens=analysis.estimated_input_tokens + tool_tokens,
            message_count=analysis.message_count,
            signals=tuple(signals),
            task_scores=analysis.task_scores,
        )
        mandatory_capabilities = analysis.required_capabilities
        minimum_estimated_tokens = analysis.estimated_input_tokens
        if (
            task is None
            and self.context_analyzer is not None
            and analysis.confidence < self.analysis_threshold
        ):
            try:
                enriched = self.context_analyzer(prompt, analysis)
                if not isinstance(enriched, ContextAnalysis):
                    raise TypeError("context analyzer must return ContextAnalysis")
                if enriched.task not in self.task_labels:
                    raise ValueError("context analyzer returned an unknown task")
                analysis = replace(
                    enriched,
                    task_scores=_normalized_task_scores(
                        enriched.task, enriched.task_scores, self.task_labels
                    ),
                )
            except Exception:
                analysis = ContextAnalysis(
                    task=analysis.task,
                    complexity=analysis.complexity,
                    confidence=analysis.confidence,
                    required_capabilities=analysis.required_capabilities,
                    high_risk=analysis.high_risk,
                    estimated_input_tokens=analysis.estimated_input_tokens,
                    message_count=analysis.message_count,
                    signals=analysis.signals + ("context-analyzer-fallback",),
                    task_scores=analysis.task_scores,
                )
        if (
            not mandatory_capabilities.issubset(analysis.required_capabilities)
            or analysis.estimated_input_tokens < minimum_estimated_tokens
        ):
            analysis = ContextAnalysis(
                task=analysis.task,
                complexity=analysis.complexity,
                confidence=analysis.confidence,
                required_capabilities=analysis.required_capabilities | mandatory_capabilities,
                high_risk=analysis.high_risk,
                estimated_input_tokens=max(
                    analysis.estimated_input_tokens, minimum_estimated_tokens
                ),
                message_count=analysis.message_count,
                signals=analysis.signals + ("capability-guard",),
                task_scores=analysis.task_scores,
            )
        selected_task = analysis.task if task is None else task
        if selected_task not in self.task_labels:
            raise OpenRoutiQError(f"unknown task: {selected_task}")
        effective_complexity = (
            analysis.complexity
            if complexity is None
            else _request_number(complexity, "complexity", minimum=0, maximum=100)
        )
        if high_risk is not None and not isinstance(high_risk, bool):
            raise OpenRoutiQError("high_risk must be a boolean or None")
        effective_high_risk = analysis.high_risk if high_risk is None else high_risk
        signals = list(analysis.signals)
        if task is not None:
            signals.append("task-override")
        if complexity is not None:
            signals.append("complexity-override")
        requested_reasoning: str | None = None
        if reasoning_effort is not None:
            try:
                requested_reasoning = _reasoning_level(
                    reasoning_effort, "reasoning_effort", allow_auto=True
                )
            except CatalogError as exc:
                raise OpenRoutiQError(str(exc)) from exc
            if requested_reasoning == "auto":
                requested_reasoning = None
            else:
                signals.append(f"reasoning:{requested_reasoning}")
        if pinned_model is not None:
            if not isinstance(pinned_model, str) or not pinned_model.strip():
                raise OpenRoutiQError("pinned_model must be a non-empty model id or None")
            pinned_model = pinned_model.strip()
            if pinned_model not in self._profiles_by_id:
                raise OpenRoutiQError(f"unknown pinned model id: {pinned_model}")
            signals.append("model-pinned")
        analysis = ContextAnalysis(
            task=selected_task,
            complexity=effective_complexity,
            confidence=100.0 if task is not None else analysis.confidence,
            required_capabilities=analysis.required_capabilities,
            high_risk=effective_high_risk,
            estimated_input_tokens=analysis.estimated_input_tokens,
            message_count=analysis.message_count,
            signals=tuple(signals),
            task_scores=(
                ((selected_task, 1.0),)
                if task is not None
                else _normalized_task_scores(selected_task, analysis.task_scores, self.task_labels)
            ),
        )
        selected_strategy = strategy or self.strategy
        risk_enabled = risk_policy is not None or (
            isinstance(selected_strategy, str) and selected_strategy.lower() == "risk_aware"
        )
        effective_risk_policy = RiskPolicy.parse(risk_policy) if risk_enabled else None
        route_weights = (
            Weights.parse(weights)
            if weights is not None
            else _strategy_weights(selected_strategy, effective_complexity, self.weights)
        )
        decision_strategy = "custom" if weights is not None else selected_strategy
        if effective_risk_policy is not None:
            analysis = replace(analysis, signals=analysis.signals + ("risk-aware",))
        limits = Constraints.parse(constraints)
        limits = replace(
            limits,
            max_predicted_cost=_minimum_limit(limits.max_predicted_cost, context.budget_remaining),
            max_latency_ms=_minimum_limit(limits.max_latency_ms, context.latency_deadline_ms),
        )

        if input_tokens is not None:
            estimated_input = _request_integer(input_tokens, "input_tokens")
        elif self.token_counter is not None:
            try:
                estimated_input = _request_integer(
                    self.token_counter(context, tools), "token_counter result"
                )
            except OpenRoutiQError:
                raise
            except Exception as exc:
                raise OpenRoutiQError(f"token_counter failed: {type(exc).__name__}") from exc
        else:
            estimated_input = analysis.estimated_input_tokens
        # ``input_tokens`` retains trusted SDK exact-count semantics. Untrusted adapters such as
        # the proxy use a floor so a declaration can only make routing more conservative.
        if minimum_input_tokens is not None:
            estimated_input = max(
                estimated_input,
                _request_integer(minimum_input_tokens, "minimum_input_tokens"),
            )
        expected_output = (
            self.default_output_tokens
            if expected_output_tokens is None
            else _request_integer(expected_output_tokens, "expected_output_tokens")
        )
        review_budget = (
            None if soft_budget is None else _request_number(soft_budget, "soft_budget", minimum=0)
        )

        required_capabilities = set(limits.required_capabilities | analysis.required_capabilities)
        if required_capabilities != set(analysis.required_capabilities):
            analysis = ContextAnalysis(
                task=analysis.task,
                complexity=analysis.complexity,
                confidence=analysis.confidence,
                required_capabilities=frozenset(required_capabilities),
                high_risk=analysis.high_risk,
                estimated_input_tokens=analysis.estimated_input_tokens,
                message_count=analysis.message_count,
                signals=analysis.signals + ("capability-override",),
                task_scores=analysis.task_scores,
            )

        context_requirement = (
            estimated_input + expected_output
            if required_context_tokens is None
            else _request_integer(required_context_tokens, "required_context_tokens")
        )
        required_context = max(limits.min_context_tokens, context_requirement)
        required_parameters: set[str] = set()
        any_parameter_groups: list[frozenset[str]] = []
        if tools_enabled:
            required_parameters.add("tools")
        if tool_choice not in (None, "none"):
            required_parameters.add("tool_choice")
        if parallel_tool_calls:
            required_parameters.add("parallel_tool_calls")
        if response_format is not None:
            any_parameter_groups.append(frozenset({"response_format", "structured_outputs"}))
        if requested_reasoning is not None and requested_reasoning != "none":
            any_parameter_groups.append(frozenset({"reasoning", "reasoning_effort"}))
        capability_requirements = CapabilityRequirements(
            capabilities=frozenset(required_capabilities),
            required_parameters=frozenset(required_parameters),
            any_parameter_groups=tuple(any_parameter_groups),
            context_tokens=required_context,
            reasoning_level=requested_reasoning,
        )

        contextual_estimates: dict[str, OutcomeEstimate] = {}
        context_similarity = 0.0
        outcome_has_data = False
        out_of_domain = False
        if self.outcome_store is not None:
            contextual_estimates, context_similarity, outcome_has_data = (
                self.outcome_store.estimates(
                    embedding_id=self.embedding_id,
                    embedding=self._embed(
                        _learning_text(
                            context,
                            tools=tools,
                            parallel_tool_calls=parallel_tool_calls,
                            response_format=response_format,
                            stream=stream,
                        )
                    ),
                    model_ids=self._profiles_by_id,
                    neighbors=self.outcome_neighbors,
                    minimum_similarity=self.outcome_similarity_threshold,
                    max_rows=self.outcome_max_rows,
                )
            )
            out_of_domain = (
                outcome_has_data and context_similarity < self.outcome_similarity_threshold
            )
            outcome_signal = (
                "outcome-out-of-domain"
                if out_of_domain
                else "outcome-neighbors"
                if contextual_estimates
                else "outcomes-cold-start"
            )
            analysis = replace(analysis, signals=analysis.signals + (outcome_signal,))
            if out_of_domain and pinned_model is None and self.out_of_domain_model is not None:
                pinned_model = self.out_of_domain_model
                analysis = replace(
                    analysis,
                    signals=analysis.signals + (f"ood-fallback:{pinned_model}",),
                )

        with self._telemetry_lock:
            observed_latency = dict(self._observed_latency)
            observed_quality = dict(self._observed_quality)
        eligible: list[_EligibleCandidate] = []
        excluded: list[ExcludedModel] = []
        for profile in self._profiles:
            latency = observed_latency.get(profile.id, profile.latency_ms)
            if self.cost_estimator is None:
                cost = _predicted_cost(profile, estimated_input, expected_output)
            else:
                try:
                    cost = _request_number(
                        self.cost_estimator(profile, estimated_input, expected_output),
                        "cost_estimator result",
                        minimum=0,
                    )
                except OpenRoutiQError:
                    raise
                except Exception as exc:
                    raise OpenRoutiQError(f"cost_estimator failed: {type(exc).__name__}") from exc
            reasons: list[str] = []
            if not profile.available:
                reasons.append("unavailable")
            if pinned_model is not None and profile.id != pinned_model:
                reasons.append("not the pinned model")
            if limits.candidate_ids and profile.id not in limits.candidate_ids:
                reasons.append("not in candidate_ids")
            if limits.allowed_providers and profile.provider not in limits.allowed_providers:
                reasons.append("provider not allowed")
            if profile.provider in limits.blocked_providers:
                reasons.append("provider blocked")
            if limits.local_only and not profile.local:
                reasons.append("not local")
            gate_result = self.capability_gate.evaluate(profile, capability_requirements)
            reasons.extend(gate_result.reasons)
            if limits.max_predicted_cost is not None and cost > limits.max_predicted_cost:
                reasons.append(
                    f"predicted cost {cost:.8f} exceeds hard limit {limits.max_predicted_cost:.8f}"
                )
            catalog_quality = _effective_quality(
                profile, analysis.task_scores, required_capabilities, observed_quality
            )
            contextual_estimate = contextual_estimates.get(profile.id)
            prior_estimate: OutcomeEstimate | None = None
            if self.outcome_prior is not None:
                try:
                    prior_estimate = self.outcome_prior(profile.id, selected_task)
                except OpenRoutiQError:
                    raise
                except Exception as exc:
                    raise OpenRoutiQError(
                        f"outcome_prior failed for {profile.id}: {type(exc).__name__}"
                    ) from exc
                if prior_estimate is not None and not isinstance(prior_estimate, OutcomeEstimate):
                    raise OpenRoutiQError("outcome_prior must return OutcomeEstimate or None")
            estimate = contextual_estimate or prior_estimate
            context_quality = (
                contextual_estimate.quality_score if contextual_estimate is not None else None
            )
            context_strength = (
                _outcome_strength(
                    contextual_estimate,
                    minimum_similarity=self.outcome_similarity_threshold,
                    prior_samples=self.outcome_prior_samples,
                )
                if contextual_estimate is not None
                else 0.0
            )
            quality = (
                context_quality
                if catalog_quality is None
                else catalog_quality
                if context_quality is None
                else (1 - context_strength) * catalog_quality + context_strength * context_quality
            )
            if contextual_estimate is not None and contextual_estimate.latency_ms is not None:
                latency = (
                    1 - context_strength
                ) * latency + context_strength * contextual_estimate.latency_ms
            if quality is None:
                reasons.append(f"missing quality score for {selected_task} and general")
            elif limits.min_quality is not None and quality < limits.min_quality:
                reasons.append(f"quality {quality:.2f} < required {limits.min_quality:.2f}")
            if limits.max_latency_ms is not None and latency > limits.max_latency_ms:
                reasons.append(
                    f"latency {latency:.2f}ms exceeds limit {limits.max_latency_ms:.2f}ms"
                )
            profile_reasoning = profile.reasoning_level or "none"
            if limits.reasoning_levels and profile_reasoning not in limits.reasoning_levels:
                reasons.append("reasoning level not allowed")
            if (
                profile.reasoning_mode == "budget"
                and profile.reasoning_budget_tokens is not None
                and expected_output <= profile.reasoning_budget_tokens
            ):
                reasons.append("expected output tokens must exceed reasoning budget")
            if reasons:
                excluded.append(
                    ExcludedModel(
                        profile.id,
                        tuple(reasons),
                        gate_result.failure_type or FailureType.ROUTING_FAILURE,
                    )
                )
            else:
                if quality is None:
                    raise OpenRoutiQError(
                        f"eligible model {profile.id} has no quality score for task {selected_task}"
                    )
                candidate = _EligibleCandidate(
                    profile=profile,
                    quality=float(quality),
                    latency=latency,
                    cost=cost,
                    catalog_quality=catalog_quality,
                    context_quality=context_quality,
                    similarity=(
                        contextual_estimate.similarity if contextual_estimate is not None else 0.0
                    ),
                    samples=(contextual_estimate.samples if contextual_estimate is not None else 0),
                    context_strength=context_strength,
                    estimate=estimate,
                )
                if effective_risk_policy is not None:
                    candidate.forecast, candidate.scenarios = _outcome_forecast(
                        candidate,
                        effective_risk_policy,
                    )
                    risk_reasons = _chance_constraint_reasons(
                        candidate.forecast,
                        effective_risk_policy,
                    )
                    if risk_reasons:
                        excluded.append(ExcludedModel(profile.id, tuple(risk_reasons)))
                        continue
                eligible.append(candidate)

        if not eligible:
            raise NoEligibleModelError(excluded)

        latency_values = [
            item.forecast.expected_latency_ms if item.forecast is not None else item.latency
            for item in eligible
        ]
        cost_values = [
            item.forecast.expected_cost_usd if item.forecast is not None else item.cost
            for item in eligible
        ]
        latency_scores = _inverse_scores(latency_values)
        cost_scores = _inverse_scores(cost_values)
        total_weight = route_weights.quality + route_weights.latency + route_weights.cost
        ranked: list[CandidateScore] = []
        latency_bounds = _scenario_bounds(
            eligible,
            metric="latency",
            fallback=latency_values,
        )
        cost_bounds = _scenario_bounds(
            eligible,
            metric="cost",
            fallback=cost_values,
        )
        for candidate, latency_score, cost_score in zip(
            eligible, latency_scores, cost_scores, strict=True
        ):
            profile = candidate.profile
            expected_total = (
                route_weights.quality * candidate.quality
                + route_weights.latency * latency_score
                + route_weights.cost * cost_score
            ) / total_weight
            tail_score: float | None = None
            total = expected_total
            forecast = candidate.forecast
            if effective_risk_policy is not None and forecast is not None:
                cvar_loss = _candidate_cvar_loss(
                    candidate.scenarios,
                    route_weights,
                    effective_risk_policy,
                    latency_bounds=latency_bounds,
                    cost_bounds=cost_bounds,
                )
                tail_score = max(0.0, 100.0 - cvar_loss)
                total = (
                    1 - effective_risk_policy.risk_aversion
                ) * expected_total + effective_risk_policy.risk_aversion * tail_score
                forecast = replace(
                    forecast,
                    cvar_loss=cvar_loss,
                    cvar_alpha=effective_risk_policy.cvar_alpha,
                )
            ranked.append(
                CandidateScore(
                    model_id=profile.id,
                    provider=profile.provider,
                    provider_model=profile.model,
                    reasoning_level=profile.reasoning_level,
                    api_style=profile.api_style,
                    base_url=profile.base_url,
                    reasoning_mode=profile.reasoning_mode,
                    reasoning_budget_tokens=profile.reasoning_budget_tokens,
                    provider_options=dict(profile.provider_options),
                    total_score=total,
                    quality_score=candidate.quality,
                    latency_score=latency_score,
                    cost_score=cost_score,
                    expected_latency_ms=candidate.latency,
                    predicted_cost=candidate.cost,
                    confidence=profile.confidence,
                    catalog_quality_score=candidate.catalog_quality,
                    context_quality_score=candidate.context_quality,
                    context_similarity=candidate.similarity,
                    context_samples=candidate.samples,
                    context_weight=candidate.context_strength,
                    expected_score=expected_total if forecast is not None else None,
                    tail_score=tail_score,
                    forecast=forecast,
                )
            )
        ranked.sort(
            key=lambda item: (
                -item.total_score,
                -item.quality_score,
                item.predicted_cost,
                item.expected_latency_ms,
                item.model_id,
            )
        )

        winner = ranked[0]
        review_reasons: list[str] = []
        if effective_high_risk:
            review_reasons.append("request marked high-risk")
        if out_of_domain:
            review_reasons.append(
                f"context similarity {context_similarity:.2f} is below outcome threshold "
                f"{self.outcome_similarity_threshold:.2f}"
            )
        if (
            task is None
            and winner.context_weight < 0.1
            and analysis.confidence < self.analysis_threshold
        ):
            review_reasons.append(
                f"task confidence {analysis.confidence:.2f} is below threshold {self.analysis_threshold:.2f}"
            )
        if len(ranked) > 1:
            margin = winner.total_score - ranked[1].total_score
            if margin < self.review_margin:
                review_reasons.append(
                    f"winner margin {margin:.2f} is below review threshold {self.review_margin:.2f}"
                )
        if winner.confidence < self.min_confidence:
            review_reasons.append(
                f"winner confidence {winner.confidence:.2f} is below threshold {self.min_confidence:.2f}"
            )
        if winner.forecast is not None and winner.forecast.evidence_samples == 0:
            review_reasons.append(
                "risk forecast uses catalog priors because no similar evaluated outcomes exist"
            )
        if review_budget is not None and winner.predicted_cost > review_budget:
            review_reasons.append(
                f"predicted cost {winner.predicted_cost:.8f} exceeds soft budget {review_budget:.8f}"
            )

        decision = RouteDecision(
            selected=winner,
            task=selected_task,
            strategy=decision_strategy,
            weights=route_weights,
            analysis=analysis,
            review_required=bool(review_reasons),
            review_reasons=tuple(review_reasons),
            ranked=tuple(ranked),
            excluded=tuple(excluded),
            estimated_input_tokens=estimated_input,
            expected_output_tokens=expected_output,
            out_of_domain=out_of_domain,
            context_similarity=context_similarity,
            risk_policy=effective_risk_policy,
        )
        if self.observability is not None:
            try:
                duration_ms = (
                    None
                    if observability_started is None
                    else (perf_counter() - observability_started) * 1_000
                )
                self.observability.record_route(decision, duration_ms=duration_ms)
            except Exception:
                # Selection is final before export and must be returned unchanged.
                pass
        return decision


def _minimum_limit(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _outcome_strength(
    estimate: OutcomeEstimate,
    *,
    minimum_similarity: float,
    prior_samples: int,
) -> float:
    similarity = max(
        0.0,
        (estimate.similarity - minimum_similarity) / (1 - minimum_similarity),
    )
    support = estimate.samples / (estimate.samples + prior_samples)
    return min(1.0, similarity * support)


def _semantic_context_text(context: RouteContext, request_text: str) -> str:
    parts = [request_text]
    for label, value in (
        ("agent role", context.agent_role),
        ("workflow step", context.workflow_step),
        ("side effect level", context.side_effect_level),
    ):
        if value is not None:
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def _validate_learning_flags(
    tools: Sequence[Any] | None,
    parallel_tool_calls: bool | None,
    response_format: Mapping[str, Any] | None,
    stream: bool,
) -> None:
    if tools is not None and (
        not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray))
    ):
        raise OpenRoutiQError("tools must be a sequence")
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise OpenRoutiQError("parallel_tool_calls must be a boolean or None")
    if response_format is not None and not isinstance(response_format, Mapping):
        raise OpenRoutiQError("response_format must be an object or None")
    if not isinstance(stream, bool):
        raise OpenRoutiQError("stream must be a boolean")


def _learning_text(
    context: RouteContext,
    *,
    tools: Sequence[Any] | None,
    parallel_tool_calls: bool | None,
    response_format: Mapping[str, Any] | None,
    stream: bool,
) -> str:
    _validate_learning_flags(tools, parallel_tool_calls, response_format, stream)
    request_text, _, _, _ = _request_context(context.request)
    parts = [_semantic_context_text(context, request_text)]
    if tools:
        try:
            serialized = json.dumps(
                list(tools),
                sort_keys=True,
                default=lambda value: getattr(value, "__name__", type(value).__name__),
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            serialized = " ".join(_tool_names(tools))
        # ponytail: cap schemas before local embedding; exact token accounting already happens elsewhere.
        parts.append(f"available tools: {serialized[:12000]}")
    if parallel_tool_calls is not None:
        parts.append(f"parallel tools: {parallel_tool_calls}")
    if response_format is not None:
        parts.append("structured output: required")
    if stream:
        parts.append("streaming: required")
    return "\n".join(parts)


def _tool_names(tools: Sequence[Any] | None) -> list[str]:
    names: list[str] = []
    for tool in tools or ():
        if isinstance(tool, Mapping):
            function = tool.get("function")
            name = tool.get("name")
            if name is None and isinstance(function, Mapping):
                name = function.get("name")
        else:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return sorted(set(names))


def _outcome_metadata(
    context: RouteContext,
    *,
    tools: Sequence[Any] | None,
    parallel_tool_calls: bool | None,
    response_format: Mapping[str, Any] | None,
    stream: bool,
) -> dict[str, Any]:
    _validate_learning_flags(tools, parallel_tool_calls, response_format, stream)
    return {
        key: value
        for key, value in {
            "agent_role": context.agent_role,
            "workflow_step": context.workflow_step,
            "side_effect_level": context.side_effect_level,
            "budget_remaining": context.budget_remaining,
            "latency_deadline_ms": context.latency_deadline_ms,
            "tool_names": _tool_names(tools),
            "parallel_tool_calls": parallel_tool_calls,
            "structured_output": response_format is not None,
            "stream": stream,
        }.items()
        if value is not None
    }


def _request_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        return _number(value, name, minimum=minimum, maximum=maximum)
    except CatalogError as exc:
        raise OpenRoutiQError(str(exc)) from exc


def _request_integer(value: Any, name: str) -> int:
    try:
        return _integer(value, name)
    except CatalogError as exc:
        raise OpenRoutiQError(str(exc)) from exc


def _tool_summary(tools: Sequence[Any] | None) -> tuple[int, int]:
    if not tools:
        return 0, 0
    try:
        serialized = json.dumps(
            list(tools),
            default=lambda value: getattr(value, "__name__", type(value).__name__),
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        serialized = " ".join(type(item).__name__ for item in tools)
    return len(tools), max(1, math.ceil(len(serialized) / 4))


def _normalized_task_scores(
    task: str,
    scores: Sequence[tuple[str, float]],
    allowed_tasks: frozenset[str],
) -> tuple[tuple[str, float], ...]:
    try:
        parsed = dict(scores)
    except (TypeError, ValueError) as exc:
        raise OpenRoutiQError("task scores must be task-value pairs") from exc
    if not parsed:
        return ((task, 1.0),)
    if any(
        name not in allowed_tasks
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for name, value in parsed.items()
    ):
        raise OpenRoutiQError("task scores must be non-negative finite values for known tasks")
    if max(parsed, key=lambda name: parsed[name]) != task:
        return ((task, 1.0),)
    total = sum(parsed.values())
    if total <= 0:
        raise OpenRoutiQError("at least one task score must be greater than zero")
    return tuple(sorted((name, value / total) for name, value in parsed.items()))


def _effective_quality(
    profile: ModelProfile,
    task_scores: Sequence[tuple[str, float]],
    required_capabilities: set[str],
    observed: Mapping[tuple[str, str], float],
) -> float | None:
    weighted_quality = 0.0
    covered_weight = 0.0
    for task, weight in task_scores:
        quality = observed.get((profile.id, task), profile.quality_for(task))
        if quality is not None:
            weighted_quality += weight * quality
            covered_weight += weight
    if covered_weight == 0:
        return None
    primary = weighted_quality / covered_weight
    auxiliary_tasks: set[str] = set()
    if required_capabilities & {"tools", "parallel_tools"}:
        auxiliary_tasks.add("tool_use")
    if "vision" in required_capabilities:
        auxiliary_tasks.add("vision")
    if "json_schema" in required_capabilities:
        auxiliary_tasks.add("extraction")
    auxiliary_tasks.difference_update(task for task, _ in task_scores)
    auxiliary = [
        observed.get((profile.id, name), profile.quality[name])
        for name in auxiliary_tasks
        if name in profile.quality
    ]
    if not auxiliary:
        return float(primary)
    return 0.75 * float(primary) + 0.25 * min(auxiliary)


def _request_context(request: str | Sequence[Any]) -> tuple[str, int, set[str], int]:
    if isinstance(request, str):
        if not request.strip():
            raise OpenRoutiQError("prompt must be a non-empty string")
        return request, 1, set(), 0
    if not isinstance(request, Sequence) or not request:
        raise OpenRoutiQError("request must be text or a non-empty message sequence")

    text_parts: list[str] = []
    capabilities: set[str] = set()
    image_count = 0
    for index, message in enumerate(request):
        if isinstance(message, Mapping):
            content = message.get("content")
            tool_calls = (
                message.get("tool_calls")
                or message.get("function_call")
                or message.get("functionCall")
                or message.get("functionResponse")
            )
            message_type = str(message.get("type", "")).lower()
        else:
            content = getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None)
            message_type = str(getattr(message, "type", "")).lower()
        is_tool_state = message_type in {
            "function_call",
            "function_call_output",
            "function_result",
            "tool_call",
            "tool_result",
            "tool_use",
            "server_tool_use",
        }
        is_opaque_state = message_type in {"reasoning", "thinking", "redacted_thinking", "thought"}
        if content is None and not tool_calls and not is_tool_state and not is_opaque_state:
            raise OpenRoutiQError(f"message {index} has no content")
        if tool_calls or is_tool_state:
            capabilities.add("tools")
        content_text, content_images, content_tools, content_capabilities = _content_text(
            content, index
        )
        text_parts.extend(content_text)
        image_count += content_images
        capabilities.update(content_capabilities)
        if content_tools:
            capabilities.add("tools")
    if image_count:
        capabilities.add("vision")
    text = "\n".join(part for part in text_parts if part.strip()).strip()
    if not text:
        text = "[image input]" if image_count else "[tool context]"
    return text, len(request), capabilities, image_count


def _content_text(content: Any, message_index: int) -> tuple[list[str], int, bool, set[str]]:
    if content is None:
        return [], 0, False, set()
    if isinstance(content, str):
        return [content], 0, False, set()
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        raise OpenRoutiQError(f"message {message_index} content must be text or content blocks")
    texts: list[str] = []
    images = 0
    tools = False
    capabilities: set[str] = set()
    for block in content:
        if isinstance(block, str):
            texts.append(block)
            continue
        if not isinstance(block, Mapping):
            raise OpenRoutiQError(f"message {message_index} contains an invalid content block")
        block_type = str(block.get("type", "")).lower()
        nested_media: Mapping[str, Any] = {}
        for name in ("inlineData", "inline_data", "fileData", "file_data"):
            candidate = block.get(name)
            if isinstance(candidate, Mapping):
                nested_media = candidate
                break
        mime_type = str(
            block.get("mime_type")
            or block.get("mimeType")
            or nested_media.get("mime_type")
            or nested_media.get("mimeType")
            or ""
        ).lower()
        if (
            block_type in {"image", "image_url", "input_image"}
            or "image_url" in block
            or mime_type.startswith("image/")
        ):
            images += 1
        if (
            block_type in {"audio", "audio_url", "input_audio"}
            or "audio_url" in block
            or "input_audio" in block
            or mime_type.startswith("audio/")
        ):
            capabilities.add("audio")
        if (
            block_type in {"video", "video_url", "input_video"}
            or "video_url" in block
            or mime_type.startswith("video/")
        ):
            capabilities.add("video")
        if (
            block_type in {"document", "document_url", "file", "input_file"}
            or "fileData" in block
            or "file_data" in block
            or (mime_type and not mime_type.startswith(("image/", "audio/", "video/", "text/")))
        ):
            capabilities.add("documents")
        if block_type in {
            "function_call",
            "function_call_output",
            "function_result",
            "tool_call",
            "tool_result",
            "tool_use",
            "server_tool_use",
        }:
            tools = True
        if "functionCall" in block or "functionResponse" in block:
            tools = True
        text = block.get("text") or block.get("input_text")
        if isinstance(text, str):
            texts.append(text)
        nested = block.get("content") if block_type in {"tool_result", "function_result"} else None
        if isinstance(nested, str):
            texts.append(nested)
        elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
            nested_text, nested_images, nested_tools, nested_capabilities = _content_text(
                nested, message_index
            )
            texts.extend(nested_text)
            images += nested_images
            tools = tools or nested_tools
            capabilities.update(nested_capabilities)
    return texts, images, tools, capabilities


def _strategy_weights(strategy: str, complexity: float, configured: Weights) -> Weights:
    if not isinstance(strategy, str):
        raise OpenRoutiQError(
            "strategy must be auto, balanced, quality, speed, cost, or risk_aware"
        )
    name = {"fast": "speed", "cheap": "cost"}.get(strategy.lower(), strategy.lower())
    if name == "balanced":
        return configured
    if name == "quality":
        return Weights(85, 10, 5)
    if name == "speed":
        return Weights(35, 55, 10)
    if name == "cost":
        return Weights(35, 10, 55)
    if name in {"auto", "risk_aware"}:
        quality = 45 + 0.45 * complexity
        remainder = 100 - quality
        return Weights(quality, remainder * 0.6, remainder * 0.4)
    raise OpenRoutiQError("strategy must be auto, balanced, quality, speed, cost, or risk_aware")


def _predicted_cost(profile: ModelProfile, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * profile.input_price_per_million
        + output_tokens * profile.output_price_per_million
    ) / 1_000_000


def _inverse_scores(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [50.0] * len(values)
    return [100.0 * (high - value) / (high - low) for value in values]


def _weighted_mean(values: Sequence[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in values)
    if not values or total_weight <= 0:
        raise OpenRoutiQError("weighted values must have positive total weight")
    return sum(value * weight for value, weight in values) / total_weight


def _weighted_stddev(values: Sequence[tuple[float, float]], mean: float | None = None) -> float:
    if not values:
        return 0.0
    center = _weighted_mean(values) if mean is None else mean
    return math.sqrt(
        max(0.0, _weighted_mean([((value - center) ** 2, weight) for value, weight in values]))
    )


def _weighted_percentile(values: Sequence[tuple[float, float]], percentile: float) -> float:
    if not values:
        raise OpenRoutiQError("weighted percentile requires at least one value")
    quantile = _request_number(percentile, "percentile", minimum=0, maximum=1)
    ordered = sorted(values, key=lambda item: item[0])
    total_weight = sum(weight for _, weight in ordered)
    if total_weight <= 0:
        raise OpenRoutiQError("weighted values must have positive total weight")
    target = quantile * total_weight
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def _effective_sample_size(weights: Sequence[float]) -> float:
    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    return total * total / squared if squared > 0 else 0.0


def _outcome_forecast(
    candidate: _EligibleCandidate,
    policy: RiskPolicy,
) -> tuple[OutcomeForecast, tuple[OutcomeScenario, ...]]:
    estimate = candidate.estimate
    observed_metrics: set[str] = set()
    has_joint_scenarios = estimate is not None and bool(estimate.scenarios)
    if estimate is not None and estimate.scenarios:
        observed_metrics.add("quality")
        if any(item.latency_ms is not None for item in estimate.scenarios):
            observed_metrics.add("latency")
        if any(item.cost_usd is not None for item in estimate.scenarios):
            observed_metrics.add("cost")
        if any(item.success is not None for item in estimate.scenarios):
            observed_metrics.add("success")
        scenarios = tuple(
            OutcomeScenario(
                quality_score=min(
                    100.0,
                    max(
                        0.0,
                        candidate.quality
                        + candidate.context_strength
                        * (item.quality_score - estimate.quality_score),
                    ),
                ),
                latency_ms=(
                    candidate.latency
                    + candidate.context_strength
                    * (float(item.latency_ms) - float(estimate.latency_ms))
                    if item.latency_ms is not None and estimate.latency_ms is not None
                    else candidate.latency
                ),
                cost_usd=(
                    (1 - candidate.context_strength) * candidate.cost
                    + candidate.context_strength * float(item.cost_usd)
                    if item.cost_usd is not None
                    else candidate.cost
                ),
                success=item.success,
                failure_class=item.failure_class,
                weight=item.weight,
                similarity=item.similarity,
            )
            for item in estimate.scenarios
        )
        effective_samples = estimate.effective_samples or float(estimate.samples)
    elif estimate is not None and estimate.samples > 0:
        observed_metrics.add("quality")
        if estimate.latency_ms is not None:
            observed_metrics.add("latency")
        if estimate.average_cost_usd is not None:
            observed_metrics.add("cost")
        if estimate.success_probability is not None:
            observed_metrics.add("success")
        quality_spread = max(
            estimate.quality_stddev,
            (
                estimate.quality_score - estimate.quality_lower_bound
                if estimate.quality_lower_bound is not None
                else 0.0
            ),
        )
        latency_mean = estimate.latency_ms or candidate.latency
        latency_high = estimate.latency_p95_ms or latency_mean
        latency_low = max(0.0, 2 * latency_mean - latency_high)
        cost_mean = estimate.average_cost_usd or candidate.cost
        cost_high = estimate.cost_p95_usd or cost_mean
        cost_low = max(0.0, 2 * cost_mean - cost_high)
        scenarios = (
            OutcomeScenario(
                quality_score=max(0.0, estimate.quality_score - quality_spread),
                latency_ms=latency_high,
                cost_usd=cost_high,
                weight=0.2,
            ),
            OutcomeScenario(
                quality_score=estimate.quality_score,
                latency_ms=latency_mean,
                cost_usd=cost_mean,
                weight=0.6,
            ),
            OutcomeScenario(
                quality_score=min(100.0, estimate.quality_score + quality_spread),
                latency_ms=latency_low,
                cost_usd=cost_low,
                weight=0.2,
            ),
        )
        effective_samples = estimate.effective_samples or float(estimate.samples)
    else:
        spread = (100.0 - candidate.profile.confidence) * 0.2
        if spread > 0:
            scenarios = (
                OutcomeScenario(
                    quality_score=max(0.0, candidate.quality - spread),
                    latency_ms=candidate.latency,
                    cost_usd=candidate.cost,
                    weight=0.2,
                ),
                OutcomeScenario(
                    quality_score=candidate.quality,
                    latency_ms=candidate.latency,
                    cost_usd=candidate.cost,
                    weight=0.6,
                ),
                OutcomeScenario(
                    quality_score=min(100.0, candidate.quality + spread),
                    latency_ms=candidate.latency,
                    cost_usd=candidate.cost,
                    weight=0.2,
                ),
            )
        else:
            scenarios = (
                OutcomeScenario(
                    quality_score=candidate.quality,
                    latency_ms=candidate.latency,
                    cost_usd=candidate.cost,
                ),
            )
        effective_samples = 0.0

    quality_values = [(item.quality_score, item.weight) for item in scenarios]
    latency_values = [
        (float(item.latency_ms), item.weight) for item in scenarios if item.latency_ms is not None
    ]
    cost_values = [
        (float(item.cost_usd), item.weight) for item in scenarios if item.cost_usd is not None
    ]
    expected_quality = _weighted_mean(quality_values)
    quality_stddev = _weighted_stddev(quality_values, expected_quality)
    epistemic_margin = (
        (100.0 - candidate.profile.confidence) * 0.1 / math.sqrt(1.0 + effective_samples)
    )
    quality_lower_bound = max(
        0.0,
        min(
            _weighted_percentile(quality_values, 0.05),
            expected_quality - epistemic_margin,
        ),
    )
    if estimate is not None and estimate.quality_lower_bound is not None:
        blended_lower_bound = candidate.quality + candidate.context_strength * (
            estimate.quality_lower_bound - estimate.quality_score
        )
        quality_lower_bound = max(0.0, min(quality_lower_bound, blended_lower_bound))

    success_probability = estimate.success_probability if estimate is not None else None
    probabilities = tuple(
        (
            constraint.label,
            _event_probability(constraint, scenarios, success_probability),
        )
        for constraint in policy.constraints
    )
    joint_probability = (
        _joint_constraint_probability(policy.constraints, scenarios)
        if has_joint_scenarios
        else None
    )
    expected_latency = _weighted_mean(latency_values)
    expected_cost = _weighted_mean(cost_values)
    forecast = OutcomeForecast(
        expected_quality=expected_quality,
        quality_stddev=quality_stddev,
        quality_lower_bound=quality_lower_bound,
        success_probability=success_probability,
        failure_probabilities=(() if estimate is None else estimate.failure_probabilities),
        expected_latency_ms=expected_latency,
        latency_p50_ms=_weighted_percentile(latency_values, 0.5),
        latency_p95_ms=_weighted_percentile(latency_values, 0.95),
        latency_p99_ms=_weighted_percentile(latency_values, 0.99),
        expected_cost_usd=expected_cost,
        cost_p50_usd=_weighted_percentile(cost_values, 0.5),
        cost_p95_usd=_weighted_percentile(cost_values, 0.95),
        cost_p99_usd=_weighted_percentile(cost_values, 0.99),
        evidence_samples=0 if estimate is None else estimate.samples,
        effective_samples=effective_samples,
        observed_metrics=frozenset(observed_metrics),
        event_probabilities=probabilities,
        joint_constraint_probability=joint_probability,
        cvar_alpha=policy.cvar_alpha,
    )
    return forecast, scenarios


def _event_probability(
    constraint: ChanceConstraint,
    scenarios: Sequence[OutcomeScenario],
    success_probability: float | None,
) -> float | None:
    if constraint.event == "success":
        return success_probability
    outcomes = [
        (result, scenario.weight)
        for scenario in scenarios
        if (result := _scenario_event_holds(constraint, scenario)) is not None
    ]
    if not outcomes:
        return None
    total = sum(weight for _, weight in outcomes)
    return sum(weight for result, weight in outcomes if result) / total


def _joint_constraint_probability(
    constraints: Sequence[ChanceConstraint],
    scenarios: Sequence[OutcomeScenario],
) -> float | None:
    if not constraints:
        return None
    outcomes: list[tuple[bool, float]] = []
    for scenario in scenarios:
        results = [_scenario_event_holds(constraint, scenario) for constraint in constraints]
        if any(result is None for result in results):
            continue
        outcomes.append((all(bool(result) for result in results), scenario.weight))
    if not outcomes:
        return None
    total = sum(weight for _, weight in outcomes)
    return sum(weight for result, weight in outcomes if result) / total


def _scenario_event_holds(
    constraint: ChanceConstraint,
    scenario: OutcomeScenario,
) -> bool | None:
    if constraint.event == "success":
        return scenario.success
    assert constraint.threshold is not None
    threshold = float(constraint.threshold)
    if constraint.event == "quality_at_least":
        return scenario.quality_score >= threshold
    if constraint.event == "latency_at_most":
        return None if scenario.latency_ms is None else scenario.latency_ms <= threshold
    if constraint.event == "cost_at_most":
        return None if scenario.cost_usd is None else scenario.cost_usd <= threshold
    raise OpenRoutiQError(f"unsupported chance constraint event: {constraint.event}")


def _chance_constraint_reasons(
    forecast: OutcomeForecast,
    policy: RiskPolicy,
) -> list[str]:
    reasons: list[str] = []
    if forecast.evidence_samples < policy.minimum_samples:
        reasons.append(
            f"risk evidence {forecast.evidence_samples} samples < required {policy.minimum_samples}"
        )
    probabilities = dict(forecast.event_probabilities)
    observed_names = {
        "success": "success",
        "quality_at_least": "quality",
        "latency_at_most": "latency",
        "cost_at_most": "cost",
    }
    for constraint in policy.constraints:
        probability = probabilities.get(constraint.label)
        if probability is None:
            reasons.append(f"no probability estimate for {constraint.label}")
            continue
        observed_name = observed_names[constraint.event]
        if policy.require_observed_probabilities and observed_name not in forecast.observed_metrics:
            reasons.append(f"{constraint.label} probability has no observed evidence")
            continue
        if probability + 1e-12 < constraint.minimum_probability:
            reasons.append(
                f"P({constraint.label}) {probability:.4f} < required "
                f"{constraint.minimum_probability:.4f}"
            )
    return reasons


def _scenario_bounds(
    candidates: Sequence[_EligibleCandidate],
    *,
    metric: str,
    fallback: Sequence[float],
) -> tuple[float, float]:
    values: list[float] = []
    for candidate in candidates:
        for scenario in candidate.scenarios:
            value = scenario.latency_ms if metric == "latency" else scenario.cost_usd
            if value is not None:
                values.append(float(value))
    if not values:
        values = list(fallback)
    return min(values), max(values)


def _inverse_score(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if math.isclose(low, high):
        return 50.0
    return min(100.0, max(0.0, 100.0 * (high - value) / (high - low)))


def _candidate_cvar_loss(
    scenarios: Sequence[OutcomeScenario],
    weights: Weights,
    policy: RiskPolicy,
    *,
    latency_bounds: tuple[float, float],
    cost_bounds: tuple[float, float],
) -> float:
    total_weight = weights.quality + weights.latency + weights.cost
    losses: list[tuple[float, float]] = []
    for scenario in scenarios:
        if scenario.latency_ms is None or scenario.cost_usd is None:
            raise OpenRoutiQError("risk scenarios require latency and cost values")
        latency_score = _inverse_score(float(scenario.latency_ms), latency_bounds)
        cost_score = _inverse_score(float(scenario.cost_usd), cost_bounds)
        utility = (
            weights.quality * scenario.quality_score
            + weights.latency * latency_score
            + weights.cost * cost_score
        ) / total_weight
        loss = max(0.0, 100.0 - utility)
        if policy.constraints:
            results = [_scenario_event_holds(item, scenario) for item in policy.constraints]
            violations = sum(result is False for result in results)
            if violations:
                loss = min(
                    100.0,
                    loss + policy.constraint_penalty * violations / len(policy.constraints),
                )
        losses.append((loss, scenario.weight))
    return _weighted_cvar(losses, policy.cvar_alpha)


def _weighted_cvar(losses: Sequence[tuple[float, float]], alpha: float) -> float:
    if not losses:
        return 0.0
    ordered = sorted(losses, key=lambda item: item[0], reverse=True)
    total_weight = sum(weight for _, weight in ordered)
    tail_weight = total_weight * (1.0 - alpha)
    if tail_weight <= 0:
        return ordered[0][0]
    accumulated = 0.0
    weighted_loss = 0.0
    for loss, weight in ordered:
        taken = min(weight, tail_weight - accumulated)
        weighted_loss += loss * taken
        accumulated += taken
        if accumulated >= tail_weight:
            break
    return weighted_loss / tail_weight
