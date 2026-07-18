from __future__ import annotations

import json
import math
import os
import random
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlsplit

from openroutiq.router.failures import FailureType, normalize_failure_type
from openroutiq.router.core import (
    Constraints,
    OpenRoutiQError,
    ModelProfile,
    NoEligibleModelError,
    OutcomeEstimate,
    RouteContext,
    RouteDecision,
    Router,
    analyze_context,
)


OPERATING_STATES = frozenset({"active", "dormant", "quarantined"})
TASK_STATES = frozenset({"provisional", "trusted", "degraded"})
ADAPTIVE_SCHEMA_VERSION = 3
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|auth[_-]?token|"
    r"bearer[_-]?token|refresh[_-]?token|^token$)$",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(r"^(?:bearer\s+|basic\s+|sk-[a-z0-9_-]{8,})", re.IGNORECASE)


class AdaptiveStoreError(OpenRoutiQError):
    """Raised when the configured adaptive registry is unavailable or corrupt."""


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenRoutiQError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise OpenRoutiQError(f"{name} must be finite")
    if minimum is not None and parsed < minimum:
        raise OpenRoutiQError(f"{name} must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        raise OpenRoutiQError(f"{name} must be at most {maximum}")
    return parsed


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OpenRoutiQError(f"{name} must be an integer of at least {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenRoutiQError(f"{name} must be non-empty text")
    parsed = value.strip()
    if len(parsed) > 512:
        raise OpenRoutiQError(f"{name} must be at most 512 characters")
    return parsed


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AdaptiveStoreError("adaptive registry contains an invalid timestamp") from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _usd_nanos(value: float, *, rounding: str) -> int:
    try:
        parsed = Decimal(str(value)) * Decimal("1000000000")
        return int(parsed.to_integral_value(rounding=rounding))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise OpenRoutiQError("USD amount cannot be represented safely") from exc


def _profile_mapping(profile: ModelProfile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "provider": profile.provider,
        "model": profile.model,
        "api_style": profile.api_style,
        "base_url": profile.base_url,
        "quality": dict(profile.quality),
        "latency_ms": profile.latency_ms,
        "input_price_per_million": profile.input_price_per_million,
        "output_price_per_million": profile.output_price_per_million,
        "max_context_tokens": profile.max_context_tokens,
        "capabilities": sorted(profile.capabilities),
        "supported_parameters": sorted(profile.supported_parameters),
        "available": profile.available,
        "confidence": profile.confidence,
        "reasoning_level": profile.reasoning_level,
        "reasoning_mode": profile.reasoning_mode,
        "reasoning_budget_tokens": profile.reasoning_budget_tokens,
        "provider_options": dict(profile.provider_options),
        "pricing": dict(profile.pricing),
        "local": profile.local,
        "tags": sorted(profile.tags),
    }


def _stored_profile(raw: Any, model_id: str) -> ModelProfile:
    """Parse a persisted profile while preserving store-vs-request error semantics."""

    try:
        decoded = json.loads(str(raw))
        if not isinstance(decoded, Mapping):
            raise ValueError("profile JSON must contain an object")
        return ModelProfile.from_mapping(decoded)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, OpenRoutiQError) as exc:
        raise AdaptiveStoreError(
            f"adaptive registry contains an invalid profile for {model_id}"
        ) from exc


def _contains_secret(value: Any, *, path: str = "profile") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            location = f"{path}.{name}"
            if _SECRET_KEY.search(name):
                return location
            found = _contains_secret(item, path=location)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _contains_secret(item, path=f"{path}[{index}]")
            if found is not None:
                return found
    elif isinstance(value, str) and _SECRET_VALUE.search(value.strip()):
        return path
    return None


@dataclass(frozen=True)
class AdaptivePolicy:
    """Safety and learning policy for opaque, customer-local model variants."""

    automatic_promotion: bool = False
    cold_start_quality: float = 50.0
    cold_start_confidence: float = 5.0
    prior_samples: int = 4
    promotion_samples: int = 8
    promotion_quality_floor: float = 50.0
    promotion_success_rate: float = 0.7
    promotion_effective_sample_fraction: float = 0.9
    uncertainty_scale: float = 25.0
    consecutive_failure_limit: int = 3
    drift_window: int = 5
    drift_min_history: int = 8
    drift_quality_drop: float = 20.0
    exploration_rate: float = 0.0
    exploration_daily_budget_usd: float = 0.0
    exploration_max_request_cost_usd: float = 0.0
    random_seed: int = 3407
    max_observations_per_model: int = 2_000
    exploration_retention_days: int = 35
    telemetry_refresh_interval: int = 10
    cost_ratio_floor: float = 0.01
    cost_ratio_ceiling: float = 100.0
    maximum_learned_price_per_million: float = 1_000_000.0
    max_models: int = 10_000
    max_tasks_per_model: int = 256
    evidence_half_life_days: float = 30.0
    trusted_evidence_stale_after_days: float = 90.0
    telemetry_recent_samples: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.automatic_promotion, bool):
            raise OpenRoutiQError("automatic_promotion must be a boolean")
        _finite_number(self.cold_start_quality, "cold_start_quality", minimum=0, maximum=100)
        _finite_number(
            self.cold_start_confidence,
            "cold_start_confidence",
            minimum=0,
            maximum=100,
        )
        _positive_integer(self.prior_samples, "prior_samples")
        _positive_integer(self.promotion_samples, "promotion_samples")
        _finite_number(
            self.promotion_quality_floor,
            "promotion_quality_floor",
            minimum=0,
            maximum=100,
        )
        _finite_number(
            self.promotion_success_rate,
            "promotion_success_rate",
            minimum=0,
            maximum=1,
        )
        _finite_number(
            self.promotion_effective_sample_fraction,
            "promotion_effective_sample_fraction",
            minimum=0,
            maximum=1,
        )
        if self.promotion_effective_sample_fraction <= 0:
            raise OpenRoutiQError("promotion_effective_sample_fraction must be positive")
        _finite_number(self.uncertainty_scale, "uncertainty_scale", minimum=0)
        _positive_integer(self.consecutive_failure_limit, "consecutive_failure_limit")
        _positive_integer(self.drift_window, "drift_window", minimum=2)
        _positive_integer(self.drift_min_history, "drift_min_history")
        _finite_number(self.drift_quality_drop, "drift_quality_drop", minimum=0, maximum=100)
        _finite_number(self.exploration_rate, "exploration_rate", minimum=0, maximum=1)
        _finite_number(
            self.exploration_daily_budget_usd,
            "exploration_daily_budget_usd",
            minimum=0,
        )
        _finite_number(
            self.exploration_max_request_cost_usd,
            "exploration_max_request_cost_usd",
            minimum=0,
        )
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise OpenRoutiQError("random_seed must be an integer")
        _positive_integer(self.max_observations_per_model, "max_observations_per_model")
        _positive_integer(self.exploration_retention_days, "exploration_retention_days")
        _positive_integer(self.telemetry_refresh_interval, "telemetry_refresh_interval")
        _finite_number(self.cost_ratio_floor, "cost_ratio_floor", minimum=0)
        _finite_number(self.cost_ratio_ceiling, "cost_ratio_ceiling", minimum=0)
        if self.cost_ratio_floor <= 0 or self.cost_ratio_floor > self.cost_ratio_ceiling:
            raise OpenRoutiQError(
                "cost_ratio_floor must be positive and no greater than cost_ratio_ceiling"
            )
        _finite_number(
            self.maximum_learned_price_per_million,
            "maximum_learned_price_per_million",
            minimum=0,
        )
        if self.maximum_learned_price_per_million <= 0:
            raise OpenRoutiQError("maximum_learned_price_per_million must be positive")
        _positive_integer(self.max_models, "max_models")
        _positive_integer(self.max_tasks_per_model, "max_tasks_per_model")
        _finite_number(self.evidence_half_life_days, "evidence_half_life_days", minimum=0)
        if self.evidence_half_life_days <= 0:
            raise OpenRoutiQError("evidence_half_life_days must be positive")
        _finite_number(
            self.trusted_evidence_stale_after_days,
            "trusted_evidence_stale_after_days",
            minimum=0,
        )
        if self.trusted_evidence_stale_after_days <= 0:
            raise OpenRoutiQError("trusted_evidence_stale_after_days must be positive")
        _positive_integer(self.telemetry_recent_samples, "telemetry_recent_samples")
        if self.exploration_rate > 0 and (
            self.exploration_daily_budget_usd <= 0 or self.exploration_max_request_cost_usd <= 0
        ):
            raise OpenRoutiQError(
                "positive exploration_rate requires positive daily and per-request budgets"
            )


@dataclass(frozen=True)
class AdaptiveModelStatus:
    model_id: str
    operating_state: str
    task: str
    task_state: str
    source: str
    samples: int
    effective_samples: float
    posterior_quality: float
    quality_lower_bound: float
    confidence: float
    success_rate: float | None
    average_latency_ms: float | None
    average_cost_usd: float | None
    consecutive_failures: int
    first_seen_at: str
    last_seen_at: str
    state_reason: str | None = None
    quality_variance: float = 0.0
    latency_p95_ms: float | None = None
    cost_p95_usd: float | None = None
    failure_probabilities: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "operating_state": self.operating_state,
            "task": self.task,
            "task_state": self.task_state,
            "source": self.source,
            "samples": self.samples,
            "effective_samples": round(self.effective_samples, 4),
            "posterior_quality": round(self.posterior_quality, 4),
            "quality_lower_bound": round(self.quality_lower_bound, 4),
            "quality_variance": round(self.quality_variance, 4),
            "confidence": round(self.confidence, 2),
            "success_rate": (None if self.success_rate is None else round(self.success_rate, 4)),
            "average_latency_ms": (
                None if self.average_latency_ms is None else round(self.average_latency_ms, 2)
            ),
            "latency_p95_ms": (
                None if self.latency_p95_ms is None else round(self.latency_p95_ms, 2)
            ),
            "average_cost_usd": (
                None if self.average_cost_usd is None else round(self.average_cost_usd, 8)
            ),
            "cost_p95_usd": (None if self.cost_p95_usd is None else round(self.cost_p95_usd, 8)),
            "consecutive_failures": self.consecutive_failures,
            "failure_probabilities": {
                name: round(probability, 4) for name, probability in self.failure_probabilities
            },
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "state_reason": self.state_reason,
        }


@runtime_checkable
class AdaptiveRegistryBackend(Protocol):
    """Storage boundary for enterprise registry implementations."""

    policy: AdaptivePolicy

    @property
    def revision(self) -> int: ...

    def encounter(
        self,
        profile: ModelProfile | Mapping[str, Any],
        *,
        source: str = "encounter",
        trusted: bool = False,
    ) -> AdaptiveModelStatus: ...

    def encounter_opaque(
        self,
        *,
        model_id: str,
        provider: str,
        model: str,
        max_context_tokens: int,
        capabilities: Sequence[str],
        supported_parameters: Sequence[str] = (),
        tasks: Sequence[str] = (),
        latency_ms: float,
        input_price_per_million: float,
        output_price_per_million: float,
        api_style: str = "openai_compatible",
        base_url: str | None = None,
        reasoning_level: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_budget_tokens: int | None = None,
        local: bool = False,
        source: str = "private",
        prior_quality: float | None = None,
    ) -> AdaptiveModelStatus: ...

    def record(
        self,
        model_id: str,
        task: str,
        *,
        quality_score: float | None = None,
        latency_ms: float | None = None,
        actual_cost_usd: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        success: bool | None = None,
        failure_type: FailureType | str | None = None,
    ) -> AdaptiveModelStatus: ...

    def status(self, model_id: str, *, task: str = "general") -> AdaptiveModelStatus: ...

    def profiles(self, *, include_inactive: bool = True) -> tuple[ModelProfile, ...]: ...

    def routing_states(self, task: str) -> dict[str, tuple[str, str]]: ...

    def reserve_exploration(self, model_id: str, task: str, predicted_cost: float) -> bool: ...


class AdaptiveModelRegistry:
    """Local registry for models learned when they are encountered.

    The registry stores model contracts and aggregate outcome telemetry. It never stores
    prompts, responses, provider credentials, or model weights.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        policy: AdaptivePolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.policy = policy or AdaptivePolicy()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection:
                connection.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    PRAGMA foreign_keys=ON;
                    CREATE TABLE IF NOT EXISTS adaptive_meta (
                        key TEXT PRIMARY KEY,
                        value INTEGER NOT NULL
                    );
                    INSERT OR IGNORE INTO adaptive_meta (key, value) VALUES ('revision', 0);
                    INSERT OR IGNORE INTO adaptive_meta (key, value) VALUES ('schema_version', 0);

                    CREATE TABLE IF NOT EXISTS adaptive_models (
                        model_id TEXT PRIMARY KEY,
                        profile_json TEXT NOT NULL,
                        source TEXT NOT NULL,
                        operating_state TEXT NOT NULL CHECK (
                            operating_state IN ('active', 'dormant', 'quarantined')
                        ),
                        default_task_state TEXT NOT NULL CHECK (
                            default_task_state IN ('provisional', 'trusted', 'degraded')
                        ),
                        first_seen_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        observation_count INTEGER NOT NULL DEFAULT 0 CHECK (
                            observation_count >= 0
                        )
                    );

                    CREATE TABLE IF NOT EXISTS adaptive_task_states (
                        model_id TEXT NOT NULL,
                        task TEXT NOT NULL,
                        state TEXT NOT NULL CHECK (
                            state IN ('provisional', 'trusted', 'degraded')
                        ),
                        reason TEXT,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (model_id, task),
                        FOREIGN KEY (model_id) REFERENCES adaptive_models(model_id)
                            ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS adaptive_observations (
                        id INTEGER PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        task TEXT NOT NULL,
                        quality_score REAL CHECK (
                            quality_score IS NULL OR quality_score BETWEEN 0 AND 100
                        ),
                        latency_ms REAL CHECK (latency_ms IS NULL OR latency_ms >= 0),
                        actual_cost_usd REAL CHECK (
                            actual_cost_usd IS NULL OR actual_cost_usd >= 0
                        ),
                        input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
                        output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
                        success INTEGER CHECK (success IS NULL OR success IN (0, 1)),
                        failure_type TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (model_id) REFERENCES adaptive_models(model_id)
                            ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS adaptive_observations_lookup
                    ON adaptive_observations (model_id, task, id DESC);

                    CREATE TABLE IF NOT EXISTS adaptive_explorations (
                        id INTEGER PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        task TEXT NOT NULL,
                        reserved_cost_usd REAL NOT NULL CHECK (reserved_cost_usd >= 0),
                        reserved_cost_nano_usd INTEGER NOT NULL CHECK (
                            reserved_cost_nano_usd >= 0
                        ),
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (model_id) REFERENCES adaptive_models(model_id)
                            ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS adaptive_explorations_created
                    ON adaptive_explorations (created_at);
                    """
                )
                # Serialize additive migrations across multiple starting workers.
                connection.execute("BEGIN IMMEDIATE")
                version_row = connection.execute(
                    "SELECT value FROM adaptive_meta WHERE key = 'schema_version'"
                ).fetchone()
                schema_version = 0 if version_row is None else int(version_row[0])
                if schema_version > ADAPTIVE_SCHEMA_VERSION:
                    raise AdaptiveStoreError(
                        "adaptive registry schema is newer than this OpenRoutiQ version"
                    )
                observation_columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(adaptive_observations)"
                    ).fetchall()
                }
                if "input_tokens" not in observation_columns:
                    connection.execute(
                        "ALTER TABLE adaptive_observations ADD COLUMN input_tokens INTEGER "
                        "CHECK (input_tokens IS NULL OR input_tokens >= 0)"
                    )
                if "output_tokens" not in observation_columns:
                    connection.execute(
                        "ALTER TABLE adaptive_observations ADD COLUMN output_tokens INTEGER "
                        "CHECK (output_tokens IS NULL OR output_tokens >= 0)"
                    )
                if "failure_type" not in observation_columns:
                    connection.execute(
                        "ALTER TABLE adaptive_observations ADD COLUMN failure_type TEXT"
                    )
                exploration_columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(adaptive_explorations)"
                    ).fetchall()
                }
                if "reserved_cost_nano_usd" not in exploration_columns:
                    connection.execute(
                        "ALTER TABLE adaptive_explorations ADD COLUMN "
                        "reserved_cost_nano_usd INTEGER NOT NULL DEFAULT 0 "
                        "CHECK (reserved_cost_nano_usd >= 0)"
                    )
                    legacy_rows = connection.execute(
                        "SELECT id, reserved_cost_usd FROM adaptive_explorations"
                    ).fetchall()
                    connection.executemany(
                        """
                        UPDATE adaptive_explorations
                        SET reserved_cost_nano_usd = ? WHERE id = ?
                        """,
                        [
                            (
                                _usd_nanos(float(cost), rounding=ROUND_CEILING),
                                int(row_id),
                            )
                            for row_id, cost in legacy_rows
                        ],
                    )
                model_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(adaptive_models)").fetchall()
                }
                if "observation_count" not in model_columns:
                    connection.execute(
                        "ALTER TABLE adaptive_models ADD COLUMN observation_count "
                        "INTEGER NOT NULL DEFAULT 0 CHECK (observation_count >= 0)"
                    )
                    connection.execute(
                        """
                        UPDATE adaptive_models
                        SET observation_count = (
                            SELECT COUNT(*) FROM adaptive_observations
                            WHERE adaptive_observations.model_id = adaptive_models.model_id
                        )
                        """
                    )
                connection.execute(
                    "UPDATE adaptive_meta SET value = ? WHERE key = 'schema_version'",
                    (ADAPTIVE_SCHEMA_VERSION,),
                )
                connection.commit()
            if os.name != "nt":
                os.chmod(self.path, 0o600)
        except AdaptiveStoreError:
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError, OpenRoutiQError) as exc:
            raise AdaptiveStoreError(
                f"cannot initialize adaptive registry {self.path}: {exc}"
            ) from exc

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise AdaptiveStoreError("adaptive registry clock must return a datetime")
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    def _now_text(self) -> str:
        return self._now().isoformat(timespec="microseconds")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _bump_revision(connection: sqlite3.Connection) -> None:
        connection.execute("UPDATE adaptive_meta SET value = value + 1 WHERE key = 'revision'")

    @property
    def revision(self) -> int:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT value FROM adaptive_meta WHERE key = 'revision'"
                ).fetchone()
                return 0 if row is None else int(row[0])
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise AdaptiveStoreError(f"cannot read adaptive registry {self.path}: {exc}") from exc

    def encounter(
        self,
        profile: ModelProfile | Mapping[str, Any],
        *,
        source: str = "encounter",
        trusted: bool = False,
    ) -> AdaptiveModelStatus:
        parsed = (
            profile if isinstance(profile, ModelProfile) else ModelProfile.from_mapping(profile)
        )
        source = _text(source, "source")
        raw_profile = _profile_mapping(parsed)
        secret_path = _contains_secret(
            parsed.provider_options,
            path="profile.provider_options",
        )
        if secret_path is not None:
            raise OpenRoutiQError(
                f"adaptive registry refuses credential-like field {secret_path}; use environment "
                "variables or the framework's secret store"
            )
        if parsed.base_url is not None:
            parsed_url = urlsplit(parsed.base_url)
            query_has_secret = any(
                _SECRET_KEY.search(name) for name, _ in parse_qsl(parsed_url.query)
            )
            if (
                parsed_url.username is not None
                or parsed_url.password is not None
                or query_has_secret
            ):
                raise OpenRoutiQError("adaptive registry refuses credentials embedded in base_url")
        default_task_state = "trusted" if trusted else "provisional"
        if not trusted:
            parsed = replace(
                parsed,
                confidence=min(parsed.confidence, self.policy.cold_start_confidence),
            )
            raw_profile = _profile_mapping(parsed)
        serialized = json.dumps(raw_profile, sort_keys=True, separators=(",", ":"))
        now = self._now_text()
        with self._lock:
            try:
                with closing(self._connect()) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = connection.execute(
                        """
                        SELECT profile_json, operating_state, default_task_state
                        FROM adaptive_models WHERE model_id = ?
                        """,
                        (parsed.id,),
                    ).fetchone()
                    if existing is not None:
                        previous = _stored_profile(existing[0], parsed.id)
                        identity_before = (
                            previous.provider,
                            previous.model,
                            previous.reasoning_level,
                            previous.reasoning_mode,
                        )
                        identity_after = (
                            parsed.provider,
                            parsed.model,
                            parsed.reasoning_level,
                            parsed.reasoning_mode,
                        )
                        if identity_before != identity_after:
                            raise OpenRoutiQError(
                                f"model id {parsed.id} is already bound to a different "
                                "provider/model/reasoning variant"
                            )
                        operating_state = "active" if existing[1] == "dormant" else existing[1]
                        preserved_default = (
                            "trusted" if existing[2] == "trusted" or trusted else "provisional"
                        )
                        connection.execute(
                            """
                            UPDATE adaptive_models
                            SET profile_json = ?, source = ?, operating_state = ?,
                                default_task_state = ?, last_seen_at = ?, updated_at = ?
                            WHERE model_id = ?
                            """,
                            (
                                serialized,
                                source,
                                operating_state,
                                preserved_default,
                                now,
                                now,
                                parsed.id,
                            ),
                        )
                    else:
                        model_count = int(
                            connection.execute("SELECT COUNT(*) FROM adaptive_models").fetchone()[0]
                        )
                        if model_count >= self.policy.max_models:
                            raise OpenRoutiQError(
                                f"adaptive registry model limit {self.policy.max_models} reached"
                            )
                        connection.execute(
                            """
                            INSERT INTO adaptive_models (
                                model_id, profile_json, source, operating_state,
                                default_task_state, first_seen_at, last_seen_at, updated_at
                            ) VALUES (?, ?, ?, 'active', ?, ?, ?, ?)
                            """,
                            (
                                parsed.id,
                                serialized,
                                source,
                                default_task_state,
                                now,
                                now,
                                now,
                            ),
                        )
                    self._bump_revision(connection)
                    connection.commit()
            except sqlite3.Error as exc:
                raise AdaptiveStoreError(
                    f"cannot write adaptive registry {self.path}: {exc}"
                ) from exc
        return self.status(parsed.id, task="general")

    def encounter_opaque(
        self,
        *,
        model_id: str,
        provider: str,
        model: str,
        max_context_tokens: int,
        capabilities: Sequence[str],
        supported_parameters: Sequence[str] = (),
        tasks: Sequence[str] = (),
        latency_ms: float,
        input_price_per_million: float,
        output_price_per_million: float,
        api_style: str = "openai_compatible",
        base_url: str | None = None,
        reasoning_level: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_budget_tokens: int | None = None,
        local: bool = False,
        source: str = "private",
        prior_quality: float | None = None,
    ) -> AdaptiveModelStatus:
        quality = (
            self.policy.cold_start_quality
            if prior_quality is None
            else _finite_number(prior_quality, "prior_quality", minimum=0, maximum=100)
        )
        if isinstance(tasks, (str, bytes, bytearray)) or any(
            not isinstance(task, str) or not task.strip() for task in tasks
        ):
            raise OpenRoutiQError("tasks must be a sequence of non-empty task labels")
        quality_priors = {"general": quality}
        quality_priors.update({task.strip(): quality for task in tasks})
        raw: dict[str, Any] = {
            "id": model_id,
            "provider": provider,
            "model": model,
            "api_style": api_style,
            "base_url": base_url,
            "quality": quality_priors,
            "latency_ms": latency_ms,
            "input_price_per_million": input_price_per_million,
            "output_price_per_million": output_price_per_million,
            "max_context_tokens": max_context_tokens,
            "capabilities": list(capabilities),
            "supported_parameters": list(supported_parameters),
            "confidence": self.policy.cold_start_confidence,
            "reasoning_level": reasoning_level,
            "local": local,
            "tags": ["adaptive", "opaque"],
        }
        if reasoning_mode is not None:
            raw["reasoning_mode"] = reasoning_mode
        if reasoning_budget_tokens is not None:
            raw["reasoning_budget_tokens"] = reasoning_budget_tokens
        return self.encounter(raw, source=source, trusted=False)

    def _model_row(self, connection: sqlite3.Connection, model_id: str) -> tuple[Any, ...]:
        row = connection.execute(
            """
            SELECT profile_json, source, operating_state, default_task_state,
                   first_seen_at, last_seen_at
            FROM adaptive_models WHERE model_id = ?
            """,
            (model_id,),
        ).fetchone()
        if row is None:
            raise OpenRoutiQError(f"unknown adaptive model id: {model_id}")
        if row[2] not in OPERATING_STATES or row[3] not in TASK_STATES:
            raise AdaptiveStoreError(
                f"adaptive registry contains an invalid lifecycle state for {model_id}"
            )
        return row

    def record(
        self,
        model_id: str,
        task: str,
        *,
        quality_score: float | None = None,
        latency_ms: float | None = None,
        actual_cost_usd: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        success: bool | None = None,
        failure_type: FailureType | str | None = None,
    ) -> AdaptiveModelStatus:
        model_id = _text(model_id, "model_id")
        task = _text(task, "task")
        quality = (
            None
            if quality_score is None
            else _finite_number(quality_score, "quality_score", minimum=0, maximum=100)
        )
        latency = (
            None if latency_ms is None else _finite_number(latency_ms, "latency_ms", minimum=0)
        )
        cost = (
            None
            if actual_cost_usd is None
            else _finite_number(actual_cost_usd, "actual_cost_usd", minimum=0)
        )
        input_count = (
            None
            if input_tokens is None
            else _positive_integer(input_tokens, "input_tokens", minimum=0)
        )
        output_count = (
            None
            if output_tokens is None
            else _positive_integer(output_tokens, "output_tokens", minimum=0)
        )
        if success is not None and not isinstance(success, bool):
            raise OpenRoutiQError("success must be a boolean or None")
        try:
            normalized_failure = normalize_failure_type(failure_type)
        except ValueError as exc:
            raise OpenRoutiQError(str(exc)) from exc
        if success is True and normalized_failure is not None:
            raise OpenRoutiQError("failure_type requires success=false or success=None")
        if (
            quality is None
            and latency is None
            and cost is None
            and input_count is None
            and output_count is None
            and success is None
            and normalized_failure is None
        ):
            raise OpenRoutiQError("an adaptive observation must contain at least one measurement")
        now = self._now_text()
        with self._lock:
            try:
                with closing(self._connect()) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    model_row = self._model_row(connection, model_id)
                    profile = _stored_profile(model_row[0], model_id)
                    known_task = (
                        task in profile.quality
                        or connection.execute(
                            """
                        SELECT 1 FROM adaptive_observations
                        WHERE model_id = ? AND task = ?
                        UNION ALL
                        SELECT 1 FROM adaptive_task_states
                        WHERE model_id = ? AND task = ?
                        LIMIT 1
                        """,
                            (model_id, task, model_id, task),
                        ).fetchone()
                        is not None
                    )
                    if not known_task:
                        learned_tasks = {
                            str(row[0])
                            for row in connection.execute(
                                """
                                SELECT task FROM adaptive_observations WHERE model_id = ?
                                UNION
                                SELECT task FROM adaptive_task_states WHERE model_id = ?
                                """,
                                (model_id, model_id),
                            ).fetchall()
                        }
                        learned_tasks.update(profile.quality)
                        if len(learned_tasks) >= self.policy.max_tasks_per_model:
                            raise OpenRoutiQError(
                                f"adaptive task limit {self.policy.max_tasks_per_model} "
                                f"reached for model {model_id}"
                            )
                    connection.execute(
                        """
                        INSERT INTO adaptive_observations (
                            model_id, task, quality_score, latency_ms,
                            actual_cost_usd, input_tokens, output_tokens,
                            success, created_at
                            , failure_type
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            model_id,
                            task,
                            quality,
                            latency,
                            cost,
                            input_count,
                            output_count,
                            None if success is None else int(success),
                            now,
                            None if normalized_failure is None else normalized_failure.value,
                        ),
                    )
                    connection.execute(
                        """
                        DELETE FROM adaptive_observations
                        WHERE model_id = ? AND id NOT IN (
                            SELECT id FROM adaptive_observations
                            WHERE model_id = ? ORDER BY id DESC LIMIT ?
                        )
                        """,
                        (model_id, model_id, self.policy.max_observations_per_model),
                    )
                    connection.execute(
                        """
                        UPDATE adaptive_models
                        SET last_seen_at = ?, updated_at = ?,
                            observation_count = observation_count + 1
                        WHERE model_id = ?
                        """,
                        (now, now, model_id),
                    )
                    state_changed = self._update_task_state(connection, model_id, task, now)
                    count_row = connection.execute(
                        "SELECT observation_count FROM adaptive_models WHERE model_id = ?",
                        (model_id,),
                    ).fetchone()
                    observation_count = int(count_row[0])
                    if (
                        quality is not None
                        or success is False
                        or state_changed
                        or observation_count % self.policy.telemetry_refresh_interval == 0
                    ):
                        self._bump_revision(connection)
                    connection.commit()
            except sqlite3.Error as exc:
                raise AdaptiveStoreError(
                    f"cannot record adaptive observation in {self.path}: {exc}"
                ) from exc
        return self.status(model_id, task=task)

    def _quality_statistics(
        self,
        connection: sqlite3.Connection,
        model_id: str,
        task: str,
        prior: float,
    ) -> tuple[int, float, float, float, float]:
        rows = connection.execute(
            """
            SELECT quality_score, created_at
            FROM adaptive_observations
            WHERE model_id = ? AND task = ? AND quality_score IS NOT NULL
            """,
            (model_id, task),
        ).fetchall()
        return self._decayed_quality(rows, prior)

    def _decayed_quality(
        self,
        rows: Sequence[tuple[Any, Any]],
        prior: float,
    ) -> tuple[int, float, float, float, float]:
        samples = len(rows)
        now = self._now()
        weighted: list[tuple[float, float]] = []
        for raw_score, raw_created_at in rows:
            age_days = max(
                0.0,
                (now - _timestamp(str(raw_created_at))).total_seconds() / 86_400,
            )
            weight = 0.5 ** (age_days / self.policy.evidence_half_life_days)
            weighted.append((weight, float(raw_score)))
        effective_samples = sum(weight for weight, _ in weighted)
        mean = (
            sum(weight * score for weight, score in weighted) / effective_samples
            if effective_samples > 0
            else prior
        )
        second_moment = (
            sum(weight * score * score for weight, score in weighted) / effective_samples
            if effective_samples > 0
            else prior * prior
        )
        squared_mean = mean * mean
        variance = (
            0.0
            if math.isclose(second_moment, squared_mean, rel_tol=1e-12, abs_tol=1e-12)
            else max(0.0, second_moment - squared_mean)
        )
        posterior = (self.policy.prior_samples * prior + effective_samples * mean) / (
            self.policy.prior_samples + effective_samples
        )
        standard_error = math.sqrt(variance / max(effective_samples, 1.0))
        prior_uncertainty = self.policy.uncertainty_scale / math.sqrt(
            self.policy.prior_samples + effective_samples
        )
        lower_bound = max(0.0, posterior - prior_uncertainty - 1.96 * standard_error)
        return samples, effective_samples, posterior, lower_bound, variance

    def _success_statistics(
        self, connection: sqlite3.Connection, model_id: str, task: str
    ) -> tuple[float | None, int]:
        values = connection.execute(
            """
            SELECT success FROM adaptive_observations
            WHERE model_id = ? AND task = ? AND success IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (model_id, task, self.policy.telemetry_recent_samples),
        ).fetchall()
        if not values:
            return None, 0
        parsed = [int(row[0]) for row in values]
        consecutive_failures = 0
        for value in parsed:
            if value:
                break
            consecutive_failures += 1
        return sum(parsed) / len(parsed), consecutive_failures

    def _has_drift(self, connection: sqlite3.Connection, model_id: str, task: str) -> bool:
        values = [
            float(row[0])
            for row in connection.execute(
                """
                SELECT quality_score FROM adaptive_observations
                WHERE model_id = ? AND task = ? AND quality_score IS NOT NULL
                ORDER BY id DESC
                """,
                (model_id, task),
            ).fetchall()
        ]
        window = self.policy.drift_window
        if len(values) < window + self.policy.drift_min_history:
            return False
        recent = values[:window]
        history = values[window:]
        return (
            sum(history) / len(history) - sum(recent) / len(recent)
            >= self.policy.drift_quality_drop
        )

    def _update_task_state(
        self,
        connection: sqlite3.Connection,
        model_id: str,
        task: str,
        now: str,
    ) -> bool:
        model_row = self._model_row(connection, model_id)
        profile = _stored_profile(model_row[0], model_id)
        prior = profile.quality_for(task)
        if prior is None:
            prior = self.policy.cold_start_quality
        samples, effective_samples, _, lower_bound, _ = self._quality_statistics(
            connection, model_id, task, prior
        )
        success_rate, consecutive_failures = self._success_statistics(connection, model_id, task)
        current = connection.execute(
            "SELECT state, reason FROM adaptive_task_states WHERE model_id = ? AND task = ?",
            (model_id, task),
        ).fetchone()
        current_state = model_row[3] if current is None else str(current[0])
        current_reason = None if current is None else current[1]
        state = current_state
        reason: str | None = None
        if consecutive_failures >= self.policy.consecutive_failure_limit:
            state = "degraded"
            reason = f"{consecutive_failures} consecutive execution failures"
        elif self._has_drift(connection, model_id, task):
            state = "degraded"
            reason = "recent quality is below historical quality"
        elif (
            current_state == "trusted"
            and current_reason == "promotion thresholds satisfied"
            and effective_samples
            < self.policy.promotion_samples * self.policy.promotion_effective_sample_fraction
        ):
            state = "provisional"
            reason = "evaluated evidence decayed below promotion support"
        elif (
            self.policy.automatic_promotion
            and samples >= self.policy.promotion_samples
            and effective_samples
            >= self.policy.promotion_samples * self.policy.promotion_effective_sample_fraction
        ):
            success_ok = success_rate is None or success_rate >= self.policy.promotion_success_rate
            if lower_bound >= self.policy.promotion_quality_floor and success_ok:
                state = "trusted"
                reason = "promotion thresholds satisfied"
            elif current_state == "trusted":
                state = "trusted"
        state_changed = state != current_state
        if current is None and state == model_row[3] and reason is None:
            return state_changed
        if current is not None and state == current[0] and reason == current[1]:
            return state_changed
        connection.execute(
            """
            INSERT INTO adaptive_task_states (model_id, task, state, reason, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_id, task) DO UPDATE SET
                state = excluded.state,
                reason = excluded.reason,
                updated_at = excluded.updated_at
            """,
            (model_id, task, state, reason, now),
        )
        return state_changed

    def status(self, model_id: str, *, task: str = "general") -> AdaptiveModelStatus:
        model_id = _text(model_id, "model_id")
        task = _text(task, "task")
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN")
                model_row = self._model_row(connection, model_id)
                profile = _stored_profile(model_row[0], model_id)
                prior = profile.quality_for(task)
                if prior is None:
                    prior = self.policy.cold_start_quality
                samples, effective_samples, posterior, lower_bound, variance = (
                    self._quality_statistics(connection, model_id, task, prior)
                )
                success_rate, consecutive_failures = self._success_statistics(
                    connection, model_id, task
                )
                task_row = connection.execute(
                    """
                    SELECT state, reason FROM adaptive_task_states
                    WHERE model_id = ? AND task = ?
                    """,
                    (model_id, task),
                ).fetchone()
                state = model_row[3] if task_row is None else str(task_row[0])
                reason = None if task_row is None else task_row[1]
                latest_quality = connection.execute(
                    """
                    SELECT MAX(created_at) FROM adaptive_observations
                    WHERE model_id = ? AND task = ? AND quality_score IS NOT NULL
                    """,
                    (model_id, task),
                ).fetchone()[0]
                if (
                    state == "trusted"
                    and reason == "promotion thresholds satisfied"
                    and self._evidence_is_stale(latest_quality)
                ):
                    state = "provisional"
                    reason = "evaluated evidence is stale"
                telemetry_rows = connection.execute(
                    """
                    SELECT latency_ms, actual_cost_usd, success, failure_type
                    FROM adaptive_observations
                    WHERE model_id = ? AND task = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (model_id, task, self.policy.telemetry_recent_samples),
                ).fetchall()
        except sqlite3.Error as exc:
            raise AdaptiveStoreError(f"cannot read adaptive registry {self.path}: {exc}") from exc
        confidence = 100 * effective_samples / (effective_samples + self.policy.prior_samples)
        latency_values = [float(row[0]) for row in telemetry_rows if row[0] is not None]
        cost_values = [float(row[1]) for row in telemetry_rows if row[1] is not None]
        failed_rows = [row for row in telemetry_rows if row[2] == 0]
        failure_counts: dict[str, int] = {}
        for row in failed_rows:
            name = str(row[3] or FailureType.UNKNOWN_FAILURE.value)
            failure_counts[name] = failure_counts.get(name, 0) + 1
        failure_probabilities = tuple(
            sorted((name, count / len(failed_rows)) for name, count in failure_counts.items())
        )
        return AdaptiveModelStatus(
            model_id=model_id,
            operating_state=str(model_row[2]),
            task=task,
            task_state=state,
            source=str(model_row[1]),
            samples=samples,
            effective_samples=effective_samples,
            posterior_quality=posterior,
            quality_lower_bound=lower_bound,
            quality_variance=variance,
            confidence=confidence,
            success_rate=success_rate,
            average_latency_ms=(
                None if not latency_values else sum(latency_values) / len(latency_values)
            ),
            latency_p95_ms=_percentile(latency_values, 0.95),
            average_cost_usd=(None if not cost_values else sum(cost_values) / len(cost_values)),
            cost_p95_usd=_percentile(cost_values, 0.95),
            failure_probabilities=failure_probabilities,
            consecutive_failures=consecutive_failures,
            first_seen_at=str(model_row[4]),
            last_seen_at=str(model_row[5]),
            state_reason=None if reason is None else str(reason),
        )

    def task_state(self, model_id: str, task: str) -> str:
        return self.status(model_id, task=task).task_state

    def _evidence_is_stale(self, created_at: Any) -> bool:
        if created_at is None:
            return True
        age = self._now() - _timestamp(str(created_at))
        return age > timedelta(days=self.policy.trusted_evidence_stale_after_days)

    def routing_states(self, task: str) -> dict[str, tuple[str, str]]:
        task = _text(task, "task")
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN")
                rows = connection.execute(
                    """
                    SELECT models.model_id, models.operating_state,
                           COALESCE(tasks.state, models.default_task_state),
                           tasks.reason,
                           (
                               SELECT MAX(observations.created_at)
                               FROM adaptive_observations AS observations
                               WHERE observations.model_id = models.model_id
                                 AND observations.task = ?
                                 AND observations.quality_score IS NOT NULL
                           )
                    FROM adaptive_models AS models
                    LEFT JOIN adaptive_task_states AS tasks
                      ON tasks.model_id = models.model_id AND tasks.task = ?
                    ORDER BY models.model_id
                    """,
                    (task, task),
                ).fetchall()
        except sqlite3.Error as exc:
            raise AdaptiveStoreError(f"cannot read adaptive registry {self.path}: {exc}") from exc
        result: dict[str, tuple[str, str]] = {}
        for model_id, operating_state, task_state, reason, latest_quality in rows:
            if operating_state not in OPERATING_STATES or task_state not in TASK_STATES:
                raise AdaptiveStoreError(
                    f"adaptive registry contains an invalid lifecycle state for {model_id}"
                )
            effective_state = str(task_state)
            if (
                effective_state == "trusted"
                and reason == "promotion thresholds satisfied"
                and self._evidence_is_stale(latest_quality)
            ):
                effective_state = "provisional"
            result[str(model_id)] = (str(operating_state), effective_state)
        return result

    def set_operating_state(self, model_id: str, state: str) -> AdaptiveModelStatus:
        model_id = _text(model_id, "model_id")
        state = _text(state, "state").lower()
        if state not in OPERATING_STATES:
            raise OpenRoutiQError(
                f"operating state must be one of: {', '.join(sorted(OPERATING_STATES))}"
            )
        now = self._now_text()
        with self._lock:
            try:
                with closing(self._connect()) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    self._model_row(connection, model_id)
                    connection.execute(
                        """
                        UPDATE adaptive_models
                        SET operating_state = ?, updated_at = ? WHERE model_id = ?
                        """,
                        (state, now, model_id),
                    )
                    self._bump_revision(connection)
                    connection.commit()
            except sqlite3.Error as exc:
                raise AdaptiveStoreError(
                    f"cannot update adaptive registry {self.path}: {exc}"
                ) from exc
        return self.status(model_id)

    def set_task_state(self, model_id: str, task: str, state: str) -> AdaptiveModelStatus:
        model_id = _text(model_id, "model_id")
        task = _text(task, "task")
        state = _text(state, "state").lower()
        if state not in TASK_STATES:
            raise OpenRoutiQError(f"task state must be one of: {', '.join(sorted(TASK_STATES))}")
        now = self._now_text()
        with self._lock:
            try:
                with closing(self._connect()) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    self._model_row(connection, model_id)
                    connection.execute(
                        """
                        INSERT INTO adaptive_task_states (model_id, task, state, reason, updated_at)
                        VALUES (?, ?, ?, 'manual override', ?)
                        ON CONFLICT(model_id, task) DO UPDATE SET
                            state = excluded.state,
                            reason = excluded.reason,
                            updated_at = excluded.updated_at
                        """,
                        (model_id, task, state, now),
                    )
                    self._bump_revision(connection)
                    connection.commit()
            except sqlite3.Error as exc:
                raise AdaptiveStoreError(
                    f"cannot update adaptive registry {self.path}: {exc}"
                ) from exc
        return self.status(model_id, task=task)

    def profiles(self, *, include_inactive: bool = True) -> tuple[ModelProfile, ...]:
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN")
                rows = connection.execute(
                    """
                    SELECT model_id, profile_json, default_task_state, operating_state
                    FROM adaptive_models ORDER BY model_id
                    """
                ).fetchall()
                quality_stats: dict[str, dict[str, list[float]]] = {}
                quality_now = self._now()
                for model_id, task, quality_score, created_at in connection.execute(
                    """
                    SELECT model_id, task, quality_score, created_at
                    FROM adaptive_observations
                    WHERE quality_score IS NOT NULL
                    ORDER BY model_id, task, id
                    """
                ):
                    age_days = max(
                        0.0,
                        (quality_now - _timestamp(str(created_at))).total_seconds() / 86_400,
                    )
                    weight = 0.5 ** (age_days / self.policy.evidence_half_life_days)
                    aggregate = quality_stats.setdefault(str(model_id), {}).setdefault(
                        str(task), [0.0, 0.0]
                    )
                    aggregate[0] += weight
                    aggregate[1] += weight * float(quality_score)
                latency_values: dict[str, list[float]] = {}
                for model_id, latency_ms in connection.execute(
                    """
                    SELECT model_id, latency_ms
                    FROM (
                        SELECT model_id, latency_ms,
                               ROW_NUMBER() OVER (
                                   PARTITION BY model_id ORDER BY id DESC
                               ) AS recent_rank
                        FROM adaptive_observations
                        WHERE latency_ms IS NOT NULL
                    )
                    WHERE recent_rank <= ?
                    ORDER BY model_id, recent_rank
                    """,
                    (self.policy.telemetry_recent_samples,),
                ):
                    latency_values.setdefault(str(model_id), []).append(float(latency_ms))
                cost_stats: dict[str, list[tuple[float, int, int]]] = {}
                for model_id, actual_cost, input_tokens, output_tokens in connection.execute(
                    """
                    SELECT model_id, actual_cost_usd, input_tokens, output_tokens
                    FROM (
                        SELECT model_id, actual_cost_usd, input_tokens, output_tokens,
                               ROW_NUMBER() OVER (
                                   PARTITION BY model_id ORDER BY id DESC
                               ) AS recent_rank
                        FROM adaptive_observations
                        WHERE actual_cost_usd IS NOT NULL
                          AND input_tokens IS NOT NULL AND output_tokens IS NOT NULL
                          AND input_tokens + output_tokens > 0
                    )
                    WHERE recent_rank <= ?
                    ORDER BY model_id, recent_rank
                    """,
                    (self.policy.telemetry_recent_samples,),
                ):
                    cost_stats.setdefault(str(model_id), []).append(
                        (float(actual_cost), int(input_tokens), int(output_tokens))
                    )
                profiles: list[ModelProfile] = []
                for model_id, raw_profile, default_state, operating_state in rows:
                    if not include_inactive and operating_state != "active":
                        continue
                    if operating_state not in OPERATING_STATES or default_state not in TASK_STATES:
                        raise AdaptiveStoreError(
                            f"adaptive registry contains an invalid lifecycle state for {model_id}"
                        )
                    profile = _stored_profile(raw_profile, str(model_id))
                    quality = dict(profile.quality)
                    confidences: list[float] = []
                    for task, aggregate in quality_stats.get(str(model_id), {}).items():
                        prior = profile.quality_for(task)
                        if prior is None:
                            prior = self.policy.cold_start_quality
                        effective_samples, weighted_total = aggregate
                        posterior = (self.policy.prior_samples * prior + weighted_total) / (
                            self.policy.prior_samples + effective_samples
                        )
                        quality[task] = posterior
                        confidences.append(
                            100
                            * effective_samples
                            / (effective_samples + self.policy.prior_samples)
                        )
                    recent_latencies = latency_values.get(str(model_id), [])
                    latency_samples = len(recent_latencies)
                    observed_latency = (
                        sum(recent_latencies) / latency_samples
                        if latency_samples
                        else profile.latency_ms
                    )
                    latency = profile.latency_ms
                    if latency_samples:
                        latency = (
                            self.policy.prior_samples * profile.latency_ms
                            + latency_samples * observed_latency
                        ) / (self.policy.prior_samples + latency_samples)
                    learned_confidence = max(confidences, default=0.0)
                    confidence = (
                        max(profile.confidence, learned_confidence)
                        if default_state == "trusted"
                        else learned_confidence
                    )
                    input_price = profile.input_price_per_million
                    output_price = profile.output_price_per_million
                    ratios: list[float] = []
                    effective_prices: list[float] = []
                    for actual_cost, input_tokens, output_tokens in cost_stats.get(
                        str(model_id), []
                    ):
                        predicted = (
                            input_tokens * input_price + output_tokens * output_price
                        ) / 1_000_000
                        if predicted > 0:
                            ratio = actual_cost / predicted
                            if math.isfinite(ratio):
                                ratios.append(
                                    min(
                                        self.policy.cost_ratio_ceiling,
                                        max(self.policy.cost_ratio_floor, ratio),
                                    )
                                )
                        else:
                            effective_price = (
                                actual_cost * 1_000_000 / (input_tokens + output_tokens)
                            )
                            if math.isfinite(effective_price):
                                effective_prices.append(
                                    min(
                                        self.policy.maximum_learned_price_per_million,
                                        effective_price,
                                    )
                                )
                    if ratios:
                        ordered = sorted(ratios)
                        middle = len(ordered) // 2
                        median_ratio = (
                            ordered[middle]
                            if len(ordered) % 2
                            else (ordered[middle - 1] + ordered[middle]) / 2
                        )
                        learned_ratio = (self.policy.prior_samples + len(ratios) * median_ratio) / (
                            self.policy.prior_samples + len(ratios)
                        )
                        input_price *= learned_ratio
                        output_price *= learned_ratio
                    elif effective_prices:
                        ordered = sorted(effective_prices)
                        middle = len(ordered) // 2
                        effective_price = (
                            ordered[middle]
                            if len(ordered) % 2
                            else (ordered[middle - 1] + ordered[middle]) / 2
                        )
                        input_price = effective_price
                        output_price = effective_price
                    profiles.append(
                        replace(
                            profile,
                            quality=quality,
                            latency_ms=latency,
                            input_price_per_million=input_price,
                            output_price_per_million=output_price,
                            confidence=confidence,
                            tags=profile.tags | frozenset({"adaptive"}),
                        )
                    )
        except sqlite3.Error as exc:
            raise AdaptiveStoreError(f"cannot read adaptive registry {self.path}: {exc}") from exc
        return tuple(profiles)

    def reserve_exploration(self, model_id: str, task: str, predicted_cost: float) -> bool:
        model_id = _text(model_id, "model_id")
        task = _text(task, "task")
        cost = _finite_number(predicted_cost, "predicted_cost", minimum=0)
        cost_nanos = _usd_nanos(cost, rounding=ROUND_CEILING)
        daily_budget_nanos = _usd_nanos(
            self.policy.exploration_daily_budget_usd,
            rounding=ROUND_FLOOR,
        )
        request_budget_nanos = _usd_nanos(
            self.policy.exploration_max_request_cost_usd,
            rounding=ROUND_FLOOR,
        )
        if (
            daily_budget_nanos <= 0
            or request_budget_nanos <= 0
            or cost_nanos > request_budget_nanos
        ):
            return False
        now = self._now()
        start = datetime(now.year, now.month, now.day, tzinfo=UTC).isoformat()
        end = (datetime(now.year, now.month, now.day, tzinfo=UTC) + timedelta(days=1)).isoformat()
        retention_boundary = (
            now - timedelta(days=self.policy.exploration_retention_days)
        ).isoformat()
        created_at = now.isoformat(timespec="microseconds")
        with self._lock:
            try:
                with closing(self._connect()) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    self._model_row(connection, model_id)
                    connection.execute(
                        "DELETE FROM adaptive_explorations WHERE created_at < ?",
                        (retention_boundary,),
                    )
                    spent_row = connection.execute(
                        """
                        SELECT COALESCE(SUM(reserved_cost_nano_usd), 0)
                        FROM adaptive_explorations
                        WHERE created_at >= ? AND created_at < ?
                        """,
                        (start, end),
                    ).fetchone()
                    spent_nanos = int(spent_row[0])
                    if spent_nanos + cost_nanos > daily_budget_nanos:
                        connection.rollback()
                        return False
                    connection.execute(
                        """
                        INSERT INTO adaptive_explorations (
                            model_id, task, reserved_cost_usd,
                            reserved_cost_nano_usd, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (model_id, task, cost_nanos / 1_000_000_000, cost_nanos, created_at),
                    )
                    connection.commit()
                    return True
            except sqlite3.Error as exc:
                raise AdaptiveStoreError(
                    f"cannot reserve adaptive exploration budget in {self.path}: {exc}"
                ) from exc


def _pinned_model(
    request: str | Sequence[Any] | RouteContext, options: Mapping[str, Any]
) -> str | None:
    pinned = options.get("pinned_model")
    if pinned is None and isinstance(request, RouteContext):
        pinned = request.pinned_model
    return None if pinned is None else str(pinned)


def _candidate_constraints(value: Any, candidate_ids: Iterable[str]) -> Constraints:
    parsed = Constraints.parse(value)
    candidates = frozenset(candidate_ids)
    if parsed.candidate_ids:
        candidates &= parsed.candidate_ids
    return replace(parsed, candidate_ids=candidates)


def _annotate(
    decision: RouteDecision,
    status: AdaptiveModelStatus,
    *,
    exploration: bool,
    selection_probability: float = 1.0,
) -> RouteDecision:
    signal = "adaptive:exploration" if exploration else f"adaptive:{status.task_state}"
    analysis = replace(decision.analysis, signals=decision.analysis.signals + (signal,))
    reasons = list(decision.review_reasons)
    if status.operating_state != "active":
        reasons.append(f"adaptive model operating state is {status.operating_state}")
    if status.task_state != "trusted":
        reasons.append(
            f"adaptive model is {status.task_state} for {status.task} "
            f"with {status.samples} evaluated outcomes"
        )
    if exploration:
        reasons.append("bounded exploration selected a provisional model")
    return replace(
        decision,
        analysis=analysis,
        review_required=bool(reasons),
        review_reasons=tuple(dict.fromkeys(reasons)),
        selection_probability=selection_probability,
    )


class AdaptiveRouter:
    """Encounter-driven router for public, private, fine-tuned, and future models."""

    def __init__(
        self,
        profiles: Iterable[ModelProfile | Mapping[str, Any]] = (),
        *,
        registry: str | Path | AdaptiveRegistryBackend,
        policy: AdaptivePolicy | None = None,
        **router_options: Any,
    ) -> None:
        if isinstance(registry, AdaptiveRegistryBackend):
            if policy is not None and policy != registry.policy:
                raise OpenRoutiQError("policy conflicts with the adaptive registry policy")
            self.registry = registry
            self.policy = registry.policy
        elif isinstance(registry, (str, Path)):
            self.policy = policy or AdaptivePolicy()
            self.registry = AdaptiveModelRegistry(registry, policy=self.policy)
        else:
            raise OpenRoutiQError(
                "registry must be a path or an AdaptiveRegistryBackend implementation"
            )
        for profile in profiles:
            self.registry.encounter(profile, source="catalog", trusted=True)
        self._router_options = dict(router_options)
        self._router: Router | None = None
        self._router_revision = -1
        self._router_lock = RLock()
        self._random = random.Random(self.policy.random_seed)
        self._random_lock = Lock()

    @classmethod
    def from_file(
        cls,
        catalog: str | Path,
        *,
        registry: str | Path | AdaptiveRegistryBackend,
        policy: AdaptivePolicy | None = None,
        **router_options: Any,
    ) -> AdaptiveRouter:
        path = Path(catalog).expanduser().resolve()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise OpenRoutiQError(f"cannot read catalog {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OpenRoutiQError(f"invalid JSON in catalog {path}: {exc}") from exc
        models = raw.get("models") if isinstance(raw, Mapping) else raw
        if not isinstance(models, list):
            raise OpenRoutiQError("catalog root must be a model list or an object with models")
        if isinstance(raw, Mapping) and "task_examples" not in router_options:
            router_options["task_examples"] = raw.get("task_examples")
        return cls(
            models,
            registry=registry,
            policy=policy,
            **router_options,
        )

    def _core(self) -> Router:
        revision = self.registry.revision
        with self._router_lock:
            if self._router is None or revision != self._router_revision:
                profiles = self.registry.profiles(include_inactive=True)
                if not profiles:
                    raise OpenRoutiQError(
                        "adaptive registry contains no models; encounter a model first"
                    )
                router_options = dict(self._router_options)
                router_options.setdefault("outcome_prior", self._outcome_prior)
                self._router = Router(profiles, **router_options)
                self._router_revision = revision
            return self._router

    def _outcome_prior(self, model_id: str, task: str) -> OutcomeEstimate:
        status = self.registry.status(model_id, task=task)
        return OutcomeEstimate(
            quality_score=status.posterior_quality,
            latency_ms=status.average_latency_ms,
            similarity=0.0,
            samples=status.samples,
            quality_stddev=math.sqrt(max(0.0, status.quality_variance)),
            quality_lower_bound=status.quality_lower_bound,
            success_probability=status.success_rate,
            average_cost_usd=status.average_cost_usd,
            latency_p95_ms=status.latency_p95_ms,
            cost_p95_usd=status.cost_p95_usd,
            effective_samples=status.effective_samples,
        )

    @property
    def profiles(self) -> tuple[ModelProfile, ...]:
        return self.registry.profiles(include_inactive=True)

    def encounter(
        self,
        profile: ModelProfile | Mapping[str, Any],
        *,
        source: str = "encounter",
        trusted: bool = False,
    ) -> AdaptiveModelStatus:
        return self.registry.encounter(profile, source=source, trusted=trusted)

    def encounter_opaque(self, **contract: Any) -> AdaptiveModelStatus:
        return self.registry.encounter_opaque(**contract)

    def status(self, model_id: str, *, task: str = "general") -> AdaptiveModelStatus:
        return self.registry.status(model_id, task=task)

    def route(
        self,
        request: str | Sequence[Any] | RouteContext,
        *,
        explore: bool | None = None,
        **options: Any,
    ) -> RouteDecision:
        if explore is not None and not isinstance(explore, bool):
            raise OpenRoutiQError("explore must be a boolean or None")
        core = self._core()
        preliminary = core.route(request, **options)
        task = preliminary.task
        pinned = _pinned_model(request, options)
        if pinned is not None:
            return _annotate(
                preliminary,
                self.registry.status(preliminary.selected.model_id, task=task),
                exploration=False,
                selection_probability=1.0,
            )

        constrained = Constraints.parse(options.get("constraints"))
        active_ids: set[str] = set()
        trusted_ids: set[str] = set()
        provisional_ids: set[str] = set()
        states = self.registry.routing_states(task)
        for profile in core.profiles:
            operating_state, task_state = states[profile.id]
            if operating_state != "active":
                continue
            active_ids.add(profile.id)
            if task_state == "trusted":
                trusted_ids.add(profile.id)
            elif task_state == "provisional":
                provisional_ids.add(profile.id)
        if constrained.candidate_ids:
            active_ids &= constrained.candidate_ids
            trusted_ids &= constrained.candidate_ids
            provisional_ids &= constrained.candidate_ids

        baseline: RouteDecision | None = None
        if trusted_ids:
            baseline_options = dict(options)
            baseline_options["constraints"] = _candidate_constraints(
                options.get("constraints"), trusted_ids
            )
            try:
                baseline = core.route(request, **baseline_options)
            except NoEligibleModelError:
                baseline = None

        if explore is True:
            should_explore = True
        elif explore is False or self.policy.exploration_rate <= 0:
            should_explore = False
        else:
            with self._random_lock:
                should_explore = self._random.random() < self.policy.exploration_rate
        if preliminary.analysis.high_risk:
            should_explore = False

        if should_explore and provisional_ids:
            exploration_options = dict(options)
            exploration_options["constraints"] = _candidate_constraints(
                options.get("constraints"), provisional_ids
            )
            try:
                candidate = core.route(request, **exploration_options)
            except NoEligibleModelError:
                candidate = None
            if candidate is not None and self.registry.reserve_exploration(
                candidate.selected.model_id,
                candidate.task,
                candidate.selected.predicted_cost,
            ):
                return _annotate(
                    candidate,
                    self.registry.status(candidate.selected.model_id, task=task),
                    exploration=True,
                    selection_probability=(
                        1.0 if explore is True else self.policy.exploration_rate
                    ),
                )

        if baseline is None:
            available = sorted(active_ids)
            detail = f" Registered active candidates: {', '.join(available)}." if available else ""
            raise OpenRoutiQError(
                f"no trusted adaptive model is eligible for task {task}; explicitly pin a "
                f"provisional model or enable budgeted exploration.{detail}"
            )
        return _annotate(
            baseline,
            self.registry.status(baseline.selected.model_id, task=task),
            exploration=False,
            selection_probability=(
                1.0 - self.policy.exploration_rate
                if explore is None
                and provisional_ids
                and not preliminary.analysis.high_risk
                and self.policy.exploration_rate > 0
                else 1.0
            ),
        )

    def observe_execution(
        self,
        model_id: str,
        *,
        task: str = "general",
        latency_ms: float | None = None,
        actual_cost_usd: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        success: bool | None = None,
        failure_type: FailureType | str | None = None,
    ) -> AdaptiveModelStatus:
        return self.registry.record(
            model_id,
            task,
            latency_ms=latency_ms,
            actual_cost_usd=actual_cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            failure_type=failure_type,
        )

    def record_evaluation(
        self,
        request: str | Sequence[Any] | RouteContext,
        model_id: str,
        quality_score: float,
        *,
        task: str | None = None,
        latency_ms: float | None = None,
        actual_cost_usd: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        success: bool | None = None,
        failure_class: str | None = None,
        failure_type: FailureType | str | None = None,
        selection_probability: float | None = None,
        tools: Sequence[Any] | None = None,
        parallel_tool_calls: bool | None = None,
        response_format: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> AdaptiveModelStatus:
        core = self._core()
        selected_task = (
            _text(task, "task")
            if task is not None
            else analyze_context(request, task_classifier=core.task_classifier).task
        )
        if core.outcome_store is not None:
            core.record_evaluation(
                request,
                model_id,
                quality_score,
                latency_ms=latency_ms,
                actual_cost_usd=actual_cost_usd,
                success=success,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                failure_class=failure_class,
                selection_probability=selection_probability,
                tools=tools,
                parallel_tool_calls=parallel_tool_calls,
                response_format=response_format,
                stream=stream,
            )
        return self.registry.record(
            model_id,
            selected_task,
            quality_score=quality_score,
            latency_ms=latency_ms,
            actual_cost_usd=actual_cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            failure_type=failure_type,
        )
