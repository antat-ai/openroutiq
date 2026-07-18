from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from openroutiq.router.failures import FailureType


class CapabilityProfile(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def capabilities(self) -> frozenset[str]: ...

    @property
    def supported_parameters(self) -> frozenset[str]: ...

    @property
    def max_context_tokens(self) -> int: ...

    @property
    def reasoning_level(self) -> str | None: ...


@dataclass(frozen=True)
class CapabilityRequirements:
    """Hard execution requirements derived before a model is scored."""

    capabilities: frozenset[str] = field(default_factory=frozenset)
    required_parameters: frozenset[str] = field(default_factory=frozenset)
    any_parameter_groups: tuple[frozenset[str], ...] = ()
    context_tokens: int = 0
    reasoning_level: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.context_tokens, bool) or self.context_tokens < 0:
            raise ValueError("context_tokens must be an integer >= 0")
        if any(not group for group in self.any_parameter_groups):
            raise ValueError("any_parameter_groups cannot contain an empty group")


@dataclass(frozen=True)
class CapabilityGateResult:
    model_id: str
    eligible: bool
    reasons: tuple[str, ...]
    failure_type: FailureType | None


class CapabilityGate:
    """Fail-closed pre-scoring filter for model and provider execution contracts.

    ``supported_parameters`` is optional for backward-compatible catalogs. When it is
    declared, the gate enforces it exactly; semantic capabilities and context limits are
    always enforced.
    """

    def evaluate(
        self,
        profile: CapabilityProfile,
        requirements: CapabilityRequirements,
    ) -> CapabilityGateResult:
        reasons: list[str] = []
        missing = sorted(requirements.capabilities - profile.capabilities)
        if missing:
            reasons.append(f"missing capabilities: {', '.join(missing)}")
        if profile.max_context_tokens < requirements.context_tokens:
            reasons.append(
                f"context {profile.max_context_tokens} < required {requirements.context_tokens}"
            )

        supported = profile.supported_parameters
        if supported:
            missing_parameters = sorted(requirements.required_parameters - supported)
            if missing_parameters:
                reasons.append("missing supported parameters: " + ", ".join(missing_parameters))
            for group in requirements.any_parameter_groups:
                if not supported.intersection(group):
                    reasons.append(
                        "missing any supported parameter from: " + ", ".join(sorted(group))
                    )

        requested_reasoning = requirements.reasoning_level
        if (
            requested_reasoning is not None
            and (profile.reasoning_level or "none") != requested_reasoning
        ):
            reasons.append(f"reasoning level is not {requested_reasoning}")
        return CapabilityGateResult(
            model_id=profile.id,
            eligible=not reasons,
            reasons=tuple(reasons),
            failure_type=FailureType.CAPABILITY_MISMATCH if reasons else None,
        )

    def eligible(
        self,
        profiles: Iterable[CapabilityProfile],
        requirements: CapabilityRequirements,
    ) -> tuple[CapabilityGateResult, ...]:
        return tuple(self.evaluate(profile, requirements) for profile in profiles)
