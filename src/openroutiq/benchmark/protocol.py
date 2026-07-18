from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


# This object is embedded in every reviewed live-flow plan. Changing the relay
# contract therefore changes the plan SHA-256 and requires a fresh approval.
OPENROUTER_BENCHMARK_PROTOCOL: dict[str, Any] = {
    "id": "openrouter-chat-completions-price-capped-v4",
    "chat_output_token_parameter": "max_tokens",
    "provider_require_parameters": True,
    "provider_sort": "price",
    "provider_max_price_enforced": True,
    "maximum_consecutive_upstream_failures": 3,
    "maximum_parallel_cases": 6,
    "parallelism_scope": "one_inflight_case_per_system",
    "single_settlement_per_reservation": True,
    "downstream_disconnect_after_settlement": "ignore",
}


def deterministic_selection_splits(
    case_ids: Sequence[str],
    *,
    seed: int,
    track_id: str,
    split_counts: Mapping[str, int],
) -> dict[str, str]:
    """Assign predeclared benchmark cases to leakage-safe deterministic splits."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("selection split seed must be an integer")
    if not track_id:
        raise ValueError("selection split track id must be non-empty")
    ids = list(case_ids)
    if not ids or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("selection split case ids must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("selection split case ids must be unique")
    if not split_counts:
        raise ValueError("selection split counts must not be empty")
    counts: list[tuple[str, int]] = []
    for name, count in split_counts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("selection split names must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("selection split counts must be positive integers")
        counts.append((name, count))
    if sum(count for _, count in counts) != len(ids):
        raise ValueError("selection split counts must equal the number of cases")
    ranked = sorted(
        ids,
        key=lambda case_id: hashlib.sha256(
            f"{seed}:split:{track_id}:{case_id}".encode()
        ).hexdigest(),
    )
    assignments: dict[str, str] = {}
    cursor = 0
    for name, count in counts:
        for case_id in ranked[cursor : cursor + count]:
            assignments[case_id] = name
        cursor += count
    return assignments
