from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from openroutiq.router.core import OpenRoutiQError, RouteDecision, Router


_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "model": "YOUR_OPENAI_MODEL",
        "api_style": "openai_responses",
        "reasoning_mode": "effort",
    },
    "anthropic": {
        "model": "YOUR_ANTHROPIC_MODEL",
        "api_style": "anthropic_messages",
        "reasoning_mode": "adaptive",
    },
    "openrouter": {
        "model": "YOUR_PROVIDER/YOUR_MODEL",
        "api_style": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "reasoning_mode": "effort",
        "provider_options": {"allow_fallbacks": True},
    },
    "requesty": {
        "model": "YOUR_PROVIDER/YOUR_MODEL",
        "api_style": "requesty",
        "base_url": "https://router.requesty.ai/v1",
        "reasoning_mode": "effort",
    },
    "litellm": {
        "model": "YOUR_PROVIDER/YOUR_MODEL",
        "api_style": "litellm",
        "reasoning_mode": "effort",
    },
}

_TASK_EXAMPLES = {
    "general": [
        "Hello there",
        "Answer this question clearly",
        "Help me understand this topic",
        "Summarize our conversation",
        "Explain this in simple terms",
    ],
    "coding": [
        "Repair the race condition in this worker",
        "Implement the missing cache eviction behavior",
        "Review this Python service and add tests",
        "Find the defect in this SQL query",
    ],
    "reasoning": [
        "Solve this proof step by step",
        "Compare the tradeoffs under these constraints",
        "Work through this logic puzzle",
        "Plan a multi-stage migration",
    ],
    "writing": [
        "Polish this customer announcement",
        "Rewrite this paragraph for clarity",
        "Draft an article for our audience",
        "Edit this product copy",
    ],
    "research": [
        "Investigate the evidence and cite reliable sources",
        "Find recent papers about this subject",
        "Compare the available sources",
        "Produce a sourced literature review",
    ],
    "extraction": [
        "Pull the named fields into structured data",
        "Convert this document into a table",
        "Parse these records into JSON",
        "Identify every date and amount",
    ],
    "vision": [
        "Inspect the attached screenshot",
        "Describe what appears in this photograph",
        "Read this architecture diagram",
        "Compare the two images",
    ],
    "tool_use": [
        "Look up the order using the available function",
        "Call the service and return its result",
        "Schedule the deployment with the provided action",
        "Use the available tools to complete the request",
    ],
}


def _profile(provider: str) -> dict[str, Any]:
    return {
        "id": f"template/{provider}:high",
        "provider": provider,
        **_PROVIDERS[provider],
        "reasoning_level": "high",
        "quality": {
            "general": 80,
            "coding": 80,
            "reasoning": 80,
            "tool_use": 80,
            "extraction": 80,
        },
        "latency_ms": 1000,
        "input_price_per_million": 1,
        "output_price_per_million": 1,
        "max_context_tokens": 100_000,
        "capabilities": ["text", "vision", "tools", "parallel_tools", "json_schema", "streaming"],
        "confidence": 1,
    }


def init_catalog(
    path: str | Path = "models.json",
    *,
    provider: str = "all",
    force: bool = False,
) -> Path:
    """Create an editable starter catalog without pretending its values are benchmarks."""
    if provider != "all" and provider not in _PROVIDERS:
        choices = ", ".join(("all", *_PROVIDERS))
        raise OpenRoutiQError(f"unknown provider {provider!r}; choose one of: {choices}")
    target = Path(path).expanduser()
    if target.exists() and not force:
        raise OpenRoutiQError(f"catalog already exists: {target}; pass --force to replace it")
    providers = _PROVIDERS if provider == "all" else {provider: _PROVIDERS[provider]}
    catalog = {
        "note": (
            "Starter values only. Replace model names, quality scores, prices, latency, limits, "
            "capabilities, confidence, and task examples with verified data for your deployment."
        ),
        "task_examples": _TASK_EXAMPLES,
        "models": [_profile(name) for name in providers],
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise OpenRoutiQError(f"cannot create catalog {target}: {exc}") from exc
    return target


def _catalog_path(catalog: str | Path | None) -> Path:
    configured = catalog or os.environ.get("OPENROUTIQ_CATALOG") or "models.json"
    return Path(configured).expanduser().resolve()


@lru_cache(maxsize=8)
def _load_router(path: str, modified_ns: int) -> Router:
    del modified_ns
    return Router.from_file(path)


def get_router(catalog: str | Path | None = None) -> Router:
    """Load and cache a catalog, refreshing automatically when the file changes."""
    path = _catalog_path(catalog)
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError as exc:
        raise OpenRoutiQError(
            f"cannot read catalog {path}; run 'openroutiq init' or pass catalog=..."
        ) from exc
    return _load_router(str(path), modified_ns)


def route(request: Any, *, catalog: str | Path | None = None, **options: Any) -> RouteDecision:
    """Route with models.json (or OPENROUTIQ_CATALOG) in one function call."""
    return get_router(catalog).route(request, **options)
