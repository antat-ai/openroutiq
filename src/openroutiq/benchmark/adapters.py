from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openroutiq.benchmark.core import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkError,
    RouterSelection,
)


def _message_list(request: str | Sequence[Any]) -> list[Any]:
    return [{"role": "user", "content": request}] if isinstance(request, str) else list(request)


def _response_text(payload: Mapping[str, Any]) -> str | None:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if not isinstance(block, Mapping):
                        continue
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                return "\n".join(parts) if parts else None
        text = choices[0].get("text")
        return text if isinstance(text, str) else None
    output_text = payload.get("output_text")
    return output_text if isinstance(output_text, str) else None


class OpenAICompatibleBenchmarkRouter:
    """Benchmark any OpenAI-compatible router that returns the selected model.

    This adapter performs model inference and is always marked live. It works with
    hosted routers such as OpenRouter Auto and with local RouteLLM-compatible servers.
    Credentials are read only from the named environment variable.
    """

    is_live = True
    model_calls_per_case = 1
    router_calls_per_case = 0
    timing_scope = "end_to_end"

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        router_model: str,
        api_key_env: str,
        timeout_seconds: float = 120,
        max_output_tokens: int | None = None,
        allowed_models_field: str | None = None,
        headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.router_model = router_model
        self.api_key_env = api_key_env
        if not self.base_url or not self.router_model or not self.api_key_env:
            raise BenchmarkError("OpenAI-compatible router requires URL, model, and key env")
        if timeout_seconds <= 0:
            raise BenchmarkError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.allowed_models_field = allowed_models_field
        self.headers = dict(headers or {})
        self.extra_body = dict(extra_body or {})

    def prepare(self, dataset: BenchmarkDataset) -> None:
        del dataset
        if not os.environ.get(self.api_key_env):
            raise BenchmarkError(f"{self.name} requires environment variable {self.api_key_env}")

    def _body(self, case: BenchmarkCase) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.router_model,
            "messages": _message_list(case.request),
            "temperature": 0,
            **self.extra_body,
        }
        output_tokens = self.max_output_tokens or case.expected_output_tokens
        if output_tokens is not None:
            body["max_tokens"] = output_tokens
        allowed = [candidate.model for candidate in case.eligible_candidates]
        if self.allowed_models_field == "openrouter_plugins":
            raw_plugins = body.get("plugins", [])
            if not isinstance(raw_plugins, list) or any(
                not isinstance(plugin, Mapping) for plugin in raw_plugins
            ):
                raise BenchmarkError("OpenRouter plugins must be an object list")
            plugins: list[dict[str, Any]] = []
            auto_router_seen = False
            for plugin in raw_plugins:
                item = dict(plugin)
                if item.get("id") == "auto-router":
                    if auto_router_seen:
                        raise BenchmarkError(
                            "OpenRouter config contains duplicate auto-router plugins"
                        )
                    auto_router_seen = True
                    # The benchmark case is authoritative for the common candidate
                    # pool. Preserve options such as cost_tier while replacing any
                    # stale allowed_models value from the static config.
                    item["allowed_models"] = allowed
                plugins.append(item)
            if not auto_router_seen:
                plugins.append({"id": "auto-router", "allowed_models": allowed})
            body["plugins"] = plugins
        elif self.allowed_models_field:
            body[self.allowed_models_field] = allowed
        return body

    def select(self, case: BenchmarkCase) -> RouterSelection:
        api_key = os.environ[self.api_key_env]
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(self._body(case)).encode("utf-8"),
            headers={
                **self.headers,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise BenchmarkError(f"{self.name} returned HTTP {exc.code}: {detail}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkError(f"{self.name} request failed: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise BenchmarkError(f"{self.name} returned a non-object response")
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise BenchmarkError(f"{self.name} response did not identify the selected model")
        usage = payload.get("usage", {})
        if not isinstance(usage, Mapping):
            usage = {}
        reported_cost = usage.get("cost")
        return RouterSelection(
            model_id=model,
            output=_response_text(payload),
            prompt_tokens=usage.get("prompt_tokens")
            if isinstance(usage.get("prompt_tokens"), int)
            else None,
            completion_tokens=(
                usage.get("completion_tokens")
                if isinstance(usage.get("completion_tokens"), int)
                else None
            ),
            cost=float(reported_cost) if isinstance(reported_cost, (int, float)) else None,
            metadata={
                "response_id": payload.get("id"),
                "router_model": self.router_model,
                "usage": dict(usage),
            },
        )


class OpenAIExecutionBenchmarkRouter:
    """Execute a selector's chosen candidate through one common model gateway.

    This keeps live comparisons fair: local and hosted selection-only routers use
    the same inference endpoint, request shape, usage accounting, and grader.
    """

    is_live = True
    model_calls_per_case = 1
    timing_scope = "end_to_end"

    def __init__(
        self,
        selector: Any,
        *,
        name: str,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float = 120,
        max_output_tokens: int | None = None,
        reasoning_style: str = "openrouter",
        headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        if reasoning_style not in {"openrouter", "reasoning_effort", "none"}:
            raise BenchmarkError("reasoning_style must be openrouter, reasoning_effort, or none")
        if int(getattr(selector, "model_calls_per_case", 0)):
            raise BenchmarkError("model executor requires a selection-only nested router")
        self.selector = selector
        self.router_calls_per_case = int(getattr(selector, "router_calls_per_case", 0))
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        if not self.base_url or not self.api_key_env:
            raise BenchmarkError("model executor requires a URL and key environment name")
        if timeout_seconds <= 0:
            raise BenchmarkError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.reasoning_style = reasoning_style
        self.headers = dict(headers or {})
        self.extra_body = dict(extra_body or {})

    def prepare(self, dataset: BenchmarkDataset) -> None:
        if not os.environ.get(self.api_key_env):
            raise BenchmarkError(f"{self.name} requires environment variable {self.api_key_env}")
        self.selector.prepare(dataset)

    def _body(self, case: BenchmarkCase, candidate: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": candidate.model,
            "messages": _message_list(case.request),
            "temperature": 0,
            **self.extra_body,
        }
        output_tokens = self.max_output_tokens or case.expected_output_tokens
        if output_tokens is not None:
            body["max_tokens"] = output_tokens
        effort = candidate.reasoning_level
        if effort not in {"none", "default"}:
            if self.reasoning_style == "openrouter":
                body["reasoning"] = {"effort": effort}
            elif self.reasoning_style == "reasoning_effort":
                body["reasoning_effort"] = effort
        overrides = candidate.metadata.get("request_overrides", {})
        if not isinstance(overrides, Mapping):
            raise BenchmarkError(
                f"candidate {candidate.id} metadata.request_overrides must be an object"
            )
        protected = {"model", "messages"} & set(overrides)
        if protected:
            raise BenchmarkError(
                f"candidate {candidate.id} cannot override: {', '.join(sorted(protected))}"
            )
        body.update(overrides)
        return body

    def select(self, case: BenchmarkCase) -> RouterSelection:
        routing = self.selector.select(case)
        candidate = case.candidate_for(routing.model_id) if routing.model_id is not None else None
        if candidate is None:
            return routing
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(self._body(case, candidate)).encode("utf-8"),
            headers={
                **self.headers,
                "Authorization": f"Bearer {os.environ[self.api_key_env]}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise BenchmarkError(f"{self.name} returned HTTP {exc.code}: {detail}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkError(f"{self.name} execution failed: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise BenchmarkError(f"{self.name} returned a non-object response")
        usage = payload.get("usage", {})
        if not isinstance(usage, Mapping):
            usage = {}
        reported_cost = usage.get("cost")
        return RouterSelection(
            model_id=candidate.id,
            output=_response_text(payload),
            prompt_tokens=(
                usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None
            ),
            completion_tokens=(
                usage.get("completion_tokens")
                if isinstance(usage.get("completion_tokens"), int)
                else None
            ),
            cost=float(reported_cost) if isinstance(reported_cost, (int, float)) else None,
            metadata={
                **dict(routing.metadata),
                "selector_model_id": routing.model_id,
                "response_id": payload.get("id"),
                "executed_model": payload.get("model"),
                "usage": dict(usage),
            },
        )


class NotDiamondBenchmarkRouter:
    """Use Not Diamond for selection against a recorded outcome matrix."""

    is_live = True
    model_calls_per_case = 0
    router_calls_per_case = 1
    timing_scope = "routing"

    def __init__(
        self,
        *,
        name: str = "not-diamond",
        api_key_env: str = "NOTDIAMOND_API_KEY",
        tradeoff: str | None = None,
    ) -> None:
        self.name = name
        self.api_key_env = api_key_env
        self.tradeoff = tradeoff
        self._client: Any = None

    def prepare(self, dataset: BenchmarkDataset) -> None:
        del dataset
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise BenchmarkError(f"{self.name} requires environment variable {self.api_key_env}")
        try:
            from notdiamond import NotDiamond
        except ImportError as exc:
            raise BenchmarkError("Not Diamond adapter requires 'pip install notdiamond'") from exc
        self._client = NotDiamond(api_key=api_key)

    def select(self, case: BenchmarkCase) -> RouterSelection:
        providers = [
            {"provider": candidate.provider, "model": candidate.model}
            for candidate in case.eligible_candidates
        ]
        kwargs: dict[str, Any] = {
            "messages": _message_list(case.request),
            "llm_providers": providers,
        }
        if self.tradeoff is not None:
            kwargs["tradeoff"] = self.tradeoff
        result = self._client.model_router.select_model(**kwargs)
        selected = getattr(result, "provider", None)
        model = getattr(selected, "model", None)
        provider = getattr(selected, "provider", None) or getattr(selected, "name", None)
        if not isinstance(model, str) or not model:
            raise BenchmarkError("Not Diamond response did not identify the selected model")
        candidate = case.candidate_for(model)
        if candidate is None and isinstance(provider, str):
            candidate = case.candidate_for(f"{provider}/{model}")
        return RouterSelection(
            model_id=candidate.id if candidate is not None else model,
            metadata={"session_id": getattr(result, "session_id", None)},
        )


class RouteLLMBenchmarkRouter:
    """Use RouteLLM's public two-model Controller.route selection API."""

    is_live = True
    model_calls_per_case = 0
    router_calls_per_case = 0
    timing_scope = "routing"

    def __init__(
        self,
        *,
        strong_model_id: str,
        weak_model_id: str,
        router: str = "mf",
        threshold: float = 0.5,
        name: str = "RouteLLM",
        config: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self.strong_model_id = strong_model_id
        self.weak_model_id = weak_model_id
        self.router_name = router
        self.threshold = threshold
        self.config = None if config is None else dict(config)
        self._controller: Any = None

    def prepare(self, dataset: BenchmarkDataset) -> None:
        for case in dataset.cases:
            candidate_ids = {candidate.id for candidate in case.eligible_candidates}
            missing = {self.strong_model_id, self.weak_model_id} - candidate_ids
            if missing:
                raise BenchmarkError(
                    f"RouteLLM pair is missing from case {case.id}: {', '.join(sorted(missing))}"
                )
        try:
            from routellm.controller import Controller
        except ImportError as exc:
            raise BenchmarkError(
                "RouteLLM adapter requires 'pip install openroutiq[benchmark-routellm]'"
            ) from exc
        self._controller = Controller(
            routers=[self.router_name],
            strong_model=self.strong_model_id,
            weak_model=self.weak_model_id,
            config=self.config,
        )

    def select(self, case: BenchmarkCase) -> RouterSelection:
        messages = _message_list(case.request)
        last = messages[-1]
        content = last.get("content") if isinstance(last, Mapping) else None
        if not isinstance(content, str):
            raise BenchmarkError("RouteLLM adapter requires text in the final message")
        selected = self._controller.route(content, self.router_name, self.threshold)
        return RouterSelection(
            model_id=selected,
            metadata={"threshold": self.threshold, "router": self.router_name},
        )


class SelectionFileBenchmarkRouter:
    """Replay model selections exported by any router or framework."""

    is_live = False
    model_calls_per_case = 0
    timing_scope = "replay_lookup"

    def __init__(self, path: str | Path, *, name: str) -> None:
        self.path = Path(path).resolve()
        self.name = name
        self._selections: dict[str, Any] = {}

    def prepare(self, dataset: BenchmarkDataset) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise BenchmarkError(f"cannot read selection file {self.path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"invalid selection file {self.path}: {exc}") from exc
        selections = raw.get("selections", raw) if isinstance(raw, Mapping) else None
        if not isinstance(selections, Mapping):
            raise BenchmarkError("selection file must be an object or contain selections")
        self._selections = dict(selections)
        unknown = set(self._selections) - {case.id for case in dataset.cases}
        if unknown:
            raise BenchmarkError(
                f"selection file contains unknown case ids: {', '.join(sorted(unknown)[:5])}"
            )

    def select(self, case: BenchmarkCase) -> RouterSelection:
        value = self._selections.get(case.id)
        if isinstance(value, str):
            return RouterSelection(model_id=value)
        if isinstance(value, Mapping):
            return RouterSelection(
                model_id=value.get("model_id"),
                output=value.get("output"),
                prompt_tokens=value.get("prompt_tokens"),
                completion_tokens=value.get("completion_tokens"),
                cost=value.get("cost"),
                metadata=value.get("metadata", {}),
            )
        return RouterSelection(model_id=None, metadata={"missing_selection": True})


class CommandBenchmarkRouter:
    """Bridge any framework through one JSON object on stdin/stdout per case.

    The command is executed without a shell. It must return at least
    ``{"model_id": "candidate-id"}`` and may also return output/token/cost fields.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        name: str,
        timeout_seconds: float = 120,
        live: bool = False,
        model_calls_per_case: int | None = None,
        router_calls_per_case: int | None = None,
        cwd: str | Path | None = None,
        pass_env: Sequence[str] = (),
    ) -> None:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise BenchmarkError("command adapter requires a non-empty argument list")
        if timeout_seconds <= 0:
            raise BenchmarkError("timeout_seconds must be positive")
        self.command = tuple(command)
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.is_live = live
        for field_name, value in (
            ("model_calls_per_case", model_calls_per_case),
            ("router_calls_per_case", router_calls_per_case),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise BenchmarkError(f"command {field_name} must be a non-negative integer")
        if live and model_calls_per_case is None and router_calls_per_case is None:
            raise BenchmarkError(
                "live command router must declare model_calls_per_case and/or router_calls_per_case"
            )
        self.model_calls_per_case = model_calls_per_case or 0
        self.router_calls_per_case = router_calls_per_case or 0
        if not live and (self.model_calls_per_case or self.router_calls_per_case):
            raise BenchmarkError("offline command router cannot declare live calls")
        self.timing_scope = "end_to_end" if live else "routing"
        self.cwd = None if cwd is None else str(cwd)
        if (
            not isinstance(pass_env, Sequence)
            or isinstance(pass_env, (str, bytes, bytearray))
            or any(not isinstance(name, str) or not name for name in pass_env)
        ):
            raise BenchmarkError("command pass_env must contain non-empty environment names")
        system_names = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TMP", "TEMP"}
        names = system_names | set(pass_env)
        self.environment = {name: os.environ[name] for name in names if name in os.environ}

    def prepare(self, dataset: BenchmarkDataset) -> None:
        del dataset

    def select(self, case: BenchmarkCase) -> RouterSelection:
        payload = {
            "id": case.id,
            "request": case.request,
            "task": case.task,
            "constraints": dict(case.constraints),
            "input_tokens": case.input_tokens,
            "expected_output_tokens": case.expected_output_tokens,
            "candidates": [candidate.__dict__ for candidate in case.candidates],
            "metadata": dict(case.metadata),
        }
        completed = subprocess.run(
            self.command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            cwd=self.cwd,
            env=self.environment,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            raise BenchmarkError(
                f"{self.name} command exited {completed.returncode}: {completed.stderr[-2000:]}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"{self.name} command returned invalid JSON") from exc
        if not isinstance(result, Mapping):
            raise BenchmarkError(f"{self.name} command response must be an object")
        return RouterSelection(
            model_id=result.get("model_id"),
            output=result.get("output"),
            prompt_tokens=result.get("prompt_tokens"),
            completion_tokens=result.get("completion_tokens"),
            cost=result.get("cost"),
            metadata=result.get("metadata", {}),
        )
