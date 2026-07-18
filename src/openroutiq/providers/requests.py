from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from openroutiq.router.core import (
    API_STYLES,
    OpenRoutiQError,
    REASONING_LEVELS,
    RouteDecision,
    _credential_like_path,
)


_PROTECTED_EXTRA_KEYS = frozenset(
    {
        "api_base",
        "api_key",
        "base_url",
        "input",
        "max_completion_tokens",
        "max_output_tokens",
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "output_config",
        "parallel_tool_calls",
        "provider",
        "reasoning",
        "reasoning_effort",
        "requesty",
        "response_format",
        "stream",
        "text",
        "thinking",
        "tool_choice",
        "tools",
    }
)


DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "requesty": "https://router.requesty.ai/v1",
}


@dataclass(frozen=True)
class ProviderRequest:
    """A selected endpoint plus provider-native arguments.

    The original message and tool objects are intentionally retained rather than
    normalized, so opaque reasoning signatures and tool-call IDs survive.
    """

    api_style: str
    provider: str
    model: str
    base_url: str | None
    kwargs: Mapping[str, Any]
    metadata: Mapping[str, Any]
    invoker: Callable[[Any, Mapping[str, Any]], Any] | None = field(
        default=None, repr=False, compare=False
    )

    def invoke(self, client: Any) -> Any:
        target = self.invoker
        if target is not None:
            result = target(client, dict(self.kwargs))
        else:
            client_target = _client_target(client, self.api_style)
            if inspect.iscoroutinefunction(client_target):
                raise OpenRoutiQError("async client requires 'await plan.ainvoke(client)'")
            result = client_target(**dict(self.kwargs))
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise OpenRoutiQError("async client requires 'await plan.ainvoke(client)'")
        return result

    async def ainvoke(self, client: Any) -> Any:
        if self.invoker is not None:
            if inspect.iscoroutinefunction(self.invoker):
                result = await self.invoker(client, dict(self.kwargs))
            else:
                result = await asyncio.to_thread(self.invoker, client, dict(self.kwargs))
        else:
            target = _client_target(client, self.api_style)
            if inspect.iscoroutinefunction(target):
                result = await target(**dict(self.kwargs))
            else:
                result = await asyncio.to_thread(target, **dict(self.kwargs))
        return await result if inspect.isawaitable(result) else result

    def summary(self) -> dict[str, Any]:
        return {
            "api_style": self.api_style,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "parameters": sorted(key for key in self.kwargs if key not in {"input", "messages"}),
            "metadata": dict(self.metadata),
        }


def prepare_request(
    decision: RouteDecision,
    request: str | Sequence[Any],
    *,
    tools: Sequence[Any] | None = None,
    tool_choice: Any = None,
    parallel_tool_calls: bool | None = None,
    response_format: Mapping[str, Any] | None = None,
    stream: bool = False,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    provider_options: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    adapter: Callable[..., ProviderRequest] | None = None,
) -> ProviderRequest:
    selected = decision.selected
    style = selected.api_style
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise OpenRoutiQError("parallel_tool_calls must be a boolean or None")
    if not isinstance(stream, bool):
        raise OpenRoutiQError("stream must be a boolean")
    if response_format is not None and not isinstance(response_format, Mapping):
        raise OpenRoutiQError("response_format must be an object or None")
    if tools is not None and (
        not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray))
    ):
        raise OpenRoutiQError("tools must be a sequence")
    if not tools and tool_choice not in (None, "none"):
        raise OpenRoutiQError("tool_choice requires tools")
    if extra is not None and not isinstance(extra, Mapping):
        raise OpenRoutiQError("extra must be an object or None")

    output_tokens = (
        decision.expected_output_tokens if max_output_tokens is None else max_output_tokens
    )
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 1:
        raise OpenRoutiQError("max_output_tokens must be an integer >= 1")
    level = _reasoning(reasoning_effort, selected.reasoning_level)
    agent_metadata = _metadata(agent_id, run_id, parent_run_id, tags, metadata)
    options = dict(selected.provider_options)
    if provider_options is not None:
        if not isinstance(provider_options, Mapping):
            raise OpenRoutiQError("provider_options must be an object or None")
        secret_path = _credential_like_path(provider_options, path="provider_options")
        if secret_path is not None:
            raise OpenRoutiQError(
                f"provider_options cannot contain credential-like field {secret_path}; "
                "use environment variables or the provider SDK's secret store"
            )
        options.update(provider_options)

    kwargs: dict[str, Any] = {"model": selected.provider_model}
    if style == "openai_responses":
        kwargs["input"] = request
        kwargs["max_output_tokens"] = output_tokens
        _reasoning_kwargs(
            kwargs, style, selected.reasoning_mode, level, selected.reasoning_budget_tokens
        )
        if response_format is not None:
            kwargs["text"] = {"format": _responses_format(response_format)}
        if agent_metadata:
            kwargs["metadata"] = {key: str(value) for key, value in agent_metadata.items()}
        _openai_tools(kwargs, tools, tool_choice, parallel_tool_calls)
    elif style == "anthropic_messages":
        kwargs["messages"] = request
        kwargs["max_tokens"] = output_tokens
        _reasoning_kwargs(
            kwargs, style, selected.reasoning_mode, level, selected.reasoning_budget_tokens
        )
        if response_format is not None:
            kwargs["output_config"] = {"format": _anthropic_format(response_format)}
        _anthropic_tools(kwargs, tools, tool_choice, parallel_tool_calls)
    else:
        kwargs["messages"] = request
        kwargs["max_tokens"] = output_tokens
        _reasoning_kwargs(
            kwargs, style, selected.reasoning_mode, level, selected.reasoning_budget_tokens
        )
        _openai_tools(kwargs, tools, tool_choice, parallel_tool_calls)
        if response_format is not None:
            kwargs["response_format"] = response_format
        if style == "openrouter":
            if tools or response_format is not None or level not in {None, "none"}:
                options.setdefault("require_parameters", True)
            if options:
                kwargs["provider"] = options
        elif style == "requesty":
            requesty = options
            if agent_metadata:
                requesty = {
                    **requesty,
                    "tags": list(tags),
                    "trace_id": run_id,
                    "extra": {**dict(requesty.get("extra", {})), **agent_metadata},
                }
            if requesty:
                kwargs["requesty"] = {
                    key: value for key, value in requesty.items() if value is not None
                }
        elif style == "litellm" and agent_metadata:
            kwargs["metadata"] = agent_metadata

    if stream:
        kwargs["stream"] = True
    if extra:
        protected = _PROTECTED_EXTRA_KEYS & set(extra)
        if protected:
            raise OpenRoutiQError(f"extra cannot override: {', '.join(sorted(protected))}")
        kwargs.update(extra)
    if style not in API_STYLES:
        if adapter is None:
            raise OpenRoutiQError(f"api_style {style!r} requires a custom request adapter")
        if not callable(adapter):
            raise OpenRoutiQError("custom request adapter must be callable")
        custom = adapter(
            decision=decision,
            request=request,
            kwargs=kwargs,
            metadata=agent_metadata,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            response_format=response_format,
            stream=stream,
            provider_options=options,
        )
        if not isinstance(custom, ProviderRequest):
            raise OpenRoutiQError("custom request adapter must return ProviderRequest")
        return custom
    return ProviderRequest(
        api_style=style,
        provider=selected.provider,
        model=selected.provider_model,
        base_url=selected.base_url or DEFAULT_BASE_URLS.get(style),
        kwargs=kwargs,
        metadata=agent_metadata,
    )


def _reasoning(requested: str | None, selected: str | None) -> str | None:
    if requested is None or requested == "auto":
        return selected
    if not isinstance(requested, str):
        raise OpenRoutiQError("reasoning_effort must be a reasoning level or None")
    level = {"min": "minimal"}.get(requested.strip().lower(), requested.strip().lower())
    if level not in REASONING_LEVELS:
        raise OpenRoutiQError(
            f"reasoning_effort must be one of: {', '.join(sorted(REASONING_LEVELS))}"
        )
    if level != (selected or "none"):
        raise OpenRoutiQError("reasoning_effort must match the routed model variant")
    return level


def _reasoning_kwargs(
    kwargs: dict[str, Any],
    style: str,
    mode: str,
    level: str | None,
    budget: int | None,
) -> None:
    if mode == "none" or level is None:
        return
    if style == "anthropic_messages":
        if mode == "adaptive":
            kwargs["thinking"] = {"type": "disabled" if level == "none" else "adaptive"}
        elif mode == "budget":
            if budget is None:
                raise OpenRoutiQError("budget reasoning requires reasoning_budget_tokens")
            if kwargs["max_tokens"] <= budget:
                raise OpenRoutiQError(
                    "max_output_tokens must exceed the Anthropic reasoning budget"
                )
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if level != "none":
            kwargs["effort"] = level
    elif style in {"openai_responses", "openrouter"}:
        kwargs["reasoning"] = {"effort": level}
    else:
        kwargs["reasoning_effort"] = level


def _openai_tools(
    kwargs: dict[str, Any],
    tools: Sequence[Any] | None,
    tool_choice: Any,
    parallel: bool | None,
) -> None:
    if tools and tool_choice != "none":
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if parallel is not None:
            kwargs["parallel_tool_calls"] = parallel


def _anthropic_tools(
    kwargs: dict[str, Any],
    tools: Sequence[Any] | None,
    tool_choice: Any,
    parallel: bool | None,
) -> None:
    if not tools or tool_choice == "none":
        return
    kwargs["tools"] = tools
    choice: Any = tool_choice
    if isinstance(tool_choice, str):
        choice = {"type": "any" if tool_choice in {"required", "any"} else "auto"}
    elif isinstance(tool_choice, Mapping) and tool_choice.get("type") == "function":
        function = tool_choice.get("function", {})
        choice = {"type": "tool", "name": function.get("name")}
    elif tool_choice is None and parallel is False:
        choice = {"type": "auto"}
    if parallel is False:
        choice = {**dict(choice or {"type": "auto"}), "disable_parallel_tool_use": True}
    if choice is not None:
        kwargs["tool_choice"] = choice


def _responses_format(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    nested = result.pop("json_schema", None)
    if result.get("type") == "json_schema" and isinstance(nested, Mapping):
        result.update(nested)
    return result


def _anthropic_format(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    nested = result.pop("json_schema", None)
    if result.get("type") == "json_schema" and isinstance(nested, Mapping):
        schema = nested.get("schema")
        if not isinstance(schema, Mapping):
            raise OpenRoutiQError("json_schema.schema must be an object")
        return {"type": "json_schema", "schema": dict(schema)}
    return result


def _metadata(
    agent_id: str | None,
    run_id: str | None,
    parent_run_id: str | None,
    tags: Sequence[str],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(tags, (str, bytes, bytearray)) or any(not isinstance(tag, str) for tag in tags):
        raise OpenRoutiQError("tags must be a sequence of strings")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise OpenRoutiQError("metadata must be an object or None")
    result = dict(metadata or {})
    for key, value in {
        "agent_id": agent_id,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
    }.items():
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise OpenRoutiQError(f"{key} must be a non-empty string or None")
            result[key] = value
    if tags:
        result["tags"] = list(tags)
    return result


def _client_target(client: Any, style: str):
    if style == "litellm":
        target = client if callable(client) else getattr(client, "completion", None)
    elif style == "openai_responses":
        target = getattr(getattr(client, "responses", None), "create", None)
    elif style == "anthropic_messages":
        target = getattr(getattr(client, "messages", None), "create", None)
    else:
        target = getattr(
            getattr(getattr(client, "chat", None), "completions", None), "create", None
        )
    if not callable(target):
        raise OpenRoutiQError(f"client does not support {style}")
    return target
