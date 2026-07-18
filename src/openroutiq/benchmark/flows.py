from __future__ import annotations

import base64
import json
import math
import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openroutiq.benchmark.core import BenchmarkError


@dataclass(frozen=True)
class FlowCase:
    case_id: str
    dataset_id: str
    scenario: str
    framework: str
    messages: tuple[dict[str, Any], ...]
    expected: Mapping[str, Any]
    tools: tuple[dict[str, Any], ...] = ()
    documents: tuple[str, ...] = ()
    source: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def routing_request(self) -> list[dict[str, Any]]:
        request = [dict(message) for message in self.messages]
        if self.documents:
            request.append(
                {
                    "role": "user",
                    "content": "Available local documents:\n" + "\n\n".join(self.documents),
                }
            )
        return request


@dataclass(frozen=True)
class FlowResponse:
    text: str | None
    tool_calls: tuple[dict[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{name} must be non-empty text")
    return value


def _record(snapshot: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    case_id = _text(snapshot.get("benchmark_id"), "snapshot benchmark_id")
    dataset_id = _text(snapshot.get("dataset_id"), "snapshot dataset_id")
    record = snapshot.get("record")
    if not isinstance(record, Mapping):
        raise BenchmarkError(f"snapshot {case_id} record must be an object")
    return case_id, dataset_id, dict(record)


_GORILLA_TO_JSON_SCHEMA = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}


def _openai_schema(value: Any) -> Any:
    if isinstance(value, Mapping):
        converted = {str(key): _openai_schema(item) for key, item in value.items()}
        raw_type = converted.get("type")
        if isinstance(raw_type, str):
            converted["type"] = _GORILLA_TO_JSON_SCHEMA.get(raw_type, "string")
            if raw_type == "float":
                converted.setdefault("format", "float")
        return converted
    if isinstance(value, list):
        return [_openai_schema(item) for item in value]
    return value


def _openai_tools(functions: Any, case_id: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(functions, list) or any(not isinstance(item, Mapping) for item in functions):
        raise BenchmarkError(f"BFCL case {case_id} function must be an object list")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for function in functions:
        item = dict(function)
        raw_name = _text(item.get("name"), f"BFCL case {case_id} function name")
        normalized_name = raw_name.replace(".", "_")
        if normalized_name in names:
            raise BenchmarkError(
                f"BFCL case {case_id} function names collide after OpenAI normalization"
            )
        names.add(normalized_name)
        item["name"] = normalized_name
        parameters = item.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise BenchmarkError(f"BFCL case {case_id} has invalid function parameters")
        schema = _openai_schema(parameters)
        if not isinstance(schema, dict):
            raise BenchmarkError(f"BFCL case {case_id} has invalid function schema")
        schema["type"] = "object"
        item["parameters"] = schema
        tools.append({"type": "function", "function": item})
    return tuple(tools)


def _choice_prompt(question: str, choices: Sequence[Any]) -> str:
    labels = string.ascii_uppercase
    rendered = "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(choices))
    return f"{question}\n\n{rendered}\n\nReturn only the letter of the correct answer."


def _image_block(media_root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    relative = _text(value.get("media_path"), "media_path")
    mime_type = _text(value.get("mime_type"), "mime_type")
    target = (media_root / relative).resolve()
    if media_root.resolve() not in target.parents or not target.is_file():
        raise BenchmarkError(f"missing or unsafe benchmark media path {relative}")
    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def build_flow_case(
    track: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    media_root: str | Path | None = None,
) -> FlowCase:
    case_id, dataset_id, record = _record(snapshot)
    scenario = _text(track.get("scenario"), "track scenario")
    framework = _text(track.get("framework"), "track framework")
    messages: list[dict[str, Any]] = []
    expected: dict[str, Any] = {}
    tools: tuple[dict[str, Any], ...] = ()
    documents: tuple[str, ...] = ()

    if dataset_id == "livecodebench/code_generation_lite":
        problem = _text(record.get("question_content"), f"LiveCodeBench {case_id} question")
        starter = record.get("starter_code") or ""
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the programming problem. Return only a complete Python solution, "
                    "without Markdown fences or explanatory prose."
                ),
            },
            {
                "role": "user",
                "content": f"{problem}\n\nStarter code:\n{starter}",
            },
        ]
        expected = {
            "grader": "livecodebench_official",
            "public_test_cases": record.get("public_test_cases"),
            "private_test_cases": record.get("private_test_cases"),
            "metadata": record.get("metadata"),
        }
    elif dataset_id == "allenai/IFBench_test":
        messages = [{"role": "user", "content": _text(record.get("prompt"), case_id)}]
        expected = {
            "grader": "ifbench_official",
            "instruction_id_list": record.get("instruction_id_list"),
            "kwargs": record.get("kwargs"),
            "key": record.get("key"),
        }
    elif dataset_id == "openai/gsm8k":
        question = _text(record.get("question"), case_id)
        messages = [
            {
                "role": "user",
                "content": (
                    f"Solve this problem carefully. End with `#### <number>`.\n\n{question}"
                ),
            }
        ]
        answer = _text(record.get("answer"), f"GSM8K {case_id} answer")
        expected = {"grader": "gsm8k_numeric", "answer": answer.rsplit("####", 1)[-1].strip()}
    elif dataset_id == "TIGER-Lab/MMLU-Pro":
        question = _text(record.get("question"), case_id)
        options = record.get("options")
        if not isinstance(options, list):
            raise BenchmarkError(f"MMLU-Pro {case_id} options must be a list")
        messages = [{"role": "user", "content": _choice_prompt(question, options)}]
        expected = {"grader": "choice_exact", "answer": record.get("answer")}
    elif dataset_id == "THUDM/LongBench-v2":
        choices = [record.get(f"choice_{letter}") for letter in "ABCD"]
        prompt = (
            "Read the complete context and answer the multiple-choice question.\n\n"
            f"Context:\n{record.get('context', '')}\n\n"
            + _choice_prompt(_text(record.get("question"), case_id), choices)
        )
        messages = [{"role": "user", "content": prompt}]
        expected = {"grader": "choice_exact", "answer": record.get("answer")}
    elif dataset_id == "MMMU/MMMU":
        options = record.get("options", [])
        if isinstance(options, str):
            try:
                import ast

                options = ast.literal_eval(options)
            except (SyntaxError, ValueError) as exc:
                raise BenchmarkError(f"MMMU {case_id} options are invalid") from exc
        if not isinstance(options, list):
            raise BenchmarkError(f"MMMU {case_id} options must be a list")
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _choice_prompt(_text(record.get("question"), case_id), options),
            }
        ]
        root = Path(media_root) if media_root is not None else None
        for index in range(1, 8):
            image = record.get(f"image_{index}")
            if image is not None:
                if root is None or not isinstance(image, Mapping):
                    raise BenchmarkError(f"MMMU {case_id} requires frozen media files")
                content.append(_image_block(root, image))
        messages = [{"role": "user", "content": content}]
        expected = {"grader": "choice_exact", "answer": record.get("answer")}
    elif dataset_id == "hotpotqa/hotpot_qa":
        context = record.get("context")
        if not isinstance(context, Mapping):
            raise BenchmarkError(f"HotpotQA {case_id} context must be an object")
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        if not isinstance(titles, list) or not isinstance(sentences, list):
            raise BenchmarkError(f"HotpotQA {case_id} context fields must be lists")
        documents = tuple(
            f"Title: {title}\n{''.join(parts)}"
            for title, parts in zip(titles, sentences, strict=True)
        )
        messages = [
            {
                "role": "user",
                "content": (
                    "Answer the question using the retrieved local evidence. Return only the "
                    f"short answer.\n\nQuestion: {_text(record.get('question'), case_id)}"
                ),
            }
        ]
        expected = {"grader": "hotpotqa", "answer": record.get("answer")}
    elif dataset_id == "github:LLLeoLi/bfcl":
        turns = record.get("question")
        if (
            not isinstance(turns, list)
            or not turns
            or not isinstance(turns[0], list)
            or any(not isinstance(message, Mapping) for message in turns[0])
        ):
            raise BenchmarkError(f"BFCL {case_id} question is invalid")
        messages = [dict(message) for message in turns[0]]
        tools = _openai_tools(record.get("function"), case_id)
        expected = {
            "grader": "bfcl_official_ast",
            "possible_answer": record.get("possible_answer"),
            "category": record.get("bfcl_category"),
        }
    elif dataset_id in {
        "princeton-nlp/SWE-bench_Verified",
        "SWE-bench/SWE-bench_Verified",
    }:
        messages = [
            {
                "role": "user",
                "content": _text(record.get("problem_statement"), f"SWE-bench {case_id} problem"),
            }
        ]
        expected = {
            "grader": "swebench_official",
            "instance_id": record.get("instance_id"),
            "repo": record.get("repo"),
            "base_commit": record.get("base_commit"),
            "fail_to_pass": record.get("FAIL_TO_PASS"),
            "pass_to_pass": record.get("PASS_TO_PASS"),
        }
    else:
        raise BenchmarkError(f"unsupported real-flow dataset {dataset_id}")

    return FlowCase(
        case_id=case_id,
        dataset_id=dataset_id,
        scenario=scenario,
        framework=framework,
        messages=tuple(messages),
        expected=expected,
        tools=tools,
        documents=documents,
        source=record,
    )


def _normalize_answer(value: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punctuation(text: str) -> str:
        return "".join(character for character in text if character not in string.punctuation)

    return white_space_fix(remove_articles(remove_punctuation(value.lower())))


def _hotpot_score(prediction: str, answer: str) -> tuple[float, float]:
    normalized_prediction = _normalize_answer(prediction)
    normalized_answer = _normalize_answer(answer)
    exact = float(normalized_prediction == normalized_answer)
    prediction_tokens = normalized_prediction.split()
    answer_tokens = normalized_answer.split()
    if not prediction_tokens or not answer_tokens:
        return exact, exact
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    shared = sum(common.values())
    if shared == 0:
        return exact, 0.0
    precision = shared / len(prediction_tokens)
    recall = shared / len(answer_tokens)
    return exact, 2 * precision * recall / (precision + recall)


def _choice(text: str | None) -> str | None:
    if not text:
        return None
    matches = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", text.upper())
    return matches[-1] if matches else None


def _number(value: str | None) -> float | None:
    if not value:
        return None
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", value)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def grade_flow_response(case: FlowCase, response: FlowResponse) -> dict[str, Any]:
    grader = case.expected.get("grader")
    if grader == "choice_exact":
        choice_prediction = _choice(response.text)
        answer = str(case.expected.get("answer", "")).upper()
        score = float(choice_prediction == answer)
        return {
            "score": score,
            "exact_match": score,
            "prediction": choice_prediction,
        }
    if grader == "gsm8k_numeric":
        numeric_prediction = _number(response.text)
        numeric_answer = _number(str(case.expected.get("answer", "")))
        score = float(
            numeric_prediction is not None
            and numeric_answer is not None
            and math.isclose(
                numeric_prediction,
                numeric_answer,
                rel_tol=0,
                abs_tol=1e-9,
            )
        )
        return {
            "score": score,
            "exact_match": score,
            "prediction": numeric_prediction,
        }
    if grader == "hotpotqa":
        prediction = response.text or ""
        exact, f1 = _hotpot_score(prediction, str(case.expected.get("answer", "")))
        return {"score": exact, "exact_match": exact, "token_f1": f1}
    return {
        "score": None,
        "pending_official_evaluator": True,
        "official_evaluator": grader,
    }


def flow_response_from_openai(message: Any) -> FlowResponse:
    content = getattr(message, "content", None)
    if isinstance(message, Mapping):
        content = message.get("content")
        raw_calls = message.get("tool_calls", [])
    else:
        raw_calls = getattr(message, "tool_calls", []) or []
    calls: list[dict[str, Any]] = []
    for call in raw_calls:
        if isinstance(call, Mapping):
            function = call.get("function", {})
            call_id = call.get("id")
        else:
            function = getattr(call, "function", None)
            call_id = getattr(call, "id", None)
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments")
        else:
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        calls.append({"id": call_id, "name": name, "arguments": arguments})
    return FlowResponse(
        text=content if isinstance(content, str) else None,
        tool_calls=tuple(calls),
    )
