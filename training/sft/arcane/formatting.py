from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PROBLEM_COLUMNS = ("problem", "question", "prompt", "input", "instruction")
SOLUTION_COLUMNS = ("solution", "answer", "completion", "output", "response")
MESSAGES_COLUMNS = ("messages", "conversation", "conversations")
PROMPT_COLUMNS = ("prompt", "problem", "question", "input", "instruction")
CHOSEN_COLUMNS = (
    "chosen",
    "chosen_response",
    "chosen_completion",
    "chosen_answer",
    "chosen_solution",
    "accepted",
    "winner",
)
REJECTED_COLUMNS = (
    "rejected",
    "rejected_response",
    "rejected_completion",
    "rejected_answer",
    "rejected_solution",
    "loser",
)


def first_present(example: Mapping[str, Any], columns: tuple[str, ...]) -> str:
    value, _ = first_present_value_with_name(example, columns)
    return stringify_field(value)


def first_present_with_name(example: Mapping[str, Any], columns: tuple[str, ...]) -> tuple[str, str]:
    value, column = first_present_value_with_name(example, columns)
    return stringify_field(value), column


def first_present_value(example: Mapping[str, Any], columns: tuple[str, ...]) -> Any:
    value, _ = first_present_value_with_name(example, columns)
    return value


def first_present_value_with_name(example: Mapping[str, Any], columns: tuple[str, ...]) -> tuple[Any, str]:
    for column in columns:
        value = example.get(column)
        if field_has_value(value):
            return value, column
    available = ", ".join(example.keys())
    expected = ", ".join(columns)
    raise KeyError(f"Could not find any of [{expected}] in dataset columns: {available}")


def field_has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple):
        return bool(value)
    return bool(str(value).strip())


def stringify_field(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return messages_to_text(value)
    return str(value)


def messages_to_text(messages: list[Any] | tuple[Any, ...]) -> str:
    assistant_messages = [
        message for message in messages if isinstance(message, Mapping) and message.get("role") == "assistant" and message.get("content")
    ]
    candidates = assistant_messages or list(messages)
    if not candidates:
        return ""

    message = candidates[-1]
    if isinstance(message, Mapping):
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content)
    return str(message)


def append_eos_if_available(text: str, tokenizer: Any) -> str:
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token and not text.endswith(eos_token):
        return f"{text}{eos_token}"
    return text


def apply_chat_template_if_available(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str | None:
    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        return None

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list | tuple):
        raise TypeError("messages must be a list of role/content dictionaries")

    validated: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"messages[{index}] must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            raise TypeError(f"messages[{index}].role must be a string")
        if not isinstance(content, str):
            raise TypeError(f"messages[{index}].content must be a string")
        validated.append({"role": role, "content": content})
    return validated


def format_conversational_sft_example(messages: Any, enable_thinking: bool) -> dict[str, Any]:
    return {
        "messages": validate_messages(messages),
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }


def format_messages_example(messages: Any, tokenizer: Any, enable_thinking: bool) -> dict[str, str]:
    validated = validate_messages(messages)
    text = apply_chat_template_if_available(
        tokenizer,
        validated,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    if text is None:
        text = "\n\n".join(f"{message['role']}: {message['content']}" for message in validated)
    return {"text": append_eos_if_available(text, tokenizer)}


def format_prompt_completion_example(prompt: str, completion: str, tokenizer: Any = None) -> dict[str, str]:
    text = f"{prompt}\n\n{completion}"
    return {"text": append_eos_if_available(text, tokenizer)}


def format_chat_prompt_completion_example(
    prompt_text: str,
    completion: str,
    tokenizer: Any,
    enable_thinking: bool,
) -> dict[str, str]:
    messages = [{"role": "user", "content": prompt_text}]
    prompt = apply_chat_template_if_available(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if prompt is None:
        return format_prompt_completion_example(prompt_text, completion, tokenizer)
    return {"text": append_eos_if_available(f"{prompt}{completion}", tokenizer)}


def format_sft_example(example: Mapping[str, Any], tokenizer: Any, enable_thinking: bool) -> dict[str, str]:
    try:
        messages = first_present_value(example, MESSAGES_COLUMNS)
    except KeyError:
        messages = None
    if messages is not None:
        return format_messages_example(messages, tokenizer, enable_thinking)

    prompt_text = first_present(example, PROBLEM_COLUMNS)
    completion = first_present(example, SOLUTION_COLUMNS)
    return format_chat_prompt_completion_example(prompt_text, completion, tokenizer, enable_thinking)


def format_text_sft_example(example: Mapping[str, Any], tokenizer: Any, enable_thinking: bool) -> dict[str, str]:
    prompt_text = first_present(example, PROBLEM_COLUMNS)
    completion = first_present(example, SOLUTION_COLUMNS)
    return format_chat_prompt_completion_example(prompt_text, completion, tokenizer, enable_thinking)


def format_dpo_example(example: Mapping[str, Any], tokenizer: Any, enable_thinking: bool) -> dict[str, str]:
    prompt = first_present(example, PROMPT_COLUMNS)
    chosen = first_present(example, CHOSEN_COLUMNS)
    rejected = first_present(example, REJECTED_COLUMNS)

    messages = [{"role": "user", "content": prompt}]
    templated_prompt = apply_chat_template_if_available(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if templated_prompt is None:
        templated_prompt = f"{prompt}\n\n"

    return {
        "prompt": templated_prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
