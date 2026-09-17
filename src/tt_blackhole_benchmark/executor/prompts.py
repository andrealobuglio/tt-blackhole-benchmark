"""Generation of prompts with an exact encoded token length."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PromptGenerationError(ValueError):
    """Raised when an exact token-length prompt cannot be built."""


@dataclass(frozen=True)
class GeneratedPrompt:
    """Prompt text and directly observed encoded token IDs."""

    text: str
    token_ids: tuple[int, ...]


PromptEncoder = Callable[[str], Sequence[int]]


def _extract_input_ids(encoded: Any) -> tuple[int, ...]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "ids"):
        encoded = encoded.ids

    if not isinstance(encoded, Sequence):
        raise PromptGenerationError("Tokenizer returned unsupported encoded data")

    if isinstance(encoded, (str, bytes)):
        raise PromptGenerationError("Tokenizer returned text instead of token IDs")

    token_ids = tuple(encoded)

    if not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in token_ids
    ):
        raise PromptGenerationError("Tokenizer returned non-integer token IDs")

    return token_ids


def create_hf_chat_encoder(
    model: str,
) -> PromptEncoder:
    """Create an encoder matching TT-Metal instruct encoding."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)

    def encode(prompt: str) -> tuple[int, ...]:
        chat = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
        )
        return _extract_input_ids(encoded)

    return encode


def generate_exact_length_prompt(
    *,
    target_tokens: int,
    encoder: PromptEncoder,
    fragment: str = " benchmark",
) -> GeneratedPrompt:
    """Generate and verify a repeated-fragment prompt."""

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    if not fragment:
        raise ValueError("fragment must not be empty")

    first_ids = tuple(encoder(fragment))
    second_ids = tuple(encoder(fragment * 2))
    token_increment = len(second_ids) - len(first_ids)

    if token_increment <= 0:
        raise PromptGenerationError("Prompt fragment does not increase token length")

    difference = target_tokens - len(first_ids)

    if difference < 0 or difference % token_increment != 0:
        raise PromptGenerationError(
            f"Cannot generate exactly {target_tokens} tokens with fragment {fragment!r}"
        )

    repetitions = 1 + difference // token_increment
    prompt_text = fragment * repetitions
    token_ids = tuple(encoder(prompt_text))

    if len(token_ids) != target_tokens:
        raise PromptGenerationError(
            f"Generated prompt has {len(token_ids)} tokens, expected {target_tokens}"
        )

    return GeneratedPrompt(
        text=prompt_text,
        token_ids=token_ids,
    )


def write_prompt_file(
    path: str | Path,
    *,
    prompt: GeneratedPrompt,
    request_count: int,
) -> Path:
    """Write the prompt format consumed by the TT-Metal demo."""

    if request_count <= 0:
        raise ValueError("request_count must be positive")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = [{"prompt": prompt.text} for _ in range(request_count)]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    return output_path
