"""Unit tests for exact-length prompt generation."""

import json
from pathlib import Path

import pytest

from tt_blackhole_benchmark.executor.prompts import (
    PromptGenerationError,
    generate_exact_length_prompt,
    write_prompt_file,
)


def fake_encoder(prompt: str) -> tuple[int, ...]:
    repetitions = prompt.count(" benchmark")
    return tuple(range(29 + repetitions))


def test_generate_exact_length_prompt() -> None:
    generated = generate_exact_length_prompt(
        target_tokens=64,
        encoder=fake_encoder,
    )

    assert generated.text == " benchmark" * 35
    assert len(generated.token_ids) == 64


@pytest.mark.parametrize("target_tokens", [64, 128, 256])
def test_generate_multiple_exact_lengths(
    target_tokens: int,
) -> None:
    generated = generate_exact_length_prompt(
        target_tokens=target_tokens,
        encoder=fake_encoder,
    )

    assert len(generated.token_ids) == target_tokens


def test_rejects_unreachable_token_length() -> None:
    def encoder(prompt: str) -> tuple[int, ...]:
        repetitions = prompt.count(" benchmark")
        return tuple(range(10 + repetitions * 2))

    with pytest.raises(
        PromptGenerationError,
        match="Cannot generate exactly",
    ):
        generate_exact_length_prompt(
            target_tokens=11,
            encoder=encoder,
        )


def test_detects_non_linear_tokenization() -> None:
    def encoder(prompt: str) -> tuple[int, ...]:
        repetitions = prompt.count(" benchmark")

        if repetitions <= 2:
            length = 29 + repetitions
        else:
            length = 30 + repetitions

        return tuple(range(length))

    with pytest.raises(
        PromptGenerationError,
        match="Generated prompt has",
    ):
        generate_exact_length_prompt(
            target_tokens=64,
            encoder=encoder,
        )


def test_write_prompt_file(tmp_path: Path) -> None:
    generated = generate_exact_length_prompt(
        target_tokens=64,
        encoder=fake_encoder,
    )
    output_path = write_prompt_file(
        tmp_path / "prompts.json",
        prompt=generated,
        request_count=4,
    )

    content = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(content) == 4
    assert all(entry == {"prompt": generated.text} for entry in content)


def test_rejects_invalid_request_count(
    tmp_path: Path,
) -> None:
    generated = generate_exact_length_prompt(
        target_tokens=64,
        encoder=fake_encoder,
    )

    with pytest.raises(
        ValueError,
        match="request_count must be positive",
    ):
        write_prompt_file(
            tmp_path / "prompts.json",
            prompt=generated,
            request_count=0,
        )
