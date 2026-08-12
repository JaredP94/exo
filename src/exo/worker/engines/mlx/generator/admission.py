from exo.shared.constants import (
    EXO_PROMPT_ADMISSION_CEILING,
    PROMPT_ADMISSION_CEILING_DEFAULT,
)


class PromptTooLongError(ValueError):
    """A prompt exceeded the admission ceiling and was refused before allocation.

    The default ceiling of 40,960 is a safety margin above the demonstrated
    successful lengths through 32,828 and below the 65,536-token MLX panic
    boundary. The 33,000-63,999 range remains untested; this is an admission
    guard, not a claim that every prompt below the ceiling is supported. Raising
    rather than truncating preserves the caller's prompt semantics.
    """

    def __init__(self, prompt_tokens: int, limit_tokens: int) -> None:
        self.prompt_tokens: int = prompt_tokens
        self.limit_tokens: int = limit_tokens
        super().__init__(
            f"prompt of {prompt_tokens} tokens exceeds the admission ceiling of "
            f"{limit_tokens} tokens"
        )


def assert_prompt_within_ceiling(prompt_tokens: int, limit_tokens: int) -> None:
    """Refuse an over-ceiling prompt before model or cache work can begin.

    Non-positive limits disable the guard so a missing or deliberately disabled
    configuration does not reject every request. A positive limit raises the
    typed error rather than truncating the caller's prompt.
    """
    if limit_tokens > 0 and prompt_tokens > limit_tokens:
        raise PromptTooLongError(prompt_tokens, limit_tokens)


__all__ = [
    "EXO_PROMPT_ADMISSION_CEILING",
    "PROMPT_ADMISSION_CEILING_DEFAULT",
    "PromptTooLongError",
    "assert_prompt_within_ceiling",
]
