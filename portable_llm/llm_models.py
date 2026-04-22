"""Data classes for LLM responses and token usage."""

from dataclasses import dataclass, field


@dataclass
class LLMUsage:
    """Token usage from a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.thinking_tokens


@dataclass
class LLMResponse:
    """Response from an LLM call carrying text and token usage."""

    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    tool_calls: list = field(default_factory=list)
    raw_message: object = None

    def __str__(self):
        return self.text