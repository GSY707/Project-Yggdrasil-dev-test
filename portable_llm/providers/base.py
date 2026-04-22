"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Generator

from ..llm_models import LLMResponse


class BaseProvider(ABC):
    """Interface that every LLM provider must implement."""

    @abstractmethod
    def call(self, messages: list[dict], temperature: float, tools=None, max_tokens: int | None = None) -> LLMResponse:
        """Send a chat completion request and return an LLMResponse."""

    def call_stream(self, messages: list[dict], temperature: float, tools=None, max_tokens: int | None = None) -> Generator[dict, None, None]:
        """Stream a chat completion. Yields event dicts. Default falls back to non-streaming call."""
        result = self.call(messages, temperature, tools, max_tokens)
        if result.text:
            yield {"type": "content_delta", "content": result.text}
        yield {"type": "done", "usage": result.usage, "response": result}