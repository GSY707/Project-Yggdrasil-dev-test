"""DeepSeek provider implementation (OpenAI-compatible SDK)."""

from openai import OpenAI

from ..llm_models import LLMResponse, LLMUsage
from .base import BaseProvider


class DeepSeekProvider(BaseProvider):
    """Calls the DeepSeek API using the OpenAI SDK."""

    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        if not api_key:
            raise ValueError("DeepSeek key not found or client init failed")
        self._client = OpenAI(api_key=api_key, base_url=base_url or "https://api.deepseek.com")
        self._model_name = model_name

    def call(self, messages: list[dict], temperature: float, tools=None, max_tokens: int | None = None) -> LLMResponse:
        kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)

        usage = LLMUsage()
        if response.usage:
            usage.prompt_tokens = response.usage.prompt_tokens or 0
            usage.completion_tokens = response.usage.completion_tokens or 0

        if not response.choices:
            return LLMResponse(text="", usage=usage, tool_calls=[], raw_message=None)

        msg = response.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            tool_calls = [
                {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]

        return LLMResponse(
            text=msg.content or "",
            usage=usage,
            tool_calls=tool_calls,
            raw_message=msg,
        )