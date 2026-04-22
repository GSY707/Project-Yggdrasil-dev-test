"""LongCat provider implementation (OpenAI-compatible SDK)."""

from openai import OpenAI

from ..llm_models import LLMResponse, LLMUsage
from .base import BaseProvider


class LongCatProvider(BaseProvider):
    """Calls the LongCat API using the OpenAI SDK."""

    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("LongCat key not found or client init failed")
        self._client = OpenAI(api_key=api_key, base_url="https://api.longcat.chat/openai/v1")
        self._model_name = model_name

    def call(self, messages: list[dict], temperature: float, tools=None, max_tokens: int | None = None) -> LLMResponse:
        kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 16_000,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if "thinking" in self._model_name.lower():
            kwargs["extra_body"] = {"reasoning": {"effort": "low"}}

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