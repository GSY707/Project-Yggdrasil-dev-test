"""OpenRouter provider implementation (OpenAI-compatible SDK)."""

import json
from collections.abc import Generator

from openai import OpenAI

from ..llm_models import LLMResponse, LLMUsage
from .base import BaseProvider


class OpenRouterProvider(BaseProvider):
    """Calls the OpenRouter API using the OpenAI SDK."""

    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        if not api_key:
            raise ValueError("OpenRouter key not found or client init failed")
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
        )
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

    def call_stream(self, messages: list[dict], temperature: float, tools=None, max_tokens: int | None = None) -> Generator[dict, None, None]:
        """
        流式调用 OpenRouter API。yield SSE 事件字典：
        - {"type": "content_delta", "content": "..."}
        - {"type": "tool_call_delta", "index": i, "id": "...", "name": "...", "arguments_delta": "..."}
        - {"type": "done", "usage": LLMUsage, "response": LLMResponse}
        """
        kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools

        stream = self._client.chat.completions.create(**kwargs)

        collected_content = ""
        collected_tool_calls: dict[int, dict] = {}  # index → {id, name, arguments}
        usage = LLMUsage()

        for chunk in stream:
            # Usage info (final chunk)
            if chunk.usage:
                usage.prompt_tokens = chunk.usage.prompt_tokens or 0
                usage.completion_tokens = chunk.usage.completion_tokens or 0

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Content delta
            if delta.content:
                collected_content += delta.content
                yield {"type": "content_delta", "content": delta.content}

            # Tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}

                    entry = collected_tool_calls[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments

                    yield {
                        "type": "tool_call_delta",
                        "index": idx,
                        "id": tc_delta.id or "",
                        "name": (tc_delta.function.name if tc_delta.function else "") or "",
                        "arguments_delta": (tc_delta.function.arguments if tc_delta.function else "") or "",
                    }

        # Build final tool_calls list
        tool_calls = []
        for idx in sorted(collected_tool_calls):
            entry = collected_tool_calls[idx]
            tool_calls.append({
                "id": entry["id"],
                "function": {"name": entry["name"], "arguments": entry["arguments"]},
            })

        final_response = LLMResponse(
            text=collected_content,
            usage=usage,
            tool_calls=tool_calls,
            raw_message=None,
        )
        yield {"type": "done", "usage": usage, "response": final_response}
