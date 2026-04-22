"""VectorEngine provider implementation (REST API, OpenAI proxy)."""

import requests

from ..llm_models import LLMResponse, LLMUsage
from .base import BaseProvider


class VectorEngineProvider(BaseProvider):
    """Calls the VectorEngine REST API."""

    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("VectorEngine key not found")
        self._api_key = api_key
        self._model_name = model_name

    def call(self, messages: list[dict], temperature: float, tools=None, max_tokens: int | None = None) -> LLMResponse:
        payload = {
            "model": self._model_name,
            "input": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            "https://api.vectorengine.ai/v1/responses",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        response_json = response.json()
        usage_json = response_json.get("usage", {})

        usage = LLMUsage(
            prompt_tokens=usage_json.get("input_tokens", 0),
            completion_tokens=usage_json.get("output_tokens", 0),
        )

        # 解析文本和 tool_calls (标准 OpenAI 格式)
        text = ""
        tool_calls = []

        output_items = response_json.get("output", [])
        for item in output_items:
            item_type = item.get("type", "")

            # 文本消息
            if item_type == "message":
                for content_part in item.get("content", []):
                    if content_part.get("type") == "text":
                        text += content_part.get("text", "")

            # 工具调用 (OpenAI 格式)
            elif item_type == "function_call":
                tc = {
                    "id": item.get("call_id") or item.get("id", ""),
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
                if tc["function"]["name"]:  # 只添加有效的 tool_call
                    tool_calls.append(tc)

        return LLMResponse(
            text=text,
            usage=usage,
            tool_calls=tool_calls,
        )