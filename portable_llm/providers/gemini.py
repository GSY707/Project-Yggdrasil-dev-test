"""Gemini provider implementation (Google AI API + Vertex AI REST)."""

import base64
import json
import logging

from google import genai as google_genai
from google.genai import types as google_genai_types

from ..gcloud_auth import GCloudAuth
from ..llm_models import LLMResponse, LLMUsage
from .base import BaseProvider

logger = logging.getLogger(__name__)


def _openai_tools_to_gemini(tools: list[dict]) -> list[dict]:
    """将 OpenAI function calling 格式的 tools 转换为 Gemini REST API 格式。

    OpenAI 格式:
      [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Gemini REST 格式:
      [{"functionDeclarations": [{"name": "...", "description": "...", "parameters": {...}}]}]
    """
    declarations = []
    for tool in tools:
        func = tool.get("function", tool)  # 兼容有无 "type" 包裹
        decl = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
        }
        params = func.get("parameters")
        if params:
            decl["parameters"] = params
        if decl["name"]:
            declarations.append(decl)

    if not declarations:
        return []
    return [{"functionDeclarations": declarations}]


def _parse_gemini_tool_calls(candidates: list[dict]) -> list[dict]:
    """从 Gemini REST 响应的 candidates 中提取 functionCall 并转换为 OpenAI tool_calls 格式。

    Gemini 格式 (在 candidate.content.parts 中):
      {"functionCall": {"name": "tool_name", "args": {"key": "value"}}}

    转换为:
      {"id": "call_xxx", "function": {"name": "tool_name", "arguments": '{"key":"value"}'}}
    """
    tool_calls = []
    call_idx = 0
    for candidate in candidates:
        for part in candidate.get("content", {}).get("parts", []):
            fc = part.get("functionCall")
            if fc:
                name = fc.get("name", "")
                args = fc.get("args", {})
                if name:
                    tool_calls.append({
                        "id": f"call_gemini_{call_idx}",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                        },
                    })
                    call_idx += 1
    return tool_calls


class GeminiProvider(BaseProvider):
    """Calls Gemini via the Google AI API or Vertex AI."""

    def __init__(self, model_name: str, *, api_key: str | None = None, config: dict | None = None):
        self._model_name = model_name
        self._api_key = api_key
        self._config = config or {}

    @staticmethod
    def _build_parts_from_content(content) -> list:
        """将 OpenAI 格式的 content（字符串或数组）转换为 Gemini Parts 列表。"""
        if isinstance(content, str):
            return [google_genai_types.Part.from_text(text=content)]

        # content 是数组（多模态）
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(google_genai_types.Part.from_text(text=item["text"]))
            elif item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("data:"):
                    # 解析 data URL: data:image/png;base64,...
                    header, _, b64_data = url.partition(",")
                    mime_type = header.split(";")[0].replace("data:", "")
                    raw_bytes = base64.b64decode(b64_data)
                    parts.append(google_genai_types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))
                else:
                    # 普通 URL — Gemini 支持 from_uri
                    parts.append(google_genai_types.Part.from_uri(file_uri=url, mime_type="image/jpeg"))
        return parts or [google_genai_types.Part.from_text(text="")]

    @staticmethod
    def _build_vertex_parts(content) -> list[dict]:
        """将 OpenAI 格式的 content 转换为 Vertex AI REST 格式的 parts。"""
        if isinstance(content, str):
            return [{"text": content}]

        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append({"text": item["text"]})
            elif item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("data:"):
                    header, _, b64_data = url.partition(",")
                    mime_type = header.split(";")[0].replace("data:", "")
                    parts.append({"inline_data": {"mime_type": mime_type, "data": b64_data}})
                else:
                    parts.append({"file_data": {"file_uri": url, "mime_type": "image/jpeg"}})
        return parts or [{"text": ""}]

    def call(self, messages: list[dict], temperature: float, tools=None, max_tokens: int | None = None) -> LLMResponse:
        if self._api_key:
            return self._call_gemini_api(messages, temperature, tools, max_tokens)
        return self._call_gemini_vertex(messages, temperature, tools, max_tokens)

    def _call_gemini_api(self, messages: list[dict], temperature: float, tools=None, max_tokens: int | None = None) -> LLMResponse:
        client = google_genai.Client(api_key=self._api_key)

        system_instruction = None
        contents = []
        for message in messages:
            if message["role"] == "system":
                content = message["content"]
                system_instruction = content if isinstance(content, str) else content
            elif message["role"] == "user":
                contents.append(
                    google_genai_types.Content(
                        role="user",
                        parts=self._build_parts_from_content(message["content"]),
                    )
                )
            elif message["role"] == "assistant":
                contents.append(
                    google_genai_types.Content(
                        role="model",
                        parts=self._build_parts_from_content(message["content"]),
                    )
                )

        config_args: dict = {"temperature": temperature}
        if max_tokens:
            config_args["max_output_tokens"] = max_tokens
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if tools:
            if isinstance(tools[0], dict) and "function" in tools[0]:
                pass
            else:
                config_args["tools"] = tools

        response = client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=google_genai_types.GenerateContentConfig(**config_args),
        )

        usage = LLMUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata = response.usage_metadata
            usage.prompt_tokens = getattr(metadata, "prompt_token_count", 0) or 0
            usage.completion_tokens = getattr(metadata, "candidates_token_count", 0) or 0
            usage.thinking_tokens = getattr(metadata, "thoughts_token_count", 0) or 0

        return LLMResponse(text=response.text or "", usage=usage)

    def _call_gemini_vertex(self, messages: list[dict], temperature: float, tools=None, max_tokens: int | None = None) -> LLMResponse:
        project_id = GCloudAuth.resolve_gemini_project_id(self._config)
        access_token = GCloudAuth.get_gemini_access_token(self._config)

        system_instruction = None
        contents = []
        for message in messages:
            role = message["role"]
            if role == "system":
                content = message["content"]
                system_instruction = content if isinstance(content, str) else str(content)
            elif role == "user":
                contents.append({"role": "user", "parts": self._build_vertex_parts(message["content"])})
            elif role == "assistant":
                # 处理带 tool_calls 的 assistant 消息
                assistant_parts = []
                msg_content = message.get("content")
                if msg_content:
                    assistant_parts.extend(self._build_vertex_parts(msg_content))

                tc_list = message.get("tool_calls")
                if tc_list:
                    for tc in tc_list:
                        func = tc.get("function", {})
                        fname = func.get("name", "")
                        fargs_raw = func.get("arguments", "{}")
                        try:
                            fargs = json.loads(fargs_raw) if isinstance(fargs_raw, str) else fargs_raw
                        except (json.JSONDecodeError, TypeError):
                            fargs = {}
                        if fname:
                            assistant_parts.append({"functionCall": {"name": fname, "args": fargs}})

                if assistant_parts:
                    contents.append({"role": "model", "parts": assistant_parts})

            elif role == "tool":
                # tool 结果 → Gemini functionResponse 格式
                tool_call_id = message.get("tool_call_id", "")
                tool_content = message.get("content", "")
                # 尝试从最近的 assistant 消息中匹配 function name
                func_name = tool_call_id  # fallback
                # 尝试 JSON 解析 content
                try:
                    response_data = json.loads(tool_content) if tool_content else {}
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": tool_content}

                contents.append({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": func_name,
                            "response": response_data,
                        }
                    }]
                })

        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        # 工具转换: OpenAI 格式 → Gemini 格式
        if tools:
            gemini_tools = _openai_tools_to_gemini(tools)
            if gemini_tools:
                payload["tools"] = gemini_tools

        response = GCloudAuth.get_gemini_session().post(
            (
                "https://aiplatform.googleapis.com/v1/projects/"
                f"{project_id}/locations/global/publishers/google/models/{self._model_name}:generateContent"
            ),
            headers={"Authorization": f"Bearer {access_token}"},
            json=payload,
            timeout=120,
        )
        if response.status_code >= 400:
            message = response.text
            retry_after = response.headers.get("Retry-After")
            try:
                error_payload = response.json()
                message = error_payload.get("error", {}).get("message", message)
            except ValueError:
                pass
            error = RuntimeError(f"Gemini API error ({response.status_code}): {message}")
            setattr(error, "status_code", response.status_code)
            if retry_after:
                setattr(error, "retry_after", retry_after)
            raise error

        response_json = response.json()
        usage_json = response_json.get("usageMetadata", {})

        usage = LLMUsage(
            prompt_tokens=usage_json.get("promptTokenCount", 0) or 0,
            completion_tokens=usage_json.get("candidatesTokenCount", 0) or 0,
            thinking_tokens=usage_json.get("thoughtsTokenCount", 0) or 0,
        )

        # 解析文本 + tool_calls
        text_parts = []
        candidates = response_json.get("candidates", [])

        for candidate in candidates:
            for part in candidate.get("content", {}).get("parts", []):
                text = part.get("text")
                if text:
                    text_parts.append(text)

        tool_calls = _parse_gemini_tool_calls(candidates)

        return LLMResponse(
            text="".join(text_parts),
            usage=usage,
            tool_calls=tool_calls,
        )