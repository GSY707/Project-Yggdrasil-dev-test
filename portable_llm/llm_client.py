"""Portable unified LLM client."""

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .error_handler import ProviderErrorHandler, _MAX_RETRY_WINDOW_SECONDS
from .llm_models import LLMResponse, LLMUsage
from .providers.deepseek import DeepSeekProvider
from .providers.gemini import GeminiProvider
from .providers.longcat import LongCatProvider
from .providers.openrouter import OpenRouterProvider
from .providers.vectorengine import VectorEngineProvider

logger = logging.getLogger(__name__)

__all__ = ["UnifiedLLMClient", "LLMUsage", "LLMResponse", "GenericVannaLLM"]


class UnifiedLLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(self, model_name: str = "gpt-4o-mini", config: dict | None = None, keys_path: str | None = None):
        self.model_name = model_name
        self.config = config or {}
        self.keys_path = keys_path
        self.last_usage = LLMUsage()
        self.conversation_log: list[str] = []
        self._logged_msg_count = 0
        self._load_keys()
        self._init_clients()

    def _resolve_keys_path(self) -> Path | None:
        candidates: list[Path] = []

        if self.keys_path:
            candidates.append(Path(self.keys_path))

        config_keys_path = self.config.get("keys_path")
        if config_keys_path:
            candidates.append(Path(config_keys_path))

        env_keys_path = os.environ.get("PORTABLE_LLM_KEYS_PATH")
        if env_keys_path:
            candidates.append(Path(env_keys_path))

        module_dir = Path(__file__).resolve().parent
        candidates.append(module_dir / "keys.yaml")
        candidates.append(Path.cwd() / "keys.yaml")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _load_keys(self) -> None:
        yaml_keys = {}
        resolved_keys_path = self._resolve_keys_path()
        if resolved_keys_path is not None:
            with resolved_keys_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            yaml_keys = data.get("api_keys", {}) or {}
        else:
            logger.warning("No keys.yaml found for portable_llm. Falling back to environment variables only.")

        self.vectorengine_key = os.environ.get("VECTORENGINE_API_KEY") or yaml_keys.get("vectorengine")
        self.vertex_custom_key = os.environ.get("VERTEX_CUSTOM_API_KEY") or yaml_keys.get("vertex_custom")
        self.deepseek_key = os.environ.get("DEEPSEEK_API_KEY") or yaml_keys.get("deepseek_direct")
        self.longcat_key = os.environ.get("LONGCAT_API_KEY") or yaml_keys.get("longcat")
        self.gemini_api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GEMINI_KEY")
            or yaml_keys.get("gemini_api_key")
            or yaml_keys.get("gemini")
        )
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY") or yaml_keys.get("openrouter")

    def _init_clients(self) -> None:
        model_lower = self.model_name.lower()
        deepseek_api_base = (
            self.config.get("deepseek", {}).get("api_base")
            or self.config.get("model", {}).get("api_base")
            or os.environ.get("DEEPSEEK_API_BASE")
        )

        openrouter_api_base = (
            self.config.get("openrouter", {}).get("api_base")
            or os.environ.get("OPENROUTER_API_BASE")
        )

        if "longcat" in model_lower and self.longcat_key:
            self._provider_impl = LongCatProvider(self.longcat_key, self.model_name)
        elif "gemini" in model_lower:
            self._provider_impl = GeminiProvider(
                self.model_name,
                api_key=self.gemini_api_key,
                config=self.config,
            )
        elif "deepseek" in model_lower and self.deepseek_key:
            self._provider_impl = DeepSeekProvider(self.deepseek_key, self.model_name, deepseek_api_base)
        elif self.openrouter_key:
            self._provider_impl = OpenRouterProvider(self.openrouter_key, self.model_name, openrouter_api_base)
        elif self.vectorengine_key:
            self._provider_impl = VectorEngineProvider(self.vectorengine_key, self.model_name)
        else:
            self._provider_impl = None

    def reset_conversation_log(self) -> None:
        self.conversation_log = []
        self._logged_msg_count = 0

    def log_tool_call(self, tool_name: str, args: dict, result: str) -> None:
        try:
            args_str = json.dumps(args, ensure_ascii=False)
        except Exception:
            args_str = str(args)
        self.conversation_log.append(f"LLM: call_tool({tool_name}, {args_str})")
        self.conversation_log.append(f"tool: {result}")

    def _get_provider(self) -> str:
        model_lower = self.model_name.lower()
        if "longcat" in model_lower:
            return "longcat"
        if "gemini" in model_lower:
            return "gemini"
        if "deepseek" in model_lower:
            return "deepseek"
        if isinstance(self._provider_impl, OpenRouterProvider):
            return "openrouter"
        return "vectorengine"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        if system_prompt:
            self.conversation_log.append(f"sys: {system_prompt}")
        self.conversation_log.append(f"user: {user_prompt}")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        result = self._call_internal(messages, temperature)
        self.conversation_log.append(f"LLM: {result.text}")
        return result

    def call_chat(self, messages, temperature: float = 0.7, tools=None) -> str:
        for message in messages[self._logged_msg_count :]:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                self.conversation_log.append(f"sys: {content}")
            elif role == "user":
                self.conversation_log.append(f"user: {content}")
            elif role == "assistant":
                self.conversation_log.append(f"LLM (history): {content}")

        self._logged_msg_count = len(messages)
        result = self._call_internal(messages, temperature, tools)
        self.conversation_log.append(f"LLM: {result.text}")
        self._logged_msg_count += 1
        return result.text

    def call_with_tools(self, messages, temperature: float = 0.7, tools=None, max_tokens: int | None = None) -> LLMResponse:
        """Call the LLM and return the full LLMResponse including tool_calls."""
        return self._call_internal(messages, temperature, tools, max_tokens=max_tokens)

    def call_with_tools_stream(self, messages, temperature: float = 0.7, tools=None, max_tokens: int | None = None):
        """Stream LLM response. Yields event dicts from the provider's call_stream method.
        Events: content_delta, tool_call_delta, done (with final LLMResponse)."""
        if self._provider_impl is None:
            raise RuntimeError(
                f"No provider configured for model '{self.model_name}'. "
                "Check keys.yaml, environment variables, and model name."
            )
        for event in self._provider_impl.call_stream(messages, temperature, tools, max_tokens):
            if event.get("type") == "done" and "response" in event:
                self.last_usage = event["response"].usage
            yield event

    def _call_internal(self, messages, temperature: float = 0.7, tools=None, max_tokens: int | None = None) -> LLMResponse:
        provider = self._get_provider()
        retries = 0
        started_at = time.monotonic()

        if self._provider_impl is None:
            raise RuntimeError(
                f"No provider configured for model '{self.model_name}'. "
                "Check keys.yaml, environment variables, and model name."
            )

        while True:
            ProviderErrorHandler.wait_until_available(provider, self.model_name)

            try:
                result = self._provider_impl.call(messages, temperature, tools, max_tokens=max_tokens)
                break
            except Exception as exc:
                if not ProviderErrorHandler.is_transient_error(exc):
                    ProviderErrorHandler.handle_api_error(provider, self.model_name, exc)
                    raise

                retries += 1
                retry_delay = ProviderErrorHandler.compute_retry_delay(retries, exc, started_at)
                if retry_delay <= 0:
                    ProviderErrorHandler.handle_api_error(provider, self.model_name, exc)
                    raise RuntimeError(
                        f"Exceeded transient retry window ({_MAX_RETRY_WINDOW_SECONDS:.0f}s) for "
                        f"provider='{provider}', model='{self.model_name}'. Last error: {exc}"
                    ) from exc

                ProviderErrorHandler.handle_api_error(provider, self.model_name, exc, disable_seconds=retry_delay)
                logger.warning(
                    "Transient API error on provider='%s', model='%s' (retry %d, waiting %.1fs): %s",
                    provider,
                    self.model_name,
                    retries,
                    retry_delay,
                    exc,
                )
                time.sleep(retry_delay)

        self.last_usage = result.usage
        return result


if TYPE_CHECKING:
    from vanna.legacy.base.base import VannaBase as _VannaBase
else:
    try:
        from vanna.legacy.base.base import VannaBase as _VannaBase
    except ImportError:
        _VannaBase = object


class GenericVannaLLM(_VannaBase):
    def __init__(self, client: UnifiedLLMClient):
        self.client = client

    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        return self.client.call_chat(prompt)