"""Error classification, retry logic, and provider availability tracking."""

import logging
import random
import time
from typing import ClassVar

logger = logging.getLogger(__name__)

_PERMANENT_ERROR_PATTERNS = [
    "model not found",
    "was not found or your project does not have access to it",
    "publisher model",
    "no such model",
    "model_not_found",
    "does not exist",
    "quota exceeded",
    "insufficient_quota",
    "insufficient funds",
    "payment required",
    "payment_required",
    "billing hard limit",
    "account has been deactivated",
    "organization has been disabled",
    "没有此模型",
    "配额达到上限",
    "余额不足",
]

_RESOURCE_EXHAUSTED_PATTERNS = [
    "resource exhausted",
    "rate limit",
    "too many requests",
    "429",
    "requests per",
    "资源耗尽",
]

_DEFAULT_TRANSIENT_COOLDOWN_SECONDS = 5.0
_INITIAL_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 16.0
_MAX_RETRY_WINDOW_SECONDS = 120.0

_TRANSIENT_ERROR_PATTERNS = [
    "temporarily disabled",
    "timeout",
    "timed out",
    "deadline exceeded",
    "service unavailable",
    "internal server error",
    "connection reset",
    "connection aborted",
    "connection refused",
    "read timed out",
    "unexpected_eof_while_reading",
    "eof occurred in violation of protocol",
    "ssl",
    "503",
    "500",
    "超时",
    "服务不可用",
]


class ProviderErrorHandler:
    """Tracks provider/model availability and classifies API errors."""

    _permanently_disabled_models: ClassVar[set[str]] = set()
    _temporarily_disabled_providers: ClassVar[dict[str, float]] = {}
    _temporarily_disabled_provider_models: ClassVar[dict[tuple[str, str], float]] = {}
    _temporarily_disabled_models: ClassVar[dict[str, float]] = {}

    @classmethod
    def check_provider_available(cls, provider: str, model_name: str) -> None:
        if model_name in cls._permanently_disabled_models:
            raise RuntimeError(
                f"Model '{model_name}' is permanently disabled due to a previous fatal error "
                f"(model not found / quota exceeded)."
            )

        if provider in cls._temporarily_disabled_providers:
            expiry = cls._temporarily_disabled_providers[provider]
            remaining = expiry - time.monotonic()
            if remaining > 0:
                raise RuntimeError(
                    f"Provider '{provider}' is temporarily disabled for "
                    f"{int(remaining) + 1}s due to a recent API error."
                )
            del cls._temporarily_disabled_providers[provider]

        provider_model_key = (provider, model_name)
        if provider_model_key in cls._temporarily_disabled_provider_models:
            expiry = cls._temporarily_disabled_provider_models[provider_model_key]
            remaining = expiry - time.monotonic()
            if remaining > 0:
                raise RuntimeError(
                    f"Provider/model '{provider}/{model_name}' is temporarily disabled for "
                    f"{int(remaining) + 1}s due to a recent API error."
                )
            del cls._temporarily_disabled_provider_models[provider_model_key]

        if model_name in cls._temporarily_disabled_models:
            expiry = cls._temporarily_disabled_models[model_name]
            remaining = expiry - time.monotonic()
            if remaining > 0:
                raise RuntimeError(
                    f"Model '{model_name}' is temporarily disabled for "
                    f"{int(remaining) + 1}s due to a recent API error."
                )
            del cls._temporarily_disabled_models[model_name]

    @classmethod
    def wait_until_available(cls, provider: str, model_name: str) -> float:
        waited = 0.0

        while True:
            if model_name in cls._permanently_disabled_models:
                raise RuntimeError(
                    f"Model '{model_name}' is permanently disabled due to a previous fatal error "
                    f"(model not found / quota exceeded)."
                )

            now = time.monotonic()
            provider_remaining = 0.0
            provider_model_remaining = 0.0
            model_remaining = 0.0

            if provider in cls._temporarily_disabled_providers:
                provider_remaining = cls._temporarily_disabled_providers[provider] - now
                if provider_remaining <= 0:
                    del cls._temporarily_disabled_providers[provider]
                    provider_remaining = 0.0

            provider_model_key = (provider, model_name)
            if provider_model_key in cls._temporarily_disabled_provider_models:
                provider_model_remaining = cls._temporarily_disabled_provider_models[provider_model_key] - now
                if provider_model_remaining <= 0:
                    del cls._temporarily_disabled_provider_models[provider_model_key]
                    provider_model_remaining = 0.0

            if model_name in cls._temporarily_disabled_models:
                model_remaining = cls._temporarily_disabled_models[model_name] - now
                if model_remaining <= 0:
                    del cls._temporarily_disabled_models[model_name]
                    model_remaining = 0.0

            remaining = max(provider_remaining, provider_model_remaining, model_remaining)
            if remaining <= 0:
                return waited

            logger.warning(
                "Cooldown active for provider='%s', model='%s'. Waiting %.1fs before retry.",
                provider,
                model_name,
                remaining,
            )
            time.sleep(remaining)
            waited += remaining

    @staticmethod
    def is_transient_error(error: Exception) -> bool:
        error_str = str(error).lower()

        if any(pattern in error_str for pattern in _PERMANENT_ERROR_PATTERNS):
            return False

        if any(pattern in error_str for pattern in _RESOURCE_EXHAUSTED_PATTERNS):
            return True

        return any(pattern in error_str for pattern in _TRANSIENT_ERROR_PATTERNS)

    @classmethod
    def handle_api_error(
        cls,
        provider: str,
        model_name: str,
        error: Exception,
        disable_seconds: float | None = None,
    ) -> None:
        error_str = str(error).lower()
        cooldown = disable_seconds if disable_seconds is not None else _DEFAULT_TRANSIENT_COOLDOWN_SECONDS

        if any(pattern in error_str for pattern in _PERMANENT_ERROR_PATTERNS):
            logger.error("Permanently disabling model '%s' due to fatal error: %s", model_name, error)
            cls._permanently_disabled_models.add(model_name)
        elif any(pattern in error_str for pattern in _RESOURCE_EXHAUSTED_PATTERNS):
            logger.warning(
                "Temporarily disabling provider/model '%s/%s' for %.1fs (resource exhausted): %s",
                provider,
                model_name,
                cooldown,
                error,
            )
            cls._temporarily_disabled_provider_models[(provider, model_name)] = time.monotonic() + cooldown
        else:
            logger.warning(
                "Temporarily disabling model '%s' for %.1fs (API error): %s",
                model_name,
                cooldown,
                error,
            )
            cls._temporarily_disabled_models[model_name] = time.monotonic() + cooldown

    @staticmethod
    def extract_retry_after_seconds(error: Exception) -> float | None:
        retry_after = getattr(error, "retry_after", None)
        if retry_after is None:
            return None
        try:
            retry_after_value = float(retry_after)
        except (TypeError, ValueError):
            return None
        return retry_after_value if retry_after_value > 0 else None

    @classmethod
    def compute_retry_delay(cls, retry_index: int, error: Exception, started_at: float) -> float:
        elapsed = time.monotonic() - started_at
        remaining_budget = _MAX_RETRY_WINDOW_SECONDS - elapsed
        if remaining_budget <= 0:
            return 0.0

        retry_after = cls.extract_retry_after_seconds(error)
        if retry_after is not None:
            return min(retry_after, remaining_budget)

        base_delay = min(_INITIAL_BACKOFF_SECONDS * (2 ** max(retry_index - 1, 0)), _MAX_BACKOFF_SECONDS)
        jitter = random.uniform(0.0, min(1.0, base_delay / 2))
        return min(base_delay + jitter, remaining_budget)