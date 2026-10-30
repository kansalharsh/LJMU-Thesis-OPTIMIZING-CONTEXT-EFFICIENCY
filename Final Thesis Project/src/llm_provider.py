"""Provider-agnostic LLM call wrapper.

Gemini and OpenAI are wired up; Claude 3.5 Sonnet could be added as a
sibling class behind the same `LLMResponse` contract without touching
agent code.

Rate-limit backoff time is tracked separately from `latency_s` so a
free-tier 429 doesn't silently inflate the System Latency metric with
throttling delay that has nothing to do with the pruning gateway.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from . import config


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    rate_limit_wait_s: float = 0.0


def _with_backoff(call, is_daily_quota_error, is_rate_limit_error, max_retries: int = 5):
    """Runs `call()` with exponential backoff on transient rate limits.

    Raises immediately (no retry) on a daily-quota error, since retrying
    a per-day cap only burns more of the same exhausted budget.
    """
    rate_limit_wait = 0.0
    attempt = 0
    while True:
        start = time.perf_counter()
        try:
            result = call()
            return result, time.perf_counter() - start, rate_limit_wait
        except Exception as e:  # noqa: BLE001 - re-raised below if not a handled rate limit
            if is_daily_quota_error(e):
                raise RuntimeError(
                    "Daily free-tier request/token quota exhausted for this model. "
                    "Retrying would only waste tomorrow's quota too - stopping. "
                    f"Original error: {e}"
                ) from e
            attempt += 1
            if not is_rate_limit_error(e) or attempt > max_retries:
                raise
            backoff = min(2 ** attempt, 60)
            rate_limit_wait += backoff
            print(
                f"  [rate limit] attempt {attempt}/{max_retries}, backing off {backoff}s",
                file=sys.stderr,
            )
            time.sleep(backoff)


class GeminiProvider:
    def __init__(self, model: str | None = None):
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in the environment."
            )
        from google import genai  # deferred import so config-only usage doesn't need the SDK installed

        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._model = model or config.GEMINI_MODEL

    def generate(self, prompt: str, max_retries: int = 5) -> LLMResponse:
        from google.genai import errors as genai_errors

        def call():
            return self._client.models.generate_content(model=self._model, contents=prompt)

        def is_daily_quota(e):
            return isinstance(e, genai_errors.ClientError) and "PerDay" in str(e)

        def is_rate_limit(e):
            return isinstance(e, genai_errors.ClientError) and (
                getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
            )

        resp, latency, wait = _with_backoff(call, is_daily_quota, is_rate_limit, max_retries)
        usage = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_s=latency,
            rate_limit_wait_s=wait,
        )


class OpenAIProvider:
    def __init__(self, model: str | None = None):
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        from openai import OpenAI  # deferred import

        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = model or config.OPENAI_MODEL

    def generate(self, prompt: str, max_retries: int = 5) -> LLMResponse:
        from openai import APIStatusError

        def call():
            kwargs = {}
            if config.TEMPERATURE != "default":
                kwargs["temperature"] = float(config.TEMPERATURE)
            return self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

        def is_daily_quota(e):
            return isinstance(e, APIStatusError) and e.status_code == 429 and (
                "requests per day" in str(e).lower() or "tokens per day" in str(e).lower()
            )

        def is_rate_limit(e):
            return isinstance(e, APIStatusError) and e.status_code == 429

        resp, latency, wait = _with_backoff(call, is_daily_quota, is_rate_limit, max_retries)
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
            rate_limit_wait_s=wait,
        )


def get_provider():
    if config.LLM_PROVIDER == "gemini":
        return GeminiProvider()
    if config.LLM_PROVIDER == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")
