"""OpenAI-compatible and Gemini-native Data Center Twin transport."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import asyncio
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

from .base import (
    zero_token_usage,
)
from .token_accounting import TokenUsageLedger
from .gemini_native import generate_content_payload, generate_content_url


@dataclass
class OpenAICompatibleSettings:
    """Chat-completions or ``provider='gemini_native'`` settings.

    Secrets must come from CLI/environment at runtime. Keep api_key blank in
    committed code.
    """

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    provider: str = "openai_compatible"
    api_key_env: str = "DC_TWIN_LLM_API_KEY"
    temperature: float | None = 0.0
    tool_choice: str = "required"
    max_tokens: int = 1024
    use_max_completion_tokens: bool = False
    timeout_seconds: float = 60.0
    reasoning_effort: str | None = None
    thinking_mode: str | None = None
    rate_limit_max_retries: int = 6
    rate_limit_initial_delay_seconds: float = 1.0
    rate_limit_max_delay_seconds: float = 60.0


class LLMProviderRequestError(RuntimeError):
    """Provider HTTP error with the original response body preserved."""

    def __init__(
        self,
        *,
        status_code: int,
        reason: str,
        body: str,
        message: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason
        self.body = body
        self.retry_after_seconds = retry_after_seconds


class LLMProviderTransportError(RuntimeError):
    """Terminal provider transport/response error after a request was attempted."""


class LLMProviderModelMismatchError(RuntimeError):
    """The gateway returned a different model from the requested exact model."""


class OpenAICompatibleAgent:
    """Agent using an OpenAI-compatible chat-completions endpoint.

    The class is deliberately tiny: it owns chat history, one blocking HTTP
    request wrapped in `asyncio.to_thread`, and provider token accounting. The
    benchmark loop remains provider-agnostic and only calls `get_action(...)`.
    """

    def __init__(self, settings: OpenAICompatibleSettings) -> None:
        self.settings = settings
        self.messages: list[dict[str, str]] = []
        self._token_usage_ledger = TokenUsageLedger(
            provider=settings.provider,
            model=settings.model,
        )
        # Public per-call records are evaluation-only metadata. They never enter
        # the agent prompt or environment observations.
        self.model_call_token_usage = self._token_usage_ledger.call_records
        self.provider_transport_attempts: list[dict[str, Any]] = []
        self.token_usage = zero_token_usage()
        self._provider_call_tasks: list[asyncio.Task[Any]] = []
        self._last_issued_call_index: int | None = None
        self._episode_termination_event = threading.Event()
        # This lock is held only for the instantaneous attempt-authorization
        # transition, never for network I/O.  It gives the episode cutoff and
        # each retry a single linearization point: an attempt is either
        # authorized before cutoff or rejected after it.
        self._provider_attempt_authorization_lock = threading.Lock()
        self._turn_decision_checkpoint: dict[str, Any] | None = None

    def _initialize_messages(
        self,
        system_content: str,
    ) -> None:
        """Stage only static instructions; the runner sends observations."""
        self.messages = [{"role": "system", "content": system_content}]
        self._turn_decision_checkpoint = None

    def _capture_turn_decision_checkpoint(self) -> None:
        """Remember agent state before a provider response can mutate it."""

        emitted_calls = getattr(self, "emitted_calls", None)
        self._turn_decision_checkpoint = {
            "message_count": len(self.messages),
            "emitted_call_count": (
                len(emitted_calls) if isinstance(emitted_calls, list) else None
            ),
            "last_normalization_status": deepcopy(
                getattr(self, "last_normalization_status", None)
            ),
        }

    def discard_late_provider_response(self) -> None:
        """Undo response-only agent mutations at the absolute deadline race."""

        checkpoint = self._turn_decision_checkpoint
        if not isinstance(checkpoint, dict):
            return
        message_count = checkpoint.get("message_count")
        if isinstance(message_count, int) and not isinstance(message_count, bool):
            del self.messages[message_count:]
        emitted_calls = getattr(self, "emitted_calls", None)
        emitted_count = checkpoint.get("emitted_call_count")
        if (
            isinstance(emitted_calls, list)
            and isinstance(emitted_count, int)
            and not isinstance(emitted_count, bool)
        ):
            del emitted_calls[emitted_count:]
        if hasattr(self, "last_normalization_status"):
            self.last_normalization_status = deepcopy(
                checkpoint.get("last_normalization_status")
            )

    async def _request_chat_completion_async(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Run provider I/O in a tracked task that survives turn cancellation.

        The outer ``get_action`` coroutine remains cancellable at the episode
        deadline.  Shielding only the transport task lets its eventual usage
        settle the audit ledger without allowing the response to resume action
        normalization, history mutation, or environment dispatch.
        """

        issued_payload = self._provider_request_payload(payload)
        call_index = self._token_usage_ledger.begin_call(request_payload=issued_payload)
        self._last_issued_call_index = call_index
        task = asyncio.create_task(
            asyncio.to_thread(
                self._request_chat_completion_json,
                issued_payload,
                call_index=call_index,
            ),
            name=f"provider-accounting-call-{call_index}",
        )
        self._provider_call_tasks.append(task)
        try:
            return await asyncio.shield(task)
        finally:
            # On cancellation this exposes the pending input estimate.  The
            # runner refreshes it again after accounting settlement.
            self.token_usage = self._token_usage_ledger.usage

    def mark_episode_terminated_for_accounting(
        self,
        termination_reason: str,
        *,
        termination_timestamp: str | None = None,
    ) -> int:
        """Close provider retries and quarantine the current turn's response."""

        with self._provider_attempt_authorization_lock:
            self._episode_termination_event.set()
        marked = self._token_usage_ledger.mark_in_flight_calls_accounting_only(
            termination_reason=termination_reason,
            termination_timestamp=termination_timestamp,
        )
        if self._last_issued_call_index is not None:
            self._token_usage_ledger.mark_call_accounting_only(
                self._last_issued_call_index,
                termination_reason=termination_reason,
                termination_timestamp=termination_timestamp,
            )
        self.token_usage = self._token_usage_ledger.usage
        return marked

    async def settle_provider_accounting(self) -> dict[str, Any]:
        """Await tracked provider work after the episode state is frozen."""

        settlement_started_at = _utc_timestamp()
        monotonic_started_at = time.monotonic()
        tasks = list(self._provider_call_tasks)
        unfinished_at_start = sum(not task.done() for task in tasks)
        if tasks:
            # Results and exceptions are intentionally discarded here.  Their
            # only permitted late effect is the ledger transition performed by
            # the provider worker itself.
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=True,
            )
        self.token_usage = self._token_usage_ledger.usage
        pending = int(self.token_usage.get("pending_call_count") or 0)
        return {
            "schema_version": "controlled_provider.accounting_settlement.v1",
            "supported": True,
            "status": "settled" if pending == 0 else "incomplete",
            "tracked_call_count": len(tasks),
            "unfinished_call_count_at_settlement_start": unfinished_at_start,
            "pending_call_count_at_settlement_end": pending,
            "late_completed_call_count": int(
                self.token_usage.get("late_completed_call_count") or 0
            ),
            "late_failed_call_count": int(
                self.token_usage.get("late_failed_call_count") or 0
            ),
            "settlement_started_at": settlement_started_at,
            "settlement_completed_at": _utc_timestamp(),
            "settlement_wait_seconds": time.monotonic() - monotonic_started_at,
            "episode_state_frozen_before_settlement": True,
        }

    def _chat_completion_payload(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            # Freeze the exact issued history.  A worker must never hash or
            # estimate a list that later turns can mutate by reference.
            "messages": deepcopy(messages),
        }
        if self.settings.temperature is not None:
            payload["temperature"] = self.settings.temperature
        # Approved profiles may pin the provider-native modern field up front;
        # the generic adapter retains its compatibility retry by default.
        if self.settings.use_max_completion_tokens:
            payload["max_completion_tokens"] = self.settings.max_tokens
        else:
            payload["max_tokens"] = self.settings.max_tokens
        if self.settings.reasoning_effort is not None:
            payload["reasoning_effort"] = self.settings.reasoning_effort
        if self.settings.thinking_mode is not None:
            if self.settings.thinking_mode not in {"enabled", "disabled"}:
                raise ValueError("thinking_mode must be 'enabled', 'disabled', or None")
            if self.settings.provider == "dashscope":
                payload["enable_thinking"] = self.settings.thinking_mode == "enabled"
            else:
                payload["thinking"] = {"type": self.settings.thinking_mode}
        return payload

    def _request_chat_completion_json(
        self,
        payload: dict[str, Any],
        *,
        call_index: int | None = None,
    ) -> dict[str, Any]:
        """POST with bounded compatibility and rate-limit retries."""
        url = self._chat_completion_url()
        current_payload = self._provider_request_payload(payload)
        if call_index is None:
            call_index = self._token_usage_ledger.begin_call(
                request_payload=current_payload
            )
            self._last_issued_call_index = call_index
        retried_adjustments: set[str] = set()
        rate_limit_retries = 0

        while True:
            if not self._authorize_provider_attempt():
                error = LLMProviderTransportError(
                    "episode terminated before another provider transport attempt"
                )
                self._record_unavailable_provider_call(
                    current_payload,
                    error,
                    call_index=call_index,
                )
                raise error
            try:
                transport_attempt = {
                    "transport_attempt_index": len(self.provider_transport_attempts)
                    + 1,
                    "model_call_index": call_index,
                    "started_at": _utc_timestamp(),
                    "completed_at": None,
                    "status": "in_progress",
                    "http_status": None,
                    "requested_model": current_payload.get(
                        "model", self.settings.model
                    ),
                    "request_api": (
                        "generateContent"
                        if self.settings.provider == "gemini_native"
                        else "chat/completions"
                    ),
                    "request_native_generation_config": deepcopy(
                        current_payload.get("generationConfig")
                    ),
                    "request_reasoning_effort": current_payload.get("reasoning_effort"),
                    "request_thinking_mode": (
                        (
                            "enabled"
                            if current_payload["enable_thinking"]
                            else "disabled"
                        )
                        if isinstance(current_payload.get("enable_thinking"), bool)
                        else current_payload.get("thinking", {}).get("type")
                        if isinstance(current_payload.get("thinking"), dict)
                        else None
                    ),
                    "request_uses_max_tokens": "max_tokens" in current_payload,
                    "request_uses_max_completion_tokens": (
                        "max_completion_tokens" in current_payload
                    ),
                    "request_temperature_present": "temperature" in current_payload,
                    "server_reported_model": None,
                    "provider_response_id": None,
                    "provider_service_tier": None,
                    "failure_type": None,
                    "failure_reason": None,
                }
                response = self._request_json(url, current_payload)
                transport_attempt.update(
                    {
                        "completed_at": _utc_timestamp(),
                        "status": "completed",
                        "http_status": 200,
                        "server_reported_model": response.get(
                            "model", response.get("modelVersion")
                        ),
                        "provider_response_id": response.get(
                            "id", response.get("responseId")
                        ),
                        "provider_service_tier": response.get("service_tier"),
                    }
                )
                if self.settings.provider == "gemini_native":
                    # Retain every native response, including late or rejected
                    # responses that never enter action-normalization traces.
                    transport_attempt["native_response"] = deepcopy(response)
                self.provider_transport_attempts.append(transport_attempt)
                self._token_usage_ledger.complete_call(
                    call_index,
                    request_payload=current_payload,
                    response_payload=response,
                )
                self.token_usage = self._token_usage_ledger.usage
                if self.settings.provider == "gemini_native":
                    reported_model = response.get("modelVersion")
                    if reported_model is not None and (
                        not isinstance(reported_model, str)
                        or reported_model.removeprefix("models/")
                        != self.settings.model.removeprefix("models/")
                    ):
                        # Usage has already settled: this is a real response
                        # whose decision must not enter the benchmark episode.
                        raise LLMProviderModelMismatchError(
                            f"Gemini returned model {reported_model!r}; "
                            f"requested {self.settings.model!r}"
                        )
                return response
            except LLMProviderModelMismatchError:
                raise
            except LLMProviderRequestError as error:
                try:
                    error_payload = json.loads(error.body)
                except (TypeError, ValueError):
                    error_payload = None
                error_detail = (
                    error_payload.get("error")
                    if isinstance(error_payload, dict)
                    else None
                )
                transport_attempt.update(
                    {
                        "completed_at": _utc_timestamp(),
                        "status": "failed",
                        "http_status": error.status_code,
                        "failure_type": type(error).__name__,
                        "failure_reason": error.reason,
                        "provider_error_code": (
                            error_detail.get("code") or error_detail.get("type")
                            if isinstance(error_detail, dict)
                            else None
                        ),
                    }
                )
                self.provider_transport_attempts.append(transport_attempt)
                unsupported_parameter = self._unsupported_request_parameter(error)
                if (
                    unsupported_parameter == "max_tokens"
                    and "max_tokens" in current_payload
                    and "max_completion_tokens" not in current_payload
                    and "max_tokens" not in retried_adjustments
                    and not self._episode_termination_event.is_set()
                ):
                    current_payload = dict(current_payload)
                    current_payload["max_completion_tokens"] = current_payload.pop(
                        "max_tokens"
                    )
                    retried_adjustments.add("max_tokens")
                    continue
                if (
                    unsupported_parameter == "temperature"
                    and "temperature" in current_payload
                    and "temperature" not in retried_adjustments
                    and not self._episode_termination_event.is_set()
                ):
                    current_payload = dict(current_payload)
                    current_payload.pop("temperature", None)
                    retried_adjustments.add("temperature")
                    continue

                if (
                    error.status_code == 429
                    and rate_limit_retries < self.settings.rate_limit_max_retries
                    and not self._episode_termination_event.is_set()
                ):
                    delay = self._rate_limit_retry_delay(
                        error,
                        retry_index=rate_limit_retries,
                    )
                    rate_limit_retries += 1
                    # Wake immediately at episode cutoff.  The next loop
                    # iteration still performs the atomic authorization check,
                    # so no retry can be classified as issued after cutoff.
                    self._wait_for_retry_delay(delay)
                    continue
                self._record_unavailable_provider_call(
                    current_payload,
                    error,
                    call_index=call_index,
                )
                raise
            except LLMProviderTransportError as error:
                transport_attempt.update(
                    {
                        "completed_at": _utc_timestamp(),
                        "status": "failed",
                        "failure_type": type(error).__name__,
                        "failure_reason": str(error),
                    }
                )
                self.provider_transport_attempts.append(transport_attempt)
                self._record_unavailable_provider_call(
                    current_payload,
                    error,
                    call_index=call_index,
                )
                raise
            except Exception as error:
                transport_attempt.update(
                    {
                        "completed_at": _utc_timestamp(),
                        "status": "failed",
                        "failure_type": type(error).__name__,
                        "failure_reason": str(error),
                    }
                )
                self.provider_transport_attempts.append(transport_attempt)
                self._record_unavailable_provider_call(
                    current_payload,
                    error,
                    call_index=call_index,
                )
                raise

    def _authorize_provider_attempt(self) -> bool:
        """Atomically classify one transport attempt relative to cutoff."""

        with self._provider_attempt_authorization_lock:
            return not self._episode_termination_event.is_set()

    def _wait_for_retry_delay(self, delay: float) -> bool:
        """Wait for backoff, returning early when the episode is closed."""

        return self._episode_termination_event.wait(max(0.0, delay))

    def _record_unavailable_provider_call(
        self,
        request_payload: dict[str, Any],
        error: Exception,
        *,
        call_index: int | None = None,
    ) -> None:
        status_code = getattr(error, "status_code", None)
        status_suffix = f", HTTP {status_code}" if status_code is not None else ""
        warning = (
            "provider call ended without authoritative output-token usage "
            f"({type(error).__name__}{status_suffix})"
        )
        if call_index is None:
            self._token_usage_ledger.record_unavailable_call(
                request_payload=request_payload,
                warning=warning,
            )
        else:
            self._token_usage_ledger.fail_call(
                call_index,
                request_payload=request_payload,
                warning=warning,
            )
        self.token_usage = self._token_usage_ledger.usage

    def _rate_limit_retry_delay(
        self,
        error: LLMProviderRequestError,
        *,
        retry_index: int,
    ) -> float:
        """Return capped exponential backoff honoring provider retry hints."""
        initial_delay = max(0.0, self.settings.rate_limit_initial_delay_seconds)
        max_delay = max(initial_delay, self.settings.rate_limit_max_delay_seconds)
        exponential_delay = min(max_delay, initial_delay * (2**retry_index))
        provider_delay = error.retry_after_seconds
        if provider_delay is None:
            provider_delay = self._retry_after_from_body(error.body)
        delay = min(max_delay, max(exponential_delay, provider_delay or 0.0))
        jitter_ceiling = min(1.0, delay * 0.25)
        return min(max_delay, delay + random.uniform(0.0, jitter_ceiling))

    def _retry_after_from_body(self, body: str) -> float | None:
        match = re.search(
            r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?)",
            body,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value / 1000.0 if unit.startswith("m") else value

    def _unsupported_request_parameter(
        self, error: LLMProviderRequestError
    ) -> str | None:
        try:
            parsed = json.loads(error.body)
        except json.JSONDecodeError:
            parsed = {}

        error_payload = parsed.get("error") if isinstance(parsed, dict) else None
        if not isinstance(error_payload, dict):
            return None

        code = error_payload.get("code")
        param = error_payload.get("param")
        message = error_payload.get("message")
        if code == "unsupported_parameter" and isinstance(param, str) and param:
            return param
        if isinstance(message, str):
            lowered = message.lower()
            if (
                "unsupported" in lowered
                and "max_tokens" in lowered
                and "max_completion_tokens" in lowered
            ):
                return "max_tokens"
            if "unsupported" in lowered and "temperature" in lowered:
                return "temperature"
        return None

    def _request_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON and convert provider/network failures into RuntimeError.

        The caller records these as agent infrastructure errors. Keeping the
        original HTTP body short avoids dumping large or sensitive provider
        responses into result artifacts.
        """
        api_key = self._api_key()
        body = json.dumps(payload).encode("utf-8")
        auth_header = (
            {"x-goog-api-key": api_key}
            if self.settings.provider == "gemini_native"
            else {"Authorization": f"Bearer {api_key}"}
        )
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                **auth_header,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.timeout_seconds,
            ) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            message = (
                "LLM provider request failed "
                f"({error.code} {error.reason}): {self._short_text(error_body)}"
            )
            retry_after_seconds = None
            retry_after_header = (
                error.headers.get("Retry-After") if error.headers else None
            )
            if retry_after_header is not None:
                try:
                    retry_after_seconds = max(0.0, float(retry_after_header))
                except ValueError:
                    retry_after_seconds = None
            raise LLMProviderRequestError(
                status_code=error.code,
                reason=error.reason,
                body=error_body,
                message=message,
                retry_after_seconds=retry_after_seconds,
            ) from error
        except urllib.error.URLError as error:
            raise LLMProviderTransportError(
                f"LLM provider request failed: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise LLMProviderTransportError(
                f"LLM provider request timed out after {self.settings.timeout_seconds}s"
            ) from error

        try:
            if self.settings.provider == "gemini_native":
                from .json_utils import strict_json_loads

                parsed = strict_json_loads(raw_body)
            else:
                parsed = json.loads(raw_body)
        except ValueError as error:
            raise LLMProviderTransportError(
                f"LLM provider returned non-JSON response: {self._short_text(raw_body)}"
            ) from error
        if not isinstance(parsed, dict):
            raise LLMProviderTransportError(
                f"LLM provider returned unexpected JSON: {self._short_json(parsed)}"
            )
        return parsed

    def _api_key(self) -> str:
        api_key = self.settings.api_key or os.getenv(self.settings.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                "openai_compatible agent requires an API key. "
                f"Set {self.settings.api_key_env} or select a populated variable with --api-key-env."
            )
        return api_key

    def _chat_completion_url(self) -> str:
        """Accept either a provider base URL or a full chat-completions URL."""
        if self.settings.provider == "gemini_native":
            return generate_content_url(self.settings.base_url, self.settings.model)
        base_url = self.settings.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        if "api.openai.com" in base_url:
            return f"{base_url}/v1/chat/completions"
        return f"{base_url}/chat/completions"

    def _provider_request_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.settings.provider == "gemini_native":
            return generate_content_payload(payload)
        return deepcopy(payload)

    def _short_json(self, value: Any, limit: int = 1000) -> str:
        return self._short_text(
            json.dumps(value, sort_keys=True, default=str), limit=limit
        )

    def _short_text(self, value: str, limit: int = 1000) -> str:
        return value if len(value) <= limit else value[:limit] + "...<truncated>"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
