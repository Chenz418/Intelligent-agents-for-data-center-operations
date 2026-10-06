"""Per-model-call token accounting for controlled Data Center Twin agents.

Calls are registered before transport dispatch so episode cancellation cannot
erase the issued request. Provider usage is authoritative when both input and
output counts are present. When it is absent or incomplete, the ledger records
an explicit deterministic estimate instead of allowing a missing response
field to look like known zero usage. Estimation is local and never makes a
provider request.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import threading
from typing import Any


SERIALIZATION_ID = "canonical_chat_json_v1"
PROVIDER_NATIVE = "provider_native"
ESTIMATED = "estimated"
UNAVAILABLE = "unavailable"
PENDING = "pending"


class TokenUsageLedger:
    """Accumulate auditable input/output usage for every issued model call."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self.call_records: list[dict[str, Any]] = []
        # Provider work runs in executor threads while the benchmark state
        # machine remains on the asyncio thread.  A small lock makes the
        # pending -> settled transition atomic for artifact snapshots.
        self._lock = threading.RLock()

    def begin_call(self, *, request_payload: dict[str, Any]) -> int:
        """Reserve one stable call record before dispatching provider I/O.

        Registering at issuance preserves the exact request fingerprint and a
        deterministic input-token estimate even if the episode deadline fires
        while the provider request is still in flight.
        """

        warnings: list[str] = []
        try:
            input_tokens, estimator = _estimate_request_tokens(
                provider=self.provider,
                model=self.model,
                request_payload=request_payload,
            )
            input_source = ESTIMATED
        except Exception as error:  # pragma: no cover - defensive fallback
            input_tokens = None
            estimator = None
            input_source = UNAVAILABLE
            warnings.append(
                "input-token estimation was unavailable at request issuance: "
                f"{type(error).__name__}: {error}"
            )

        with self._lock:
            call_index = len(self.call_records) + 1
            self.call_records.append(
                {
                    "call_index": call_index,
                    "provider": self.provider,
                    "model": self.model,
                    "call_status": PENDING,
                    "input_tokens": input_tokens,
                    "output_tokens": None,
                    "total_tokens": None,
                    # Keep the established public source vocabulary stable;
                    # call_status distinguishes an in-flight record from a
                    # terminal unavailable response.
                    "token_count_source": UNAVAILABLE,
                    "input_token_count_source": input_source,
                    "output_token_count_source": UNAVAILABLE,
                    "estimator": estimator,
                    "provider_usage": None,
                    "request_payload_sha256": _payload_sha256(request_payload),
                    "response_payload_sha256": None,
                    "request_started_at": _utc_now(),
                    "accounting_completed_at": None,
                    "completed_after_episode_termination": False,
                    "episode_termination_reason": None,
                    "episode_termination_timestamp": None,
                    "trajectory_disposition": "eligible_for_agent_processing",
                    "warnings": warnings,
                }
            )
        return call_index

    def complete_call(
        self,
        call_index: int,
        *,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Settle an existing call with native usage or a local estimate."""

        raw_usage = response_payload.get("usage", response_payload.get("usageMetadata"))
        provider_usage = deepcopy(raw_usage) if isinstance(raw_usage, dict) else None
        native = _native_input_output(provider_usage)
        warnings: list[str] = []

        if native is not None:
            input_tokens, output_tokens = native
            source = PROVIDER_NATIVE
            input_source = PROVIDER_NATIVE
            output_source = PROVIDER_NATIVE
            estimator = None
            reported_total = (
                provider_usage.get("total_tokens", provider_usage.get("totalTokenCount"))
                if provider_usage else None
            )
            if (
                _is_token_count(reported_total)
                and reported_total != input_tokens + output_tokens
            ):
                warnings.append(
                    "provider total_tokens did not equal input plus output; "
                    "the normalized total uses input_tokens + output_tokens"
                )
        else:
            if provider_usage is not None:
                warnings.append(
                    "provider usage was incomplete or invalid; input and output were estimated"
                )
            try:
                input_tokens, output_tokens, estimator = _estimate_call_tokens(
                    provider=self.provider,
                    model=self.model,
                    request_payload=request_payload,
                    response_payload=response_payload,
                )
                source = ESTIMATED
                input_source = ESTIMATED
                output_source = ESTIMATED
            except Exception as error:  # pragma: no cover - defensive fallback
                input_tokens = None
                output_tokens = None
                estimator = None
                source = UNAVAILABLE
                input_source = UNAVAILABLE
                output_source = UNAVAILABLE
                warnings.append(
                    "token estimation was unavailable: "
                    f"{type(error).__name__}: {error}"
                )

        return self._settle_call(
            call_index,
            {
                "call_status": "completed",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else None
                ),
                "token_count_source": source,
                "input_token_count_source": input_source,
                "output_token_count_source": output_source,
                "estimator": estimator,
                "provider_usage": provider_usage,
                "native_usage_components": (
                    gemini_usage_counts(provider_usage)
                    if "usageMetadata" in response_payload else None
                ),
                "request_payload_sha256": _payload_sha256(request_payload),
                "response_payload_sha256": _payload_sha256(response_payload),
                "warnings": warnings,
            },
        )

    def fail_call(
        self,
        call_index: int,
        *,
        request_payload: dict[str, Any],
        warning: str,
    ) -> dict[str, Any]:
        """Settle a failed request while retaining reproducible input usage."""

        warnings = [warning]
        try:
            input_tokens, estimator = _estimate_request_tokens(
                provider=self.provider,
                model=self.model,
                request_payload=request_payload,
            )
            input_source = ESTIMATED
        except Exception as error:  # pragma: no cover - defensive fallback
            input_tokens = None
            estimator = None
            input_source = UNAVAILABLE
            warnings.append(
                "input-token estimation was unavailable: "
                f"{type(error).__name__}: {error}"
            )
        return self._settle_call(
            call_index,
            {
                "call_status": "failed",
                "input_tokens": input_tokens,
                "output_tokens": None,
                "total_tokens": None,
                "token_count_source": UNAVAILABLE,
                "input_token_count_source": input_source,
                "output_token_count_source": UNAVAILABLE,
                "estimator": estimator,
                "provider_usage": None,
                "request_payload_sha256": _payload_sha256(request_payload),
                "response_payload_sha256": None,
                "warnings": warnings,
            },
        )

    def mark_in_flight_calls_accounting_only(
        self,
        *,
        termination_reason: str,
        termination_timestamp: str | None = None,
    ) -> int:
        """Quarantine pending responses from the closed episode state machine."""

        marked = 0
        with self._lock:
            for record in self.call_records:
                if record.get("call_status") != PENDING:
                    continue
                record["episode_termination_reason"] = termination_reason
                record["episode_termination_timestamp"] = (
                    termination_timestamp or _utc_now()
                )
                record["trajectory_disposition"] = (
                    "accounting_only_after_episode_termination"
                )
                marked += 1
        return marked

    def mark_call_accounting_only(
        self,
        call_index: int,
        *,
        termination_reason: str,
        termination_timestamp: str | None = None,
    ) -> None:
        """Mark the timed-out turn's call even if it won a completion race."""

        with self._lock:
            record = self._record(call_index)
            record["episode_termination_reason"] = termination_reason
            record["episode_termination_timestamp"] = (
                termination_timestamp or _utc_now()
            )
            record["trajectory_disposition"] = (
                "accounting_only_after_episode_termination"
            )
            if record.get("call_status") != PENDING:
                record["completed_after_episode_termination"] = True

    def _settle_call(
        self,
        call_index: int,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            record = self._record(call_index)
            if record.get("call_status") != PENDING:
                raise RuntimeError(f"provider call {call_index} was already settled")
            prior_warnings = [
                item for item in record.get("warnings", []) if isinstance(item, str)
            ]
            record.update(fields)
            record["warnings"] = [*prior_warnings, *fields.get("warnings", [])]
            record["accounting_completed_at"] = _utc_now()
            record["completed_after_episode_termination"] = bool(
                record.get("episode_termination_reason")
            )
            return record

    def _record(self, call_index: int) -> dict[str, Any]:
        if (
            isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or call_index < 1
            or call_index > len(self.call_records)
        ):
            raise IndexError(f"unknown provider call index: {call_index!r}")
        return self.call_records[call_index - 1]

    def record_call(
        self,
        *,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Record one successful provider response exactly once.

        HTTP compatibility and rate-limit retries that do not return a model
        response are intentionally outside this boundary.  The caller passes
        the final, possibly compatibility-adjusted request payload.
        """
        call_index = self.begin_call(request_payload=request_payload)
        return self.complete_call(
            call_index,
            request_payload=request_payload,
            response_payload=response_payload,
        )

    def record_unavailable_call(
        self,
        *,
        request_payload: dict[str, Any],
        warning: str,
    ) -> dict[str, Any]:
        """Record a terminal provider call whose output usage is unknowable.

        The request side remains locally countable, but without a provider
        response neither output tokens nor episode totals can be claimed as
        exact or estimated. This prevents transport/provider failures from
        appearing as known-zero usage.
        """
        call_index = self.begin_call(request_payload=request_payload)
        return self.fail_call(
            call_index,
            request_payload=request_payload,
            warning=warning,
        )

    @property
    def usage(self) -> dict[str, Any]:
        """Return additive rich totals while retaining legacy field aliases."""
        with self._lock:
            records = [dict(record) for record in self.call_records]
        unavailable = [
            record
            for record in records
            if record.get("token_count_source") == UNAVAILABLE
            or record.get("call_status") == PENDING
            or not _is_token_count(record.get("input_tokens"))
            or not _is_token_count(record.get("output_tokens"))
        ]
        source_counts = {
            PROVIDER_NATIVE: sum(
                record.get("token_count_source") == PROVIDER_NATIVE
                for record in records
            ),
            ESTIMATED: sum(
                record.get("token_count_source") == ESTIMATED
                for record in records
            ),
            UNAVAILABLE: sum(
                record.get("token_count_source") == UNAVAILABLE
                for record in records
            ),
        }
        pending_count = sum(
            record.get("call_status") == PENDING for record in records
        )
        sources = [source for source, count in source_counts.items() if count]
        aggregate_source = (
            sources[0] if len(sources) == 1 else ("mixed" if sources else None)
        )
        estimators = sorted(
            {
                str(record["estimator"])
                for record in records
                if record.get("estimator")
            }
        )
        warnings = [
            warning
            for record in records
            for warning in record.get("warnings", [])
            if isinstance(warning, str)
        ]

        input_available = all(
            _is_token_count(record.get("input_tokens")) for record in records
        )
        output_available = all(
            _is_token_count(record.get("output_tokens")) for record in records
        )
        known_input_records = [
            record for record in records if _is_token_count(record.get("input_tokens"))
        ]
        known_output_records = [
            record for record in records if _is_token_count(record.get("output_tokens"))
        ]
        known_total_records = [
            record for record in records if _is_token_count(record.get("total_tokens"))
        ]
        known_input_tokens = sum(
            int(record["input_tokens"]) for record in known_input_records
        )
        known_output_tokens = sum(
            int(record["output_tokens"]) for record in known_output_records
        )
        known_total_tokens = sum(
            int(record["total_tokens"]) for record in known_total_records
        )
        input_tokens = (
            known_input_tokens
            if input_available
            else None
        )
        output_tokens = (
            known_output_tokens
            if output_available
            else None
        )
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )

        return {
            # Canonical manuscript terminology.
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            # Backward-compatible benchmark terminology.
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "token_usage_available": not unavailable,
            "input_token_usage_available": input_available,
            "output_token_usage_available": output_available,
            "known_input_tokens": known_input_tokens,
            "known_output_tokens": known_output_tokens,
            "known_total_tokens": known_total_tokens,
            "known_input_call_count": len(known_input_records),
            "known_output_call_count": len(known_output_records),
            "known_total_call_count": len(known_total_records),
            "token_count_source": aggregate_source,
            "token_count_sources": source_counts,
            "estimators": estimators,
            "model_call_count": len(records),
            "pending_call_count": pending_count,
            "late_completed_call_count": sum(
                record.get("call_status") == "completed"
                and record.get("completed_after_episode_termination") is True
                for record in records
            ),
            "late_failed_call_count": sum(
                record.get("call_status") == "failed"
                and record.get("completed_after_episode_termination") is True
                for record in records
            ),
            "warnings": warnings,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _native_input_output(usage: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(usage, dict):
        return None
    if any(key in usage for key in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")):
        counts = gemini_usage_counts(usage)
        if counts["input_tokens"] is not None and counts["output_tokens"] is not None:
            return counts["input_tokens"], counts["output_tokens"]
        return None
    input_tokens = _first_token_count(usage, "prompt_tokens", "input_tokens")
    output_tokens = _first_token_count(usage, "completion_tokens", "output_tokens")
    total_tokens = _first_token_count(usage, "total_tokens")

    if input_tokens is not None and output_tokens is not None:
        return input_tokens, output_tokens
    if (
        input_tokens is not None
        and total_tokens is not None
        and total_tokens >= input_tokens
    ):
        return input_tokens, total_tokens - input_tokens
    if (
        output_tokens is not None
        and total_tokens is not None
        and total_tokens >= output_tokens
    ):
        return total_tokens - output_tokens, output_tokens
    return None


def gemini_usage_counts(usage: dict[str, Any] | None) -> dict[str, int | None]:
    """Read native usage without double counting cache or losing thoughts.

    Gemini documents totalTokenCount as prompt + thoughts + candidates; unlike
    OpenAI completion_tokens, candidatesTokenCount excludes thinking. Missing
    thoughts can be derived only from a valid reported total and other counts.
    https://ai.google.dev/api/generate-content#UsageMetadata
    """
    usage = usage if isinstance(usage, dict) else {}
    prompt = _first_token_count(usage, "promptTokenCount")
    candidates = _first_token_count(usage, "candidatesTokenCount")
    thoughts = _first_token_count(usage, "thoughtsTokenCount")
    total = _first_token_count(usage, "totalTokenCount")
    output = None
    invalid_output = any(
        key in usage and not _is_token_count(usage[key])
        for key in ("candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")
    )
    if not invalid_output:
        if candidates is not None and thoughts is not None:
            output = candidates + thoughts
        elif prompt is not None and total is not None and total >= prompt:
            residual = total - prompt
            if candidates is not None and residual >= candidates:
                thoughts = residual - candidates
                output = residual
            elif candidates is None and (thoughts is None or residual >= thoughts):
                output = residual
                if thoughts is not None:
                    candidates = residual - thoughts
    return {
        "input_tokens": prompt,
        "output_tokens": output,
        "candidate_tokens": candidates,
        "reasoning_tokens": thoughts,
        "cached_input_tokens": _first_token_count(usage, "cachedContentTokenCount"),
        "provider_total_tokens": total,
        "tool_use_prompt_tokens": _first_token_count(usage, "toolUsePromptTokenCount"),
    }


def _first_token_count(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if _is_token_count(value):
            return value
    return None


def _is_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _estimate_call_tokens(
    *,
    provider: str,
    model: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> tuple[int, int, str]:
    request_text = _canonical_json(_tokenizable_request(request_payload))
    response_text = _canonical_json(_tokenizable_response(response_payload))
    counts, estimator = _count_serialized_texts(
        provider=provider,
        model=model,
        texts=(request_text, response_text),
    )
    return counts[0], counts[1], estimator


def _estimate_request_tokens(
    *,
    provider: str,
    model: str,
    request_payload: dict[str, Any],
) -> tuple[int, str]:
    request_text = _canonical_json(_tokenizable_request(request_payload))
    counts, estimator = _count_serialized_texts(
        provider=provider,
        model=model,
        texts=(request_text,),
    )
    return counts[0], estimator


def _count_serialized_texts(
    *,
    provider: str,
    model: str,
    texts: tuple[str, ...],
) -> tuple[list[int], str]:
    tokenizer = _local_tokenizer(provider=provider, model=model)
    if tokenizer is not None:
        encoding, resolution = tokenizer
        try:
            return (
                [
                    len(encoding.encode(value, disallowed_special=()))
                    for value in texts
                ],
                f"tiktoken:{encoding.name}:{resolution}:{SERIALIZATION_ID}",
            )
        except Exception:
            # A local tokenizer can reject unusual provider payload text. The
            # deterministic byte estimator remains available without a model
            # request and is explicitly identified below.
            pass
    return (
        [_utf8_div4_count(value) for value in texts],
        f"utf8_bytes_div4_ceil:{SERIALIZATION_ID}",
    )


def _local_tokenizer(*, provider: str, model: str) -> tuple[Any, str] | None:
    """Resolve an installed tokenizer without ever querying the model API."""
    try:
        tiktoken = importlib.import_module("tiktoken")
    except Exception:
        return None

    try:
        return tiktoken.encoding_for_model(model), "encoding_for_model"
    except Exception:
        pass

    if _uses_modern_openai_encoding(provider=provider, model=model):
        try:
            return tiktoken.get_encoding("o200k_base"), "modern_openai_fallback"
        except Exception:
            return None
    return None


def _uses_modern_openai_encoding(*, provider: str, model: str) -> bool:
    normalized_provider = provider.strip().lower().replace("-", "_")
    normalized_model = model.strip().lower()
    modern_prefixes = (
        "gpt-4o",
        "gpt-5",
        "chatgpt-4o",
        "o1",
        "o3",
        "o4",
    )
    return normalized_model.startswith(modern_prefixes) or (
        "openai" in normalized_provider and normalized_model.startswith(("gpt", "o"))
    )


def _tokenizable_request(payload: dict[str, Any]) -> dict[str, Any]:
    token_bearing_keys = (
        "contents",
        "systemInstruction",
        "toolConfig",
        "messages",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "response_format",
    )
    selected = {key: payload[key] for key in token_bearing_keys if key in payload}
    return selected if selected else payload


def _tokenizable_response(payload: dict[str, Any]) -> Any:
    if "candidates" in payload:
        return {"candidates": payload["candidates"]}
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload
    token_bearing_choices: list[Any] = []
    for choice in choices:
        if not isinstance(choice, dict):
            token_bearing_choices.append(choice)
            continue
        if isinstance(choice.get("message"), dict):
            token_bearing_choices.append({"message": choice["message"]})
        elif "text" in choice:
            token_bearing_choices.append({"text": choice["text"]})
        else:
            token_bearing_choices.append(choice)
    return {"choices": token_bearing_choices}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _utf8_div4_count(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text.encode("utf-8")) / 4)
