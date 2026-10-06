"""The two supported DC-Bench agent types."""

from argparse import Namespace
import os
from .openai_compatible import OpenAICompatibleSettings
from .raw_tool_calling import RawToolCallingAgent


def list_agent_names():
    return ["tool-calling", "codex"]


def create_agent(name, args):
    if name != "tool-calling":
        raise ValueError(
            "Only tool-calling runs in process; Codex uses the isolated bridge"
        )
    return RawToolCallingAgent(_openai_settings(args, name))


def _openai_settings(args: Namespace, agent_name: str) -> OpenAICompatibleSettings:
    api_key_env = getattr(args, "api_key_env", "DC_TWIN_LLM_API_KEY")
    api_key = getattr(args, "api_key", "") or os.getenv(api_key_env, "")
    if not api_key:
        raise SystemExit(
            f"{agent_name} agent requires an API key. "
            f"Set {api_key_env} or pass --api-key-env with an environment variable that is set."
        )
    return OpenAICompatibleSettings(
        api_key=api_key,
        base_url=getattr(args, "base_url", "https://api.openai.com/v1"),
        model=getattr(args, "model", "gpt-4o-mini"),
        provider=getattr(args, "provider", "openai_compatible"),
        api_key_env=api_key_env,
        temperature=getattr(args, "temperature", 0.0),
        tool_choice=getattr(args, "tool_choice", "required"),
        max_tokens=getattr(args, "max_tokens", 1024),
        use_max_completion_tokens=getattr(args, "use_max_completion_tokens", False),
        timeout_seconds=getattr(args, "timeout_seconds", 60.0),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        thinking_mode=getattr(args, "thinking_mode", None),
        rate_limit_max_retries=getattr(args, "rate_limit_max_retries", 6),
        rate_limit_initial_delay_seconds=getattr(
            args,
            "rate_limit_initial_delay_seconds",
            1.0,
        ),
        rate_limit_max_delay_seconds=getattr(
            args,
            "rate_limit_max_delay_seconds",
            60.0,
        ),
    )
