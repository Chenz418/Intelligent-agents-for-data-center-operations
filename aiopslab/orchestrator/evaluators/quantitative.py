# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper functions for quantiative evaluation of solutions."""

from aiopslab.session import SessionItem

# Constants
token_model = "gpt-3.5-turbo"
_TOKENIZER_UNAVAILABLE = object()
_tokenizer = None


def _get_tokenizer():
    """Load the tokenizer lazily so importing benchmark registries has no network side effects."""
    global _tokenizer
    if _tokenizer is _TOKENIZER_UNAVAILABLE:
        return None
    if _tokenizer is None:
        try:
            import tiktoken

            _tokenizer = tiktoken.encoding_for_model(token_model)
        except Exception:
            _tokenizer = _TOKENIZER_UNAVAILABLE
            return None
    return _tokenizer


def _count_tokens(text: str, *, disallowed_special=()) -> int:
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return len(text.split()) if text else 0
    return len(tokenizer.encode(text, disallowed_special=disallowed_special))


def num_steps_taken(trace: list[SessionItem]) -> int:
    """Return the number of steps taken in the trace."""
    return len([item for item in trace if item.role == "assistant"])


def out_tokens(trace: list[SessionItem]) -> int:
    """Return the (approx) total token cost of the agent's output."""
    # NOTE: not dollar value, since depends on Agent's model

    agent_steps = "".join([item.content for item in trace if item.role == "assistant"])
    return _count_tokens(agent_steps, disallowed_special=())


def in_tokens(trace: list[SessionItem]) -> int:
    """Return the (approx) total token cost of the env's input."""
    # NOTE: not dollar value, since depends on Agent's model

    user_steps = "".join([item.content for item in trace if item.role != "assistant"])
    return _count_tokens(user_steps)
