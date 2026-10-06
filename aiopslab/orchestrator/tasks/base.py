"""Shared submission and accounting for DC-Bench tasks."""

from aiopslab.orchestrator.evaluators.quantitative import (
    num_steps_taken,
    in_tokens,
    out_tokens,
)
from aiopslab.utils.status import InvalidActionError


class Task:
    def __init__(self, app):
        self.results = {}
        self.app = app
        self.app_summary = app.get_app_summary()

    def add_result(self, key, value):
        self.results[key] = value

    def get_available_actions(self):
        return {
            "submit": self.actions.submit.__doc__ or "Submit the final task answer."
        }

    def perform_action(self, action_name, *args, **kwargs):
        if action_name != "submit":
            raise InvalidActionError(action_name)
        return self.actions.submit(*args, **kwargs)

    def eval(self, soln, trace, duration):
        self.results.update(
            steps=num_steps_taken(trace),
            in_tokens=in_tokens(trace),
            out_tokens=out_tokens(trace),
        )
        return self.results
