"""DC-Bench localization task submission and accounting."""

from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.actions.localization import LocalizationActions


class LocalizationTask(Task):
    def __init__(self, app):
        super().__init__(app)
        self.actions = LocalizationActions()

    def eval(self, soln, trace, duration):
        self.add_result("TTL", duration)
        return super().eval(soln, trace, duration)
