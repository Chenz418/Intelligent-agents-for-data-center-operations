"""DC-Bench mitigation task submission and accounting."""

from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.actions.mitigation import MitigationActions


class MitigationTask(Task):
    def __init__(self, app):
        super().__init__(app)
        self.actions = MitigationActions()

    def eval(self, soln, trace, duration):
        self.add_result("TTM", duration)
        return super().eval(soln, trace, duration)
