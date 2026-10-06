"""DC-Bench analysis task submission and accounting."""

from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.actions.analysis import AnalysisActions


class AnalysisTask(Task):
    def __init__(self, app):
        super().__init__(app)
        self.actions = AnalysisActions()

    def eval(self, soln, trace, duration):
        self.add_result("TTA", duration)
        return super().eval(soln, trace, duration)
