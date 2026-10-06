"""DC-Bench detection task submission and accounting."""

from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.actions.detection import DetectionActions


class DetectionTask(Task):
    def __init__(self, app):
        super().__init__(app)
        self.actions = DetectionActions()

    def eval(self, soln, trace, duration):
        self.add_result("TTD", duration)
        return super().eval(soln, trace, duration)
