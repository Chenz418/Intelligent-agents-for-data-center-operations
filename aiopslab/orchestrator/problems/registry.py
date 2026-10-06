from aiopslab.orchestrator.problems import data_center_twin
from aiopslab.orchestrator.problems.data_center_twin.scenarios import (
    ScenarioValidationError,
    validate_scenario_manifest,
)


def _load_data_center_twin_problem_registry():
    registry = {}
    for problem_id, scenario in validate_scenario_manifest().items():
        problem_class = getattr(data_center_twin, scenario.class_name, None)
        if problem_class is None:
            raise ScenarioValidationError(
                f"{problem_id}: registered class {scenario.class_name} is not exported"
            )
        class_scenario_id = getattr(problem_class, "SCENARIO_ID", None)
        if class_scenario_id != problem_id:
            raise ScenarioValidationError(
                f"{problem_id}: {scenario.class_name}.SCENARIO_ID is {class_scenario_id!r}"
            )
        registry[problem_id] = problem_class
    return registry


class ProblemRegistry:
    def __init__(self):
        self.PROBLEM_REGISTRY = _load_data_center_twin_problem_registry()

    def get_problem_instance(self, problem_id: str):
        if problem_id not in self.PROBLEM_REGISTRY:
            raise ValueError(f"Problem ID {problem_id} not found in registry.")

        return self.PROBLEM_REGISTRY.get(problem_id)()

    def get_problem(self, problem_id: str):
        return self.PROBLEM_REGISTRY.get(problem_id)

    def get_problem_ids(self, task_type: str = None):
        if task_type:
            return [k for k in self.PROBLEM_REGISTRY.keys() if task_type in k]
        return list(self.PROBLEM_REGISTRY.keys())

    def get_problem_count(self, task_type: str = None):
        if task_type:
            return len([k for k in self.PROBLEM_REGISTRY.keys() if task_type in k])
        return len(self.PROBLEM_REGISTRY)
