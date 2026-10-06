# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

from aiopslab.orchestrator.parser import ResponseParser
from aiopslab.orchestrator.tasks.detection import DetectionTask


class FakeDataCenterApp:
    helm_configs = {}

    def get_app_summary(self):
        return (
            "Service Name: data-center-twin\n"
            "Namespace: dc-twin\n"
            "Description: parser test application"
        )


class TestParser(unittest.TestCase):
    def setUp(self):
        self.app = FakeDataCenterApp()
        self.task = DetectionTask(self.app)
        self.parser = ResponseParser()

    def test_non_shell(self):
        input = """
        Action:
        ```
        submit('No')
        ```
        """
        resp = self.parser.parse(input)
        api = resp["api_name"]
        args = resp["args"]
        self.assertEqual(api, "submit")
        self.assertEqual(args, ["No"])
        resp = self.task.perform_action(api, *args)
        self.assertTrue(resp)


if __name__ == "__main__":
    unittest.main()
