# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from aiopslab.orchestrator.parser import ResponseParser
from aiopslab.utils.status import ResponseParsingError


class TestParser(unittest.TestCase):
    def test_no_args(self):
        input = """
        Action:
        ```
        myApi()
        ```
        """
        parser = ResponseParser()
        parsed = parser.parse(input)
        self.assertEqual(parsed["api_name"], "myApi")
        self.assertEqual(parsed["args"], [])

    def test_single_arg(self):
        input = """
        Action:
        ```
        myApi(10)
        ```
        """
        parser = ResponseParser()
        parsed = parser.parse(input)
        self.assertEqual(parsed["api_name"], "myApi")
        self.assertEqual(parsed["args"], [10])

    def test_multiple_args(self):
        input = """
        Action:
        ```
        myApi(10, 'error')
        ```
        """
        parser = ResponseParser()
        parsed = parser.parse(input)
        self.assertEqual(parsed["api_name"], "myApi")
        self.assertEqual(parsed["args"], [10, "error"])

    def test_signed_numeric_literals_in_nested_tool_arguments(self):
        input = """
        Action:
        ```
        dc_twin_action('calibrate_sensor', parameters={'target': 'rack-1', 'calibration_offset_c': -12.6, 'trim': +2})
        ```
        """
        parsed = ResponseParser().parse(input)

        self.assertEqual(
            parsed["kwargs"]["parameters"],
            {
                "target": "rack-1",
                "calibration_offset_c": -12.6,
                "trim": 2,
            },
        )

    def test_unary_sign_rejects_non_numeric_literals(self):
        input = """
        Action:
        ```
        myApi(value=-True)
        ```
        """

        with self.assertRaises(ResponseParsingError):
            ResponseParser().parse(input)

    def test_shell_cmd_arg(self):
        input = """
        Action:
        ```
        exec_shell("kubectl get pods -n dc-twin")
        ```
        """
        parser = ResponseParser()
        parsed = parser.parse(input)
        self.assertEqual(parsed["api_name"], "exec_shell")
        self.assertEqual(parsed["args"], ["kubectl get pods -n dc-twin"])

    def test_shell_cmd_arg_with_nested_args(self):
        input = """
        Action:
        ```
        exec_shell("kubectl patch svc dc-twin -n dc-twin --type='json' -p='[{\"op\": \"replace\", \"path\": \"/spec/ports/0/targetPort\", \"value\": 8080}]'")
        ```
        """
        parser = ResponseParser()
        parsed = parser.parse(input)
        self.assertEqual(parsed["api_name"], "exec_shell")
        self.assertEqual(
            parsed["args"],
            [
                'kubectl patch svc dc-twin -n dc-twin --type=\'json\' -p=\'[{"op": "replace", "path": "/spec/ports/0/targetPort", "value": 8080}]\''
            ],
        )

    def test_context_extraction(self):
        input = """
        Thought: Before taking actions, I need to gather initial logs from the `dc-twin-api` pod (in the `dc-twin` namespace).
        
        Action:
        ```
        get_logs('dc-twin-api', 'dc-twin')
        ```

        This will help me understand the current state of the application.
        """
        parser = ResponseParser()
        parsed = parser.parse(input)
        self.assertEqual(parsed["api_name"], "get_logs")
        self.assertEqual(
            parsed["args"], ["dc-twin-api", "dc-twin"]
        )
        self.assertEqual(len(parsed["context"]), 2)
        self.assertEqual(
            parsed["context"][0],
            "Thought: Before taking actions, I need to gather initial logs from the `dc-twin-api` pod (in the `dc-twin` namespace).\n        \n        Action:",
        )
        self.assertEqual(
            parsed["context"][1],
            "This will help me understand the current state of the application.",
        )

    def test_non_api(self):
        input = """
        Action:
        ```
        echo "Hello World"
        ```
        """
        parser = ResponseParser()
        self.assertRaises(ResponseParsingError, parser.parse, input)


if __name__ == "__main__":
    unittest.main()
