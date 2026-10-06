"""Task submission status and protocol errors."""

from enum import Enum


class SubmissionStatus(Enum):
    VALID_SUBMISSION = 1
    INVALID_SUBMISSION = 2


class InvalidActionError(Exception):
    def __init__(self, action_name):
        super().__init__(f"Invalid action: {action_name}")
        self.action_name = action_name


class ResponseParsingError(Exception):
    def __init__(self, message):
        super().__init__(f"Error parsing response: {message}")
        self.message = message
