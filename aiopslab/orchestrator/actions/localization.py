# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Actions for the localization task."""

from aiopslab.utils.status import SubmissionStatus


class LocalizationActions:
    """
    Class for localization task's actions.
    """

    @staticmethod
    def submit(faulty_components: list[str]) -> SubmissionStatus:
        """
        Submit the detected faulty components to the orchestrator for evaluation.

        Args:
            faulty_components (list[str]): List of faulty components (i.e., service names).

        Returns:
            SubmissionStatus: The status of the submission.
        """
        return SubmissionStatus.VALID_SUBMISSION
