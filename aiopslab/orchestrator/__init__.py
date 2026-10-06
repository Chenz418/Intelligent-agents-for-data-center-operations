# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AIOpsLab orchestrator package.

Import the Kubernetes-backed Orchestrator lazily so lightweight submodules such
as parser, registry, and Data Center Twin evaluation helpers can be imported in
simulator-only environments.
"""


def __getattr__(name):
    if name == "Orchestrator":
        from .orchestrator import Orchestrator

        return Orchestrator
    raise AttributeError(name)


__all__ = ["Orchestrator"]
