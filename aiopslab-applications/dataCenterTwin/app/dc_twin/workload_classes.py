"""Deterministic resource-demand definitions for workload classes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadClassProfile:
    workload_class: str
    cpu_demand: float
    memory_demand: float
    network_demand: float
    storage_demand: float
    gpu_demand: float = 0.0
    latency_sensitivity: float = 1.0
    burst_sensitivity: float = 0.0
    storage_latency_sensitivity: float = 1.0
    network_latency_sensitivity: float = 1.0

    def resource_demand(self) -> dict[str, float]:
        return {
            "cpu": self.cpu_demand,
            "memory": self.memory_demand,
            "network": self.network_demand,
            "storage": self.storage_demand,
            "gpu": self.gpu_demand,
        }


WORKLOAD_CLASS_ALIASES = {
    "web": "web_service",
    "web-service": "web_service",
    "training": "ai_training",
    "ai-training": "ai_training",
    "inference": "ai_inference",
    "ai-inference": "ai_inference",
    "storage": "storage_heavy",
    "storage-heavy": "storage_heavy",
    "network": "network_heavy",
    "network-heavy": "network_heavy",
}


WORKLOAD_CLASSES = {
    "web_service": WorkloadClassProfile(
        workload_class="web_service",
        cpu_demand=0.45,
        memory_demand=0.35,
        network_demand=0.35,
        storage_demand=0.20,
        latency_sensitivity=1.15,
    ),
    "ai_training": WorkloadClassProfile(
        workload_class="ai_training",
        cpu_demand=0.95,
        memory_demand=0.85,
        network_demand=0.75,
        storage_demand=0.70,
        gpu_demand=0.95,
        latency_sensitivity=0.80,
        storage_latency_sensitivity=1.10,
        network_latency_sensitivity=1.05,
    ),
    "ai_inference": WorkloadClassProfile(
        workload_class="ai_inference",
        cpu_demand=0.55,
        memory_demand=0.60,
        network_demand=0.45,
        storage_demand=0.30,
        gpu_demand=0.65,
        latency_sensitivity=1.35,
        burst_sensitivity=0.45,
        network_latency_sensitivity=1.10,
    ),
    "storage_heavy": WorkloadClassProfile(
        workload_class="storage_heavy",
        cpu_demand=0.35,
        memory_demand=0.45,
        network_demand=0.40,
        storage_demand=0.95,
        latency_sensitivity=0.95,
        storage_latency_sensitivity=1.75,
    ),
    "network_heavy": WorkloadClassProfile(
        workload_class="network_heavy",
        cpu_demand=0.35,
        memory_demand=0.35,
        network_demand=0.95,
        storage_demand=0.25,
        latency_sensitivity=1.00,
        network_latency_sensitivity=1.80,
    ),
}


SUPPORTED_WORKLOAD_CLASSES = set(WORKLOAD_CLASSES)
BASELINE_WORKLOAD_CLASS = WORKLOAD_CLASSES["web_service"]


def build_workload_class(workload_class: str | None) -> WorkloadClassProfile:
    normalized = WORKLOAD_CLASS_ALIASES.get(workload_class or "web_service", workload_class or "web_service")
    if normalized not in WORKLOAD_CLASSES:
        raise ValueError(f"unsupported workload_class: {normalized}")
    return WORKLOAD_CLASSES[normalized]
