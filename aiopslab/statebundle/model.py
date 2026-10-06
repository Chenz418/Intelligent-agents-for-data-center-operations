"""StateBundle's learned hierarchical representation for inference.

Five modality encoders feed subsystem/global attention, propagation membership,
and relevance/aspect projections. Inputs contain only observable canonical data.
The complete parameter architecture is retained for strict checkpoint loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from aiopslab.statebundle.config import StateBundleModelConfig
from aiopslab.statebundle.encoders import ModalityEncoderRegistry
from aiopslab.statebundle.types import (
    CanonicalObservation,
    IncidentBatch,
    ModelOutput,
    RootCauseCatalog,
)


class TrainingStage(StrEnum):
    EVIDENCE_SELECTION = "stage3"


@dataclass
class _PooledIncident:
    subsystem_ids: tuple[str, ...]
    observation_subsystem_indices: Tensor
    subsystem_embeddings: Tensor
    global_embedding: Tensor
    subsystem_attention: Tensor
    global_attention: Tensor


class StateBundleModel(nn.Module):
    """Paper-faithful hierarchical representation and selector network."""

    def __init__(
        self,
        config: StateBundleModelConfig,
        root_cause_catalog: RootCauseCatalog,
        *,
        modality_encoders: ModalityEncoderRegistry | None = None,
    ):
        super().__init__()
        if not root_cause_catalog.classes:
            raise ValueError("root_cause_catalog must contain at least one class")
        self.config = config
        self.root_cause_catalog = root_cause_catalog
        self.modality_encoders = modality_encoders or ModalityEncoderRegistry(config)

        self.subsystem_attention_projection = nn.Linear(
            config.shared_dimension, config.attention_dimension
        )
        self.subsystem_attention_vector = nn.Linear(
            config.attention_dimension, 1, bias=False
        )
        self.global_attention_projection = nn.Linear(
            config.shared_dimension, config.attention_dimension
        )
        self.global_attention_vector = nn.Linear(
            config.attention_dimension, 1, bias=False
        )

        self.relevance_projection = nn.Linear(
            config.shared_dimension, config.relevance_dimension
        )
        self.aspect_projection = nn.Linear(
            config.shared_dimension, config.aspect_dimension
        )
        self.global_relevance_projection = nn.Linear(
            config.shared_dimension, config.relevance_dimension
        )
        self.subsystem_relevance_projection = nn.Linear(
            config.shared_dimension, config.relevance_dimension
        )

        propagation_input_dimension = config.shared_dimension * 3
        self.propagation_backbone = nn.Sequential(
            nn.Linear(propagation_input_dimension, config.propagation_hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.propagation_dropout),
        )
        self.propagation_membership_head = nn.Linear(
            config.propagation_hidden_dimension, 1
        )

        self.root_cause_projection = nn.Linear(
            config.encoder_dimension, config.relevance_dimension
        )

    def _subsystem_scores(self, embeddings: Tensor) -> Tensor:
        return self.subsystem_attention_vector(
            torch.tanh(self.subsystem_attention_projection(embeddings))
        ).squeeze(-1)

    def _global_scores(self, embeddings: Tensor) -> Tensor:
        return self.global_attention_vector(
            torch.tanh(self.global_attention_projection(embeddings))
        ).squeeze(-1)

    def _pool_incident(
        self,
        embeddings: Tensor,
        observations: Sequence[CanonicalObservation],
    ) -> _PooledIncident:
        """Apply the Appendix-B subsystem and global attention equations."""

        if embeddings.ndim != 2 or embeddings.shape[0] != len(observations):
            raise ValueError("embedding/observation cardinality mismatch")
        if not observations:
            raise ValueError("each incident must contain at least one observation")
        subsystem_ids = tuple(
            sorted(
                {
                    observation.metadata.primary_subsystem or "unknown"
                    for observation in observations
                }
            )
        )
        subsystem_lookup = {name: index for index, name in enumerate(subsystem_ids)}
        assignment = torch.tensor(
            [
                subsystem_lookup[observation.metadata.primary_subsystem or "unknown"]
                for observation in observations
            ],
            dtype=torch.long,
            device=embeddings.device,
        )
        scaled = embeddings
        subsystem_embeddings: list[Tensor] = []
        observation_attention = torch.zeros(
            len(observations), dtype=embeddings.dtype, device=embeddings.device
        )
        for subsystem_index in range(len(subsystem_ids)):
            members = torch.nonzero(
                assignment == subsystem_index, as_tuple=False
            ).squeeze(-1)
            member_embeddings = scaled[members]
            scores = self._subsystem_scores(member_embeddings)
            attention = torch.softmax(scores, dim=0)
            observation_attention[members] = attention
            subsystem_embeddings.append(
                torch.sum(attention.unsqueeze(-1) * member_embeddings, dim=0)
            )
        stacked_subsystems = torch.stack(subsystem_embeddings)
        global_attention = torch.softmax(self._global_scores(stacked_subsystems), dim=0)
        global_embedding = torch.sum(
            global_attention.unsqueeze(-1) * stacked_subsystems, dim=0
        )
        return _PooledIncident(
            subsystem_ids=subsystem_ids,
            observation_subsystem_indices=assignment,
            subsystem_embeddings=stacked_subsystems,
            global_embedding=global_embedding,
            subsystem_attention=observation_attention,
            global_attention=global_attention,
        )

    def _pad_observations(
        self,
        per_incident: Sequence[Tensor],
        maximum: int,
        trailing_dimension: int | None = None,
        *,
        fill: float = 0.0,
    ) -> Tensor:
        batch_size = len(per_incident)
        device = per_incident[0].device
        dtype = per_incident[0].dtype
        if trailing_dimension is None:
            output = torch.full((batch_size, maximum), fill, dtype=dtype, device=device)
        else:
            output = torch.full(
                (batch_size, maximum, trailing_dimension),
                fill,
                dtype=dtype,
                device=device,
            )
        for index, value in enumerate(per_incident):
            output[index, : value.shape[0]] = value
        return output

    def forward(self, batch: IncidentBatch) -> ModelOutput:
        if not isinstance(batch, IncidentBatch):
            raise TypeError(
                "StateBundleModel.forward accepts observable IncidentBatch only"
            )
        if not batch.incidents:
            raise ValueError("incident batch must not be empty")
        if any(not incident.observations for incident in batch.incidents):
            raise ValueError(
                "every incident requires at least one canonical observation"
            )

        observations_by_incident = [
            incident.observations for incident in batch.incidents
        ]
        flat_observations = [
            observation
            for observations in observations_by_incident
            for observation in observations
        ]
        flat_embeddings = self.modality_encoders(flat_observations)
        per_incident_embeddings: list[Tensor] = []
        offset = 0
        for observations in observations_by_incident:
            per_incident_embeddings.append(
                flat_embeddings[offset : offset + len(observations)]
            )
            offset += len(observations)

        pooled = [
            self._pool_incident(
                embeddings,
                observations,
            )
            for embeddings, observations in zip(
                per_incident_embeddings, observations_by_incident
            )
        ]
        maximum_observations = max(len(value) for value in observations_by_incident)
        maximum_subsystems = max(len(value.subsystem_ids) for value in pooled)
        batch_size = len(batch.incidents)
        device = flat_embeddings.device

        observation_mask = torch.zeros(
            (batch_size, maximum_observations), dtype=torch.bool, device=device
        )
        subsystem_mask = torch.zeros(
            (batch_size, maximum_subsystems), dtype=torch.bool, device=device
        )
        for index, (observations, value) in enumerate(
            zip(observations_by_incident, pooled)
        ):
            observation_mask[index, : len(observations)] = True
            subsystem_mask[index, : len(value.subsystem_ids)] = True

        observation_embeddings = self._pad_observations(
            per_incident_embeddings,
            maximum_observations,
            self.config.shared_dimension,
        )
        subsystem_embeddings = self._pad_observations(
            [value.subsystem_embeddings for value in pooled],
            maximum_subsystems,
            self.config.shared_dimension,
        )
        global_embeddings = torch.stack([value.global_embedding for value in pooled])
        observation_subsystem_indices = self._pad_observations(
            [value.observation_subsystem_indices for value in pooled],
            maximum_observations,
            fill=-1,
        ).to(torch.long)
        relevance_embeddings = F.normalize(
            self.relevance_projection(observation_embeddings), dim=-1
        )
        aspect_embeddings = F.normalize(
            self.aspect_projection(observation_embeddings), dim=-1
        )
        subsystem_relevance_embeddings = F.normalize(
            self.subsystem_relevance_projection(subsystem_embeddings), dim=-1
        )
        global_relevance_embeddings = F.normalize(
            self.global_relevance_projection(global_embeddings), dim=-1
        )

        expanded_global = global_embeddings.unsqueeze(1).expand(
            -1, maximum_subsystems, -1
        )
        propagation_features = torch.cat(
            (
                subsystem_embeddings,
                expanded_global,
                subsystem_embeddings * expanded_global,
            ),
            dim=-1,
        )
        propagation_hidden = self.propagation_backbone(propagation_features)
        propagation_logits = self.propagation_membership_head(
            propagation_hidden
        ).squeeze(-1)
        propagation_logits = propagation_logits.masked_fill(~subsystem_mask, 0.0)
        propagation_memberships = torch.sigmoid(propagation_logits) * subsystem_mask

        safe_assignment = observation_subsystem_indices.clamp_min(0)
        gather_index = safe_assignment.unsqueeze(-1).expand(
            -1, -1, self.config.relevance_dimension
        )
        observation_subsystem_relevance = subsystem_relevance_embeddings.gather(
            1, gather_index
        )
        observation_propagation = propagation_memberships.gather(1, safe_assignment)
        global_similarity = torch.sum(
            relevance_embeddings * global_relevance_embeddings.unsqueeze(1), dim=-1
        )
        subsystem_similarity = torch.sum(
            relevance_embeddings * observation_subsystem_relevance, dim=-1
        )
        relevance_scores = (
            self.config.global_relevance_weight * global_similarity
            + self.config.subsystem_relevance_weight * subsystem_similarity
            + self.config.propagation_relevance_weight * observation_propagation
        )
        relevance_scores = relevance_scores.masked_fill(~observation_mask, 0.0)
        return ModelOutput(
            aspect_embeddings=aspect_embeddings, relevance_scores=relevance_scores
        )
