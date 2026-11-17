#!/usr/bin/env python3
"""
Structured Incident Adapter for PAADU/CHARM.
Converts raw numerical anomaly scoring and XAI feature attributions from the
original CHARM engine into a typed, validated incident record.
Preserves complete raw telemetry for scientific traceability.
"""
from __future__ import annotations

import uuid
import datetime
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass(frozen=True)
class FeatureAttributionRecord:
    feature: str
    hybrid_contribution: float
    reconstruction_contribution: float
    latent_contribution: float
    observed_mean: float
    benign_mean: float
    standardized_deviation: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureAttributionRecord:
        return cls(
            feature=str(data.get('feature', 'unknown')),
            hybrid_contribution=float(data.get('hybrid_contribution', 0.0)),
            reconstruction_contribution=float(data.get('reconstruction_contribution', 0.0)),
            latent_contribution=float(data.get('latent_contribution', 0.0)),
            observed_mean=float(data.get('observed_mean', 0.0)),
            benign_mean=float(data.get('benign_mean', 0.0)),
            standardized_deviation=float(data.get('standardized_deviation', 0.0))
        )

@dataclass
class CHARMIncident:
    incident_id: str
    model: str
    input_reference: str
    anomaly_detected: bool
    anomaly_score: float
    decision_threshold: float
    trigger_reasons: List[str]
    reconstruction_error: float
    mahalanobis_distance: float
    reconstruction_z: float
    latent_z: float
    top_xai_features: List[FeatureAttributionRecord]
    xai_summary_text: Optional[str]
    timestamp: str
    model_checkpoint_hash: str
    feature_schema: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_charm_output: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['top_xai_features'] = [asdict(f) for f in self.top_xai_features]
        return d

    def validate(self) -> bool:
        """Enforces schema consistency and non-negativity where mathematically required."""
        if not self.incident_id:
            raise ValueError("incident_id cannot be empty")
        if not isinstance(self.anomaly_detected, bool):
            raise TypeError("anomaly_detected must be boolean")
        if self.decision_threshold < 0:
            raise ValueError("decision_threshold must be non-negative")
        if self.reconstruction_error < 0:
            raise ValueError("reconstruction_error must be non-negative")
        if self.mahalanobis_distance < 0:
            raise ValueError("mahalanobis_distance must be non-negative")
        return True


class CHARMIncidentAdapter:
    """Transforms raw dictionary outputs of CHARMEngine and XAIInterpreter into CHARMIncident."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_hash = ""
        if checkpoint_path:
            self.checkpoint_hash = self._compute_file_hash(checkpoint_path)

    @staticmethod
    def _compute_file_hash(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    def create_incident(
        self,
        charm_record: Dict[str, Any],
        block_identifier: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        checkpoint_hash: Optional[str] = None
    ) -> CHARMIncident:
        """
        Converts one CHARM evaluation output (with decision and XAI explanation)
        into a validated CHARMIncident object.
        """
        idx = charm_record.get('index', 0)
        ref = block_identifier or f"telemetry_block_{idx}"
        inc_id = f"INC-CHARM-{uuid.uuid4().hex[:8].upper()}"
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        attributions = []
        raw_attrs = charm_record.get('feature_attributions', [])
        for a in raw_attrs:
            attributions.append(FeatureAttributionRecord.from_dict(a))

        causes = [str(c) for c in charm_record.get('causes', [])]
        
        incident = CHARMIncident(
            incident_id=inc_id,
            model="PAADU/CHARM",
            input_reference=str(ref),
            anomaly_detected=bool(charm_record.get('anomaly', False)),
            anomaly_score=float(charm_record.get('hybrid_score', 0.0)),
            decision_threshold=float(charm_record.get('threshold', 0.0)),
            trigger_reasons=causes,
            reconstruction_error=float(charm_record.get('reconstruction_error', 0.0)),
            mahalanobis_distance=float(charm_record.get('mahalanobis', 0.0)),
            reconstruction_z=float(charm_record.get('reconstruction_z', 0.0)),
            latent_z=float(charm_record.get('latent_z', 0.0)),
            top_xai_features=attributions,
            xai_summary_text=charm_record.get('explanation'),
            timestamp=ts,
            model_checkpoint_hash=checkpoint_hash or self.checkpoint_hash,
            feature_schema="CICIoT2023_46_features",
            metadata=custom_metadata or charm_record.get('metadata', {}),
            raw_charm_output=dict(charm_record)
        )
        incident.validate()
        return incident
