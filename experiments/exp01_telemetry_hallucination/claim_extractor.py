#!/usr/bin/env python3
"""
Atomic Factual Claim Extractor for Network Telemetry Incident Explanations.
Segments generated incident narratives into discrete verifiable propositions
and extracts numerical/categorical entities for telemetry fact-checking.
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ExtractedTelemetryEntities:
    status: str = ""
    hybrid_score: float = -1.0
    threshold: float = -1.0
    reconstruction_error: float = -1.0
    mahalanobis_distance: float = -1.0
    features: Dict[str, float] = field(default_factory=dict)
    means: List[tuple] = field(default_factory=list)
    raw_entities: Dict[str, Any] = field(default_factory=dict)

def extract_entities_from_text(claim_text: str) -> ExtractedTelemetryEntities:
    """Extracts verifiable telemetry entities from an atomic claim proposition."""
    entities = ExtractedTelemetryEntities()
    
    # Anomaly classification status
    status_match = re.search(r'evaluated \S+ as (ANOMALOUS|BENIGN)', claim_text, re.IGNORECASE)
    if status_match:
        entities.status = status_match.group(1).upper()
        entities.raw_entities['status'] = entities.status

    # Hybrid anomaly score
    score_match = re.search(r'score of ([0-9]+\.[0-9]+)', claim_text)
    if score_match:
        entities.hybrid_score = float(score_match.group(1))
        entities.raw_entities['hybrid_score'] = entities.hybrid_score

    # Decision threshold
    thresh_match = re.search(r'threshold of ([0-9]+\.[0-9]+)', claim_text)
    if thresh_match:
        entities.threshold = float(thresh_match.group(1))
        entities.raw_entities['threshold'] = entities.threshold

    # Reconstruction error
    recon_match = re.search(r'reconstruction error of ([0-9]+\.[0-9]+)', claim_text)
    if recon_match:
        entities.reconstruction_error = float(recon_match.group(1))
        entities.raw_entities['reconstruction_error'] = entities.reconstruction_error

    # Mahalanobis latent distance
    mahal_match = re.search(r'Mahalanobis distance of ([0-9]+\.[0-9]+)', claim_text)
    if mahal_match:
        entities.mahalanobis_distance = float(mahal_match.group(1))
        entities.raw_entities['mahalanobis_distance'] = entities.mahalanobis_distance

    # Top occlusion features and attributions (supports multi-word features like 'Protocol Type', 'Tot sum', 'Tot size')
    feat_matches = re.findall(r'(?:as:\s*|;\s*)?([a-zA-Z0-9_\- ]+?)\s*\(contribution:\s*([+-]?[0-9]+\.[0-9]+)', claim_text)
    if feat_matches:
        entities.features = {f[0].strip(): float(f[1]) for f in feat_matches if f[0].strip()}
        entities.raw_entities['features'] = entities.features

    # Observed vs benign mean comparisons
    obs_matches = re.findall(r'observed mean:\s*([0-9]+\.[0-9]+)\s*vs benign mean:\s*([0-9]+\.[0-9]+)', claim_text)
    if obs_matches:
        entities.means = [(float(m[0]), float(m[1])) for m in obs_matches]
        entities.raw_entities['means'] = entities.means

    return entities
