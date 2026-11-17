#!/usr/bin/env python3
"""
Telemetry Validator: Validates atomic factual claims against ground-truth
CHARMIncident dataclass records and raw continuous flow telemetry.
Labels each claim as: SUPPORTED, CONTRADICTED, UNSUPPORTED, or UNVERIFIABLE.
"""
from typing import Dict, Any, List
from .claim_extractor import extract_entities_from_text, ExtractedTelemetryEntities

def validate_claim_against_incident(
    claim_text: str,
    claim_type: str,
    incident: Any,
    cited_evidence: List[Any] = None
) -> Dict[str, Any]:
    """
    Validates a claim proposition against incident ground truth and retrieved evidence.
    Returns:
        dict with validation_label, entity_matches, entity_errors, verifiable_count
    """
    entities = extract_entities_from_text(claim_text)
    cited_evidence = cited_evidence or []

    entity_matches = 0
    entity_errors = 0
    verifiable_count = 0

    # 1. Evaluate Observed Telemetry Facts
    if claim_type == "OBSERVED_FACT":
        # Check hybrid score
        if entities.hybrid_score >= 0:
            verifiable_count += 1
            if abs(entities.hybrid_score - incident.anomaly_score) < 0.05:
                entity_matches += 1
            else:
                entity_errors += 1

        # Check threshold
        if entities.threshold >= 0:
            verifiable_count += 1
            if abs(entities.threshold - incident.decision_threshold) < 0.05:
                entity_matches += 1
            else:
                entity_errors += 1

        # Check reconstruction error
        if entities.reconstruction_error >= 0:
            verifiable_count += 1
            if abs(entities.reconstruction_error - incident.reconstruction_error) < 0.05:
                entity_matches += 1
            else:
                entity_errors += 1

        # Check Mahalanobis distance
        if entities.mahalanobis_distance >= 0:
            verifiable_count += 1
            if abs(entities.mahalanobis_distance - incident.mahalanobis_distance) < 0.5:
                entity_matches += 1
            else:
                entity_errors += 1

        # Check features
        top_feature_names = [f.feature for f in incident.top_xai_features]
        for feat_name in entities.features.keys():
            verifiable_count += 1
            if feat_name in top_feature_names:
                entity_matches += 1
            else:
                entity_errors += 1

        if entity_errors > 0:
            label = "CONTRADICTED"
        elif entity_matches > 0:
            label = "SUPPORTED"
        else:
            # Check general text overlap with incident triggers
            if any(t in claim_text for t in incident.trigger_reasons):
                label = "SUPPORTED"
                verifiable_count += 1
                entity_matches += 1
            else:
                label = "UNSUPPORTED"

    # 2. Evaluate Evidence Interpretation Claims
    elif claim_type == "EVIDENCE_INTERPRETATION":
        # Must be grounded in cited evidence documents
        has_evidence_support = any(
            any(word in ev.text.lower() for word in claim_text.lower().split() if len(word) > 4)
            for ev in cited_evidence
        )
        if has_evidence_support:
            label = "SUPPORTED"
        else:
            label = "UNSUPPORTED"

    # 3. Evaluate Operational Hypotheses
    elif claim_type in ("HYPOTHESIS", "OPERATIONAL_HYPOTHESIS"):
        # An operational hypothesis without explicit uncertainty tags or unverified against KB is UNSUPPORTED.
        # If it carries an explicit uncertainty flag, it is classified as a valid bounded hypothesis.
        has_uncertainty_badge = (
            "[OPERATIONAL HYPOTHESIS]" in claim_text or
            "[UNVERIFIED" in claim_text or
            "cannot be conclusively attributed" in claim_text or
            "Further investigation is recommended" in claim_text
        )
        if has_uncertainty_badge:
            label = "SUPPORTED"  # Properly guarded with epistemic uncertainty
            verifiable_count += 1
            entity_matches += 1
        else:
            label = "UNSUPPORTED"
            verifiable_count += 1
            entity_errors += 1

    else:
        label = "UNVERIFIABLE"

    return {
        'claim_text': claim_text,
        'claim_type': claim_type,
        'validation_label': label,
        'entity_matches': entity_matches,
        'entity_errors': entity_errors,
        'verifiable_count': verifiable_count
    }
