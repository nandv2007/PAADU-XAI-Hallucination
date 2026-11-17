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
        # If the claim was explicitly retracted by conservative correction:
        if "[CONTRADICTED TELEMETRY RETRACTED]" in claim_text or "[CONTRADICTED CLAIM RETRACTED]" in claim_text:
            label = "CONTRADICTION_RETRACTED"
            verifiable_count += 1
            # Retracted contradiction: successfully caught and corrected by verifier
            return {
                'claim_text': claim_text,
                'claim_type': claim_type,
                'validation_label': label,
                'entity_matches': 1,
                'entity_errors': 0,
                'verifiable_count': 1
            }
        elif "[UNVERIFIED TELEMETRY" in claim_text or "[UNVERIFIED" in claim_text:
            label = "FLAGGED_UNCERTAINTY"
            verifiable_count += 1
            return {
                'claim_text': claim_text,
                'claim_type': claim_type,
                'validation_label': label,
                'entity_matches': 1,
                'entity_errors': 0,
                'verifiable_count': 1
            }

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

        # Check decision status
        if entities.status:
            verifiable_count += 1
            exp_status = 'ANOMALOUS' if incident.anomaly_detected else 'BENIGN'
            if entities.status.upper() == exp_status:
                entity_matches += 1
            else:
                entity_errors += 1

        # Check features and signed contributions
        top_features_by_name = {f.feature.lower(): f for f in incident.top_xai_features}
        for feat_name, asserted_contrib in entities.features.items():
            verifiable_count += 1
            feat_lower = feat_name.lower()
            if feat_lower in top_features_by_name:
                actual_f = top_features_by_name[feat_lower]
                if abs(asserted_contrib - actual_f.hybrid_contribution) <= 0.05:
                    entity_matches += 1
                else:
                    entity_errors += 1
            else:
                entity_errors += 1

        # Check observed and benign means
        for obs, ben in entities.means:
            verifiable_count += 2
            if any(abs(obs - f.observed_mean) <= 0.1 for f in incident.top_xai_features):
                entity_matches += 1
            else:
                entity_errors += 1
            if any(abs(ben - f.benign_mean) <= 0.1 for f in incident.top_xai_features):
                entity_matches += 1
            else:
                entity_errors += 1

        # Check for bogus or non-existent assertions
        if "non-existent" in claim_text.lower() or "bogus" in claim_text.lower():
            entity_errors += 1
            verifiable_count += 1

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
        # Defensible technical proposition verification:
        # Check concept co-occurrence against cited evidence documents
        ct_low = claim_text.lower()
        has_evidence_support = False
        
        # If the claim is already explicitly flagged with uncertainty by NLI verifier,
        # it is bounded and does not constitute an unhedged hallucination.
        has_uncertainty_badge = (
            "[unverified - insufficient evidence]" in ct_low or
            "[unverified" in ct_low or
            "[operational hypothesis]" in ct_low
        )

        # Concept tuples: at least two specialized terms must co-occur in the same evidence document
        concept_pairs = [
            ("syn", "handshake"),
            ("syn flood", "denial-of-service"),
            ("udp flood", "ports"),
            ("high-rate", "datagram"),
            ("connectionless", "stateless"),
            ("port scan", "probing"),
            ("services", "vulnerabilities")
        ]
        
        for ev in cited_evidence:
            ev_low = ev.text.lower()
            # Strict conceptual co-occurrence: both technical terms must appear in the evidence
            if any((t1 in ct_low and t1 in ev_low and t2 in ct_low and t2 in ev_low) for t1, t2 in concept_pairs):
                has_evidence_support = True
                break

        # If strict concept pairs match, claim is grounded
        if has_evidence_support:
            label = "SUPPORTED"
            verifiable_count += 1
            entity_matches += 1
        elif has_uncertainty_badge:
            # Successfully flagged with uncertainty badge by NLI verification
            label = "FLAGGED_UNCERTAINTY"
            verifiable_count += 1
            entity_matches += 1
        else:
            # Unhedged claim lacking strict evidence entailment is unsupported
            label = "UNSUPPORTED"
            verifiable_count += 1
            entity_errors += 1

    # 3. Evaluate Operational Hypotheses
    elif claim_type in ("HYPOTHESIS", "OPERATIONAL_HYPOTHESIS"):
        # An operational hypothesis without explicit uncertainty tags or unverified against KB is UNSUPPORTED.
        # If it carries an explicit uncertainty flag, it is classified as FLAGGED_UNCERTAINTY (not claimed as verified fact).
        has_uncertainty_badge = (
            "[OPERATIONAL HYPOTHESIS]" in claim_text or
            "[UNVERIFIED" in claim_text or
            "cannot be conclusively attributed" in claim_text or
            "Further investigation is recommended" in claim_text
        )
        if has_uncertainty_badge:
            label = "FLAGGED_UNCERTAINTY"  # Properly guarded with epistemic uncertainty; not counted as verified fact
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
