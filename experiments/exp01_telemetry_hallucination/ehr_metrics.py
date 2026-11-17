#!/usr/bin/env python3
"""
Quantitative Hallucination Metrics for Network Intrusion Incident Explanations.
Computes:
- Entity Hallucination Rate (EHR)
- Unsupported Claim Rate (UCR)
- Contradiction Rate (CR)
- Evidence Coverage (EC)
"""
from typing import List, Dict, Any

def compute_incident_metrics(claim_evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes summary hallucination metrics from a list of validated claims."""
    total_claims = len(claim_evaluations)
    if total_claims == 0:
        return {'ehr': 0.0, 'ucr': 0.0, 'cr': 0.0, 'ec': 1.0}

    total_verifiable_entities = sum(c['verifiable_count'] for c in claim_evaluations)
    total_entity_errors = sum(c['entity_errors'] for c in claim_evaluations)

    supported_count = sum(1 for c in claim_evaluations if c['validation_label'] == 'SUPPORTED')
    contradicted_count = sum(1 for c in claim_evaluations if c['validation_label'] == 'CONTRADICTED')
    unsupported_count = sum(1 for c in claim_evaluations if c['validation_label'] in ('UNSUPPORTED', 'CONTRADICTED'))

    # 1. Entity Hallucination Rate (EHR)
    if total_verifiable_entities > 0:
        ehr = total_entity_errors / total_verifiable_entities
    else:
        ehr = 0.0

    # 2. Unsupported Claim Rate (UCR)
    ucr = unsupported_count / total_claims

    # 3. Contradiction Rate (CR)
    cr = contradicted_count / total_claims

    # 4. Evidence Coverage (EC)
    ec = supported_count / total_claims

    return {
        'ehr': float(ehr),
        'ucr': float(ucr),
        'cr': float(cr),
        'ec': float(ec),
        'total_claims': total_claims,
        'supported_claims': supported_count,
        'contradicted_claims': contradicted_count,
        'unsupported_claims': unsupported_count,
        'total_entities': total_verifiable_entities,
        'entity_errors': total_entity_errors
    }
