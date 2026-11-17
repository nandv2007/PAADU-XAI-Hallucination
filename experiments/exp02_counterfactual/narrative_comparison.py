#!/usr/bin/env python3
"""
Narrative Comparison Module: Compares original and counterfactual incident explanations.
Standardizes claim propositions, computes set intersections and differences,
and checks for threat classification flips.
"""
import re
from typing import List, Set, Dict, Any

def normalize_claim_proposition(text: str) -> str:
    """Normalizes claim text to ignore superficial whitespace, incident IDs, and timestamps."""
    t = text.lower()
    t = re.sub(r'incident\s+inc-charm-[a-f0-9]+', 'incident', t)
    t = re.sub(r'sensor-node-[0-9]+', 'sensor', t)
    t = re.sub(r'sensor-cf-[0-9]+', 'sensor', t)
    t = re.sub(r'2026-11-14t[0-9:]+z', 'timestamp', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_attack_attribution(explanation_text: str) -> str:
    """Extracts inferred attack attribution family from explanation text."""
    t = explanation_text.lower()
    if 'syn flood' in t or 'tcp syn' in t:
        return 'TCP_SYN_FLOOD'
    elif 'volumetric' in t or 'byte flood' in t or 'rate flood' in t:
        return 'VOLUMETRIC_FLOOD'
    elif 'reconnaissance' in t or 'port scan' in t or 'host scan' in t:
        return 'RECONNAISSANCE_SCAN'
    elif 'benign' in t or 'normal' in t:
        return 'BENIGN'
    return 'UNKNOWN_ANOMALY'

def compare_narrative_claims(
    orig_claims: List[str],
    cf_claims: List[str]
) -> Dict[str, Any]:
    """
    Compares two lists of claims from original and counterfactual explanations.
    Returns:
        dict with intersection_count, union_count, jaccard_similarity, variance_score
    """
    orig_set = set(normalize_claim_proposition(c) for c in orig_claims)
    cf_set = set(normalize_claim_proposition(c) for c in cf_claims)

    intersection = orig_set & cf_set
    union = orig_set | cf_set

    if len(union) == 0:
        jaccard = 1.0
        nvs = 0.0
    else:
        jaccard = len(intersection) / len(union)
        nvs = 1.0 - jaccard

    new_claims_in_cf = cf_set - orig_set

    return {
        'orig_claim_count': len(orig_set),
        'cf_claim_count': len(cf_set),
        'intersection_count': len(intersection),
        'union_count': len(union),
        'jaccard_similarity': float(jaccard),
        'narrative_variance_score': float(nvs),
        'new_claims_count': len(new_claims_in_cf)
    }
