#!/usr/bin/env python3
"""
Counterfactual Stability Metrics for Explainable NIDS.
Computes:
- Narrative Variance Score (NVS)
- Attack Attribution Change Rate (AACR)
- Counterfactual Consistency Rate (CCR)
- Unsupported Claim Introduction Rate (UCIR)
"""
from typing import List, Dict, Any
import numpy as np

def compute_stability_summary(pair_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes summary counterfactual stability metrics over all evaluated stable pairs."""
    total_pairs = len(pair_evaluations)
    if total_pairs == 0:
        return {'mean_nvs': 0.0, 'aacr': 0.0, 'ccr': 1.0, 'ucir': 0.0}

    nvs_values = [p['comparison']['narrative_variance_score'] for p in pair_evaluations]
    attribution_flips = sum(1 for p in pair_evaluations if p['attribution_flipped'])
    consistent_pairs = sum(1 for p in pair_evaluations if p['comparison']['narrative_variance_score'] == 0.0)
    pairs_with_new_claims = sum(1 for p in pair_evaluations if p['comparison']['new_claims_count'] > 0)

    mean_nvs = float(np.mean(nvs_values))
    std_nvs = float(np.std(nvs_values))
    aacr = float(attribution_flips / total_pairs)
    ccr = float(consistent_pairs / total_pairs)
    ucir = float(pairs_with_new_claims / total_pairs)

    return {
        'total_stable_pairs_evaluated': total_pairs,
        'mean_narrative_variance_score': mean_nvs,
        'std_narrative_variance_score': std_nvs,
        'attack_attribution_change_rate': aacr,
        'counterfactual_consistency_rate': ccr,
        'unsupported_claim_introduction_rate': ucir,
        'zero_flip_guarantee': bool(aacr == 0.0)
    }
