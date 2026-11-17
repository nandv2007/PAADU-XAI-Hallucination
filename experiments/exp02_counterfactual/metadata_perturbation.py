#!/usr/bin/env python3
"""
Metadata Perturbation Module for Counterfactual Explanation Stability Testing.
Generates controlled, non-causal perturbations:
1. Sensor ID shifts (e.g., sensor-node-001 -> sensor-node-999)
2. Block identifier & timestamp shifts
3. Non-causal / zero-attribution continuous feature jitter (features with zero XAI weight)
"""
import copy
from typing import Dict, Any, Tuple
import numpy as np

def apply_metadata_intervention(
    block: np.ndarray,
    metadata: Dict[str, Any],
    intervention_type: str,
    zero_attribution_features: list = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Applies a justified non-causal counterfactual intervention.
    Returns:
        (perturbed_block, perturbed_metadata)
    """
    cf_block = block.copy()
    cf_meta = copy.deepcopy(metadata)

    if intervention_type == 'sensor_id':
        cf_meta['sensor_id'] = f"sensor-cf-{metadata.get('sensor_id', 'node')[-3:]}"
        cf_meta['intervention'] = 'sensor_id_scramble'

    elif intervention_type == 'timestamp':
        orig_ts = metadata.get('timestamp', '2026-11-14T10:00:00Z')
        cf_meta['timestamp'] = orig_ts.replace('10:', '18:').replace('11:', '19:')
        cf_meta['block_id'] = f"{metadata['block_id']}-CF-TIME"
        cf_meta['intervention'] = 'timestamp_shift'

    elif intervention_type == 'zero_attribution_jitter':
        # Add slight jitter only to features that had zero XAI contribution
        cf_meta['intervention'] = 'zero_attribution_feature_jitter'
        if zero_attribution_features:
            for feat_idx in zero_attribution_features[:3]:
                cf_block[0, 0, :, feat_idx] += 1e-4

    else:
        cf_meta['block_id'] = f"{metadata['block_id']}-CF"
        cf_meta['intervention'] = 'block_id_relabel'

    return cf_block, cf_meta
