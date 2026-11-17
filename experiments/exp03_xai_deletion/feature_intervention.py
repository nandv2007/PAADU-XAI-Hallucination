#!/usr/bin/env python3
"""
Feature Intervention Module for XAI Faithfulness Evaluation.
Extracts CHARM's deterministic leave-one-feature-out occlusion attribution vector
and applies feature deletion (masking to benign baseline 0.0) across:
- Top-k features
- Random feature baseline
- Low-importance / bottom-k features
"""
from typing import List, Tuple
import numpy as np

def compute_occlusion_vector(engine, block: np.ndarray) -> Tuple[float, np.ndarray]:
    """Computes base hybrid anomaly score and 46-dimensional occlusion vector."""
    x = engine._validate(block)
    normalized = (x[0] - engine.mean[0]) / engine.std[0]
    base_score, _, _ = engine.score_one_components(normalized)

    attr_vec = np.zeros(len(engine.features), dtype=np.float32)
    for j in range(len(engine.features)):
        occluded = normalized.copy()
        occluded[:, :, j] = 0.0  # Occlude to empirical benign mean
        score_j, _, _ = engine.score_one_components(occluded)
        attr_vec[j] = base_score - score_j

    return base_score, attr_vec

def apply_feature_deletion(
    engine,
    block: np.ndarray,
    feature_indices: List[int]
) -> float:
    """Evaluates the anomaly score when specified feature indices are occluded to benign baseline."""
    x = engine._validate(block)
    normalized = (x[0] - engine.mean[0]) / engine.std[0]
    occluded = normalized.copy()
    for idx in feature_indices:
        occluded[:, :, idx] = 0.0

    score, _, _ = engine.score_one_components(occluded)
    return float(score)

def apply_feature_retention(
    engine,
    block: np.ndarray,
    retained_indices: List[int]
) -> float:
    """Evaluates anomaly score when ONLY specified features are retained (all others zeroed to benign mean)."""
    x = engine._validate(block)
    normalized = (x[0] - engine.mean[0]) / engine.std[0]
    # Start with all-benign baseline
    retained = np.zeros_like(normalized)
    for idx in retained_indices:
        retained[:, :, idx] = normalized[:, :, idx]

    score, _, _ = engine.score_one_components(retained)
    return float(score)
