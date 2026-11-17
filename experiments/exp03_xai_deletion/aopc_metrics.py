#!/usr/bin/env python3
"""
AOPC & Faithfulness Metrics Module for Explainable AI in NIDS.
Computes:
- Area Over the Perturbation Curve (AOPC@K)
- Comprehensiveness@k
- Sufficiency@k
- Rank Stability under Gaussian noise
"""
from typing import List, Dict, Any
import numpy as np
from scipy.stats import spearmanr, kendalltau

def compute_aopc(base_score: float, step_scores: List[float]) -> float:
    """
    Computes Area Over the Perturbation Curve:
    AOPC(K) = (1/K) * sum_{k=1}^K [s(x) - s(x^{(k)})]
    Positive AOPC indicates that deleting the top features steadily reduces the anomaly score.
    """
    K = len(step_scores)
    if K == 0:
        return 0.0
    drops = [base_score - s for s in step_scores]
    return float(np.mean(drops))

def compute_comprehensiveness(base_score: float, score_without_top_k: float) -> float:
    """Comprehensiveness: score drop when top-k features are removed."""
    return float(base_score - score_without_top_k)

def compute_sufficiency(base_score: float, score_with_only_top_k: float) -> float:
    """Sufficiency: score difference between full input and input with ONLY top-k features."""
    return float(base_score - score_with_only_top_k)

def evaluate_rank_stability(
    orig_attr: np.ndarray,
    noisy_attr: np.ndarray
) -> Dict[str, float]:
    """Computes Spearman rho and Kendall tau rank correlation under input perturbation."""
    rho, _ = spearmanr(orig_attr, noisy_attr)
    tau, _ = kendalltau(orig_attr, noisy_attr)
    return {
        'spearman_rho': float(rho),
        'kendall_tau': float(tau)
    }
