#!/usr/bin/env python3
"""
Attribution Comparison Module for Model Parameter Randomization Checks.
Computes alignment metrics between original trained attributions and randomized attributions:
- Spearman Rank Correlation (rho)
- Kendall Rank Correlation (tau)
- Cosine Similarity in R^46
- Top-k Jaccard Set Overlap (k in {3, 5})
"""
from typing import Dict, Any
import numpy as np
from scipy.stats import spearmanr, kendalltau

def compare_attributions(
    orig_vec: np.ndarray,
    rand_vec: np.ndarray
) -> Dict[str, float]:
    """Computes all comparison metrics between original and randomized attribution vectors."""
    rho, _ = spearmanr(orig_vec, rand_vec)
    tau, _ = kendalltau(orig_vec, rand_vec)

    norm_prod = np.linalg.norm(orig_vec) * np.linalg.norm(rand_vec) + 1e-8
    cos_sim = float(np.dot(orig_vec, rand_vec) / norm_prod)

    # Top-3 Jaccard Overlap
    top3_orig = set(np.argsort(np.abs(orig_vec))[-3:])
    top3_rand = set(np.argsort(np.abs(rand_vec))[-3:])
    jacc_3 = len(top3_orig & top3_rand) / len(top3_orig | top3_rand)

    # Top-5 Jaccard Overlap
    top5_orig = set(np.argsort(np.abs(orig_vec))[-5:])
    top5_rand = set(np.argsort(np.abs(rand_vec))[-5:])
    jacc_5 = len(top5_orig & top5_rand) / len(top5_orig | top5_rand)

    return {
        'spearman_rho': float(rho),
        'kendall_tau': float(tau),
        'cosine_similarity': cos_sim,
        'top3_overlap': float(jacc_3),
        'top5_overlap': float(jacc_5)
    }
