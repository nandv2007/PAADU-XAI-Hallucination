"""Experiment 3: XAI Faithfulness & AOPC Perturbation Evaluation."""
from .feature_intervention import compute_occlusion_vector, apply_feature_deletion, apply_feature_retention
from .aopc_metrics import compute_aopc, compute_comprehensiveness, compute_sufficiency, evaluate_rank_stability
