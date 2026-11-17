#!/usr/bin/env python3
"""
Statistical Evaluation Module: Computes formal significance tests,
confidence intervals, effect sizes (Cohen's d), and Wilcoxon signed-rank tests
across all experimental modules.
"""
import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from scipy import stats

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cohen's d effect size for paired samples."""
    diff = x - y
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))

def bootstrap_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> tuple:
    """Computes non-parametric bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)

def run_statistical_analysis() -> Dict[str, Any]:
    print("=" * 72)
    print("  RUNNING STATISTICAL SIGNIFICANCE TESTS & EFFECT SIZE COMPUTATIONS")
    print("=" * 72)

    stat_results = {}

    # 1. Experiment 1: UCR Reduction Wilcoxon Signed-Rank Test
    exp1_csv = PACKAGE_ROOT / 'results' / 'EXPERIMENT_1_RESULTS.csv'
    if exp1_csv.exists():
        with open(exp1_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        unv_ucr = np.array([float(r['unverified_ucr']) for r in rows])
        ver_ucr = np.array([float(r['verified_ucr']) for r in rows])

        w_stat, w_pval = stats.wilcoxon(unv_ucr, ver_ucr)
        d_ucr = compute_cohens_d(unv_ucr, ver_ucr)
        ci_unv = bootstrap_ci(unv_ucr)
        ci_ver = bootstrap_ci(ver_ucr)

        stat_results['experiment_1_ucr'] = {
            'wilcoxon_statistic': float(w_stat),
            'p_value': float(w_pval),
            'statistically_significant': bool(w_pval < 0.001),
            'cohens_d': d_ucr,
            'unverified_95ci': ci_unv,
            'verified_95ci': ci_ver
        }
        print(f"  Exp 1 UCR Reduction: Wilcoxon W={w_stat:.1f}, p={w_pval:.2e}, Cohen's d={d_ucr:.2f}")

    # 2. Experiment 3: AOPC Top-k vs Random Paired t-test
    exp3_csv = PACKAGE_ROOT / 'results' / 'EXPERIMENT_3_RESULTS.csv'
    if exp3_csv.exists():
        with open(exp3_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        aopc_top = np.array([float(r['aopc_top10']) for r in rows])
        aopc_rand = np.array([float(r['aopc_random10']) for r in rows])

        t_stat, t_pval = stats.ttest_rel(aopc_top, aopc_rand)
        w_stat, w_pval = stats.wilcoxon(aopc_top, aopc_rand)
        d_aopc = compute_cohens_d(aopc_top, aopc_rand)

        stat_results['experiment_3_aopc'] = {
            'paired_t_statistic': float(t_stat),
            'paired_t_p_value': float(t_pval),
            'wilcoxon_statistic': float(w_stat),
            'wilcoxon_p_value': float(w_pval),
            'statistically_significant': bool(t_pval < 0.001),
            'cohens_d': d_aopc,
            'top_k_95ci': bootstrap_ci(aopc_top),
            'random_95ci': bootstrap_ci(aopc_rand)
        }
        print(f"  Exp 3 AOPC Faithfulness: Paired t={t_stat:.2f}, p={t_pval:.2e}, Cohen's d={d_aopc:.2f}")

    # Save to both recomputed_results and results
    for out_dir in [PACKAGE_ROOT / 'recomputed_results', PACKAGE_ROOT / 'results']:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'STATISTICAL_TESTS.json', 'w') as f:
            json.dump(stat_results, f, indent=2)

    print(f"✓ Statistical evaluation saved to STATISTICAL_TESTS.json")
    return stat_results

if __name__ == '__main__':
    run_statistical_analysis()
