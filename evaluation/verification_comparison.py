#!/usr/bin/env python3
"""
Verification Comparison Module: Detailed Claim-Level Verification Analysis.
Evaluates claim categorization, contradiction detection rates, and
conservative correction retention rates before and after NLI verification.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

def analyze_verification_performance(results_json_path: Path) -> Dict[str, Any]:
    """Analyzes detailed verification metrics from Experiment 1 results."""
    with open(results_json_path, 'r') as f:
        data = json.load(f)

    unv = data['unverified_configuration_B']
    ver = data['verified_configuration_D']
    rel = data['relative_reductions']

    summary = {
        'total_evaluated_incidents': data['total_incidents_evaluated'],
        'unsupported_claim_rate_before': unv['mean_ucr'],
        'unsupported_claim_rate_after': ver['mean_ucr'],
        'unsupported_reduction_percent': rel['ucr_reduction_percent'],
        'entity_hallucination_rate_before': unv['mean_ehr'],
        'entity_hallucination_rate_after': ver['mean_ehr'],
        'evidence_coverage_before': unv['mean_ec'],
        'evidence_coverage_after': ver['mean_ec'],
        'coverage_improvement_percent': rel['evidence_coverage_improvement_percent'],
        'contradiction_elimination_rate': 100.0 if ver['mean_cr'] == 0.0 else (1.0 - ver['mean_cr']) * 100.0
    }
    return summary

def main():
    json_path = PACKAGE_ROOT / 'results' / 'EXPERIMENT_1_METRICS.json'
    if not json_path.exists():
        print(f"Error: {json_path} does not exist. Run experiment 1 first.")
        sys.exit(1)

    perf = analyze_verification_performance(json_path)
    print("=" * 72)
    print("  VERIFICATION PERFORMANCE SUMMARY")
    print("=" * 72)
    for k, v in perf.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
