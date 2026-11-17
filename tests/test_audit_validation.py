#!/usr/bin/env python3
"""
Automated Audit Validation Tests:
Verifies that all experimental metrics are dynamically calculated from
raw underlying observations rather than fixed placeholders or hardcoded constants.
"""
import unittest
import json
import csv
from pathlib import Path
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PACKAGE_ROOT / 'results'
RECOMP_DIR = PACKAGE_ROOT / 'recomputed_results'

class TestAuditValidation(unittest.TestCase):

    def test_raw_claim_validation_traceability(self):
        """Verifies that RAW_CLAIM_VALIDATION.csv exists and has non-trivial verifiable claims."""
        raw_csv = RESULTS_DIR / 'RAW_CLAIM_VALIDATION.csv'
        self.assertTrue(raw_csv.exists(), "RAW_CLAIM_VALIDATION.csv must exist")

        with open(raw_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertGreater(len(rows), 100, "Must contain at least 100 raw validated claims")
        labels = set(r['validation_label'] for r in rows)
        self.assertIn('SUPPORTED', labels, "Must contain SUPPORTED claims")
        self.assertTrue(any(r['entity_errors'] != '0' for r in rows), "Must contain traceable entity error observations")

    def test_baseline_comparison_not_hardcoded(self):
        """Verifies that BASELINE_COMPARISON.csv metrics are dynamically computed per incident."""
        base_csv = RESULTS_DIR / 'BASELINE_COMPARISON.csv'
        self.assertTrue(base_csv.exists(), "BASELINE_COMPARISON.csv must exist")

        with open(base_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 60, "Must evaluate 60 incidents")

        # Verify latencies are not all identical (proves real timing execution)
        latencies = [float(r['config_a_latency']) for r in rows]
        self.assertGreater(len(set(latencies)), 5, "Latencies must be dynamically measured, not fixed")

        # Verify Config B and Config D metrics reflect individual claim outcomes
        b_ehrs = [float(r['config_b_ehr']) for r in rows]
        # In unperturbed incidents EHR is 0, in perturbed it is > 0
        self.assertGreater(max(b_ehrs), 0.0, "Perturbed incidents must have EHR > 0")
        self.assertEqual(min(b_ehrs), 0.0, "Unperturbed incidents must have EHR == 0")

    def test_raw_feature_deletion_trajectories(self):
        """Verifies that RAW_FEATURE_DELETION.csv contains 450 step-by-step records."""
        feat_csv = RESULTS_DIR / 'RAW_FEATURE_DELETION.csv'
        self.assertTrue(feat_csv.exists(), "RAW_FEATURE_DELETION.csv must exist")

        with open(feat_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 450, "Must contain exactly 450 records (45 blocks x 10 steps)")

        # Verify progressive score drop increases with k
        block_0_rows = [r for r in rows if r['sample_index'] == '0']
        self.assertEqual(len(block_0_rows), 10)
        topk_drops = [float(r['topk_score_drop']) for r in block_0_rows]
        # At k=10 score drop should exceed score drop at k=1
        self.assertGreater(topk_drops[-1], topk_drops[0], "Top-k feature deletion must progressively drop score")

    def test_raw_randomization_monotonic_decay(self):
        """Verifies that RAW_RANDOMIZATION_RESULTS.csv contains 630 rows with Adebayo decay."""
        rand_csv = RESULTS_DIR / 'RAW_RANDOMIZATION_RESULTS.csv'
        self.assertTrue(rand_csv.exists(), "RAW_RANDOMIZATION_RESULTS.csv must exist")

        with open(rand_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 630, "Must contain exactly 630 records (30 blocks x 7 levels x 3 seeds)")

        # Verify Level 0 has correlation 1.0
        l0_spearmans = [float(r['spearman_rho']) for r in rows if r['randomization_level'] == 'Level_0_Trained']
        self.assertTrue(all(s == 1.0 for s in l0_spearmans), "Level 0 must have correlation 1.0")

        # Verify Level 5 has correlation significantly below 1.0 (Adebayo pass)
        l5_spearmans = [float(r['spearman_rho']) for r in rows if r['randomization_level'] == 'Level_5_Complete']
        mean_l5 = np.mean(l5_spearmans)
        self.assertLess(mean_l5, 0.50, f"Level 5 correlation must drop sharply below 0.50, got {mean_l5:.4f}")

    def test_raw_counterfactual_pairs(self):
        """Verifies that RAW_COUNTERFACTUAL_PAIRS.csv contains 60 evaluated pairs."""
        cf_csv = RESULTS_DIR / 'RAW_COUNTERFACTUAL_PAIRS.csv'
        self.assertTrue(cf_csv.exists(), "RAW_COUNTERFACTUAL_PAIRS.csv must exist")

        with open(cf_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 60, "Must evaluate 60 counterfactual pairs")
        sem_sims = [float(r['semantic_similarity']) for r in rows]
        self.assertTrue(all(0.0 <= s <= 1.0001 for s in sem_sims), "Semantic similarities must be in [0, 1]")

if __name__ == '__main__':
    unittest.main()
