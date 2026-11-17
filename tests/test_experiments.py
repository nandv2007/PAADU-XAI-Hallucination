#!/usr/bin/env python3
"""
Unit and Integration Tests for the Four Experimental Modules.
Tests:
- Test 1: Telemetry Claim Extraction & Validation
- Test 2: Counterfactual Metadata Perturbation & Consistency
- Test 3: Occlusion Vector & AOPC Calculation
- Test 4: Cascading Layer Randomization & Correlation
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ORIG_SYSTEM = PACKAGE_ROOT / 'original_system'
INTEG_PROJECT = ORIG_SYSTEM / 'integrated_project' if ORIG_SYSTEM.exists() else PACKAGE_ROOT / 'integrated_project'
CHARM_ORIG = INTEG_PROJECT / 'charm_original'

for p in [str(PACKAGE_ROOT), str(ORIG_SYSTEM), str(CHARM_ORIG / 'project'), str(CHARM_ORIG / 'project' / 'original'), str(INTEG_PROJECT / 'project')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.exp01_telemetry_hallucination.claim_extractor import extract_entities_from_text
from experiments.exp01_telemetry_hallucination.ehr_metrics import compute_incident_metrics
from experiments.exp02_counterfactual.metadata_perturbation import apply_metadata_intervention
from experiments.exp02_counterfactual.narrative_comparison import compare_narrative_claims
from experiments.exp03_xai_deletion.aopc_metrics import compute_aopc, compute_comprehensiveness, compute_sufficiency
from experiments.exp04_model_randomization.attribution_comparison import compare_attributions

class TestResearchExperiments(unittest.TestCase):

    def test_claim_extractor(self):
        """Test entity extraction from atomic claim text."""
        claim = "PAADU/CHARM evaluated block as ANOMALOUS with score of 42.150 exceeding threshold of 2.500."
        ent = extract_entities_from_text(claim)
        self.assertEqual(ent.status, "ANOMALOUS")
        self.assertAlmostEqual(ent.hybrid_score, 42.150)
        self.assertAlmostEqual(ent.threshold, 2.500)

    def test_ehr_metrics_computation(self):
        """Test EHR, UCR, and Evidence Coverage math."""
        evals = [
            {'validation_label': 'SUPPORTED', 'verifiable_count': 2, 'entity_errors': 0},
            {'validation_label': 'SUPPORTED', 'verifiable_count': 1, 'entity_errors': 0},
            {'validation_label': 'CONTRADICTED', 'verifiable_count': 1, 'entity_errors': 1},
            {'validation_label': 'UNSUPPORTED', 'verifiable_count': 0, 'entity_errors': 0}
        ]
        m = compute_incident_metrics(evals)
        self.assertEqual(m['total_claims'], 4)
        self.assertEqual(m['supported_claims'], 2)
        self.assertEqual(m['contradicted_claims'], 1)
        self.assertEqual(m['unsupported_claims'], 2)
        self.assertAlmostEqual(m['ehr'], 1 / 4)
        self.assertAlmostEqual(m['ucr'], 2 / 4)
        self.assertAlmostEqual(m['ec'], 2 / 4)

    def test_counterfactual_metadata_intervention(self):
        """Test non-causal metadata shift without changing data."""
        block = np.zeros((1, 1, 10, 46), dtype=np.float32)
        meta = {'block_id': 'TEST-01', 'sensor_id': 'sensor-node-001', 'timestamp': '2026-11-14T10:00:00Z'}
        cf_block, cf_meta = apply_metadata_intervention(block, meta, 'sensor_id')

        self.assertEqual(cf_meta['sensor_id'], 'sensor-cf-001')
        self.assertEqual(cf_meta['intervention'], 'sensor_id_scramble')
        np.testing.assert_array_equal(block, cf_block)

    def test_narrative_comparison(self):
        """Test Jaccard narrative variance score."""
        c1 = ["Anomaly detected on TCP", "syn_count elevated"]
        c2 = ["Anomaly detected on TCP", "syn_count elevated"]
        comp = compare_narrative_claims(c1, c2)
        self.assertAlmostEqual(comp['narrative_variance_score'], 0.0)
        self.assertAlmostEqual(comp['jaccard_similarity'], 1.0)

    def test_aopc_calculation(self):
        """Test AOPC, Comprehensiveness, and Sufficiency math."""
        base_score = 100.0
        step_scores = [80.0, 60.0, 40.0]
        aopc = compute_aopc(base_score, step_scores)
        # drops: 20, 40, 60 -> mean is 40.0
        self.assertAlmostEqual(aopc, 40.0)
        self.assertAlmostEqual(compute_comprehensiveness(base_score, 40.0), 60.0)
        self.assertAlmostEqual(compute_sufficiency(base_score, 90.0), 10.0)

    def test_attribution_comparison_identity(self):
        """Test attribution comparison under identity (Spearman rho == 1.0)."""
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = compare_attributions(vec, vec)
        self.assertAlmostEqual(res['spearman_rho'], 1.0)
        self.assertAlmostEqual(res['cosine_similarity'], 1.0)
        self.assertAlmostEqual(res['top3_overlap'], 1.0)

if __name__ == '__main__':
    unittest.main()
