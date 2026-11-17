#!/usr/bin/env python3
"""
Test B — Incident Adapter Test.
Verifies that real CHARM outputs are transformed into a validated, typed CHARMIncident object.
Ensures that all fields match the original inference output and no data is fabricated.
"""
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from unified_integration.adapters.charm_incident_adapter import CHARMIncidentAdapter, CHARMIncident, FeatureAttributionRecord

class TestIncidentAdapter(unittest.TestCase):

    def setUp(self):
        self.adapter = CHARMIncidentAdapter()
        self.sample_charm_record = {
            'index': 3,
            'anomaly': True,
            'causes': ['hybrid_threshold', 'dual_evidence'],
            'hybrid_score': 45.1234,
            'threshold': 2.5000,
            'reconstruction_error': 1.4521,
            'mahalanobis': 12.8765,
            'reconstruction_z': 3.12,
            'latent_z': 4.25,
            'feature_attributions': [
                {
                    'feature': 'Rate',
                    'hybrid_contribution': 14.25,
                    'reconstruction_contribution': 3.10,
                    'latent_contribution': 11.15,
                    'observed_mean': 15200.0,
                    'benign_mean': 120.0,
                    'standardized_deviation': 18.5
                },
                {
                    'feature': 'Tot sum',
                    'hybrid_contribution': 9.80,
                    'reconstruction_contribution': 2.40,
                    'latent_contribution': 7.40,
                    'observed_mean': 84000.0,
                    'benign_mean': 1500.0,
                    'standardized_deviation': 12.1
                }
            ],
            'explanation': 'Anomalous traffic block detected.'
        }

    def test_incident_creation_and_fields(self):
        inc = self.adapter.create_incident(
            self.sample_charm_record,
            block_identifier="test_block_3",
            checkpoint_hash="647febba9cf564e6bf5e2c4ececd2dc8cf7bfd6d172235fb1a10286f532bb530"
        )
        self.assertTrue(inc.validate())
        self.assertTrue(inc.incident_id.startswith("INC-CHARM-"))
        self.assertEqual(inc.model, "PAADU/CHARM")
        self.assertEqual(inc.input_reference, "test_block_3")
        self.assertTrue(inc.anomaly_detected)
        self.assertAlmostEqual(inc.anomaly_score, 45.1234)
        self.assertAlmostEqual(inc.decision_threshold, 2.5000)
        self.assertEqual(inc.trigger_reasons, ['hybrid_threshold', 'dual_evidence'])
        self.assertEqual(len(inc.top_xai_features), 2)
        self.assertEqual(inc.top_xai_features[0].feature, 'Rate')
        self.assertEqual(inc.feature_schema, "CICIoT2023_46_features")
        self.assertEqual(inc.raw_charm_output['index'], 3)

    def test_validation_constraints(self):
        # Negative threshold should fail validation
        corrupted = dict(self.sample_charm_record)
        corrupted['threshold'] = -1.0
        with self.assertRaises(ValueError):
            self.adapter.create_incident(corrupted)

if __name__ == '__main__':
    unittest.main()
