#!/usr/bin/env python3
"""
Test C — Investigation Query Builder Test.
Verifies that:
1. Queries are derived strictly from actual incident decisions and XAI feature attributions.
2. Anomaly queries focus on observed deviating features and cautious distinction from benign bursts.
3. Benign queries focus on baseline telemetry.
4. Generated query references the incident_id for traceability.
"""
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from unified_integration.adapters.charm_incident_adapter import CHARMIncident, FeatureAttributionRecord
from unified_integration.adapters.incident_query_builder import IncidentQueryBuilder

class TestQueryBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = IncidentQueryBuilder(max_features=3)

    def test_anomalous_query_generation(self):
        incident = CHARMIncident(
            incident_id="INC-CHARM-A1B2C3D4",
            model="PAADU/CHARM",
            input_reference="telemetry_block_1",
            anomaly_detected=True,
            anomaly_score=58.2,
            decision_threshold=2.5,
            trigger_reasons=['hybrid_threshold'],
            reconstruction_error=1.2,
            mahalanobis_distance=15.3,
            reconstruction_z=3.0,
            latent_z=4.5,
            top_xai_features=[
                FeatureAttributionRecord('syn_count', 25.0, 3.0, 22.0, 850.0, 10.0, 24.0),
                FeatureAttributionRecord('TCP', 10.0, 2.0, 8.0, 1.0, 0.4, 5.0)
            ],
            xai_summary_text="SYN flood indicators observed.",
            timestamp="2026-11-10T12:00:00Z",
            model_checkpoint_hash="abc",
            feature_schema="CICIoT2023_46_features"
        )

        query = self.builder.build_query(incident)
        self.assertEqual(query.incident_id, "INC-CHARM-A1B2C3D4")
        self.assertTrue(query.is_anomaly_query)
        self.assertIn("syn_count", query.target_features)
        self.assertIn("TCP", query.target_features)
        self.assertIn("syn_count", query.query_text)
        self.assertIn("TCP", query.query_text)
        # Check cautious language
        self.assertIn("possible", query.query_text)
        self.assertIn("distinguish", query.query_text)

    def test_benign_query_generation(self):
        incident = CHARMIncident(
            incident_id="INC-CHARM-BENIGN01",
            model="PAADU/CHARM",
            input_reference="telemetry_block_0",
            anomaly_detected=False,
            anomaly_score=1.1,
            decision_threshold=2.5,
            trigger_reasons=[],
            reconstruction_error=0.2,
            mahalanobis_distance=1.0,
            reconstruction_z=0.1,
            latent_z=0.2,
            top_xai_features=[],
            xai_summary_text=None,
            timestamp="2026-11-10T12:00:00Z",
            model_checkpoint_hash="abc",
            feature_schema="CICIoT2023_46_features"
        )

        query = self.builder.build_query(incident)
        self.assertFalse(query.is_anomaly_query)
        self.assertIn("normal IoT network baseline telemetry", query.query_text)

if __name__ == '__main__':
    unittest.main()
