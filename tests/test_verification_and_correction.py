#!/usr/bin/env python3
"""
Tests F & G — NLI Verification & Correction Safety Tests.
Verifies that:
1. Factual claims generated in explanations are checked against retrieved evidence using NLI (Test F).
2. Telemetry facts are verified against the real CHARMIncident record.
3. Unsupported replacements are never introduced during correction (Test G).
4. Contradicted claims are flagged or retracted rather than rewritten with fabrications.
"""
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from unified_integration.adapters.charm_incident_adapter import CHARMIncident, FeatureAttributionRecord
from unified_integration.generation.incident_explainer import FactualClaim, IncidentExplanation
from unified_integration.retrieval.adaptive_controller import EnrichedEvidence
from unified_integration.verification.verification_orchestrator import VerificationOrchestrator

class TestVerificationAndCorrection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize orchestrator with real transformer NLI model
        cls.orchestrator = VerificationOrchestrator(enable_nli=True)

    def test_nli_claim_verification(self):
        """Test F: Real NLI model verifies factual claims against evidence."""
        incident = CHARMIncident(
            incident_id="INC-CHARM-V1",
            model="PAADU/CHARM",
            input_reference="test_sample_1",
            anomaly_detected=True,
            anomaly_score=35.500,
            decision_threshold=2.500,
            trigger_reasons=['hybrid_threshold'],
            reconstruction_error=1.5,
            mahalanobis_distance=10.0,
            reconstruction_z=3.0,
            latent_z=3.5,
            top_xai_features=[FeatureAttributionRecord('syn_count', 10.0, 2.0, 8.0, 500.0, 5.0, 15.0)],
            xai_summary_text="SYN flood alert.",
            timestamp="2026-11-10T12:00:00Z",
            model_checkpoint_hash="hash123",
            feature_schema="CICIoT2023_46_features"
        )

        evidence = [
            EnrichedEvidence(
                doc_id="DOC-KB-002",
                title="TCP SYN Flood and Handshake Exhaustion",
                text="A TCP SYN flood is a denial-of-service attack exploiting the TCP three-way handshake by leaving half-open connection entries in the target backlog queue.",
                similarity_score=0.75,
                source_citation="RFC 793",
                provenance="IETF",
                topics=["TCP SYN flood"]
            )
        ]

        # Case 1: Telemetry fact
        claim_telemetry = FactualClaim(
            claim_id="C1",
            claim_text="PAADU/CHARM evaluated test_sample_1 as ANOMALOUS with a hybrid anomaly score of 35.500 against the calibrated baseline threshold of 2.500.",
            claim_type="OBSERVED_FACT"
        )

        # Case 2: Supported evidence interpretation
        claim_supported = FactualClaim(
            claim_id="C2",
            claim_text="A TCP SYN flood is a denial-of-service attack exploiting the TCP three-way handshake.",
            claim_type="EVIDENCE_INTERPRETATION",
            cited_doc_title="TCP SYN Flood and Handshake Exhaustion",
            cited_doc_id="DOC-KB-002"
        )

        # Case 3: Contradicted claim
        claim_contradicted = FactualClaim(
            claim_id="C3",
            claim_text="A TCP SYN flood is completely harmless and establishes full three-way handshakes with zero backlog usage.",
            claim_type="EVIDENCE_INTERPRETATION",
            cited_doc_title="TCP SYN Flood and Handshake Exhaustion",
            cited_doc_id="DOC-KB-002"
        )

        explanation = IncidentExplanation(
            explanation_id="EXP-1",
            incident_id=incident.incident_id,
            observed_facts=[claim_telemetry.claim_text],
            supported_interpretations=[claim_supported.claim_text, claim_contradicted.claim_text],
            unverified_hypotheses=[],
            claims=[claim_telemetry, claim_supported, claim_contradicted],
            full_narrative="",
            citations=[]
        )

        summary = self.orchestrator.verify_and_correct(explanation, incident, evidence)

        # Telemetry claim checked directly
        res_telemetry = summary.claim_results[0]
        self.assertEqual(res_telemetry.verification_status, "VERIFIED_OBSERVED_TELEMETRY")
        self.assertEqual(res_telemetry.correction_action, "RETAINED_VERIFIED")

        # Supported claim checked by NLI
        res_supported = summary.claim_results[1]
        self.assertEqual(res_supported.verification_status, "SUPPORTED")
        self.assertEqual(res_supported.correction_action, "RETAINED_VERIFIED")
        self.assertIn("[VERIFIED EVIDENCE]", res_supported.final_presented_text)

        # Contradicted claim checked and corrected safely (Test G)
        res_contra = summary.claim_results[2]
        self.assertEqual(res_contra.verification_status, "CONTRADICTED")
        self.assertEqual(res_contra.correction_action, "CONTRADICTION_CORRECTED")
        self.assertIn("CONTRADICTED CLAIM RETRACTED", res_contra.final_presented_text)

if __name__ == '__main__':
    unittest.main()
