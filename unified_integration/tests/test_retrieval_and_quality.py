#!/usr/bin/env python3
"""
Tests D & E — Retrieval Connection, Quality Assessment & Refinement Tests.
Verifies that:
1. Investigation queries are passed into the adaptive retriever (Test D).
2. Query complexity adjusts retrieval depth and parameters dynamically.
3. Retrieval quality assessor measures relevance, coverage, consistency, and provenance.
4. Bounded refinement executes when quality is insufficient and strictly terminates (Test E).
"""
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from unified_integration.knowledge_base.cybersecurity_kb import CybersecurityKnowledgeBase
from unified_integration.retrieval.query_complexity import QueryComplexityAnalyzer
from unified_integration.retrieval.adaptive_controller import AdaptiveRetrievalController
from unified_integration.retrieval.quality_assessor import RetrievalQualityAssessor
from unified_integration.retrieval.refinement_controller import RefinementController

class TestRetrievalAndQuality(unittest.TestCase):

    def setUp(self):
        self.kb = CybersecurityKnowledgeBase()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.retrieval_controller = AdaptiveRetrievalController(kb=self.kb)
        self.quality_assessor = RetrievalQualityAssessor()
        self.refinement_controller = RefinementController(
            retrieval_controller=self.retrieval_controller,
            quality_assessor=self.quality_assessor,
            max_iterations=2
        )

    def test_query_complexity_categorization(self):
        simple_q = "What is TCP protocol?"
        complex_q = (
            "What network behaviors are associated with elevated Rate, syn_count, and UDP, "
            "and what evidence helps distinguish a volumetric DDoS from benign IoT telemetry bursts?"
        )

        res_simple = self.complexity_analyzer.analyze(simple_q)
        res_complex = self.complexity_analyzer.analyze(complex_q)

        self.assertIn(res_simple.category, ['SIMPLE', 'MODERATE'])
        self.assertEqual(res_complex.category, 'COMPLEX')
        self.assertTrue(res_complex.is_comparative)
        self.assertGreater(res_complex.complexity_score, res_simple.complexity_score)
        self.assertGreater(res_complex.recommended_k, res_simple.recommended_k)

    def test_retrieval_connection(self):
        """Test D: Verify query is ingested by HybridRetriever and returns relevant documents."""
        query = "TCP SYN flood handshake exhaustion"
        ret_result = self.retrieval_controller.retrieve(query)

        self.assertGreater(len(ret_result.retrieved_evidence), 0)
        top_title = ret_result.retrieved_evidence[0].title
        self.assertIn("SYN", top_title)
        self.assertGreater(ret_result.retrieved_evidence[0].similarity_score, 0.0)
        cit = ret_result.retrieved_evidence[0].source_citation
        self.assertTrue(any(auth in cit for auth in ["NIST", "RFC", "CERT", "OASIS", "CISA", "IEEE"]))

    def test_evidence_quality_and_refinement_termination(self):
        """Test E: Verify quality scoring and bounded refinement termination."""
        query = "What causes volumetric byte saturation in IoT?"
        target_facets = ["volumetric attack", "rate flood", "bandwidth saturation", "MQTT", "ARP"]

        initial_retrieval = self.retrieval_controller.retrieve(query, override_k=2)
        initial_quality = self.quality_assessor.assess(initial_retrieval.retrieved_evidence, target_facets)

        # Refine evidence
        refine_result = self.refinement_controller.refine_evidence(initial_retrieval, target_facets)

        # Termination within bounded limit
        self.assertLessEqual(refine_result.iterations_executed, 2)
        self.assertGreaterEqual(len(refine_result.final_evidence), len(initial_retrieval.retrieved_evidence))
        self.assertIsNotNone(refine_result.termination_reason)
        # Quality score is in [0.0, 1.0]
        self.assertGreaterEqual(refine_result.final_quality.composite_score, 0.0)
        self.assertLessEqual(refine_result.final_quality.composite_score, 1.0)

if __name__ == '__main__':
    unittest.main()
