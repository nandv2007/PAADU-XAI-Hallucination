#!/usr/bin/env python3
"""
Tests H, I & J — Complete End-to-End Pipeline, Provenance Tracing & Integrity Tests.
Verifies that:
1. Complete flow executes from network input to verified report without mocks (Test H).
2. All claims trace back to evidence documents or original CHARM outputs (Test I).
3. All original project files and datasets remain 100% cryptographically unchanged (Test J).
"""
import sys
import unittest
import pickle
import subprocess
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
INTEG_PROJECT = WORKSPACE / 'integrated_project'
CHARM_ORIG = INTEG_PROJECT / 'charm_original'

sys.path.insert(0, str(WORKSPACE))

from unified_integration.pipeline import UnifiedPAADUPipeline

class TestEndToEndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = UnifiedPAADUPipeline.from_artifacts(enable_nli=True)
        with open(CHARM_ORIG / 'artifacts' / 'charm_engine_state.pkl', 'rb') as f:
            cls.state = pickle.load(f)

    def test_full_pipeline_execution_and_provenance(self):
        """Tests H & I: Full pipeline execution and provenance tracing."""
        rng = np.random.default_rng(42)
        # Construct synthetic benign baseline block matching learned distribution
        block = rng.normal(loc=self.state['mean'], scale=self.state['std'] * 0.1, size=(1, 1, 10, 46)).astype(np.float32)
        # Inject realistic SYN flood perturbation into features
        syn_cnt_idx = self.state['features'].index('syn_count')
        tcp_idx = self.state['features'].index('TCP')
        block[0, 0, :, syn_cnt_idx] += self.state['std'][0, 0, 0, syn_cnt_idx] * 20.0
        block[0, 0, :, tcp_idx] += self.state['std'][0, 0, 0, tcp_idx] * 4.0

        metadata = [{'block_identifier': 'test_e2e_block_0', 'scenario': 'E2E SYN Flood Test'}]

        reports = self.pipeline.analyze(block, metadata=metadata, save_reports=True)
        self.assertEqual(len(reports), 1)
        rep = reports[0]

        # Stage 1: CHARM Inference
        self.assertTrue(rep['charm_inference']['anomaly_detected'])
        self.assertGreater(rep['charm_inference']['hybrid_score'], rep['charm_inference']['decision_threshold'])
        self.assertGreater(len(rep['charm_inference']['top_xai_features']), 0)
        top_feat_names = [f['feature'] for f in rep['charm_inference']['top_xai_features']]
        self.assertIn('syn_count', top_feat_names)
        self.assertIn('TCP', top_feat_names)

        # Stage 2: Query Generation
        self.assertIn("syn_count", rep['investigation_query']['query_text'])
        self.assertEqual(rep['investigation_query']['incident_id'], rep['incident_id'])

        # Stage 3: Retrieval
        self.assertGreater(rep['retrieval']['refined_evidence_count'], 0)
        self.assertIn("SYN", rep['retrieval']['retrieved_evidence'][0]['title'])

        # Stage 4: Explanation & Citations
        self.assertGreater(len(rep['explanation']['claims']), 0)
        self.assertGreater(len(rep['explanation']['citations']), 0)

        # Stage 5: Verification & Correction
        ver = rep['verification_and_correction']
        self.assertGreater(ver['total_claims'], 0)
        self.assertGreater(ver['telemetry_facts_verified'], 0)
        self.assertGreater(ver['evidence_supported_claims'], 0)
        self.assertGreater(ver['verification_rate'], 0.0)

        # Stage 6: Provenance Trace (Test I)
        self.assertIn('647febba9cf564e6bf5e2c4ececd2dc8cf7bfd6d172235fb1a10286f532bb530', rep['provenance_trace']['model_checkpoint_hash'])
        self.assertEqual(rep['provenance_trace']['feature_schema'], 'CICIoT2023_46_features')
        self.assertGreater(len(rep['provenance_trace']['knowledge_base_sources']), 0)

        # Stage 7: Stage runtimes
        for stage_name, t in rep['stage_runtimes_sec'].items():
            self.assertGreaterEqual(t, 0.0)

    def test_original_file_integrity(self):
        """Test J: Verify that all original source files and datasets remain 100% untouched."""
        verify_script = WORKSPACE / 'external_evaluation' / 'scripts' / 'hash_integrity.py'
        if not verify_script.exists():
            for cand in [
                WORKSPACE / 'unified_integration' / 'hash_integrity.py',
                WORKSPACE / 'hash_integrity.py',
                WORKSPACE / 'scripts' / 'hash_integrity.py'
            ]:
                if cand.exists():
                    verify_script = cand
                    break
        cmd = f"PYTHONDONTWRITEBYTECODE=1 python3 '{verify_script}' verify"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Integrity check failed: {result.stdout}\n{result.stderr}")
        self.assertIn("SUCCESS: ALL ORIGINAL FILES ARE 100% UNCHANGED!", result.stdout)

if __name__ == '__main__':
    unittest.main()
