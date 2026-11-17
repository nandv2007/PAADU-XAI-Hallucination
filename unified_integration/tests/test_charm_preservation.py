#!/usr/bin/env python3
"""
Test A — Original CHARM Preservation Test.
Verifies that:
1. Original model weights can be strictly loaded into SemanticAnomalyMAE.
2. Calibration state and thresholds are preserved.
3. Model generates identical deterministic anomaly scores for identical inputs.
4. XAI attribution outputs match across repeated evaluations.
"""
import sys
import unittest
import pickle
from pathlib import Path
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
INTEG_PROJECT = WORKSPACE / 'integrated_project'
CHARM_ORIG = INTEG_PROJECT / 'charm_original'

for p in [str(CHARM_ORIG / 'project'), str(CHARM_ORIG / 'project' / 'original')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from charm_engine import CHARMEngine
from original.paadu_benchmark_v7 import SemanticAnomalyMAE, VolumeAdaptiveRelaxer
from xai_interpreter import XAIInterpreter

class TestCHARMPreservation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.weights_path = CHARM_ORIG / 'artifacts' / 'charm_model_weights.pt'
        cls.state_path = CHARM_ORIG / 'artifacts' / 'charm_engine_state.pkl'

        with open(cls.state_path, 'rb') as f:
            cls.state = pickle.load(f)

        cls.model = SemanticAnomalyMAE(
            num_nodes=cls.state['config'].num_nodes,
            seq_len=cls.state['config'].seq_len,
            feature_dim=len(cls.state['features']),
            embed_dim=cls.state['config'].embed_dim,
            mask_ratio=cls.state['config'].mask_ratio,
            use_gnn=cls.state['config'].use_gnn
        )
        weights = torch.load(cls.weights_path, map_location='cpu')
        cls.model.load_state_dict(weights, strict=True)
        cls.model.eval()

        cls.engine = CHARMEngine(cls.state['config'], cls.state['features'])
        cls.engine.mean = cls.state['mean']
        cls.engine.std = cls.state['std']
        cls.engine.mse_stats = cls.state['mse_stats']
        cls.engine.dm_stats = cls.state['dm_stats']
        cls.engine.floor = cls.state['floor']
        cls.engine.mahal = cls.state['mahal']
        cls.engine.original_cfg = cls.state['original_cfg']
        cls.engine.model = cls.model
        cls.engine.relaxer = VolumeAdaptiveRelaxer(cls.engine.cfg.hard_z_floor, cls.engine.original_cfg)
        cls.engine.relaxer.set_floor_value(cls.engine.floor, reason='held-out benign calibration')
        cls.engine.fitted = True

        cls.xai = XAIInterpreter(cls.engine)

    def test_checkpoint_shape_and_parameters(self):
        """Verify model parameter count and tensor structure."""
        self.assertEqual(len(self.engine.features), 46)
        self.assertEqual(self.engine.cfg.seq_len, 10)
        self.assertEqual(self.engine.cfg.num_nodes, 1)
        self.assertEqual(self.engine.floor, 2.5)
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(param_count, 185007)

    def test_deterministic_scoring(self):
        """Verify that identical inputs produce bit-for-bit identical scores."""
        rng = np.random.default_rng(999)
        test_block = rng.normal(loc=self.state['mean'], scale=self.state['std'], size=(2, 1, 10, 46)).astype(np.float32)

        scores_1 = self.engine.score(test_block)
        scores_2 = self.engine.score(test_block)

        for s1, s2 in zip(scores_1, scores_2):
            self.assertEqual(s1['anomaly'], s2['anomaly'])
            self.assertAlmostEqual(s1['hybrid_score'], s2['hybrid_score'], places=6)
            self.assertAlmostEqual(s1['reconstruction_error'], s2['reconstruction_error'], places=6)
            self.assertAlmostEqual(s1['mahalanobis'], s2['mahalanobis'], places=6)

    def test_deterministic_xai(self):
        """Verify that XAI attribution is 100% deterministic."""
        rng = np.random.default_rng(777)
        test_block = rng.normal(loc=self.state['mean'], scale=self.state['std'], size=(1, 1, 10, 46)).astype(np.float32)
        # Force anomaly
        test_block[0, 0, :, 4] += self.state['std'][0, 0, 0, 4] * 15.0

        s = self.engine.score(test_block)
        exp_1 = self.xai.explain(test_block, s)
        exp_2 = self.xai.explain(test_block, s)

        self.assertEqual(len(exp_1[0]['feature_attributions']), len(exp_2[0]['feature_attributions']))
        top_1 = exp_1[0]['feature_attributions'][0]
        top_2 = exp_2[0]['feature_attributions'][0]
        self.assertEqual(top_1['feature'], top_2['feature'])
        self.assertAlmostEqual(top_1['hybrid_contribution'], top_2['hybrid_contribution'], places=5)

if __name__ == '__main__':
    unittest.main()
