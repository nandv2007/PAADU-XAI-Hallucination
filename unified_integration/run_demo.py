#!/usr/bin/env python3
"""
Complete End-to-End Demonstration of PAADU/CHARM + SVA-GRAG-Core Unified Pipeline.
Demonstrates the full sequence:
Network Telemetry -> CHARM Engine -> XAI Occlusion -> Structured Incident ->
Query Builder -> Adaptive GraphRAG -> Quality Assessor & Refinement ->
Evidence-Grounded Explainer -> NLI Verification & Correction -> Final Verified Report.
"""
from __future__ import annotations

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
INTEG_PROJECT = WORKSPACE / 'integrated_project'
CHARM_ORIG = INTEG_PROJECT / 'charm_original'

sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / 'unified_integration'))

from unified_integration.pipeline import UnifiedPAADUPipeline

def main():
    parser = argparse.ArgumentParser(description="Run complete unified PAADU/CHARM + SVA-GRAG pipeline demo")
    parser.add_argument('--no-nli', action='store_true', help="Disable transformer NLI model (fast offline mode)")
    args = parser.parse_args()

    enable_nli = not args.no_nli
    print("=" * 80)
    print("  PAADU/CHARM + SVA-GRAG-Core Unified End-to-End Pipeline Demonstration")
    print(f"  NLI Verification: {'ENABLED (cross-encoder/nli-MiniLM2-L6-H768)' if enable_nli else 'DISABLED'}")
    print("=" * 80)

    # 1. Initialize pipeline from preserved original artifacts
    pipeline = UnifiedPAADUPipeline.from_artifacts(enable_nli=enable_nli)
    print("✓ Pipeline successfully initialized from original model weights and calibration state.\n")

    # 2. Prepare test telemetry blocks matching the 46-feature schema
    state_path = CHARM_ORIG / 'artifacts' / 'charm_engine_state.pkl'
    with open(state_path, 'rb') as f:
        state = pickle.load(f)

    rng = np.random.default_rng(2026)
    
    # Block 1: Benign baseline IoT telemetry (small variance around benign training mean)
    benign_block = rng.normal(loc=state['mean'], scale=state['std'] * 0.1, size=(1, 1, 10, 46)).astype(np.float32)

    # Block 2: Anomalous TCP SYN Flood attack (syn_count and syn_flag_number severely elevated)
    syn_flood_block = benign_block.copy()
    syn_cnt_idx = state['features'].index('syn_count')
    syn_flg_idx = state['features'].index('syn_flag_number')
    tcp_idx = state['features'].index('TCP')
    syn_flood_block[0, 0, :, syn_cnt_idx] += state['std'][0, 0, 0, syn_cnt_idx] * 22.0
    syn_flood_block[0, 0, :, syn_flg_idx] += state['std'][0, 0, 0, syn_flg_idx] * 8.0
    syn_flood_block[0, 0, :, tcp_idx] += state['std'][0, 0, 0, tcp_idx] * 4.0

    # Block 3: Anomalous Volumetric DoS attack (Rate, Tot sum, AVG surged)
    volumetric_block = benign_block.copy()
    rate_idx = state['features'].index('Rate')
    totsum_idx = state['features'].index('Tot sum')
    volumetric_block[0, 0, :, rate_idx] += state['std'][0, 0, 0, rate_idx] * 16.0
    volumetric_block[0, 0, :, totsum_idx] += state['std'][0, 0, 0, totsum_idx] * 18.0

    test_batch = np.concatenate([benign_block, syn_flood_block, volumetric_block], axis=0)
    metadata = [
        {'block_identifier': 'sample_benign_iot_flow', 'scenario': 'Benign IoT Sensor Telemetry'},
        {'block_identifier': 'sample_syn_flood_burst', 'scenario': 'TCP SYN Flood Attack Simulation'},
        {'block_identifier': 'sample_volumetric_rate_spike', 'scenario': 'Volumetric Rate Denial-of-Service'}
    ]

    print(f"Input Telemetry Batch Shape: {test_batch.shape} (3 blocks, 1 node, 10 steps, 46 features)\n")

    # 3. Execute Unified Analysis
    reports = pipeline.analyze(test_batch, metadata=metadata)

    # 4. Display results
    for idx, rep in enumerate(reports):
        inc = rep['charm_inference']
        qry = rep['investigation_query']
        ret = rep['retrieval']
        ver = rep['verification_and_correction']
        rt = rep['stage_runtimes_sec']

        print("-" * 80)
        print(f"BLOCK {idx + 1}: {rep['input_reference']} [{metadata[idx]['scenario']}]")
        print(f"  Incident ID: {rep['incident_id']}")
        print(f"  CHARM Decision: {'ANOMALY DETECTED' if inc['anomaly_detected'] else 'BENIGN'} (Score: {inc['hybrid_score']:.4f}, Threshold: {inc['decision_threshold']:.4f})")
        if inc['anomaly_detected']:
            top_feats = ", ".join([f"{f['feature']} ({f['hybrid_contribution']:+.2f})" for f in inc['top_xai_features'][:3]])
            print(f"  Top XAI Features: {top_feats}")
            print(f"  Trigger Causes: {inc['trigger_reasons']}")
        print(f"  Synthesized Query: \"{qry['query_text']}\"")
        print(f"  Adaptive Strategy: {ret['strategy']} -> Quality: {ret['evidence_quality']['quality_status']} ({ret['evidence_quality']['composite_score']:.3f})")
        print(f"  Retrieved Documents: {ret['refined_evidence_count']} docs (Refinement: {ret['refinement_iterations']} iterations)")
        print(f"  Claims Verified: {ver['total_claims']} total ({ver['telemetry_facts_verified']} telemetry, {ver['evidence_supported_claims']} evidence supported)")
        print(f"  Block Pipeline Latency: {rt['total_block_sec']:.4f}s")
        print("\n  Verified Incident Summary:")
        for line in ver['final_verified_narrative'].split('\n'):
            if line.strip():
                print(f"    {line}")
        print()

    print("=" * 80)
    print("SUCCESS: Full end-to-end integrated pipeline executed cleanly.")
    print("Reports written to: unified_integration/outputs/")
    print("=" * 80)

if __name__ == '__main__':
    main()
