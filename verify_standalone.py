#!/usr/bin/env python3
"""
Standalone Independence Verification Script.
Guarantees that the standalone package:
1. Executes fully without silent dependencies on the original development workspace.
2. Loads all model weights, state pickles, and original modules strictly from local bundle paths.
3. Performs real unmocked inference through PAADU/CHARM and Hugging Face Transformers NLI.
"""
import os
import sys
import pickle
from pathlib import Path
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent

# Ensure sys.path prioritizes PACKAGE_ROOT
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / 'integrated_project' / 'charm_original' / 'project'))
sys.path.insert(0, str(PACKAGE_ROOT / 'integrated_project' / 'charm_original' / 'project' / 'original'))
sys.path.insert(0, str(PACKAGE_ROOT / 'integrated_project' / 'project'))

from unified_integration.pipeline import UnifiedPAADUPipeline

def main():
    print("=" * 72)
    print("  STANDALONE INDEPENDENCE AND EXECUTION VALIDATION")
    print(f"  Package Root: {PACKAGE_ROOT}")
    print("=" * 72)

    # 1. Initialize Pipeline
    print("[1/5] Initializing pipeline from bundled local artifacts...")
    weights_path = str(PACKAGE_ROOT / 'integrated_project' / 'charm_original' / 'artifacts' / 'charm_model_weights.pt')
    state_path = str(PACKAGE_ROOT / 'integrated_project' / 'charm_original' / 'artifacts' / 'charm_engine_state.pkl')
    
    assert os.path.exists(weights_path), f"Missing bundled weights: {weights_path}"
    assert os.path.exists(state_path), f"Missing bundled state: {state_path}"

    pipeline = UnifiedPAADUPipeline.from_artifacts(
        weights_path=weights_path,
        state_path=state_path,
        enable_nli=True
    )
    print("  ✓ Pipeline initialized successfully.")

    # 2. Check sys.modules for leakage of original development workspace
    print("[2/5] Inspecting loaded modules for workspace leakage...")
    original_workspace = "/Users/nandhinivenkatesan/Documents/IoT PAADU CHARM/PAADU_Testing"
    leaked_modules = []
    local_custom_modules = []

    for name, mod in list(sys.modules.items()):
        if mod and hasattr(mod, '__file__') and mod.__file__:
            mod_file = str(Path(mod.__file__).resolve())
            if original_workspace in mod_file and str(PACKAGE_ROOT.resolve()) not in mod_file:
                leaked_modules.append((name, mod_file))
            elif str(PACKAGE_ROOT.resolve()) in mod_file:
                local_custom_modules.append((name, mod_file))

    if leaked_modules:
        print(f"  ✗ LEAKAGE DETECTED: {len(leaked_modules)} modules loaded from original workspace:")
        for name, path in leaked_modules:
            print(f"    - {name}: {path}")
        sys.exit(1)
    else:
        print(f"  ✓ ZERO LEAKAGE: All {len(local_custom_modules)} project modules loaded strictly from standalone root!")

    # 3. Create realistic test input matching trained distribution
    print("[3/5] Executing full pipeline analysis on standalone test input...")
    with open(state_path, 'rb') as f:
        state = pickle.load(f)

    rng = np.random.default_rng(123)
    block = rng.normal(loc=state['mean'], scale=state['std'] * 0.1, size=(1, 1, 10, 46)).astype(np.float32)
    syn_idx = state['features'].index('syn_count')
    tcp_idx = state['features'].index('TCP')
    block[0, 0, :, syn_idx] += state['std'][0, 0, 0, syn_idx] * 25.0
    block[0, 0, :, tcp_idx] += state['std'][0, 0, 0, tcp_idx] * 5.0

    metadata = [{'block_identifier': 'standalone_independent_test_0', 'scenario': 'Standalone Validation'}]
    reports = pipeline.analyze(block, metadata=metadata, save_reports=False)
    assert len(reports) == 1, "Failed to produce report"
    rep = reports[0]
    print("  ✓ Pipeline execution finished successfully.")

    # 4. Verify stage outputs
    print("[4/5] Verifying stage outputs and metrics...")
    score = rep['charm_inference']['hybrid_score']
    threshold = rep['charm_inference']['decision_threshold']
    assert score > threshold, f"Anomaly not detected: {score} <= {threshold}"
    top_features = [f['feature'] for f in rep['charm_inference']['top_xai_features']]
    assert 'syn_count' in top_features, f"syn_count not in top features: {top_features}"
    assert rep['investigation_query']['query_text'], "Query text empty"
    assert len(rep['retrieval']['retrieved_evidence']) > 0, "No evidence retrieved"
    assert rep['verification_and_correction']['total_claims'] > 0, "No claims generated"
    assert rep['verification_and_correction']['evidence_supported_claims'] > 0, "No evidence claims verified"
    print(f"  ✓ CHARM Score: {score:.4f} (Threshold: {threshold:.4f})")
    print(f"  ✓ Top XAI Features: {top_features[:3]}")
    print(f"  ✓ NLI Claims Verified: {rep['verification_and_correction']['total_claims']} total "
          f"({rep['verification_and_correction']['telemetry_facts_verified']} telemetry facts, "
          f"{rep['verification_and_correction']['evidence_supported_claims']} evidence supported)")

    # 5. Summary
    print("[5/5] Standalone Independence Check Result:")
    print("  " + "=" * 68)
    print("  SUCCESS: Standalone package is 100% self-contained and independently executable!")
    print("  " + "=" * 68)

if __name__ == '__main__':
    main()
