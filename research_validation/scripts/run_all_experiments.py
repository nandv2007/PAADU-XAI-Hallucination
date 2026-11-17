#!/usr/bin/env python3
"""
Master Experiment Runner: Executes the complete research validation suite.
1. Phase 1: Experiment 1 (Ground-Truth Telemetry Fact Checking & Hallucination)
2. Phase 2: Experiment 2 (Counterfactual Metadata Testing & Stability)
3. Phase 3: Experiment 3 (XAI Faithfulness & AOPC Feature Deletion)
4. Phase 4: Experiment 4 (Model Parameter Randomization Sanity Checks)
5. Phase 5: Scientific Baseline Comparison (Configs A, B, C, D)
6. Publication Figures Generation (Figures 1 to 5)
7. Cryptographic File Integrity Verification (144 Original Files)
"""
import sys
import time
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = WORKSPACE / 'research_validation' / 'scripts'

def run_script(script_name: str, desc: str):
    print("\n" + "=" * 76)
    print(f"  STARTING: {desc}")
    print(f"  Script: {script_name}")
    print("=" * 76)
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], text=True)
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"✗ FAILED: {script_name} returned exit code {res.returncode}")
        sys.exit(res.returncode)
    else:
        print(f"✓ COMPLETED: {desc} in {dt:.2f} seconds.")

def main():
    total_t0 = time.perf_counter()
    print("============================================================================")
    print("  PAADU/CHARM + SVA-GRAG-Core Hallucination & Faithfulness Research Suite  ")
    print("============================================================================")

    # 1. Experiment 1
    run_script('run_experiment_1.py', 'Experiment 1: Ground-Truth Telemetry Fact Checking')

    # 2. Experiment 2
    run_script('run_experiment_2.py', 'Experiment 2: Counterfactual Metadata Testing')

    # 3. Experiment 3
    run_script('run_experiment_3.py', 'Experiment 3: XAI Faithfulness & AOPC Perturbation')

    # 4. Experiment 4
    run_script('run_experiment_4.py', 'Experiment 4: Model Parameter Randomization Sanity Checks')

    # 5. Baseline Comparison
    run_script('run_baseline_comparison.py', 'Scientific Baseline Comparison (Configs A, B, C, D)')

    # 6. Figures Generation
    run_script('generate_figures.py', 'Publication-Quality Figures Generation')

    # 7. Integrity Verification
    print("\n" + "=" * 76)
    print("  VERIFYING ORIGINAL FILE INTEGRITY (144 Original Files)")
    print("=" * 76)
    verify_script = WORKSPACE / 'external_evaluation' / 'scripts' / 'hash_integrity.py'
    if not verify_script.exists():
        verify_script = WORKSPACE / 'unified_integration' / 'hash_integrity.py'
    res = subprocess.run([sys.executable, str(verify_script), 'verify'], text=True)
    if res.returncode != 0:
        print("✗ INTEGRITY FAILURE! Original files were modified!")
        sys.exit(res.returncode)
    else:
        print("✓ SUCCESS: All 144 original research and dataset files remain 100% UNCHANGED!")

    total_time = time.perf_counter() - total_t0
    print("\n" + "=" * 76)
    print(f"  ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("============================================================================")

if __name__ == '__main__':
    main()
