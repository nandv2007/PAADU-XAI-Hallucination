#!/usr/bin/env python3
"""
Master Experiment Runner: Autonomous End-to-End Scientific Validation Suite.
Executes all 4 experimental research modules, scientific baseline comparisons,
statistical tests, publication figure generation, and cryptographic integrity verification.
"""
import sys
import time
import subprocess
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

def run_step(script_path: Path, description: str):
    print("\n" + "=" * 76)
    print(f"  STARTING: {description}")
    print(f"  Target: {script_path.relative_to(PACKAGE_ROOT)}")
    print("=" * 76)
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, str(script_path)], text=True)
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"✗ FAILED: {script_path.name} exited with returncode {res.returncode}")
        sys.exit(res.returncode)
    else:
        print(f"✓ COMPLETED: {description} in {dt:.2f} seconds.")

def main():
    total_t0 = time.perf_counter()
    print("============================================================================")
    print("  PAADU/CHARM + SVA-GRAG-Core Hallucination & Faithfulness Research Suite  ")
    print(f"  Workspace: {PACKAGE_ROOT}")
    print("============================================================================")

    # 1. Experiment 1: Telemetry Fact Checking & Hallucination Rate
    run_step(
        PACKAGE_ROOT / 'experiments' / 'exp01_telemetry_hallucination' / 'run_experiment.py',
        'Module 1 (Exp 1): Ground-Truth Telemetry Fact Checking & Hallucination Rate'
    )

    # 2. Experiment 2: Counterfactual Metadata Testing & Stability
    run_step(
        PACKAGE_ROOT / 'experiments' / 'exp02_counterfactual' / 'run_experiment.py',
        'Module 2 (Exp 2): Counterfactual Metadata Testing & Narrative Stability'
    )

    # 3. Experiment 3: XAI Faithfulness & AOPC Feature Deletion
    run_step(
        PACKAGE_ROOT / 'experiments' / 'exp03_xai_deletion' / 'run_experiment.py',
        'Module 3 (Exp 3): XAI Faithfulness & AOPC Cumulative Feature Deletion'
    )

    # 4. Experiment 4: Model Parameter Randomization Sanity Checks
    run_step(
        PACKAGE_ROOT / 'experiments' / 'exp04_model_randomization' / 'run_experiment.py',
        'Module 4 (Exp 4): Cascading Model Parameter Randomization (Adebayo Sanity Check)'
    )

    # 5. Scientific Baseline Comparison (Configs A, B, C, D)
    run_step(
        PACKAGE_ROOT / 'evaluation' / 'baseline_comparison.py',
        'Multi-Configuration Baseline Comparison (Configs A, B, C, D)'
    )

    # 6. Statistical Significance & Effect Sizes
    run_step(
        PACKAGE_ROOT / 'evaluation' / 'statistics.py',
        'Formal Statistical Significance Tests & Cohen\'s d Effect Sizes'
    )

    # 7. Publication Figures Generation
    run_step(
        PACKAGE_ROOT / 'evaluation' / 'generate_figures.py',
        'Publication-Quality High-Resolution Figures Generation (300 DPI)'
    )

    # 8. Cryptographic Baseline Integrity Check
    print("\n" + "=" * 76)
    print("  VERIFYING ORIGINAL FILE INTEGRITY (144 Monitored Research Files)")
    print("=" * 76)
    verify_script = PACKAGE_ROOT / 'external_evaluation' / 'scripts' / 'hash_integrity.py'
    if not verify_script.exists():
        verify_script = PACKAGE_ROOT / 'unified_integration' / 'hash_integrity.py'
    res = subprocess.run([sys.executable, str(verify_script), 'verify'], text=True)
    if res.returncode != 0:
        print("✗ INTEGRITY FAILURE: Original source files were modified!")
        sys.exit(res.returncode)
    else:
        print("✓ SUCCESS: All original research files remain 100% UNCHANGED!")

    total_time = time.perf_counter() - total_t0
    print("\n" + "=" * 76)
    print(f"  ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("============================================================================")

if __name__ == '__main__':
    main()
