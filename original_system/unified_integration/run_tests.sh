#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${WORKSPACE}:${SCRIPT_DIR}:${WORKSPACE}/integrated_project/charm_original/project:${WORKSPACE}/integrated_project/charm_original/project/original:${WORKSPACE}/integrated_project/project"
export PYTHONDONTWRITEBYTECODE=1

echo "========================================================================"
echo "  PAADU/CHARM + SVA-GRAG-Core Unified Integration Test Suite"
echo "========================================================================"

echo ""
echo "--- Running Test A: CHARM Architecture & Checkpoint Preservation ---"
python3 "${SCRIPT_DIR}/tests/test_charm_preservation.py"

echo ""
echo "--- Running Test B: Structured Incident Adapter ---"
python3 "${SCRIPT_DIR}/tests/test_incident_adapter.py"

echo ""
echo "--- Running Test C: Investigation Query Builder ---"
python3 "${SCRIPT_DIR}/tests/test_query_builder.py"

echo ""
echo "--- Running Tests D & E: Adaptive Retrieval, Complexity & Quality ---"
python3 "${SCRIPT_DIR}/tests/test_retrieval_and_quality.py"

echo ""
echo "--- Running Tests F & G: NLI Verification & Conservative Correction ---"
python3 "${SCRIPT_DIR}/tests/test_verification_and_correction.py"

echo ""
echo "--- Running Tests H, I & J: Full End-to-End Pipeline & Integrity ---"
python3 "${SCRIPT_DIR}/tests/test_end_to_end_pipeline.py"

echo ""
echo "========================================================================"
echo "  ALL TESTS PASSED: Full Functional Integration Verified Successfully"
echo "========================================================================"
