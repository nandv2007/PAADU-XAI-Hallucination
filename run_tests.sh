#!/usr/bin/env bash
# Standalone Test Suite Runner for PAADU/CHARM + SVA-GRAG-Core
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/unified_integration:${SCRIPT_DIR}/integrated_project/charm_original/project:${SCRIPT_DIR}/integrated_project/charm_original/project/original:${SCRIPT_DIR}/integrated_project/project:${PYTHONPATH:-}"

echo "========================================================================"
echo "  Executing PAADU/CHARM + SVA-GRAG-Core Standalone Test Suite"
echo "  Root: ${SCRIPT_DIR}"
echo "========================================================================"

cd "${SCRIPT_DIR}"
bash unified_integration/run_tests.sh
