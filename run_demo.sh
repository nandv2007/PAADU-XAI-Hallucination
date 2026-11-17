#!/usr/bin/env bash
# Standalone Pipeline Demo Runner for PAADU/CHARM + SVA-GRAG-Core
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/unified_integration:${SCRIPT_DIR}/integrated_project/charm_original/project:${SCRIPT_DIR}/integrated_project/charm_original/project/original:${SCRIPT_DIR}/integrated_project/project:${PYTHONPATH:-}"

echo "========================================================================"
echo "  Executing PAADU/CHARM + SVA-GRAG-Core Standalone Pipeline Demo"
echo "  Root: ${SCRIPT_DIR}"
echo "========================================================================"

cd "${SCRIPT_DIR}"
python3 unified_integration/run_demo.py
