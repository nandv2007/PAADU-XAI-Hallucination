#!/usr/bin/env bash
# Standalone Installation Script for PAADU/CHARM + SVA-GRAG-Core
set -euo pipefail

echo "========================================================================"
echo "  PAADU/CHARM + SVA-GRAG-Core Standalone Environment Setup"
echo "========================================================================"

PYTHON_BIN="${PYTHON:-python3}"
echo "Using Python executable: $(which ${PYTHON_BIN}) ($(${PYTHON_BIN} --version))"

echo "Checking / Installing dependencies..."
${PYTHON_BIN} -m pip install -r unified_integration/requirements.txt

echo "Testing PyTorch and Transformers import..."
${PYTHON_BIN} -c "import torch, transformers, sklearn, networkx; print('PyTorch:', torch.__version__, '| Transformers:', transformers.__version__)"

echo ""
echo "Installation completed successfully."
echo "You can now run tests with: ./run_tests.sh"
echo "Or run the pipeline demo with: ./run_demo.sh"
