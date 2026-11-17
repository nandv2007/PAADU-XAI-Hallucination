#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python tests/test_integration.py
python project/run_demo.py
PYTHONPATH=charm_original/project python charm_original/project/smoke_test.py
printf '\nPASS: reconstructed RAG and original CHARM smoke tests both completed.\n'
