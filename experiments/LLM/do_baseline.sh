#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Running baseline evaluation using current python environment..."
python3 test_baseline.py > baseline.log

echo "Baseline completed. Results:"
cat baseline.log
