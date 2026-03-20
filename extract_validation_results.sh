#!/bin/bash
# Extract validation results from ES and GRPO logs for comparison
# Usage: SSH_PASSWORD=... ./extract_validation_results.sh

set -euo pipefail
: "${SSH_PASSWORD:?Set SSH_PASSWORD (use sshpass -e)}"
export SSHPASS="$SSH_PASSWORD"
REMOTE="${SSH_REMOTE:-d3@100.89.206.96}"
REMOTE_DIR="${SSH_REMOTE_DIR:-/home/d3/es-fine-tuning-paper}"

echo "=== Extracting Validation Results ==="
echo ""

echo "GRPO Validation Results:"
echo "-----------------------"
sshpass -e ssh -o StrictHostKeyChecking=no "$REMOTE" "cd $REMOTE_DIR && tail -2000 results_grpo/training.log | grep -A 3 'Evaluation Results:' | grep -E '(Success rate|Format reward|Mean reward)' | head -9"

echo ""
echo "ES Validation Results:"
echo "----------------------"
sshpass -e ssh -o StrictHostKeyChecking=no "$REMOTE" "cd $REMOTE_DIR && tail -2000 results_es/training.log | grep -A 5 'Evaluation Results:' | grep -E '(Loss|Reward|Accuracy)' | head -15"
