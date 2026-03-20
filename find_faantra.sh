#!/bin/bash
# Usage: SSH_PASSWORD=... ./find_faantra.sh
set -euo pipefail
: "${SSH_PASSWORD:?Set SSH_PASSWORD}"
export SSHPASS="$SSH_PASSWORD"
REMOTE="${SSH_REMOTE:-d3@100.89.206.96}"
sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$REMOTE" << 'EOF'
echo "Searching for FAANTRA model..."
echo ""
echo "=== Searching in home directory ==="
find ~ -iname "*faantra*" 2>/dev/null | head -20
echo ""
echo "=== Searching in common model directories ==="
find ~/Desktop ~/Documents ~/Downloads -iname "*faantra*" 2>/dev/null | head -20
echo ""
echo "=== Searching for model files (safetensors, bin, etc.) ==="
find ~ -type f \( -iname "*.safetensors" -o -iname "*.bin" -o -iname "*.pt" -o -iname "*.pth" \) -path "*faantra*" 2>/dev/null | head -20
echo ""
echo "=== Checking current directory ==="
pwd
ls -la | grep -i faantra
EOF
