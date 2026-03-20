#!/bin/bash
# Setup script to copy repo to remote server and set up environment

set -e

REMOTE_USER="d3"
REMOTE_HOST="100.89.206.96"
REMOTE_PATH="/home/d3/es-fine-tuning-paper"
LOCAL_PATH="/Users/mehdiiranmanesh/Desktop/es-fine-tuning-paper"

echo "=========================================="
echo "Setting up es-fine-tuning-paper on remote server"
echo "=========================================="
echo "Remote: $REMOTE_USER@$REMOTE_HOST"
echo "Remote path: $REMOTE_PATH"
echo ""

# Step 1: Copy files to remote server
echo "Step 1: Copying files to remote server..."
rsync -avz --progress \
    --exclude 'es/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'checkpoint_*' \
    --exclude 'finetuned_*' \
    --exclude 'results_*' \
    --exclude 'logs_*' \
    --exclude 'ckpt_*' \
    "$LOCAL_PATH/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"

echo ""
echo "Step 2: Setting up virtual environment on remote server..."
ssh "$REMOTE_USER@$REMOTE_HOST" << 'ENDSSH'
cd /home/d3/es-fine-tuning-paper

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv es

# Activate and install dependencies
echo "Installing dependencies..."
source es/bin/activate
pip install --upgrade pip
pip install -r requirement.txt

# Install additional dependencies for GRPO
pip install tensorboard pyarrow jinja2 tokenizers safetensors pyyaml

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

echo ""
echo "Setup complete!"
echo "You can now run:"
echo "  ./run_es_gpu.sh"
echo "  ./run_zo_gpu.sh"
echo "  ./run_grpo_gpu.sh"
ENDSSH

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To run training on the remote server:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST"
echo "  cd $REMOTE_PATH"
echo "  ./run_es_gpu.sh    # or run_zo_gpu.sh or run_grpo_gpu.sh"
echo ""
