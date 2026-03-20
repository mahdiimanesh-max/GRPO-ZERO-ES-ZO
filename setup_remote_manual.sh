#!/bin/bash
# Manual setup instructions and script for remote server
# Run this script, or follow the manual steps below

set -e

REMOTE_USER="d3"
REMOTE_HOST="100.89.206.96"
REMOTE_PATH="/home/d3/es-fine-tuning-paper"
LOCAL_PATH="/Users/mehdiiranmanesh/Desktop/es-fine-tuning-paper"
: "${SSH_PASSWORD:?Set SSH_PASSWORD (never commit secrets)}"
export SSHPASS="$SSH_PASSWORD"

echo "=========================================="
echo "Remote Server Setup for es-fine-tuning-paper"
echo "=========================================="
echo ""
echo "This script will copy files and set up the environment."
echo "Using SSH_PASSWORD from the environment for sshpass."
echo ""

# Check if sshpass is available
if command -v sshpass &> /dev/null; then
    echo "Using sshpass -e (SSHPASS / SSH_PASSWORD)."
    USE_SSHPASS=true
else
    echo "Note: sshpass not found. You'll need to enter password manually."
    echo "Install with: brew install hudochenkov/sshpass/sshpass (macOS)"
    SSHPASS_CMD=""
    USE_SSHPASS=false
fi

# Step 1: Copy files
echo "Step 1: Copying files to remote server..."
if [ "$USE_SSHPASS" = true ]; then
    sshpass -e rsync -avz --progress \
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
else
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
fi

echo ""
echo "Step 2: Setting up virtual environment on remote server..."

# Create setup script to run on remote
cat > /tmp/remote_setup.sh << 'REMOTESCRIPT'
#!/bin/bash
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

# Make scripts executable
chmod +x run_es_gpu.sh run_zo_gpu.sh run_grpo_gpu.sh

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
REMOTESCRIPT

# Copy and run setup script on remote
if [ "$USE_SSHPASS" = true ]; then
    sshpass -e scp /tmp/remote_setup.sh "$REMOTE_USER@$REMOTE_HOST:/tmp/remote_setup.sh"
    sshpass -e ssh "$REMOTE_USER@$REMOTE_HOST" "bash /tmp/remote_setup.sh"
else
    scp /tmp/remote_setup.sh "$REMOTE_USER@$REMOTE_HOST:/tmp/remote_setup.sh"
    ssh "$REMOTE_USER@$REMOTE_HOST" "bash /tmp/remote_setup.sh"
fi

rm /tmp/remote_setup.sh

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To run training on the remote server:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST"
echo "  cd $REMOTE_PATH"
echo "  ./run_es_gpu.sh    # or run_zo_gpu.sh or run_grpo_gpu.sh"
echo ""
