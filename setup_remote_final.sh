#!/bin/bash
# Final setup script - handles password authentication more reliably

set -e

REMOTE_USER="d3"
REMOTE_HOST="100.89.206.96"
REMOTE_PATH="/home/d3/es-fine-tuning-paper"
LOCAL_PATH="/Users/mehdiiranmanesh/Desktop/es-fine-tuning-paper"

# Set in environment: export SSH_PASSWORD='...'  (never commit secrets)
: "${SSH_PASSWORD:?Set SSH_PASSWORD for sshpass}"
PASSWORD="$SSH_PASSWORD"

echo "=========================================="
echo "Remote Server Setup for es-fine-tuning-paper"
echo "=========================================="
echo "Remote: $REMOTE_USER@$REMOTE_HOST"
echo "Remote path: $REMOTE_PATH"
echo ""

# Create temporary password file for sshpass (more secure than command line)
PASSFILE=$(mktemp)
echo "$PASSWORD" > "$PASSFILE"
chmod 600 "$PASSFILE"

# Function to cleanup
cleanup() {
    rm -f "$PASSFILE"
}
trap cleanup EXIT

# Step 1: Copy files
echo "Step 1: Copying files to remote server..."
sshpass -f "$PASSFILE" rsync -avz --progress \
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

# Create setup script to run on remote
cat > /tmp/remote_setup.sh << 'REMOTESCRIPT'
#!/bin/bash
set -e
cd /home/d3/es-fine-tuning-paper

echo "Creating virtual environment..."
python3 -m venv es

echo "Activating and installing dependencies..."
source es/bin/activate
pip install --upgrade pip --quiet
pip install -r requirement.txt

echo "Installing additional dependencies for GRPO..."
pip install tensorboard pyarrow jinja2 tokenizers safetensors pyyaml

echo "Making scripts executable..."
chmod +x run_es_gpu.sh run_zo_gpu.sh run_grpo_gpu.sh

echo ""
echo "Checking GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('No CUDA devices found')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "You can now run:"
echo "  ./run_es_gpu.sh"
echo "  ./run_zo_gpu.sh"
echo "  ./run_grpo_gpu.sh"
REMOTESCRIPT

# Copy and run setup script on remote
echo "Copying setup script..."
sshpass -f "$PASSFILE" scp /tmp/remote_setup.sh "$REMOTE_USER@$REMOTE_HOST:/tmp/remote_setup.sh"

echo "Running setup on remote server..."
sshpass -f "$PASSFILE" ssh "$REMOTE_USER@$REMOTE_HOST" "bash /tmp/remote_setup.sh"

rm -f /tmp/remote_setup.sh

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To run training on the remote server:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST"
echo "  cd $REMOTE_PATH"
echo "  ./run_es_gpu.sh    # or run_zo_gpu.sh or run_grpo_gpu.sh"
echo ""
