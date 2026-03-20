# Remote Server Setup - Step by Step Guide

## Quick Setup (Automated)

If automated setup fails, follow the manual steps below.

### Automated Setup (Try This First)

```bash
cd /Users/mehdiiranmanesh/Desktop/es-fine-tuning-paper
export SSH_PASSWORD='your_ssh_password'   # never commit this
./setup_remote_final.sh
```

**Note**: If you get "Permission denied", try entering the password manually or use SSH keys. See Manual Setup below.

---

## Manual Setup (If Automated Fails)

### Step 1: Copy Files to Remote Server

Run `rsync` and enter your SSH password when prompted.

```bash
cd /Users/mehdiiranmanesh/Desktop/es-fine-tuning-paper

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
    ./ d3@100.89.206.96:/home/d3/es-fine-tuning-paper/
```

### Step 2: SSH into Server and Set Up Environment

```bash
ssh d3@100.89.206.96
```

Once connected, run:

```bash
cd /home/d3/es-fine-tuning-paper

# Create virtual environment
python3 -m venv es

# Activate and install dependencies
source es/bin/activate
pip install --upgrade pip
pip install -r requirement.txt

# Install additional dependencies for GRPO
pip install tensorboard pyarrow jinja2 tokenizers safetensors pyyaml

# Make scripts executable
chmod +x run_es_gpu.sh run_zo_gpu.sh run_grpo_gpu.sh

# Verify GPU is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## Running Training

Once setup is complete, you can run any of the three methods:

### Evolution Strategies (ES)
```bash
./run_es_gpu.sh
```

### Zero-Order Optimization (ZO)
```bash
./run_zo_gpu.sh
```

### GRPO (Group Relative Policy Optimization)
```bash
./run_grpo_gpu.sh
```

---

## Troubleshooting

### "Permission denied" Error

1. **Verify credentials**: Ensure your SSH password or key is correct.

2. **Check SSH access**: Try connecting manually first:
   ```bash
   ssh d3@100.89.206.96
   ```

3. **Password file method**: If sshpass isn't working, store the password in a **local** file (mode `600`) and use `sshpass -f`, or prefer SSH keys.

### "CUDA not available" Warning

- Make sure PyTorch with CUDA support is installed:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

### Virtual Environment Issues

- If `python3 -m venv` fails, try:
  ```bash
  python3 -m pip install --user virtualenv
  python3 -m virtualenv es
  ```

---

## Configuration Summary

All training scripts are pre-configured with:

- **Model**: Qwen2.5-1.5B-Instruct
- **Precision**: bfloat16 (for GPU)
- **Training samples**: 500
- **Eval samples**: 128
- **Max tokens**: 1024
- **Eval frequency**: Every 10 iterations

### ES Settings:
- Population: 30
- Iterations: 500
- Sigma: 0.001, Alpha: 0.0005

### ZO Settings:
- Perturbations: 30
- Iterations: 500
- Mu: 0.001, LR: 0.0005
- Method: Central difference

### GRPO Settings:
- Batch size: 16 (4 questions × 4 answers)
- LR: 1e-5
- Temperature: 0.7

---

## Output Locations

- Training logs: `results_es/training.log`, `results_zo/training.log`, `results_grpo/training.log`
- Final models: `finetuned_es_pop30_iter500_final/`, `finetuned_zo_pert30_iter500_central_final/`, `finetuned_grpo_countdown_final/`
- Checkpoints: Saved every 10 iterations (ES/ZO) or 50 steps (GRPO)
