# Remote Server Setup Instructions

## Quick Setup

### Option 1: Automated (if sshpass is installed)

```bash
# Install sshpass on macOS (if not already installed)
brew install hudochenkov/sshpass/sshpass

# Run the setup script (set your SSH password in the environment — never commit it)
export SSH_PASSWORD='your_ssh_password'
./setup_remote_manual.sh
```

### Option 2: Manual Steps

#### Step 1: Copy files to remote server

```bash
cd /Users/mehdiiranmanesh/Desktop/es-fine-tuning-paper

# Enter your SSH password when prompted.
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

#### Step 2: SSH into the server and set up environment

```bash
ssh d3@100.89.206.96

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

## Configuration

All scripts are pre-configured with proper hyperparameters for GPU training:

- **Model**: Qwen2.5-1.5B-Instruct
- **Precision**: bfloat16 (for GPU)
- **Training samples**: 500
- **Eval samples**: 128 (held-out test set)
- **Max generation length**: 1024 tokens
- **Evaluation frequency**: Every 10 iterations/steps

### ES-specific:
- Population size: 30
- Sigma: 0.001
- Alpha: 0.0005
- Iterations: 500

### ZO-specific:
- Perturbations per iteration: 30
- Mu: 0.001
- Learning rate: 0.0005
- Gradient method: central difference
- Iterations: 500

### GRPO-specific:
- Batch size: 16 (4 questions × 4 answers)
- Learning rate: 1e-5
- Temperature: 0.7
- Micro batch size: 2

## Output

Each method will create its own output directory:
- `results_es/` - ES training logs and checkpoints
- `results_zo/` - ZO training logs and checkpoints
- `results_grpo/` - GRPO training logs and checkpoints

Final models will be saved as:
- `finetuned_es_pop30_iter500_final/`
- `finetuned_zo_pert30_iter500_central_final/`
- `finetuned_grpo_countdown_final/`

## Monitoring

Training progress is logged to:
- Console output (with `tee` to log files)
- `results_*/training.log` - Full training log

Evaluation results are printed every 10 iterations/steps showing:
- Success rate (answer_reward > 0)
- Format reward
- Mean reward

## Comparing Results

After training, you can use the evaluation script to compare models:

```bash
# Evaluate base model
python evaluate_countdown.py --model_path Qwen/Qwen2.5-1.5B-Instruct --test_size 128

# Evaluate fine-tuned models
python evaluate_countdown.py --model_path finetuned_es_pop30_iter500_final --test_size 128
python evaluate_countdown.py --model_path finetuned_zo_pert30_iter500_central_final --test_size 128
python evaluate_countdown.py --model_path finetuned_grpo_countdown_final --test_size 128

# Compare two models
python evaluate_countdown.py \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --compare finetuned_grpo_countdown_final \
    --test_size 128
```
