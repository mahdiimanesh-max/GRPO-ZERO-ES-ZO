#!/bin/bash
# Zero-Order (ZO) Optimization Training Script for GPU
# Proper setup for Qwen2.5-1.5B-Instruct on Countdown Task
# Runs in screen session to persist after SSH disconnect

set -e

# Check if running in screen, if not, restart in screen
if [ -z "$STY" ] && [ -z "$TMUX" ]; then
    SESSION_NAME="zo_training_$(date +%Y%m%d_%H%M%S)"
    echo "Not running in screen/tmux. Starting in screen session: $SESSION_NAME"
    exec screen -dmS "$SESSION_NAME" bash -c "$0 $*; exec bash"
    exit 0
fi

# Configuration — match ES GPU comparison run (15 iters, same data/eval/checkpoint cadence as run_es_gpu.sh)
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
PRECISION="bfloat16"
DATA_SAMPLE=32
EVAL_DATA_SAMPLE=128
ITERATIONS=15
NUM_PERTURBATIONS=30
MU=0.001
LR=0.0005
MAX_NEW_TOKENS=256
EVAL_ITERATIONS=10
GRAD_METHOD="central"
NUM_THREADS=8
SEED=33

# Output directory
OUTPUT_DIR="results_zo"
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
if [ -d "/home/d3/miniforge3/envs/es_cuda" ]; then
    source /home/d3/miniforge3/bin/activate es_cuda
else
    echo "Error: Conda environment 'es_cuda' not found. Please run setup first."
    exit 1
fi

# Check for GPU
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "Warning: CUDA not available. Training will be slow on CPU."
fi

echo "=========================================="
echo "ZO Fine-tuning for Countdown Task (GPU)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Precision: $PRECISION"
echo "Training samples: $DATA_SAMPLE"
echo "Eval samples: $EVAL_DATA_SAMPLE"
echo "Iterations: $ITERATIONS"
echo "Perturbations per iteration: $NUM_PERTURBATIONS"
echo "Mu: $MU, Learning rate: $LR"
echo "Gradient method: $GRAD_METHOD"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Eval every: $EVAL_ITERATIONS iterations"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run training (unbuffered logs, same as ES)
python -u zo_fine-tuning_countdown_mac.py \
    --model_path "$MODEL_PATH" \
    --precision "$PRECISION" \
    --data_sample "$DATA_SAMPLE" \
    --eval_data_sample "$EVAL_DATA_SAMPLE" \
    --iterations "$ITERATIONS" \
    --num_perturbations "$NUM_PERTURBATIONS" \
    --mu "$MU" \
    --lr "$LR" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --eval_iterations "$EVAL_ITERATIONS" \
    --grad_method "$GRAD_METHOD" \
    --num_threads "$NUM_THREADS" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training complete! Check $OUTPUT_DIR/ for results."
