#!/bin/bash
# GRPO (Group Relative Policy Optimization) Training Script for GPU
# Proper setup for Qwen2.5-1.5B-Instruct on Countdown Task
# Runs in screen session to persist after SSH disconnect

set -e

# Check if running in screen, if not, restart in screen
if [ -z "$STY" ] && [ -z "$TMUX" ]; then
    SESSION_NAME="grpo_training_$(date +%Y%m%d_%H%M%S)"
    echo "Not running in screen/tmux. Starting in screen session: $SESSION_NAME"
    exec screen -dmS "$SESSION_NAME" bash -c "$0 $*; exec bash"
    exit 0
fi

# Configuration
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
PRECISION="bfloat16"
DATA_SAMPLE=500
TEST_SIZE=128
BATCH_SIZE=128
NUM_QUESTIONS=32
MICRO_BATCH_SIZE=4
MAX_GEN_LEN=1024
LR=1e-5
MAX_GRAD_NORM=1.0
EVAL_INTERVAL=10
CKPT_INTERVAL=50
TEMPERATURE=0.7
SEED=1337

# Output directory
OUTPUT_DIR="results_grpo"
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
echo "GRPO Fine-tuning for Countdown Task (GPU)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Precision: $PRECISION"
echo "Training samples: $DATA_SAMPLE"
echo "Test samples: $TEST_SIZE"
echo "Batch size: $BATCH_SIZE ($NUM_QUESTIONS questions × $((BATCH_SIZE / NUM_QUESTIONS)) answers)"
echo "Micro batch size: $MICRO_BATCH_SIZE"
echo "Max gen length: $MAX_GEN_LEN"
echo "Learning rate: $LR"
echo "Temperature: $TEMPERATURE"
echo "Eval interval: $EVAL_INTERVAL steps"
echo "Checkpoint interval: $CKPT_INTERVAL steps"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run training
python grpo_fine-tuning_countdown_mac.py \
    --model_path "$MODEL_PATH" \
    --precision "$PRECISION" \
    --data_sample "$DATA_SAMPLE" \
    --test_size "$TEST_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --num_questions "$NUM_QUESTIONS" \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --max_gen_len "$MAX_GEN_LEN" \
    --lr "$LR" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --eval_interval "$EVAL_INTERVAL" \
    --ckpt_interval "$CKPT_INTERVAL" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training complete! Check $OUTPUT_DIR/ for results."
