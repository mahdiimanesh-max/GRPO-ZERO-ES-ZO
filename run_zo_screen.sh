#!/bin/bash
# Wrapper script to run ZO training in a screen session
# Usage: ./run_zo_screen.sh

set -e

SCRIPT_DIR="/home/d3/es-fine-tuning-paper"
SESSION_NAME="zo_training_$(date +%Y%m%d_%H%M%S)"

echo "Starting ZO training in screen session: $SESSION_NAME"
echo "To attach: screen -r $SESSION_NAME"
echo "To detach: Press Ctrl+A then D"
echo ""

cd "$SCRIPT_DIR"

# Start screen session with the training script
screen -dmS "$SESSION_NAME" bash -c "
    cd $SCRIPT_DIR && \
    source /home/d3/miniforge3/bin/activate es_cuda && \
    ./run_zo_gpu.sh && \
    exec bash
"

echo "Screen session '$SESSION_NAME' started!"
echo ""
echo "To attach to the session:"
echo "  screen -r $SESSION_NAME"
echo ""
echo "To list all screen sessions:"
echo "  screen -ls"
echo ""
echo "To detach from screen (while attached):"
echo "  Press: Ctrl+A, then D"
