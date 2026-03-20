#!/bin/bash
# Check GPU memory usage for running training processes

echo "Checking GPU memory usage..."
echo ""

# Find training process
PID=$(pgrep -f "grpo_fine-tuning_countdown_mac.py" | head -1)

if [ -z "$PID" ]; then
    echo "No GRPO training process found"
    exit 1
fi

echo "Training process PID: $PID"
echo ""

# Use PyTorch to check memory
/home/d3/miniforge3/envs/es_cuda/bin/python3 << 'PYEOF'
import torch
import sys

if not torch.cuda.is_available():
    print("CUDA not available")
    sys.exit(1)

device = torch.device('cuda:0')
props = torch.cuda.get_device_properties(device)

# Get memory stats
allocated = torch.cuda.memory_allocated(device)
reserved = torch.cuda.memory_reserved(device)
max_allocated = torch.cuda.max_memory_allocated(device)
total = props.total_memory

print("=" * 60)
print("GPU Memory Status (NVIDIA GB10)")
print("=" * 60)
print(f"Total GPU Memory:        {total / 1024**3:.2f} GB")
print(f"Currently Allocated:     {allocated / 1024**3:.2f} GB ({allocated / total * 100:.1f}%)")
print(f"Currently Reserved:      {reserved / 1024**3:.2f} GB ({reserved / total * 100:.1f}%)")
print(f"Max Allocated (session): {max_allocated / 1024**3:.2f} GB ({max_allocated / total * 100:.1f}%)")
print(f"Free Memory:             {(total - reserved) / 1024**3:.2f} GB ({(total - reserved) / total * 100:.1f}%)")
print("=" * 60)
print("")
print("Note: This shows memory from THIS Python process context.")
print("      The training process has its own CUDA context.")
print("      For accurate training process memory, check the training log.")
PYEOF

echo ""
echo "To see memory in training log, wait for next step completion."
echo "The updated script now logs GPU memory usage at each step."
