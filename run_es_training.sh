#!/bin/bash

# Evolution Strategies Fine-tuning Script
# Usage: ./run_es_training.sh [options]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default parameters (CPU is default due to MPS stability issues)
USE_CPU=true
DATA_SAMPLE=20
NUM_ITERATIONS=50
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            USE_CPU=true
            shift
            ;;
        --mps)
            USE_CPU=false
            echo -e "${YELLOW}Warning: MPS may cause segmentation faults. Use --cpu if you encounter issues.${NC}"
            shift
            ;;
        --data-sample)
            DATA_SAMPLE="$2"
            shift 2
            ;;
        --iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --cpu              Force CPU usage (RECOMMENDED - stable, slower)"
            echo "  --mps              Try MPS/GPU (WARNING: May cause segmentation faults)"
            echo "  --data-sample N    Number of data samples (default: 20)"
            echo "  --iterations N     Number of ES iterations (default: 50)"
            echo "  --verbose, -v      Enable verbose output"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --cpu                    # Run on CPU with defaults"
            echo "  $0 --mps --data-sample 50   # Run on MPS with 50 samples"
            echo "  $0 --cpu --iterations 10    # Quick test: 10 iterations on CPU"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Evolution Strategies Fine-tuning${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "es" ]; then
    echo -e "${YELLOW}Error: Virtual environment 'es' not found!${NC}"
    echo "Please create it first with: python3 -m venv es"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source es/bin/activate

# Check if model exists
MODEL_PATH="$HOME/Desktop/Qwen/Qwen2.5-1.5B-Instruct"
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Warning: Model not found at $MODEL_PATH${NC}"
    echo "The script will try to download it, or you can specify a different path."
fi

# Build command
CMD="python es_fine-tuning_countdown_mac.py"
CMD="$CMD --data_sample $DATA_SAMPLE"

if [ "$USE_CPU" = true ]; then
    CMD="$CMD --use_cpu"
    echo -e "${GREEN}Device: CPU (recommended - most stable)${NC}"
else
    echo -e "${YELLOW}Device: MPS (WARNING: May cause segmentation faults)${NC}"
    echo -e "${YELLOW}If you encounter crashes, use --cpu instead${NC}"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

# Display configuration
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Data samples: $DATA_SAMPLE"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Population size: 10 (hardcoded in script)"
echo "  Device: $([ "$USE_CPU" = true ] && echo "CPU" || echo "MPS/CPU")"
echo ""

# Note: NUM_ITERATIONS is hardcoded in the Python script, so we'll mention it
echo -e "${YELLOW}Note: Iterations are currently hardcoded to 50 in the script.${NC}"
echo -e "${YELLOW}To change, edit NUM_ITERATIONS in es_fine-tuning_countdown_mac.py${NC}"
echo ""

# Estimate time
if [ "$USE_CPU" = true ]; then
    ESTIMATED_TIME="1-1.5 hours"
else
    ESTIMATED_TIME="10-20 minutes"
fi
echo -e "${BLUE}Estimated time: $ESTIMATED_TIME${NC}"
echo ""

# Ask for confirmation
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Run the command
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Check the output directory for the fine-tuned model."
else
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Training encountered an error.${NC}"
    echo -e "${YELLOW}========================================${NC}"
    exit 1
fi
