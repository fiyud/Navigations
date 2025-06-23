#!/bin/bash
# run_experiments.sh - Run complete training pipeline with single command

# Function to display usage
usage() {
    echo "Usage: $0 -d DATASET [-s STAGE] [-m MODE] [-c CHECKPOINT]"
    echo ""
    echo "Arguments:"
    echo "  -d DATASET      Dataset to use: 'ithor', 'robothor', or 'combined' (required)"
    echo "  -s STAGE        Training stage: 1 or 2 (default: 1)"
    echo "  -m MODE         Mode: 'train', 'eval', or 'test' (default: train)"
    echo "  -c CHECKPOINT   Checkpoint path for stage 2 or evaluation"
    echo ""
    echo "Examples:"
    echo "  # Train stage 1 on combined dataset"
    echo "  $0 -d combined -s 1"
    echo ""
    echo "  # Train stage 2 on iTHOR with stage 1 checkpoint"
    echo "  $0 -d ithor -s 2 -c checkpoints/unified/stage1_best.pth"
    echo ""
    echo "  # Evaluate on combined dataset validation set"
    echo "  $0 -d combined -m eval -c checkpoints/unified/stage2_best.pth"
    echo ""
    echo "  # Test on iTHOR test set"
    echo "  $0 -d ithor -m test -c checkpoints/unified/stage2_best.pth"
    exit 1
}

# Default values
DATASET=""
STAGE=1
MODE="train"
CHECKPOINT=""

# Parse command line arguments
while getopts "d:s:m:c:h" opt; do
    case $opt in
        d) DATASET="$OPTARG";;
        s) STAGE="$OPTARG";;
        m) MODE="$OPTARG";;
        c) CHECKPOINT="$OPTARG";;
        h) usage;;
        *) usage;;
    esac
done

# Check required arguments
if [ -z "$DATASET" ]; then
    echo "Error: Dataset is required!"
    usage
fi

# Validate dataset
if [ "$DATASET" != "ithor" ] && [ "$DATASET" != "robothor" ] && [ "$DATASET" != "combined" ]; then
    echo "Error: Dataset must be 'ithor', 'robothor', or 'combined'"
    usage
fi

# Validate stage
if [ "$STAGE" != "1" ] && [ "$STAGE" != "2" ]; then
    echo "Error: Stage must be 1 or 2"
    usage
fi

# Validate mode
if [ "$MODE" != "train" ] && [ "$MODE" != "eval" ] && [ "$MODE" != "test" ]; then
    echo "Error: Mode must be 'train', 'eval', or 'test'"
    usage
fi

# Check checkpoint requirement for stage 2
if [ "$STAGE" == "2" ] && [ "$MODE" == "train" ] && [ -z "$CHECKPOINT" ]; then
    echo "Error: Stage 2 training requires a stage 1 checkpoint!"
    echo "Looking for default checkpoint..."
    DEFAULT_CHECKPOINT="./checkpoints/unified/${DATASET}_stage1_best.pth"
    if [ -f "$DEFAULT_CHECKPOINT" ]; then
        CHECKPOINT="$DEFAULT_CHECKPOINT"
        echo "Found default checkpoint: $CHECKPOINT"
    else
        echo "No checkpoint found. Please specify with -c option."
        exit 1
    fi
fi

# Set up Python path
export PYTHONPATH="${PYTHONPATH}:/home/tuandang/tuandang/quanganh/visualnav-transformer"

# Create necessary directories
mkdir -p ./checkpoints/unified
mkdir -p ./config/splits

# Generate dataset splits if not exist
if [ "$DATASET" == "combined" ]; then
    if [ ! -f "./config/splits/combined_splits.yaml" ]; then
        echo "Generating combined dataset splits..."
        python train/nomad_rl/combined_dataset_splits.py
    fi
else
    if [ ! -f "./config/splits/${DATASET}_splits.yaml" ]; then
        echo "Generating dataset splits for $DATASET..."
        python train/nomad_rl/dataset_splits.py
    fi
fi

# Build command
CMD="python train/nomad_rl/training/unified_trainer.py"
CMD="$CMD --config config/unified_training.yaml"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --stage $STAGE"
CMD="$CMD --mode $MODE"

if [ ! -z "$CHECKPOINT" ]; then
    CMD="$CMD --stage1-checkpoint $CHECKPOINT" 
fi

# Display configuration
echo "======================================"
echo "NoMaD-RL Unified Training Pipeline"
echo "======================================"
echo "Dataset: $DATASET"
echo "Stage: $STAGE"
echo "Mode: $MODE"
if [ ! -z "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi
echo "======================================"
echo ""

# Run the command
echo "Executing: $CMD"
echo ""
$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Execution completed successfully!"
    echo "======================================"
    
    # Provide next steps
    if [ "$MODE" == "train" ]; then
        if [ "$STAGE" == "1" ]; then
            echo ""
            echo "Next step: Train stage 2 with:"
            echo "$0 -d $DATASET -s 2 -c ./checkpoints/unified/${DATASET}_stage1_best.pth"
        else
            echo ""
            echo "Training complete! Test the model with:"
            echo "$0 -d $DATASET -m test -c ./checkpoints/unified/${DATASET}_stage2_best.pth"
        fi
    fi
else
    echo ""
    echo "======================================"
    echo "Execution failed with error code $?"
    echo "======================================"
    exit 1
fi