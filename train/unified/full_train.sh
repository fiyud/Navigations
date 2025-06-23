#!/bin/bash
# run_complete_pipeline.sh - Run entire two-stage training pipeline automatically

# Function to run full pipeline
run_full_pipeline() {
    local DATASET=$1
    local EXPERIMENT_NAME="${DATASET}_$(date +%Y%m%d_%H%M%S)"
    
    echo "======================================"
    echo "Running Full Pipeline for $DATASET"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "======================================"
    
    # Create experiment directory
    EXPERIMENT_DIR="./experiments/$EXPERIMENT_NAME"
    mkdir -p "$EXPERIMENT_DIR"
    
    # Copy config
    cp config/unified_training.yaml "$EXPERIMENT_DIR/"
    
    # Log file
    LOG_FILE="$EXPERIMENT_DIR/training.log"
    
    echo "Logging to: $LOG_FILE"
    echo ""
    
    # Stage 1 Training
    echo "[$(date)] Starting Stage 1 Training..." | tee -a "$LOG_FILE"
    ./run_experiments.sh -d "$DATASET" -s 1 -m train 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date)] Stage 1 training failed!" | tee -a "$LOG_FILE"
        return 1
    fi
    
    # Find stage 1 checkpoint
    STAGE1_CHECKPOINT="./checkpoints/unified/${DATASET}_stage1_best.pth"
    if [ ! -f "$STAGE1_CHECKPOINT" ]; then
        STAGE1_CHECKPOINT=$(ls -t ./checkpoints/unified/*stage1*.pth | head -1)
    fi
    
    echo "[$(date)] Stage 1 completed. Checkpoint: $STAGE1_CHECKPOINT" | tee -a "$LOG_FILE"
    
    # Stage 2 Training
    echo "[$(date)] Starting Stage 2 Training..." | tee -a "$LOG_FILE"
    ./run_experiments.sh -d "$DATASET" -s 2 -m train -c "$STAGE1_CHECKPOINT" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date)] Stage 2 training failed!" | tee -a "$LOG_FILE"
        return 1
    fi
    
    # Find stage 2 checkpoint
    STAGE2_CHECKPOINT="./checkpoints/unified/${DATASET}_stage2_best.pth"
    if [ ! -f "$STAGE2_CHECKPOINT" ]; then
        STAGE2_CHECKPOINT=$(ls -t ./checkpoints/unified/*stage2*.pth | head -1)
    fi
    
    echo "[$(date)] Stage 2 completed. Checkpoint: $STAGE2_CHECKPOINT" | tee -a "$LOG_FILE"
    
    # Final Testing
    echo "[$(date)] Running final test evaluation..." | tee -a "$LOG_FILE"
    ./run_experiments.sh -d "$DATASET" -m test -c "$STAGE2_CHECKPOINT" 2>&1 | tee -a "$LOG_FILE"
    
    # Copy results to experiment directory
    cp -r ./checkpoints/unified/results/* "$EXPERIMENT_DIR/" 2>/dev/null || true
    
    echo "[$(date)] Pipeline completed!" | tee -a "$LOG_FILE"
    echo "Results saved to: $EXPERIMENT_DIR" | tee -a "$LOG_FILE"
    
    return 0
}

# Main execution
if [ $# -eq 0 ]; then
    echo "Usage: $0 DATASET [DATASET2 ...]"
    echo ""
    echo "Run complete two-stage training pipeline for specified datasets"
    echo ""
    echo "DATASET options: ithor, robothor, combined"
    echo ""
    echo "Examples:"
    echo "  # Run on combined dataset"
    echo "  $0 combined"
    echo ""
    echo "  # Run on iTHOR only"
    echo "  $0 ithor"
    echo ""
    echo "  # Run on all datasets"
    echo "  $0 all"
    echo ""
    echo "  # Run on multiple datasets"
    echo "  $0 ithor robothor combined"
    exit 1
fi

# Check for 'all' option
if [ "$1" == "all" ]; then
    DATASETS="ithor robothor combined"
else
    DATASETS="$@"
fi

# Make scripts executable
chmod +x run_experiments.sh

# Run pipelines
for DATASET in $DATASETS; do
    if [ "$DATASET" != "ithor" ] && [ "$DATASET" != "robothor" ] && [ "$DATASET" != "combined" ]; then
        echo "Error: Unknown dataset $DATASET"
        continue
    fi
    
    # Run in background for parallel execution
    if [ "$1" == "all" ]; then
        run_full_pipeline "$DATASET" &
    else
        run_full_pipeline "$DATASET"
    fi
done

# Wait for all background jobs if running in parallel
if [ "$1" == "all" ]; then
    wait
    echo ""
    echo "All pipelines completed!"
fi