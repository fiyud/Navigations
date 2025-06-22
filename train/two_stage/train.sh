#!/bin/bash
# run_two_stage_training.sh

# Stage 1: Exploration + Auxiliary Learning
echo "Starting Stage 1: Exploration Training"
python train/nomad_rl/training/two_stage_trainer.py \
    --config config/nomad_rl_enhanced.yaml \
    --stage 1

# After Stage 1 completes, find the best checkpoint
STAGE1_CHECKPOINT="./checkpoints/nomad_rl_enhanced/stage1_best.pth"

# Check if Stage 1 checkpoint exists
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "Stage 1 checkpoint not found! Using latest checkpoint instead."
    STAGE1_CHECKPOINT=$(ls -t ./checkpoints/nomad_rl_enhanced/stage1_*.pth | head -1)
fi

echo "Stage 1 completed. Best checkpoint: $STAGE1_CHECKPOINT"

# Stage 2: Goal-Directed + Safety
echo "Starting Stage 2: Goal-Directed Training with Safety"
python train/nomad_rl/training/two_stage_trainer.py \
    --config config/nomad_rl_enhanced.yaml \
    --stage 2 \
    --stage1-checkpoint "$STAGE1_CHECKPOINT"

echo "Two-stage training completed!"