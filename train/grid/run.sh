#!/bin/bash
# run_grid_curriculum.sh - Run grid-based curriculum training

# Function to display usage
usage() {
    echo "Usage: $0 -d DATASET [-s STAGE] [-c CHECKPOINT] [-g]"
    echo ""
    echo "Arguments:"
    echo "  -d DATASET      Dataset to use: 'ithor', 'robothor', or 'combined' (required)"
    echo "  -s STAGE        Training stage: 1 or 2 (default: 1)"
    echo "  -c CHECKPOINT   Stage 1 checkpoint path (required for stage 2)"
    echo "  -g              Compute grid sizes before training"
    echo ""
    echo "Examples:"
    echo "  # Compute grids and train stage 1 on combined dataset"
    echo "  $0 -d combined -s 1 -g"
    echo ""
    echo "  # Train stage 2 on iTHOR with existing checkpoint"
    echo "  $0 -d ithor -s 2 -c checkpoints/grid_curriculum/ithor_stage1_best.pth"
    exit 1
}

# Default values
DATASET=""
STAGE=1
CHECKPOINT=""
COMPUTE_GRIDS=false

# Parse command line arguments
while getopts "d:s:c:gh" opt; do
    case $opt in
        d) DATASET="$OPTARG";;
        s) STAGE="$OPTARG";;
        c) CHECKPOINT="$OPTARG";;
        g) COMPUTE_GRIDS=true;;
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

# Check checkpoint for stage 2
if [ "$STAGE" == "2" ] && [ -z "$CHECKPOINT" ]; then
    echo "Error: Stage 2 requires a checkpoint from stage 1!"
    exit 1
fi

# Set up paths
export PYTHONPATH="${PYTHONPATH}:/home/tuandang/tuandang/quanganh/visualnav-transformer"
CONFIG_FILE="config/grid_curriculum.yaml"
SPLITS_DIR="./config/splits"

# Create necessary directories
mkdir -p ./checkpoints/grid_curriculum
mkdir -p ./config/splits

# Check if config exists, if not create it
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating grid curriculum config file..."
    cat > "$CONFIG_FILE" << EOF
# Grid-based Curriculum Learning Configuration
dataset: null  # Set via command line

# Environment settings
image_size: [224, 224]
max_episode_steps: 500
success_distance: 1.0
context_size: 5
goal_prob: 0.5
eval_goal_prob: 1.0

# Model Architecture
encoding_size: 256
mha_num_attention_heads: 4
mha_num_attention_layers: 4
mha_ff_dim_factor: 4
hidden_dim: 512
lstm_hidden_size: 256
lstm_num_layers: 2

# Curriculum Settings
curriculum_window_size: 100
curriculum_success_threshold: 0.7
curriculum_min_episodes: 100
curriculum_update_freq: 10

# Two-Stage Training
training_stage: 1
stage1_timesteps: 1000000
stage2_timesteps: 2000000

# Learning rates
stage1_learning_rate: 0.001
stage2_learning_rate: 0.0001

# Reward Components
success_reward: 100.0
distance_weight: 20.0
step_penalty: 0.005
collision_penalty: 1.0
exploration_bonus: 5.0
curiosity_weight: 0.1

# Training Parameters
rollout_steps: 1024
buffer_size: 1024
batch_size: 32
ppo_epochs: 4

# PPO Parameters
gamma: 0.99
lam: 0.95
clip_ratio: 0.2
entropy_coef: 0.01
value_coef: 0.5
distance_coef: 0.1
auxiliary_coef: 0.1
max_grad_norm: 0.5

# Performance
use_amp: true
device: cuda

# Evaluation
eval_episodes: 20
val_freq: 100

# Logging
log_freq: 10
save_freq: 50
save_dir: ./checkpoints/grid_curriculum

# Weights & Biases
use_wandb: true
wandb_project: nomad-rl-grid-curriculum
run_name: grid_curriculum_training

# Splits file location
splits_file: ${SPLITS_DIR}/${DATASET}_splits.yaml
EOF
fi

# Compute grid sizes if requested
if [ "$COMPUTE_GRIDS" = true ]; then
    echo "Computing scene grid sizes for $DATASET dataset..."
    python compute_scene_grids.py --dataset "$DATASET" --output-dir "$SPLITS_DIR"
    
    if [ $? -ne 0 ]; then
        echo "Error computing grid sizes!"
        exit 1
    fi
fi

# Check if grid sizes exist
GRID_FILE="$SPLITS_DIR/scene_grid_sizes.yaml"
if [ ! -f "$GRID_FILE" ]; then
    echo "Grid sizes not found. Computing now..."
    python compute_scene_grids.py --dataset "$DATASET" --output-dir "$SPLITS_DIR"
fi

# Build training command
CMD="python grid_curriculum_trainer.py"
CMD="$CMD --config $CONFIG_FILE"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --stage $STAGE"

if [ ! -z "$CHECKPOINT" ]; then
    CMD="$CMD --stage1-checkpoint $CHECKPOINT"
fi

# Display configuration
echo "======================================"
echo "Grid-Based Curriculum Training"
echo "======================================"
echo "Dataset: $DATASET"
echo "Stage: $STAGE"
echo "Config: $CONFIG_FILE"
if [ ! -z "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi
echo "======================================"
echo ""

# Run training
echo "Executing: $CMD"
echo ""
$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Training completed successfully!"
    echo "======================================"
    
    # Provide next steps
    if [ "$STAGE" == "1" ]; then
        echo ""
        echo "Next step: Train stage 2 with:"
        echo "$0 -d $DATASET -s 2 -c ./checkpoints/grid_curriculum/${DATASET}_stage1_best.pth"
    else
        echo ""
        echo "Training complete! Results saved to ./checkpoints/grid_curriculum/results/"
    fi
else
    echo ""
    echo "======================================"
    echo "Training failed!"
    echo "======================================"
    exit 1
fi