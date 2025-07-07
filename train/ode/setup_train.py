#!/usr/bin/env python3
"""
Setup script for Neural ODE Navigation Training
Run this before starting training to create necessary files and directories.
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

def setup_directories():
    """Create all necessary directories"""
    dirs = [
        './config',
        './config/splits', 
        './checkpoints',
        './checkpoints/unified_advanced',
        './checkpoints/unified_advanced/graphs',
        './checkpoints/unified_advanced/results',
        './logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def create_scene_grid_sizes():
    """Create scene grid sizes with realistic estimates"""
    print("Creating scene grid size estimates...")
    
    np.random.seed(42)  # For reproducible estimates
    grid_sizes = {}
    
    # iTHOR Kitchen scenes (1-30) - compact layouts
    for i in range(1, 31):
        base = 50 + (i % 15) * 2
        noise = np.random.randint(-8, 12)
        grid_sizes[f'FloorPlan{i}'] = max(35, base + noise)
    
    # iTHOR Living rooms (201-230) - largest rooms
    for i in range(201, 231):
        base = 95 + (i % 20) * 3
        noise = np.random.randint(-10, 20)
        grid_sizes[f'FloorPlan{i}'] = max(75, base + noise)
    
    # iTHOR Bedrooms (301-330) - medium size
    for i in range(301, 331):
        base = 70 + (i % 15) * 2
        noise = np.random.randint(-8, 15)
        grid_sizes[f'FloorPlan{i}'] = max(50, base + noise)
    
    # iTHOR Bathrooms (401-430) - smallest rooms
    for i in range(401, 431):
        base = 40 + (i % 10) * 2
        noise = np.random.randint(-5, 10)
        grid_sizes[f'FloorPlan{i}'] = max(25, base + noise)
    
    # RoboTHOR Train scenes - large apartments
    for i in range(1, 13):
        for j in range(1, 6):
            base = 120 + i * 6 + j * 3
            noise = np.random.randint(-15, 25)
            grid_sizes[f'FloorPlan_Train{i}_{j}'] = max(90, base + noise)
    
    # RoboTHOR Val scenes
    for i in range(1, 4):
        for j in range(1, 6):
            base = 135 + i * 8 + j * 4
            noise = np.random.randint(-12, 20)
            grid_sizes[f'FloorPlan_Val{i}_{j}'] = max(100, base + noise)
    
    # Save to file
    grid_file = './config/splits/scene_grid_sizes.yaml'
    with open(grid_file, 'w') as f:
        yaml.dump(grid_sizes, f, default_flow_style=False)
    
    print(f"✓ Created grid sizes: {len(grid_sizes)} scenes")
    print(f"  Range: {min(grid_sizes.values())} - {max(grid_sizes.values())} grids")
    print(f"  Average: {np.mean(list(grid_sizes.values())):.1f} grids")
    
    return grid_sizes

def create_dataset_splits():
    """Create train/val/test splits for all datasets"""
    print("Creating dataset splits...")
    
    np.random.seed(42)
    
    # === iTHOR Splits ===
    ithor_kitchen = [f'FloorPlan{i}' for i in range(1, 31)]
    ithor_living = [f'FloorPlan{i}' for i in range(201, 231)]  
    ithor_bedroom = [f'FloorPlan{i}' for i in range(301, 331)]
    ithor_bathroom = [f'FloorPlan{i}' for i in range(401, 431)]
    
    def split_scenes(scenes, train_ratio=0.7, val_ratio=0.15):
        shuffled = scenes.copy()
        np.random.shuffle(shuffled)
        n_train = int(len(shuffled) * train_ratio)
        n_val = int(len(shuffled) * val_ratio)
        return {
            'train': shuffled[:n_train],
            'val': shuffled[n_train:n_train + n_val],
            'test': shuffled[n_train + n_val:]
        }
    
    # Split each room type
    kitchen_split = split_scenes(ithor_kitchen)
    living_split = split_scenes(ithor_living)
    bedroom_split = split_scenes(ithor_bedroom)
    bathroom_split = split_scenes(ithor_bathroom)
    
    ithor_splits = {
        'train': kitchen_split['train'] + living_split['train'] + 
                bedroom_split['train'] + bathroom_split['train'],
        'val': kitchen_split['val'] + living_split['val'] + 
              bedroom_split['val'] + bathroom_split['val'],
        'test': kitchen_split['test'] + living_split['test'] + 
               bedroom_split['test'] + bathroom_split['test']
    }
    
    # Shuffle final splits
    for split in ithor_splits:
        np.random.shuffle(ithor_splits[split])
    
    # === RoboTHOR Splits ===
    robothor_train_scenes = []
    for i in range(1, 13):
        for j in range(1, 6):
            robothor_train_scenes.append(f'FloorPlan_Train{i}_{j}')
    
    robothor_val_scenes = []
    for i in range(1, 4):
        for j in range(1, 6):
            robothor_val_scenes.append(f'FloorPlan_Val{i}_{j}')
    
    # Use some training scenes for test (RoboTHOR doesn't have official test)
    robothor_test_scenes = [
        'FloorPlan_Train1_1', 'FloorPlan_Train1_2', 'FloorPlan_Train1_3',
        'FloorPlan_Train2_1', 'FloorPlan_Train2_2', 'FloorPlan_Train2_3'
    ]
    
    # Remove test scenes from train
    robothor_train_filtered = [s for s in robothor_train_scenes if s not in robothor_test_scenes]
    
    robothor_splits = {
        'train': robothor_train_filtered,
        'val': robothor_val_scenes,
        'test': robothor_test_scenes
    }
    
    # === Combined Splits ===
    combined_splits = {
        'train': ithor_splits['train'] + robothor_splits['train'],
        'val': ithor_splits['val'] + robothor_splits['val'],
        'test': ithor_splits['test'] + robothor_splits['test']
    }
    
    # Shuffle combined splits
    for split in combined_splits:
        np.random.shuffle(combined_splits[split])
    
    # Save all splits
    splits_to_save = {
        'ithor': ithor_splits,
        'robothor': robothor_splits,
        'combined': combined_splits
    }
    
    for dataset_name, splits in splits_to_save.items():
        split_file = f'./config/splits/{dataset_name}_splits.yaml'
        with open(split_file, 'w') as f:
            yaml.dump(splits, f, default_flow_style=False)
        
        print(f"✓ Created {dataset_name} splits:")
        for split, scenes in splits.items():
            print(f"    {split}: {len(scenes)} scenes")
    
    return splits_to_save

def create_config_file():
    """Create the main configuration file"""
    print("Creating configuration file...")
    
    config = {
        # Dataset configuration
        'dataset': 'combined',
        'splits_file': './config/splits/combined_splits.yaml',
        
        # Environment settings
        'image_size': [224, 224],
        'max_episode_steps': 500,
        'success_distance': 1.0,
        'context_size': 5,
        'goal_prob': 0.5,
        'eval_goal_prob': 1.0,
        
        # Model Architecture
        'encoding_size': 256,
        'mha_num_attention_heads': 4,
        'mha_num_attention_layers': 4,
        'mha_ff_dim_factor': 4,
        'hidden_dim': 512,
        
        # Spatial Memory Graph ODE settings
        'max_nodes': 500,
        'distance_threshold': 1.0,
        'time_step': 0.1,
        
        # Counterfactual World Model
        'use_counterfactuals': True,
        'counterfactual_horizon': 5,
        
        # Curriculum Learning Settings
        'curriculum_type': 'grid',
        'curriculum_window_size': 100,
        'curriculum_success_threshold': 0.7,
        'curriculum_min_episodes': 100,
        'curriculum_update_freq': 10,
        'grid_size': 5,
        
        # Training parameters
        'total_timesteps': 1000000,  # 1M for initial testing
        'rollout_steps': 2048,
        'buffer_size': 2048,
        'batch_size': 64,
        'ppo_epochs': 10,
        'learning_rate': 0.0003,
        
        # PPO parameters
        'gamma': 0.99,
        'lam': 0.95,
        'clip_ratio': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'distance_coef': 0.1,
        'max_grad_norm': 0.5,
        
        # Auxiliary loss coefficients
        'counterfactual_coef': 0.1,
        'spatial_smoothness_coef': 0.05,
        
        # Performance optimization
        'use_amp': True,
        'device': 'cuda',
        
        # Evaluation settings
        'eval_episodes': 20,
        'val_freq': 100,
        
        # Logging and saving
        'log_freq': 10,
        'save_freq': 50,
        'save_dir': './checkpoints/unified_advanced',
        
        # Weights & Biases
        'use_wandb': False,  # Set to True if you want to use W&B
        'wandb_project': 'nomad-rl-unified-advanced',
        'run_name': 'unified_spatial_ode_experiment'
    }
    
    config_file = './config/ode_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Created config file: {config_file}")
    return config

def check_dependencies():
    """Check for required Python packages"""
    print("Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
        ('cv2', 'OpenCV (pip install opencv-python)'),
        ('gym', 'OpenAI Gym'),
    ]
    
    optional_packages = [
        ('ai2thor', 'AI2THOR (pip install ai2thor)'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('torchdiffeq', 'torchdiffeq (pip install torchdiffeq)'),
        ('wandb', 'Weights & Biases (pip install wandb)'),
        ('efficientnet_pytorch', 'EfficientNet (pip install efficientnet-pytorch)')
    ]
    
    missing_required = []
    missing_optional = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✓ {description}")
        except ImportError:
            print(f"✗ {description} - REQUIRED")
            missing_required.append(description)
    
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✓ {description}")
        except ImportError:
            print(f"⚠ {description} - OPTIONAL")
            missing_optional.append(description)
    
    if missing_required:
        print(f"\n Missing required packages: {missing_required}")
        print("Please install them before running training.")
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing optional packages: {missing_optional}")
        print("Training may not work without these packages.")
    
    return True

def create_minimal_test_config():
    """Create a minimal config for testing"""
    test_config = {
        'dataset': 'ithor',
        'splits_file': './config/splits/ithor_splits.yaml',
        'image_size': [224, 224],
        'max_episode_steps': 200,
        'success_distance': 1.0,
        'context_size': 3,
        'goal_prob': 0.8,
        'encoding_size': 128,
        'hidden_dim': 256,
        'max_nodes': 100,
        'distance_threshold': 1.0,
        'time_step': 0.1,
        'use_counterfactuals': False,  # Disable for simplicity
        'total_timesteps': 10000,  # Very short for testing
        'rollout_steps': 512,
        'buffer_size': 512,
        'batch_size': 32,
        'ppo_epochs': 4,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'lam': 0.95,
        'clip_ratio': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_amp': False,  # Disable mixed precision for debugging
        'device': 'cpu',  # Use CPU for initial testing
        'eval_episodes': 5,
        'val_freq': 20,
        'log_freq': 5,
        'save_freq': 20,
        'save_dir': './checkpoints/test',
        'use_wandb': False,
        'curriculum_type': 'grid',
        'curriculum_window_size': 20,
        'curriculum_success_threshold': 0.6,
        'curriculum_min_episodes': 20
    }
    
    test_config_file = './config/test_config.yaml'
    with open(test_config_file, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    print(f"✓ Created test config: {test_config_file}")
    return test_config

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📁 Created Files:")
    print("   ./config/ode_config.yaml - Main configuration")
    print("   ./config/test_config.yaml - Minimal test configuration")
    print("   ./config/splits/*.yaml - Dataset splits")
    print("   ./config/splits/scene_grid_sizes.yaml - Grid size estimates")
    
    print("\n🚀 Next Steps:")
    print("1. Install missing dependencies if any were reported")
    print("2. Test with minimal config:")
    print("   python train.py --config ./config/test_config.yaml --dataset ithor --mode train")
    print("\n3. Run full training:")
    print("   python train.py --config ./config/ode_config.yaml --dataset combined --mode train")
    print("\n4. Evaluate a model:")
    print("   python train.py --config ./config/ode_config.yaml --dataset combined --mode eval")
    
    print("\n⚙️ Configuration Notes:")
    print("- Edit ./config/ode_config.yaml to adjust hyperparameters")
    print("- Set 'use_wandb: true' for experiment tracking")
    print("- Adjust 'total_timesteps' based on your compute budget")
    print("- Use 'device: cpu' if no CUDA available")
    
    print("\n Troubleshooting:")
    print("- If AI2THOR fails: pip install ai2thor")
    print("- If PyTorch Geometric fails: pip install torch-geometric")
    print("- If torchdiffeq fails: pip install torchdiffeq")
    print("- Check logs in ./logs/ directory")
    
    print("="*60)

def main():
    """Main setup function"""
    print(" Setting up Neural ODE Navigation Training Environment")
    print("="*60)
    
    try:
        # 1. Check dependencies first
        if not check_dependencies():
            print("\n Please install missing required packages first!")
            return False
        
        # 2. Create directory structure
        setup_directories()
        
        # 3. Create scene grid sizes
        create_scene_grid_sizes()
        
        # 4. Create dataset splits
        create_dataset_splits()
        
        # 5. Create configuration files
        create_config_file()
        create_minimal_test_config()
        
        # 6. Print next steps
        print_next_steps()
        
        return True
        
    except Exception as e:
        print(f"\n Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)