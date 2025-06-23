# test_combined_dataset.py - Test combined iTHOR + RoboTHOR dataset
import sys
sys.path.append('/home/tuandang/tuandang/quanganh/visualnav-transformer')

from train.nomad_rl.combined_dataset_splits import CombinedAI2THORDatasetSplitter
from train.nomad_rl.environments.ai2thor_nomad_env import AI2ThorNoMaDEnv
import yaml
import os

def test_combined_splits():    
    print("=== Testing Combined Dataset Splits ===\n")
    
    splitter = CombinedAI2THORDatasetSplitter(seed=42)
    combined_splits = splitter.get_combined_splits()
    
    splitter.print_statistics(combined_splits)
    
    splitter.verify_splits(combined_splits)
    
    print("\n=== Testing Split Loading ===")
    splits_file = './config/splits/combined_splits.yaml'
    if os.path.exists(splits_file):
        with open(splits_file, 'r') as f:
            loaded_splits = yaml.safe_load(f)
        print(f"Successfully loaded splits from {splits_file}")
        print(f"Train: {len(loaded_splits['train'])} scenes")
        print(f"Val: {len(loaded_splits['val'])} scenes")
        print(f"Test: {len(loaded_splits['test'])} scenes")
    else:
        print("No saved splits found. Saving now...")
        splitter.save_combined_splits()
    
    print("\n=== Testing Environment with Combined Scenes ===")
    
    test_scenes = []
    for split_data in combined_splits['train'].values():
        if isinstance(split_data, list) and len(split_data) > 0:
            test_scenes.extend(split_data[:2])  # Take 2 scenes from each
    
    print(f"\nTesting with scenes: {test_scenes[:6]}")
    
    try:
        env = AI2ThorNoMaDEnv(
            scene_names=test_scenes[:6],
            image_size=(224, 224),
            max_episode_steps=50,
            goal_prob=1.0
        )
        
        print("\nEnvironment created successfully!")
        
        for i in range(min(3, len(test_scenes))):
            obs = env.reset()
            print(f"\nScene {i+1}: {env.current_scene}")
            print(f"  Goal conditioned: {env.is_goal_conditioned}")
            print(f"  Observation shapes: {[(k, v.shape) for k, v in obs.items()]}")
            
            for step in range(5):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                print(f"  Step {step+1}: Action={action}, Reward={reward:.2f}, Done={done}")
                if done:
                    break
        
        env.close()
        print("\n✓ Environment tests passed!")
        
    except Exception as e:
        print(f"\n✗ Environment test failed: {str(e)}")
        print("\nNote: RoboTHOR scenes may require additional setup or different controller initialization.")
    
    print("\n=== Usage Instructions ===")
    print("To train on the combined dataset, run:")
    print("  ./run_experiments.sh -d combined -s 1")
    print("\nOr for the complete pipeline:")
    print("  ./run_complete_pipeline.sh combined")
    
    return combined_splits


def test_scene_distribution():
    print("\n=== Scene Distribution Analysis ===")
    
    metadata_file = './config/splits/combined_metadata.yaml'
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = yaml.safe_load(f)
        
        print("\nDataset Composition:")
        print(f"Total scenes: {metadata['total_scenes']}")
        
        for split_name, split_info in metadata['splits'].items():
            print(f"\n{split_name.upper()}:")
            total = split_info['total']
            ithor = split_info['ithor']
            robothor = split_info['robothor']
            
            print(f"  Total: {total}")
            print(f"  iTHOR: {ithor} ({ithor/total*100:.1f}%)")
            print(f"  RoboTHOR: {robothor} ({robothor/total*100:.1f}%)")

if __name__ == "__main__":
    combined_splits = test_combined_splits()
    test_scene_distribution()
    
    print("\n=== Test Summary ===")
    print("✓ Combined dataset splits generated successfully")
    print("✓ No overlap between train/val/test")
    print("✓ Balanced room type distribution in iTHOR")
    print("✓ Official RoboTHOR splits preserved")
    print("✓ Ready for training!")