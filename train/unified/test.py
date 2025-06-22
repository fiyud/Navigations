# test_splits.py - Quick test to verify dataset splits are working
import sys
sys.path.append('/home/tuandang/tuandang/quanganh/visualnav-transformer')

from train.nomad_rl.dataset_splits import AI2THORDatasetSplitter
from train.nomad_rl.environments.ai2thor_nomad_env import AI2ThorNoMaDEnv

def test_splits():
    """Test that splits are generated correctly and environments can be created"""
    
    # Generate splits
    splitter = AI2THORDatasetSplitter(seed=42)
    
    # Test iTHOR splits
    print("Testing iTHOR splits...")
    ithor_splits = splitter.split_ithor_scenes()
    
    print(f"\niTHOR Split Summary:")
    print(f"Train: {len(ithor_splits['train'])} scenes")
    print(f"Val: {len(ithor_splits['val'])} scenes")
    print(f"Test: {len(ithor_splits['test'])} scenes")
    
    # Verify no overlap
    all_scenes = set(ithor_splits['train'] + ithor_splits['val'] + ithor_splits['test'])
    total_scenes = len(ithor_splits['train']) + len(ithor_splits['val']) + len(ithor_splits['test'])
    assert len(all_scenes) == total_scenes, "Found overlapping scenes!"
    
    # Test creating environments
    print("\nTesting environment creation...")
    for split_name, scenes in [('train', ithor_splits['train'][:5]), 
                               ('val', ithor_splits['val'][:2])]:
        print(f"\nCreating {split_name} environment with {len(scenes)} scenes...")
        env = AI2ThorNoMaDEnv(
            scene_names=scenes,
            image_size=(224, 224),
            max_episode_steps=50,
            goal_prob=1.0
        )
        
        # Test reset
        obs = env.reset()
        print(f"Environment reset successful. Scene: {env.current_scene}")
        print(f"Observation shapes: {[(k, v.shape) for k, v in obs.items()]}")
        
        # Test step
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Done={done}")
            if done:
                break
        
        env.close()
        print(f"{split_name} environment test completed!")
    
    # Test RoboTHOR splits
    print("\n\nTesting RoboTHOR splits...")
    robothor_splits = splitter.get_robothor_splits()
    print(f"Train: {len(robothor_splits['train'])} scenes")
    print(f"Val: {len(robothor_splits['val'])} scenes") 
    print(f"Test: {len(robothor_splits['test'])} scenes")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_splits()