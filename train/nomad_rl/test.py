import torch
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from typing import Dict, List
import argparse

import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer')

from train.nomad_rl.environments.ai2thor_nomad_env import AI2ThorNoMaDEnv
from train.nomad_rl.models.nomad_rl_model import NoMaDRL, prepare_observation
from train.nomad_rl.training.nomad_rl_trainer import NoMaDRLTrainer
from train.nomad_rl.evaluation.nomad_rl_eval import NoMaDRLEvaluator


def create_sample_config() -> Dict:
    return {
        'scene_names': ['FloorPlan1', 'FloorPlan2', 'FloorPlan3'],
        'image_size': [96, 96],
        'max_episode_steps': 200,  # Shorter for quick testing
        'success_distance': 1.0,
        'context_size': 5,
        'goal_prob': 0.7,
        
        # Model
        'encoding_size': 256,
        'mha_num_attention_heads': 4,
        'mha_num_attention_layers': 4,
        'mha_ff_dim_factor': 4,
        'hidden_dim': 512,
        
        # Training
        'total_timesteps': 50000,
        'rollout_steps': 1024,
        'buffer_size': 1024,
        'batch_size': 32,
        'ppo_epochs': 4,
        'learning_rate': 3e-4,
        
        # PPO
        'gamma': 0.99,
        'lam': 0.95,
        'clip_ratio': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'distance_coef': 0.1,
        'max_grad_norm': 0.5,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log_freq': 5,
        'save_freq': 20,
        'save_dir': './checkpoints/nomad_rl_example',
        'use_wandb': False,  # Disable for example
    }

def demo_environment():
    env = AI2ThorNoMaDEnv(
        scene_names=['FloorPlan1'],
        image_size=(96, 96),
        max_episode_steps=50,
        goal_prob=1.0  # Always goal-conditioned for demo
    )
    
    print("Environment initialized!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {env.observation_space.spaces.keys()}")
    
    for episode in range(2):
        print(f"\n--- Episode {episode + 1} ---")
        obs = env.reset()
        
        print(f"Goal-conditioned: {env.is_goal_conditioned}")
        print(f"Observation shapes:")
        for key, value in obs.items():
            print(f"  {key}: {value.shape}")
        
        total_reward = 0
        steps = 0
        
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, "
                  f"Distance={info.get('distance_to_goal', 0):.2f}")
            
            if done:
                break
        
        print(f"Episode finished: {steps} steps, {total_reward:.2f} total reward")
        print(f"Success: {info.get('success', False)}")
    
    env.close()
    print("Environment demo completed!")

def demo_model():
    print("\n=== NoMaD-RL Model Demo ===")
    
    config = create_sample_config()
    device = torch.device(config['device'])
    
    model = NoMaDRL(
        action_dim=6,  # AI2Thor actions
        encoding_size=config['encoding_size'],
        context_size=config['context_size'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    print(f"Model initialized on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    dummy_obs = {
        'rgb': torch.randn(1, 3, 96, 96).to(device),
        'context': torch.randn(1, 15, 96, 96).to(device),  # 3 * context_size channels
        'goal_rgb': torch.randn(1, 3, 96, 96).to(device),
        'goal_mask': torch.tensor([[0.0]]).to(device),  # Goal-conditioned
        'goal_position': torch.randn(1, 3).to(device)
    }
    
    print("\nTesting model forward pass...")
    
    try:
        with torch.no_grad():
            policy_out = model.forward(dummy_obs, mode="policy")
            print(f"Policy logits shape: {policy_out['policy_logits'].shape}")
            
            value_out = model.forward(dummy_obs, mode="value")
            print(f"Value shape: {value_out['values'].shape}")
            
            dist_out = model.forward(dummy_obs, mode="distance")
            print(f"Distance shape: {dist_out['distances'].shape}")
            
            action, log_prob = model.get_action(dummy_obs)
            print(f"Sampled action: {action.item()}, Log prob: {log_prob.item():.3f}")
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        raise
    
    print("Model demo completed!")

def quick_training_demo():    
    config = create_sample_config()
    config['total_timesteps'] = 5000
    config['log_freq'] = 2
    config['save_freq'] = 10
    
    print("Configuration:")
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)):
            print(f"  {key}: {value}")
    
    trainer = NoMaDRLTrainer(config)
    trainer.train(config['total_timesteps'])

def demo_evaluation():
    print("\n=== Evaluation Demo ===")
    
    config = create_sample_config()
    checkpoint_dir = config['save_dir']
    
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
            print(f"Found checkpoint: {latest_checkpoint}")
            
            evaluator = NoMaDRLEvaluator(config, latest_checkpoint)
            results = evaluator.evaluate(num_episodes=3, render=True)
            
            print("Evaluation Results:")
            print(f"  Success Rate: {results['success_rate']:.2%}")
            print(f"  Average Reward: {results['avg_reward']:.2f}")
            print(f"  Average Length: {results['avg_episode_length']:.1f}")
            
            evaluator.close()
        else:
            print("No checkpoints found - run training first!")
    else:
        print("Checkpoint directory not found - run training first!")

def main():
    parser = argparse.ArgumentParser(description='NoMaD-RL Complete Example')
    parser.add_argument('--demo', choices=['env', 'model', 'train', 'eval', 'all'], 
                       default='all', help='Which demo to run')
    args = parser.parse_args()
    
    print("\n=== NoMaD-RL Complete Example ===")

    if args.demo in ['env', 'all']:
        demo_environment()
    
    if args.demo in ['model', 'all']:
        demo_model()
    
    if args.demo in ['train', 'all']:
        print("\n Training demo will take several minutes...")
        response = input("Continue with training demo? (y/n): ")
        if response.lower() == 'y':
            quick_training_demo()
        else:
            print("Skipping training demo.")
    
    if args.demo in ['eval', 'all']:
        demo_evaluation()
            
if __name__ == '__main__':
    main()