import torch
import numpy as np
import argparse
import yaml
import os
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F

import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train')

from nomad_rl.environments.ai2thor_nomad_env import AI2ThorNoMaDEnv
from nomad_rl.models.nomad_rl_model import NoMaDRL, prepare_observation

class NoMaDRLEvaluator:
    def __init__(self, config: Dict, checkpoint_path: str):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.env = AI2ThorNoMaDEnv(
            scene_names=config['scene_names'],
            image_size=tuple(config['image_size']),
            max_episode_steps=config['max_episode_steps'],
            success_distance=config['success_distance'],
            context_size=config['context_size'],
            goal_prob=1.0
        )
        
        self.model = NoMaDRL(
            action_dim=self.env.action_space.n,  # Now 4
            encoding_size=config['encoding_size'],
            context_size=config['context_size'],
            mha_num_attention_heads=config['mha_num_attention_heads'],
            mha_num_attention_layers=config['mha_num_attention_layers'],
            mha_ff_dim_factor=config['mha_ff_dim_factor'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)
        
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
    
    def _load_checkpoint(self, checkpoint_path: str):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    
    def evaluate(self, num_episodes: int = 10, render: bool = False, 
                deterministic: bool = True) -> Dict[str, float]:
        print(f"Evaluating model for {num_episodes} episodes...")
        
        results = {
            'episodes': [],
            'success_rate': 0,
            'avg_reward': 0,
            'avg_episode_length': 0,
            'avg_distance_to_goal': 0,
            'collision_rate': 0
        }
        
        episode_results = []
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            obs = self.env.reset()
            torch_obs = prepare_observation(obs, self.device)
            
            episode_reward = 0
            episode_length = 0
            collisions = 0
            success = False
            distances_to_goal = []
            
            trajectory = []
            
            while True:
                with torch.no_grad():
                    action, log_prob = self.model.get_action(torch_obs, deterministic=deterministic)
                
                next_obs, reward, done, info = self.env.step(action.cpu().item())
                
                trajectory.append({
                    'action': action.cpu().item(),
                    'reward': reward,
                    'goal_conditioned': info.get('goal_conditioned', False),
                    'distance_to_goal': info.get('distance_to_goal', 0),
                    'collision': info.get('collision', False)
                })
                
                episode_reward += reward
                episode_length += 1
                
                if info.get('collision', False):
                    collisions += 1
                
                if info.get('success', False):
                    success = True
                
                distances_to_goal.append(info.get('distance_to_goal', 0))
                
                if render:
                    print(f"Step {episode_length}: Action={action.item()}, "
                          f"Reward={reward:.2f}, Distance={info.get('distance_to_goal', 0):.2f}")
                
                if done:
                    break
                
                torch_obs = prepare_observation(next_obs, self.device)
            
            episode_result = {
                'episode': episode,
                'success': success,
                'reward': episode_reward,
                'length': episode_length,
                'collisions': collisions,
                'avg_distance_to_goal': np.mean(distances_to_goal) if distances_to_goal else 0,
                'final_distance_to_goal': distances_to_goal[-1] if distances_to_goal else 0,
                'trajectory': trajectory
            }
            
            episode_results.append(episode_result)
            
            print(f"  Result: {'SUCCESS' if success else 'FAILURE'}, "
                  f"Reward: {episode_reward:.2f}, Length: {episode_length}, "
                  f"Collisions: {collisions}, Final Distance: {episode_result['final_distance_to_goal']:.2f}")
        
        successes = sum(1 for r in episode_results if r['success'])
        results['success_rate'] = successes / num_episodes
        results['avg_reward'] = np.mean([r['reward'] for r in episode_results])
        results['avg_episode_length'] = np.mean([r['length'] for r in episode_results])
        results['avg_distance_to_goal'] = np.mean([r['avg_distance_to_goal'] for r in episode_results])
        results['collision_rate'] = np.mean([r['collisions'] / r['length'] for r in episode_results])
        results['episodes'] = episode_results
        
        return results
    
    def evaluate_exploration(self, num_episodes: int = 10) -> Dict[str, float]:
        print(f"Evaluating exploration for {num_episodes} episodes...")
        
        original_goal_prob = self.env.goal_prob
        self.env.goal_prob = 0.0
        
        exploration_results = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            torch_obs = prepare_observation(obs, self.device)
            
            visited_positions = set()
            episode_length = 0
            
            while episode_length < self.config['max_episode_steps']:
                with torch.no_grad():
                    action, _ = self.model.get_action(torch_obs, deterministic=False)
                
                next_obs, reward, done, info = self.env.step(action.cpu().item())
                episode_length += 1
                
                agent_pos = self.env._get_agent_position()
                pos_key = (round(agent_pos['x'], 1), round(agent_pos['z'], 1))
                visited_positions.add(pos_key)
                
                if done:
                    break
                
                torch_obs = prepare_observation(next_obs, self.device)
            
            exploration_results.append({
                'episode': episode,
                'unique_positions': len(visited_positions),
                'coverage': len(visited_positions) / episode_length if episode_length > 0 else 0
            })
        
        self.env.goal_prob = original_goal_prob
        
        avg_coverage = np.mean([r['coverage'] for r in exploration_results])
        avg_unique_positions = np.mean([r['unique_positions'] for r in exploration_results])
        
        return {
            'avg_coverage': avg_coverage,
            'avg_unique_positions': avg_unique_positions,
            'episodes': exploration_results
        }
    
    def plot_results(self, results: Dict, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('NoMaD-RL Evaluation Results')
        
        rewards = [ep['reward'] for ep in results['episodes']]
        axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        lengths = [ep['length'] for ep in results['episodes']]
        axes[0, 1].plot(lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        successes = [ep['success'] for ep in results['episodes']]
        success_counts = [sum(successes[:i+1]) for i in range(len(successes))]
        axes[1, 0].plot(success_counts)
        axes[1, 0].set_title('Cumulative Successes')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Successes')
        
        successful_episodes = [ep for ep in results['episodes'] if ep['success']]
        if successful_episodes:
            avg_distances = []
            for step in range(max(len(ep['trajectory']) for ep in successful_episodes)):
                step_distances = []
                for ep in successful_episodes:
                    if step < len(ep['trajectory']):
                        step_distances.append(ep['trajectory'][step]['distance_to_goal'])
                if step_distances:
                    avg_distances.append(np.mean(step_distances))
            
            if avg_distances:
                axes[1, 1].plot(avg_distances)
                axes[1, 1].set_title('Average Distance to Goal (Successful Episodes)')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Distance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Results plot saved to {save_path}")
        
        plt.show()
    
    def close(self):
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate NoMaD-RL model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes')
    parser.add_argument('--exploration', action='store_true',
                       help='Evaluate exploration capability')
    parser.add_argument('--plot', action='store_true',
                       help='Plot results')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluator = NoMaDRLEvaluator(config, args.checkpoint)
    
    try:
        if args.exploration:
            exploration_results = evaluator.evaluate_exploration(args.episodes)
            print(f"\n--- Exploration Results ---")
            print(f"Average Coverage: {exploration_results['avg_coverage']:.3f}")
            print(f"Average Unique Positions: {exploration_results['avg_unique_positions']:.1f}")
        else:
            results = evaluator.evaluate(args.episodes, args.render)
            
            print(f"\n--- Evaluation Results ---")
            print(f"Episodes: {args.episodes}")
            print(f"Success Rate: {results['success_rate']:.2%}")
            print(f"Average Reward: {results['avg_reward']:.2f}")
            print(f"Average Episode Length: {results['avg_episode_length']:.1f}")
            print(f"Average Distance to Goal: {results['avg_distance_to_goal']:.2f}")
            print(f"Collision Rate: {results['collision_rate']:.3f}")
            
            if args.plot:
                evaluator.plot_results(results, 'evaluation_results.png')
    
    finally:
        evaluator.close()

if __name__ == '__main__':
    main()