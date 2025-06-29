#!/usr/bin/env python3
"""
Grid-based Curriculum Training for NoMaD-RL
Train the model progressively from small to large navigable spaces
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import argparse
import yaml
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train')

from grid_curriculum_manager import GridBasedCurriculumManager

from nomad_rl.environments.enhanced_ai2thor_env import EnhancedAI2ThorEnv
from nomad_rl.models.enhanced_nomad_rl_model import (
    EnhancedNoMaDRL, MultiComponentRewardCalculator, 
    EvaluationMetrics, prepare_observation
)
from nomad_rl.training.two_stage_trainer import TwoStageTrainer

class GridCurriculumTrainer(TwoStageTrainer):
    def __init__(self, config: Dict):
        self.dataset = config['dataset']
        self.splits = self._load_splits(config)
        
        self.curriculum_manager = GridBasedCurriculumManager(config, self.dataset)
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        self.all_train_scenes = self.splits['train']
        config['scene_names'] = self.curriculum_manager.get_current_scenes(self.all_train_scenes)
        
        config.update({
            'max_episode_steps': curriculum_settings['max_episode_steps'],
            'goal_prob': curriculum_settings['goal_prob']
        })
        
        super().__init__(config)
        
        # Override reward calculator with curriculum-aware version
        self.reward_calculator = self._create_curriculum_reward_calculator(config)
        
        self.val_scenes = self.splits['val']
        self.test_scenes = self.splits['test']
        
        self.results = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list),
            'curriculum': defaultdict(list)
        }
        
        self.best_val_success_rate = 0
        self.best_val_checkpoint = None
        
        # Curriculum update frequency
        self.last_curriculum_update = 0
        self.curriculum_update_freq = config.get('curriculum_update_freq', 10)
        
        print(f"\nInitialized Grid-Based Curriculum Learning")
        print(f"Starting with {len(config['scene_names'])} scenes from level 0")
        print(f"Grid range: {curriculum_settings['min_grid_size']:.0f} - {curriculum_settings['max_grid_size']:.0f}")
    
    def _create_curriculum_reward_calculator(self, config: Dict) -> MultiComponentRewardCalculator:
        """Create reward calculator with curriculum-adjusted settings"""
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        reward_config = config.copy()
        reward_config['training_stage'] = self.stage
        
        # Apply curriculum-specific multipliers
        if 'collision_penalty_multiplier' in curriculum_settings:
            reward_config['collision_penalty'] = (
                config.get('collision_penalty', 1.0) * 
                curriculum_settings['collision_penalty_multiplier']
            )
        
        # Adjust exploration bonus based on grid size
        # Larger environments get higher exploration bonuses
        grid_range = curriculum_settings['max_grid_size'] - curriculum_settings['min_grid_size']
        exploration_multiplier = 1.0 + (grid_range / 100.0)  # Scale by grid range
        reward_config['exploration_bonus'] = (
            config.get('exploration_bonus', 5.0) * exploration_multiplier
        )
        
        return MultiComponentRewardCalculator(reward_config)
    
    def _update_environment_curriculum(self):
        curriculum_settings = self.curriculum_manager.get_current_settings()
        new_scenes = self.curriculum_manager.get_current_scenes(self.all_train_scenes)
        
        self.env.close()
        
        self.env = self._create_environment(
            new_scenes,
            self.config,
            goal_prob=curriculum_settings['goal_prob']
        )
        
        self.env.max_episode_steps = curriculum_settings['max_episode_steps']
        self.env.goal_prob = curriculum_settings['goal_prob']
        
        self.reward_calculator = self._create_curriculum_reward_calculator(self.config)
        
        if curriculum_settings['max_distance'] != float('inf'):
            self.env.max_goal_distance = curriculum_settings['max_distance']
        
        grid_stats = self.curriculum_manager.get_progress_stats()
        print(f"\n{'='*60}")
        print(f"CURRICULUM UPDATE - Level {self.curriculum_manager.current_level}")
        print(f"Grid Range: {grid_stats['grid_range'][0]:.0f} - {grid_stats['grid_range'][1]:.0f}")
        print(f"Scenes: {len(new_scenes)}")
        print(f"Max Distance: {curriculum_settings['max_distance']}")
        print(f"Episode Steps: {curriculum_settings['max_episode_steps']}")
        print(f"{'='*60}\n")
    
    def _create_environment(self, scenes: List[str], config: Dict, goal_prob: float):
        return EnhancedAI2ThorEnv(
            scene_names=scenes,
            image_size=tuple(config['image_size']),
            max_episode_steps=config['max_episode_steps'],
            success_distance=config['success_distance'],
            context_size=config['context_size'],
            goal_prob=goal_prob
        )
    
    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        rollout_stats = super().collect_rollouts(num_steps)
        
        # Track grid size statistics
        if hasattr(self, 'curriculum_manager'):
            current_scene = self.env.current_scene
            if current_scene in self.curriculum_manager.scene_grid_sizes:
                grid_size = self.curriculum_manager.scene_grid_sizes[current_scene]
                rollout_stats['avg_grid_size'] = grid_size
        
        return rollout_stats
    
    def train_with_curriculum(self, total_timesteps: int):
        print(f"Starting grid-based curriculum training for {total_timesteps} timesteps...")
        print(f"Dataset: {self.dataset}")
        print(f"Stage: {self.stage}")
        
        timesteps_collected = 0
        update_count = 0
        
        while timesteps_collected < total_timesteps:
            if update_count > 0 and update_count % self.curriculum_update_freq == 0:
                old_level = self.curriculum_manager.current_level
                
                if self.curriculum_manager.current_level != old_level:
                    self._update_environment_curriculum()
                    self.last_curriculum_update = update_count
            
            rollout_stats = self.collect_rollouts(self.config['rollout_steps'])
            timesteps_collected += self.config['rollout_steps']
            
            update_stats = self.update_policy()
            update_count += 1
            
            if update_count % 10 == 0:
                torch.cuda.empty_cache()
            
            if update_count % self.config['log_freq'] == 0:
                self._log_training_stats_with_grid_info(
                    timesteps_collected, rollout_stats, update_stats
                )
                
                # Track curriculum progress
                curriculum_stats = self.curriculum_manager.get_progress_stats()
                self.results['curriculum']['timesteps'].append(timesteps_collected)
                self.results['curriculum']['level'].append(curriculum_stats['current_level'])
                self.results['curriculum']['grid_range'].append(curriculum_stats['grid_range'])
                self.results['curriculum']['success_rate'].append(
                    curriculum_stats['current_success_rate']
                )
            
            if update_count % self.config.get('val_freq', 100) == 0:
                val_metrics = self._evaluate_on_split('val', num_episodes=20)
                self.results['val']['timesteps'].append(timesteps_collected)
                for key, value in val_metrics.items():
                    self.results['val'][key].append(value)
                
                if val_metrics['success_rate'] > self.best_val_success_rate:
                    self.best_val_success_rate = val_metrics['success_rate']
                    self.best_val_checkpoint = self._save_model(
                        update_count, timesteps_collected, is_best=True
                    )
                    print(f"New best validation success rate: {self.best_val_success_rate:.2%}")
            
            if update_count % self.config['save_freq'] == 0:
                self._save_model(update_count, timesteps_collected)
        
        print("\nTraining completed! Running final evaluation...")
        
        if self.best_val_checkpoint and os.path.exists(self.best_val_checkpoint):
            print(f"Loading best model from {self.best_val_checkpoint}")
            checkpoint = torch.load(self.best_val_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        final_results = self._run_final_evaluation()
        self._save_final_results_with_grid_info(final_results)
        
        self.env.close()
        
        return final_results
    
    def _log_training_stats_with_grid_info(self, timesteps: int, rollout_stats: Dict, update_stats: Dict):
        curriculum_stats = self.curriculum_manager.get_progress_stats()
        
        print(f"\n--- Stage {self.stage} Update {timesteps // self.config['rollout_steps']} "
              f"(Timesteps: {timesteps}) ---")
        print(f"Curriculum Level: {curriculum_stats['current_level']} ({curriculum_stats['level_name']})")
        print(f"Grid Range: {curriculum_stats['grid_range'][0]:.0f} - {curriculum_stats['grid_range'][1]:.0f}")
        print(f"Episodes at Level: {curriculum_stats['episodes_at_level']}")
        
        if len(self.episode_rewards) > 0:
            print(f"Episode Reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
            print(f"Episode Length: {np.mean(self.episode_lengths):.1f}")
            print(f"Success Rate: {np.mean(self.success_rates):.2%}")
        
        print(f"Policy Loss: {update_stats['policy_loss']:.4f}")
        print(f"Value Loss: {update_stats['value_loss']:.4f}")
        print(f"Entropy: {-update_stats['entropy_loss']:.4f}")
        
        if self.config.get('use_wandb', False):
            log_dict = {
                'timesteps': timesteps,
                'stage': self.stage,
                'curriculum/level': curriculum_stats['current_level'],
                'curriculum/level_name': curriculum_stats['level_name'],
                'curriculum/min_grid_size': curriculum_stats['grid_range'][0],
                'curriculum/max_grid_size': curriculum_stats['grid_range'][1],
                'curriculum/episodes_at_level': curriculum_stats['episodes_at_level'],
                'episode_reward_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'episode_length_mean': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'success_rate': np.mean(self.success_rates) if self.success_rates else 0,
                'policy_loss': update_stats['policy_loss'],
                'value_loss': update_stats['value_loss'],
                'entropy': -update_stats['entropy_loss'],
                'approx_kl': update_stats['approx_kl'],
            }
            
            wandb.log(log_dict)
    
    def _evaluate_on_split(self, split_name: str, num_episodes: Optional[int] = None) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = self.config.get('eval_episodes', 20)
        
        print(f"\nEvaluating on {split_name} split ({num_episodes} episodes)...")
        
        if split_name == 'val':
            eval_scenes = self.val_scenes
        elif split_name == 'test':
            eval_scenes = self.test_scenes
        else:
            eval_scenes = self.splits[split_name]
        
        eval_env = self._create_environment(
            eval_scenes,
            self.config,
            goal_prob=self.config.get('eval_goal_prob', 1.0)
        )
        
        self.model.eval()
        metrics = EvaluationMetrics()
        episode_results = []
        
        for episode_idx in range(num_episodes):
            obs = eval_env.reset()
            torch_obs = prepare_observation(obs, self.device)
            
            if hasattr(self.model, 'reset_hidden'):
                self.model.reset_hidden()
            hidden_state = None
            
            episode_reward = 0
            episode_length = 0
            collision_count = 0
            
            current_scene = eval_env.current_scene
            grid_size = self.curriculum_manager.scene_grid_sizes.get(current_scene, 0)
            
            while episode_length < self.config['max_episode_steps']:
                with torch.no_grad():
                    if hasattr(self.model, 'get_action'):
                        action, _, hidden_state = self.model.get_action(
                            torch_obs, deterministic=True, hidden_state=hidden_state
                        )
                    else:
                        outputs = self.model.forward(torch_obs, mode="policy")
                        action = outputs['action_dist'].probs.argmax(dim=-1)
                
                next_obs, _, done, info = eval_env.step(action.cpu().item())
                event = eval_env.controller.last_event
                reward = self.reward_calculator.calculate_reward(event, next_obs, info, eval_env)
                
                episode_reward += reward
                episode_length += 1
                
                if info.get('collision', False):
                    collision_count += 1
                
                if done:
                    break
                
                torch_obs = prepare_observation(next_obs, self.device)
            
            episode_results.append({
                'success': info.get('success', False),
                'reward': episode_reward,
                'length': episode_length,
                'collisions': collision_count,
                'grid_size': grid_size,
                'scene': current_scene
            })
            
            if (episode_idx + 1) % 10 == 0:
                current_sr = sum(r['success'] for r in episode_results) / len(episode_results)
                print(f"  Completed {episode_idx + 1}/{num_episodes} episodes (SR: {current_sr:.2%})")
        
        eval_env.close()
        self.model.train()
        
        eval_metrics = {
            'success_rate': np.mean([r['success'] for r in episode_results]),
            'avg_reward': np.mean([r['reward'] for r in episode_results]),
            'avg_length': np.mean([r['length'] for r in episode_results]),
            'avg_collisions': np.mean([r['collisions'] for r in episode_results]),
            'avg_grid_size': np.mean([r['grid_size'] for r in episode_results]),
            'num_episodes': num_episodes
        }
        
        grid_sizes = [r['grid_size'] for r in episode_results]
        if grid_sizes:
            quartiles = np.percentile(grid_sizes, [25, 50, 75])
            
            small_success = np.mean([r['success'] for r in episode_results if r['grid_size'] <= quartiles[0]])
            medium_success = np.mean([r['success'] for r in episode_results if quartiles[0] < r['grid_size'] <= quartiles[2]])
            large_success = np.mean([r['success'] for r in episode_results if r['grid_size'] > quartiles[2]])
            
            eval_metrics['success_rate_small_grids'] = small_success
            eval_metrics['success_rate_medium_grids'] = medium_success
            eval_metrics['success_rate_large_grids'] = large_success
        
        print(f"\n{split_name.upper()} Results:")
        print(f"  Overall Success Rate: {eval_metrics['success_rate']:.2%}")
        print(f"  Avg Grid Size: {eval_metrics['avg_grid_size']:.1f}")
        if 'success_rate_small_grids' in eval_metrics:
            print(f"  Success by Grid Size:")
            print(f"    Small (<{quartiles[0]:.0f}): {eval_metrics['success_rate_small_grids']:.2%}")
            print(f"    Medium ({quartiles[0]:.0f}-{quartiles[2]:.0f}): {eval_metrics['success_rate_medium_grids']:.2%}")
            print(f"    Large (>{quartiles[2]:.0f}): {eval_metrics['success_rate_large_grids']:.2%}")
        
        return eval_metrics
    
    def _save_final_results_with_grid_info(self, results: Dict):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(self.config['save_dir'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(
            results_dir,
            f'{self.dataset}_stage{self.stage}_grid_curriculum_{timestamp}_results.json'
        )
        
        comprehensive_results = {
            'config': self.config,
            'final_results': results,
            'training_history': {
                split: dict(self.results[split])
                for split in ['train', 'val', 'test', 'curriculum']
            },
            'curriculum_progression': {
                'final_level': self.curriculum_manager.current_level,
                'total_levels': len(self.curriculum_manager.levels),
                'performance_history': self.curriculum_manager.performance_history,
                'levels': self.curriculum_manager.levels
            },
            'best_val_success_rate': self.best_val_success_rate,
            'timestamp': timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY - GRID-BASED CURRICULUM")
        print("="*70)
        print(f"Final Curriculum Level: {self.curriculum_manager.current_level} / {len(self.curriculum_manager.levels) - 1}")
        print(f"Final Grid Range: {self.curriculum_manager.levels[self.curriculum_manager.current_level]['min_grid_size']:.0f} - "
              f"{self.curriculum_manager.levels[self.curriculum_manager.current_level]['max_grid_size']:.0f}")
        print("-"*70)
        
        print(f"{'Metric':<30} {'Train':>10} {'Val':>10} {'Test':>10}")
        print("-"*70)
        
        metrics_to_show = [
            ('Success Rate (%)', 'success_rate', 100),
            ('Avg Reward', 'avg_reward', 1),
            ('Avg Episode Length', 'avg_length', 1),
            ('Avg Grid Size', 'avg_grid_size', 1),
        ]
        
        for metric_name, metric_key, multiplier in metrics_to_show:
            train_val = results.get('train', {}).get(metric_key, 0) * multiplier
            val_val = results.get('val', {}).get(metric_key, 0) * multiplier
            test_val = results.get('test', {}).get(metric_key, 0) * multiplier
            print(f"{metric_name:<30} {train_val:>10.2f} {val_val:>10.2f} {test_val:>10.2f}")
        
        print("="*70)
    
    def _run_final_evaluation(self) -> Dict:
        final_results = {}
        
        print("\n=== Training Set Evaluation ===")
        train_metrics = self._evaluate_on_split('train', num_episodes=50)
        final_results['train'] = train_metrics
        
        print("\n=== Validation Set Evaluation ===")
        val_metrics = self._evaluate_on_split('val', num_episodes=50)
        final_results['val'] = val_metrics
        
        print("\n=== Test Set Evaluation ===")
        test_metrics = self._evaluate_on_split('test', num_episodes=100)
        final_results['test'] = test_metrics
        
        return final_results
    
    def _load_splits(self, config: Dict) -> Dict[str, List[str]]:
        dataset = config['dataset']
        
        if dataset == 'combined':
            splits_file = config.get('splits_file', './config/splits/combined_splits.yaml')
        else:
            splits_file = config.get('splits_file', f'./config/splits/{dataset}_splits.yaml')
        
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = yaml.safe_load(f)
            print(f"Loaded splits from {splits_file}")
        else:
            raise FileNotFoundError(f"Splits file not found: {splits_file}")
        
        return splits


def main():
    parser = argparse.ArgumentParser(description='Grid-based Curriculum Training for NoMaD-RL')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['ithor', 'robothor', 'combined'],
                       required=True, help='Dataset to use')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                       help='Training stage')
    parser.add_argument('--stage1-checkpoint', type=str, default=None,
                       help='Stage 1 checkpoint for stage 2')
    parser.add_argument('--compute-grids', action='store_true',
                       help='Compute grid sizes before training')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['dataset'] = args.dataset
    config['training_stage'] = args.stage
    if args.stage1_checkpoint:
        config['stage1_checkpoint'] = args.stage1_checkpoint
    
    if args.compute_grids:
        print("Computing scene grid sizes...")
        os.system(f"python compute_scene_grids.py --dataset {args.dataset}")
    
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'nomad-rl-grid-curriculum'),
            name=f"{args.dataset}_grid_curriculum_stage{args.stage}",
            config=config
        )
    
    trainer = GridCurriculumTrainer(config)
    results = trainer.train_with_curriculum(config[f'stage{args.stage}_timesteps'])
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()