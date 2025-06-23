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

import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train')

from nomad_rl.environments.enhanced_ai2thor_env import EnhancedAI2ThorEnv
from nomad_rl.models.enhanced_nomad_rl_model import (
    EnhancedNoMaDRL, MultiComponentRewardCalculator, 
    EvaluationMetrics, prepare_observation
)
from nomad_rl.training.two_stage_trainer import TwoStageTrainer
from nomad_rl.training.enhanced_curriculum_manager import EnhancedCurriculumManager

class UnifiedTrainerWithCurriculum(TwoStageTrainer):
    def __init__(self, config: Dict):
        self.dataset = config['dataset']
        self.splits = self._load_splits(config)
        
        self.curriculum_manager = EnhancedCurriculumManager(config, self.dataset)
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        self.all_train_scenes = self.splits['train']
        config['scene_names'] = self.curriculum_manager.get_current_scenes(self.all_train_scenes)
        
        config.update({
            'max_episode_steps': curriculum_settings['max_episode_steps'],
            'goal_prob': curriculum_settings['goal_prob']
        })
        
        super().__init__(config)
        
        # Override reward calculator to use curriculum-adjusted penalties
        self.reward_calculator = self._create_curriculum_reward_calculator(config)
        
        # Create validation and test environments
        self.val_env = self._create_environment(
            self.splits['val'], 
            config, 
            goal_prob=config.get('eval_goal_prob', 1.0)
        )
        
        self.test_env = self._create_environment(
            self.splits['test'], 
            config, 
            goal_prob=config.get('eval_goal_prob', 1.0)
        )
        
        self.results = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list),
            'curriculum': defaultdict(list)
        }
        
        self.best_val_success_rate = 0
        self.best_val_checkpoint = None
        
        self.last_curriculum_update = 0
        self.curriculum_update_freq = config.get('curriculum_update_freq', 10)
    
    def _create_curriculum_reward_calculator(self, config: Dict) -> MultiComponentRewardCalculator:
        """Create reward calculator with curriculum-adjusted settings"""
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        # Adjust reward config based on curriculum
        reward_config = config.copy()
        reward_config['training_stage'] = self.stage
        
        # Apply curriculum-specific multipliers
        if 'collision_penalty_multiplier' in curriculum_settings:
            reward_config['collision_penalty'] = (
                config.get('collision_penalty', 1.0) * 
                curriculum_settings['collision_penalty_multiplier']
            )
        
        return MultiComponentRewardCalculator(reward_config)
    
    def _create_environment(self, scenes: List[str], config: Dict, goal_prob: float):
        return EnhancedAI2ThorEnv(
            scene_names=scenes,
            image_size=tuple(config['image_size']),
            max_episode_steps=config['max_episode_steps'],
            success_distance=config['success_distance'],
            context_size=config['context_size'],
            goal_prob=goal_prob
        )
    
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
        
        # Update reward calculator
        self.reward_calculator = self._create_curriculum_reward_calculator(self.config)
        
        # Update success distance based on curriculum
        if curriculum_settings['max_distance'] != float('inf'):
            # Temporarily constrain goal selection distance
            self.env.max_goal_distance = curriculum_settings['max_distance']
        
        print(f"Updated environment with {len(new_scenes)} scenes for curriculum level {self.curriculum_manager.current_level}")
    
    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """Collect rollouts with curriculum tracking"""
        obs = self.env.reset()
        torch_obs = prepare_observation(obs, self.device)
        
        # Reset LSTM hidden state at episode start
        self.model.reset_hidden()
        hidden_state = None
        
        episode_reward = 0
        episode_length = 0
        episode_success = False
        episode_collisions = 0
        
        rollout_stats = {
            'total_reward': 0,
            'episodes': 0,
            'successes': 0,
            'collisions': 0,
            'auxiliary_loss': 0
        }
        
        for step in range(num_steps):
            with torch.no_grad():
                if self.use_amp:
                    with autocast():
                        outputs = self.model.forward(
                            torch_obs, mode="all", hidden_state=hidden_state
                        )
                else:
                    outputs = self.model.forward(
                        torch_obs, mode="all", hidden_state=hidden_state
                    )
                
                action_dist = outputs['action_dist']
                value = outputs['values']
                hidden_state = outputs['hidden_state']
                
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            # Environment step
            next_obs, _, done, info = self.env.step(action.cpu().item())
            
            # Calculate multi-component reward
            event = self.env.controller.last_event
            reward = self.reward_calculator.calculate_reward(event, next_obs, info, self.env)
            
            next_torch_obs = prepare_observation(next_obs, self.device)
            
            self.buffer.store(
                obs=torch_obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )
            
            episode_reward += reward
            episode_length += 1
            if info.get('collision', False):
                episode_collisions += 1
            if info.get('success', False):
                episode_success = True
            
            rollout_stats['total_reward'] += reward
            
            # Episode termination
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.success_rates.append(1.0 if episode_success else 0.0)
                
                self.curriculum_manager.update(
                    episode_success, 
                    episode_length,
                    episode_collisions
                )
                
                # Update evaluation metrics
                agent_pos = self.env._get_agent_position()
                pos_key = (round(agent_pos['x'], 1), round(agent_pos['z'], 1))
                self.evaluation_metrics.step(info, reward, pos_key)
                self.evaluation_metrics.end_episode(episode_success)
                
                rollout_stats['episodes'] += 1
                rollout_stats['successes'] += 1 if episode_success else 0
                rollout_stats['collisions'] += episode_collisions
                
                # Reset for new episode
                obs = self.env.reset()
                torch_obs = prepare_observation(obs, self.device)
                self.model.reset_hidden()
                hidden_state = None
                episode_reward = 0
                episode_length = 0
                episode_success = False
                episode_collisions = 0
            else:
                obs = next_obs
                torch_obs = next_torch_obs
        
        # Compute GAE
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    final_value = self.model.forward(
                        torch_obs, mode="value", hidden_state=hidden_state
                    )['values'].cpu().item()
            else:
                final_value = self.model.forward(
                    torch_obs, mode="value", hidden_state=hidden_state
                )['values'].cpu().item()
        
        self.buffer.compute_gae(final_value, self.gamma, self.lam)
        
        return rollout_stats
    
    def train_val_test(self, total_timesteps: int):
        print(f"Starting unified training with curriculum learning for {total_timesteps} timesteps...")
        print(f"Dataset: {self.dataset}")
        print(f"Stage: {self.stage}")
        print(f"Initial curriculum level: {self.curriculum_manager.current_level}")
        
        timesteps_collected = 0
        update_count = 0
        
        while timesteps_collected < total_timesteps:
            # Check if curriculum should be updated
            if update_count > 0 and update_count % self.curriculum_update_freq == 0:
                old_level = self.curriculum_manager.current_level
                
                # Check if curriculum has changed
                if self.curriculum_manager.current_level != old_level:
                    self._update_environment_curriculum()
                    self.last_curriculum_update = update_count
            
            # Collect rollouts on training set
            rollout_stats = self.collect_rollouts(self.config['rollout_steps'])
            timesteps_collected += self.config['rollout_steps']
            
            update_stats = self.update_policy()
            update_count += 1
            
            if update_count % 10 == 0:
                torch.cuda.empty_cache()
            
            if update_count % self.config['log_freq'] == 0:
                self._log_training_stats_with_curriculum(
                    timesteps_collected, rollout_stats, update_stats
                )
                
                # Record curriculum progress
                curriculum_stats = self.curriculum_manager.get_progress_stats()
                self.results['curriculum']['timesteps'].append(timesteps_collected)
                self.results['curriculum']['level'].append(curriculum_stats['current_level'])
                self.results['curriculum']['success_rate'].append(
                    curriculum_stats['current_success_rate']
                )
                
                self.results['train']['timesteps'].append(timesteps_collected)
                self.results['train']['reward'].append(
                    np.mean(self.episode_rewards) if self.episode_rewards else 0
                )
                self.results['train']['success_rate'].append(
                    np.mean(self.success_rates) if self.success_rates else 0
                )
            
            if update_count % self.config.get('val_freq', 100) == 0:
                val_metrics = self._evaluate_on_split('val', self.val_env)
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
        self._save_final_results_with_curriculum(final_results)
        
        self.env.close()
        self.val_env.close()
        self.test_env.close()
        
        return final_results
    
    def _log_training_stats_with_curriculum(self, timesteps: int, rollout_stats: Dict, update_stats: Dict):
        curriculum_stats = self.curriculum_manager.get_progress_stats()
        
        print(f"\n--- Stage {self.stage} Update {timesteps // self.config['rollout_steps']} "
              f"(Timesteps: {timesteps}) ---")
        print(f"Curriculum Level: {curriculum_stats['current_level']} ({curriculum_stats['level_name']})")
        print(f"Episodes at Level: {curriculum_stats['episodes_at_level']}")
        print(f"Progress: {curriculum_stats['levels_completed']}/{curriculum_stats['total_levels']} levels")
        
        if len(self.episode_rewards) > 0:
            print(f"Episode Reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
            print(f"Episode Length: {np.mean(self.episode_lengths):.1f}")
            print(f"Success Rate: {np.mean(self.success_rates):.2%}")
            
            # Compute comprehensive metrics
            metrics = self.evaluation_metrics.compute_metrics()
            if metrics:
                print(f"SPL: {metrics.get('spl', 0):.3f}")
                print(f"Collision-Free Success Rate: {metrics.get('collision_free_success_rate', 0):.2%}")
                print(f"Exploration Coverage: {metrics.get('exploration_coverage', 0):.3f}")
        
        print(f"Policy Loss: {update_stats['policy_loss']:.4f}")
        print(f"Value Loss: {update_stats['value_loss']:.4f}")
        print(f"Entropy: {-update_stats['entropy_loss']:.4f}")
        
        if self.stage == 1 and update_stats['auxiliary_loss'] > 0:
            print(f"Auxiliary Loss: {update_stats['auxiliary_loss']:.4f}")
        
        print(f"Approx KL: {update_stats['approx_kl']:.4f}")
        
        if self.config.get('use_wandb', False):
            log_dict = {
                'timesteps': timesteps,
                'stage': self.stage,
                'curriculum/level': curriculum_stats['current_level'],
                'curriculum/level_name': curriculum_stats['level_name'],
                'curriculum/episodes_at_level': curriculum_stats['episodes_at_level'],
                'curriculum/level_success_rate': curriculum_stats['current_success_rate'],
                'episode_reward_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'episode_length_mean': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'success_rate': np.mean(self.success_rates) if self.success_rates else 0,
                'policy_loss': update_stats['policy_loss'],
                'value_loss': update_stats['value_loss'],
                'entropy': -update_stats['entropy_loss'],
                'distance_loss': update_stats['distance_loss'],
                'approx_kl': update_stats['approx_kl'],
                'clip_fraction': update_stats['clip_fraction'],
            }
            
            if self.stage == 1:
                log_dict['auxiliary_loss'] = update_stats['auxiliary_loss']
            
            metrics = self.evaluation_metrics.compute_metrics()
            if metrics:
                log_dict.update({f'train/{k}': v for k, v in metrics.items()})
            
            wandb.log(log_dict)
    
    def _run_final_evaluation(self) -> Dict:
        final_results = {}
        
        print("\n=== Training Set Evaluation ===")
        self.env.close()
        self.env = self._create_environment(
            self.splits['train'], 
            self.config,
            goal_prob=self.config.get('eval_goal_prob', 1.0)
        )
        train_metrics = self._evaluate_on_split('train', self.env, num_episodes=50)
        final_results['train'] = train_metrics
        
        print("\n=== Validation Set Evaluation ===")
        val_metrics = self._evaluate_on_split('val', self.val_env, num_episodes=50)
        final_results['val'] = val_metrics
        
        print("\n=== Test Set Evaluation ===")
        test_metrics = self._evaluate_on_split('test', self.test_env, num_episodes=100)
        final_results['test'] = test_metrics
        
        return final_results
    
    def _save_final_results_with_curriculum(self, results: Dict):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(self.config['save_dir'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        curriculum_history = self.curriculum_manager.performance_history
        results_file = os.path.join(
            results_dir, 
            f'{self.dataset}_stage{self.stage}_{timestamp}_results.json'
        )
        
        with open(results_file, 'w') as f:
            json.dump({
                'config': self.config,
                'final_results': results,
                'training_history': {
                    split: dict(self.results[split]) 
                    for split in ['train', 'val', 'test', 'curriculum']
                },
                'curriculum_progression': {
                    'final_level': self.curriculum_manager.current_level,
                    'levels_completed': self.curriculum_manager.current_level,
                    'total_levels': len(self.curriculum_manager.levels),
                    'performance_history': curriculum_history
                },
                'best_val_success_rate': self.best_val_success_rate,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        print(f"Final Curriculum Level: {self.curriculum_manager.current_level} / {len(self.curriculum_manager.levels) - 1}")
        print(f"Curriculum Progression: {self.curriculum_manager.get_current_settings()['name']}")
        print("-"*70)
        print(f"{'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10}")
        print("-"*70)
        
        metrics_to_show = [
            ('Success Rate (%)', 'success_rate', 100),
            ('SPL', 'spl', 1),
            ('Avg Reward', 'avg_reward', 1),
            ('Avg Episode Length', 'avg_length', 1),
            ('Collision-Free SR (%)', 'collision_free_success_rate', 100)
        ]
        
        for metric_name, metric_key, multiplier in metrics_to_show:
            train_val = results['train'].get(metric_key, 0) * multiplier
            val_val = results['val'].get(metric_key, 0) * multiplier
            test_val = results['test'].get(metric_key, 0) * multiplier
            print(f"{metric_name:<25} {train_val:>10.2f} {val_val:>10.2f} {test_val:>10.2f}")
        
        print("="*70)
        
        if curriculum_history:
            print("\n" + "="*70)
            print("CURRICULUM PROGRESSION")
            print("="*70)
            print(f"{'Level':<6} {'Name':<20} {'Episodes':<10} {'Success':<10} {'Collisions':<12}")
            print("-"*70)
            
            for i, hist in enumerate(curriculum_history):
                level_name = self.curriculum_manager.levels[hist['level']]['name']
                print(f"{hist['level']:<6} {level_name:<20} {hist['episodes']:<10} "
                      f"{hist['success_rate']*100:<10.1f} {hist['avg_collisions']:<12.2f}")
            
            print("="*70)
    
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
            if dataset == 'combined':
                from combined_dataset_splits import CombinedAI2THORDatasetSplitter
                splitter = CombinedAI2THORDatasetSplitter()
                splits_dict = splitter.save_combined_splits()
                splits = {k: v['combined'] for k, v in splits_dict.items()}
            else:
                from dataset_splits import AI2THORDatasetSplitter
                splitter = AI2THORDatasetSplitter()
                splits = splitter.save_splits(dataset)
        
        print(f"Dataset: {dataset}")
        print(f"Train scenes: {len(splits['train'])}")
        print(f"Val scenes: {len(splits['val'])}")
        print(f"Test scenes: {len(splits['test'])}")
        
        return splits
    
    def _save_model(self, update_count: int, timesteps: int, is_best: bool = False):
        """Save model checkpoint with curriculum information"""
        curriculum_stats = self.curriculum_manager.get_progress_stats()
        
        checkpoint_name = f'{self.dataset}_stage{self.stage}_{"best" if is_best else update_count}.pth'
        save_path = os.path.join(self.config['save_dir'], checkpoint_name)
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': update_count,
            'timesteps': timesteps,
            'stage': self.stage,
            'curriculum_level': curriculum_stats['current_level'],
            'curriculum_stats': curriculum_stats,
            'config': self.config,
            'success_rate': np.mean(self.success_rates) if self.success_rates else 0,
        }, save_path)
        
        print(f"Model saved to {save_path}")
        return save_path


def main():
    parser = argparse.ArgumentParser(description='Unified Train/Val/Test with Curriculum Learning')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['ithor', 'robothor', 'combined'],
                       required=True, help='Dataset to use')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                       help='Training stage')
    parser.add_argument('--stage1-checkpoint', type=str, default=None,
                       help='Stage 1 checkpoint for stage 2')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test'],
                       default='train', help='Mode to run')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint for evaluation/testing')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['dataset'] = args.dataset
    config['training_stage'] = args.stage
    if args.stage1_checkpoint:
        config['stage1_checkpoint'] = args.stage1_checkpoint
    
    if config.get('use_wandb', False) and args.mode == 'train':
        wandb.init(
            project=config.get('wandb_project', 'nomad-rl-curriculum'),
            name=f"{args.dataset}_{config.get('run_name', 'curriculum')}_stage{args.stage}",
            config=config
        )
    
    trainer = UnifiedTrainerWithCurriculum(config)
    
    if args.mode == 'train':
        results = trainer.train_val_test(config[f'stage{args.stage}_timesteps'])
    
    elif args.mode == 'eval':
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from curriculum level: {checkpoint.get('curriculum_level', 'unknown')}")
        
        val_metrics = trainer._evaluate_on_split('val', trainer.val_env, num_episodes=100)
        print("\nValidation metrics:", val_metrics)
    
    elif args.mode == 'test':
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from curriculum level: {checkpoint.get('curriculum_level', 'unknown')}")
        
        test_metrics = trainer._evaluate_on_split('test', trainer.test_env, num_episodes=200)
        print("\nTest metrics:", test_metrics)

if __name__ == '__main__':
    main()