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

from nomad_rl.environments.ai2thor_nomad_env import AI2ThorNoMaDEnv
from nomad_rl.models.enhanced_nomad_rl_model import (
    EnhancedNoMaDRL, MultiComponentRewardCalculator, 
    CurriculumManager, EvaluationMetrics, prepare_observation
)
from nomad_rl.training.two_stage_trainer import TwoStageTrainer

class UnifiedTrainer(TwoStageTrainer):
    """Extended trainer with train/val/test splits and comprehensive evaluation"""
    
    def __init__(self, config: Dict):
        # Load dataset splits
        self.dataset = config['dataset']  # 'ithor', 'robothor', or 'combined'
        self.splits = self._load_splits(config)
        
        # Set training scenes
        config['scene_names'] = self.splits['train']
        
        # Initialize parent trainer
        super().__init__(config)
        
        # Determine if we need RoboTHOR controller settings
        self.use_robothor = self._check_robothor_scenes(self.splits)
        
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
        
        # Results storage
        self.results = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
        
        # Best model tracking
        self.best_val_success_rate = 0
        self.best_val_checkpoint = None
    
    def _check_robothor_scenes(self, splits: Dict[str, List[str]]) -> bool:
        """Check if any splits contain RoboTHOR scenes"""
        for split_scenes in splits.values():
            for scene in split_scenes:
                if 'Train' in scene or 'Val' in scene or 'Test' in scene:
                    return True
        return False
    
    def _create_environment(self, scenes: List[str], config: Dict, goal_prob: float) -> AI2ThorNoMaDEnv:
        """Create environment that can handle both iTHOR and RoboTHOR scenes"""
        # Note: AI2ThorNoMaDEnv should be updated to handle RoboTHOR scenes
        # For now, we use the standard environment
        return AI2ThorNoMaDEnv(
            scene_names=scenes,
            image_size=tuple(config['image_size']),
            max_episode_steps=config['max_episode_steps'],
            success_distance=config['success_distance'],
            context_size=config['context_size'],
            goal_prob=goal_prob
        )
    
    def _load_splits(self, config: Dict) -> Dict[str, List[str]]:
        """Load dataset splits from config or file"""
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
            # Generate splits if not found
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
        
        # Print sample scenes from each split
        for split_name, scenes in splits.items():
            print(f"Sample {split_name} scenes: {scenes[:3]}...")
        
        return splits
    
    def train_val_test(self, total_timesteps: int):
        """Main training loop with validation and final testing"""
        print(f"Starting unified training for {total_timesteps} timesteps...")
        print(f"Dataset: {self.dataset}")
        print(f"Stage: {self.stage}")
        
        timesteps_collected = 0
        update_count = 0
        
        # Training loop with periodic validation
        while timesteps_collected < total_timesteps:
            # Collect rollouts on training set
            rollout_stats = self.collect_rollouts(self.config['rollout_steps'])
            timesteps_collected += self.config['rollout_steps']
            
            # Update policy
            update_stats = self.update_policy()
            update_count += 1
            
            # Clear GPU cache
            if update_count % 10 == 0:
                torch.cuda.empty_cache()
            
            # Log training stats
            if update_count % self.config['log_freq'] == 0:
                self._log_training_stats(timesteps_collected, rollout_stats, update_stats)
                self.results['train']['timesteps'].append(timesteps_collected)
                self.results['train']['reward'].append(np.mean(self.episode_rewards) if self.episode_rewards else 0)
                self.results['train']['success_rate'].append(np.mean(self.success_rates) if self.success_rates else 0)
            
            # Validation
            if update_count % self.config.get('val_freq', 100) == 0:
                val_metrics = self._evaluate_on_split('val', self.val_env)
                self.results['val']['timesteps'].append(timesteps_collected)
                for key, value in val_metrics.items():
                    self.results['val'][key].append(value)
                
                # Save best model based on validation
                if val_metrics['success_rate'] > self.best_val_success_rate:
                    self.best_val_success_rate = val_metrics['success_rate']
                    self.best_val_checkpoint = self._save_model(
                        update_count, timesteps_collected, is_best=True
                    )
                    print(f"New best validation success rate: {self.best_val_success_rate:.2%}")
            
            # Regular checkpointing
            if update_count % self.config['save_freq'] == 0:
                self._save_model(update_count, timesteps_collected)
        
        print("\nTraining completed! Running final evaluation...")
        
        # Load best model for final testing
        if self.best_val_checkpoint and os.path.exists(self.best_val_checkpoint):
            print(f"Loading best model from {self.best_val_checkpoint}")
            checkpoint = torch.load(self.best_val_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation on all splits
        final_results = {}
        
        # Training set evaluation
        print("\n=== Training Set Evaluation ===")
        train_metrics = self._evaluate_on_split('train', self.env, num_episodes=50)
        final_results['train'] = train_metrics
        
        # Validation set evaluation
        print("\n=== Validation Set Evaluation ===")
        val_metrics = self._evaluate_on_split('val', self.val_env, num_episodes=50)
        final_results['val'] = val_metrics
        
        # Test set evaluation
        print("\n=== Test Set Evaluation ===")
        test_metrics = self._evaluate_on_split('test', self.test_env, num_episodes=100)
        final_results['test'] = test_metrics
        
        # Save final results
        self._save_final_results(final_results)
        
        # Close environments
        self.env.close()
        self.val_env.close()
        self.test_env.close()
        
        return final_results
    
    def _evaluate_on_split(self, split_name: str, env: AI2ThorNoMaDEnv, 
                          num_episodes: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model on a specific data split"""
        if num_episodes is None:
            num_episodes = self.config.get('eval_episodes', 20)
        
        print(f"\nEvaluating on {split_name} split ({num_episodes} episodes)...")
        self.model.eval()
        
        metrics = EvaluationMetrics()
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        
        for episode_idx in range(num_episodes):
            obs = env.reset()
            torch_obs = prepare_observation(obs, self.device)
            self.model.reset_hidden()
            hidden_state = None
            
            episode_reward = 0
            episode_length = 0
            
            while episode_length < self.config['max_episode_steps']:
                with torch.no_grad():
                    action, _, hidden_state = self.model.get_action(
                        torch_obs, deterministic=True, hidden_state=hidden_state
                    )
                
                next_obs, _, done, info = env.step(action.cpu().item())
                event = env.controller.last_event
                reward = self.reward_calculator.calculate_reward(event, next_obs, info, env)
                
                # Update metrics
                agent_pos = env._get_agent_position()
                pos_key = (round(agent_pos['x'], 1), round(agent_pos['z'], 1))
                metrics.step(info, reward, pos_key)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                torch_obs = prepare_observation(next_obs, self.device)
            
            # Record episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(info.get('success', False))
            metrics.end_episode(info.get('success', False))
            
            # Progress update
            if (episode_idx + 1) % 10 == 0:
                print(f"  Completed {episode_idx + 1}/{num_episodes} episodes")
        
        self.model.train()
        
        # Compute comprehensive metrics
        eval_metrics = metrics.compute_metrics()
        eval_metrics.update({
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': np.mean(episode_successes),
            'num_episodes': num_episodes
        })
        
        # Print summary
        print(f"\n{split_name.upper()} Results:")
        print(f"  Success Rate: {eval_metrics['success_rate']:.2%}")
        print(f"  SPL: {eval_metrics.get('spl', 0):.3f}")
        print(f"  Avg Reward: {eval_metrics['avg_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        print(f"  Avg Length: {eval_metrics['avg_length']:.1f} ± {eval_metrics['std_length']:.1f}")
        print(f"  Collision-Free SR: {eval_metrics.get('collision_free_success_rate', 0):.2%}")
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log({f'{split_name}/{k}': v for k, v in eval_metrics.items()})
        
        return eval_metrics
    
    def _save_final_results(self, results: Dict):
        """Save comprehensive results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(self.config['save_dir'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
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
                    for split in ['train', 'val', 'test']
                },
                'best_val_success_rate': self.best_val_success_rate,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        # Print summary table
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"{'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10}")
        print("-"*60)
        
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
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Unified Train/Val/Test for NoMaD-RL')
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
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line args
    config['dataset'] = args.dataset
    config['training_stage'] = args.stage
    if args.stage1_checkpoint:
        config['stage1_checkpoint'] = args.stage1_checkpoint
    
    # Initialize wandb
    if config.get('use_wandb', False) and args.mode == 'train':
        wandb.init(
            project=config.get('wandb_project', 'nomad-rl-unified'),
            name=f"{args.dataset}_{config.get('run_name', 'unified')}_stage{args.stage}",
            config=config
        )
    
    # Create trainer
    trainer = UnifiedTrainer(config)
    
    if args.mode == 'train':
        # Full training with validation and testing
        results = trainer.train_val_test(config[f'stage{args.stage}_timesteps'])
    
    elif args.mode == 'eval':
        # Evaluation only on validation set
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        val_metrics = trainer._evaluate_on_split('val', trainer.val_env, num_episodes=100)
        print("\nValidation metrics:", val_metrics)
    
    elif args.mode == 'test':
        # Testing only on test set
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = trainer._evaluate_on_split('test', trainer.test_env, num_episodes=200)
        print("\nTest metrics:", test_metrics)


if __name__ == '__main__':
    main()