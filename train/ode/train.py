import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import argparse
import yaml
import os
import json
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.append(r'D:\NCKH.2025-2026\Navigations\train')

from model import (
    UnifiedAdvancedNoMaDRL, 
    UnifiedAdvancedTrainingWrapper,
    create_unified_model,
    visualize_spatial_memory_graph
)

from ode.preprocess.grid_mana import GridBasedCurriculumManager
from unified.environments import EnhancedAI2ThorEnv
from arch.nomad_rl import prepare_observation, PPOBuffer
# from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

class UnifiedAdvancedNoMaDTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        self.dataset = config['dataset']
        self.splits = self._load_splits(config)
        
        self.curriculum_manager = GridBasedCurriculumManager(config, self.dataset)
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        self.all_train_scenes = self.splits['train']
        current_scenes = self.curriculum_manager.get_current_scenes(self.all_train_scenes)
        
        self.env = self._create_environment(
            current_scenes,
            config,
            goal_prob=curriculum_settings['goal_prob'],
            max_episode_steps=curriculum_settings['max_episode_steps']
        )
        
        self.model = create_unified_model(config).to(self.device)
        self.training_wrapper = UnifiedAdvancedTrainingWrapper(self.model, config)
        
        # Optimizer with different learning rates for different components
        self.optimizer = self._create_optimizer()
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # PPO Buffer
        obs_shapes = {
            'rgb': (3, *config['image_size']),
            'context': (3 * config['context_size'], *config['image_size']),
            'goal_rgb': (3, *config['image_size']),
            'goal_mask': (1,),
            'goal_position': (3,)
        }
        
        self.buffer = PPOBuffer(
            size=config['buffer_size'],
            obs_shape=obs_shapes,
            action_dim=self.env.action_space.n,
            device=self.device
        )
        
        # Metrics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.graph_sizes = deque(maxlen=100)
        self.path_confidences = deque(maxlen=100)
        self.counterfactual_accuracy = deque(maxlen=100)
        
        # Validation and test environments
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
        
        # Results tracking
        self.results = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list),
            'curriculum': defaultdict(list),
            'graph_metrics': defaultdict(list)
        }
        
        self.best_val_success_rate = 0
        self.best_val_checkpoint = None
        
        self.gamma = config['gamma']
        self.lam = config['lam']
        self.clip_ratio = config['clip_ratio']
        self.entropy_coef = config['entropy_coef']
        self.value_coef = config['value_coef']
        self.distance_coef = config['distance_coef']
        self.counterfactual_coef = config.get('counterfactual_coef', 0.1)
        self.spatial_smoothness_coef = config.get('spatial_smoothness_coef', 0.05)
        self.max_grad_norm = config['max_grad_norm']
        self.ppo_epochs = config['ppo_epochs']
        self.batch_size = config['batch_size']
        
        print(f"\nInitialized Unified Advanced NoMaD-RL Trainer")
        print(f"Dataset: {self.dataset}")
        print(f"Using Spatial Memory Graph with Neural ODE/PDE")
        print(f"Max nodes: {config.get('max_nodes', 500)}")
        print(f"Using Counterfactuals: {config.get('use_counterfactuals', True)}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        vision_params = []
        spatial_graph_params = []
        counterfactual_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'vision_encoder' in name:
                vision_params.append(param)
            elif 'spatial_memory_graph' in name:
                spatial_graph_params.append(param)
            elif 'world_model' in name:
                counterfactual_params.append(param)
            else:
                other_params.append(param)
        
        # Different learning rates for different components
        base_lr = self.config['learning_rate']
        param_groups = [
            {'params': vision_params, 'lr': base_lr * 0.1},  # Lower LR for vision
            {'params': spatial_graph_params, 'lr': base_lr * 2},  # Higher LR for new ODE component
            {'params': counterfactual_params, 'lr': base_lr},
            {'params': other_params, 'lr': base_lr}
        ]
        
        return optim.Adam(param_groups)
    
    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """Collect rollouts with unified model"""
        obs = self.env.reset()
        torch_obs = prepare_observation(obs, self.device)
        
        # Reset episode
        self.training_wrapper.reset_episode()
        
        episode_reward = 0
        episode_length = 0
        episode_success = False
        episode_collisions = 0
        episode_graph_sizes = []
        episode_path_confs = []
        
        rollout_stats = defaultdict(float)
        rollout_stats['episodes'] = 0
        
        for step in range(num_steps):
            with torch.no_grad():
                # Get action using training wrapper (handles time)
                action, log_prob, extra_info = self.training_wrapper.step(torch_obs)
                
                outputs = self.model.forward(torch_obs, mode="all", 
                                           time_delta=self.training_wrapper.get_time_delta())
                value = outputs['values']
            
            next_obs, reward, done, info = self.env.step(action.cpu().item())
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
            episode_graph_sizes.append(extra_info['num_nodes'])
            episode_path_confs.append(extra_info['path_confidence'].item())
            
            if info.get('collision', False):
                episode_collisions += 1
            if info.get('success', False):
                episode_success = True
            
            rollout_stats['total_reward'] += reward
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.success_rates.append(1.0 if episode_success else 0.0)
                self.graph_sizes.append(np.mean(episode_graph_sizes))
                self.path_confidences.append(np.mean(episode_path_confs))
                
                # Update curriculum
                self.curriculum_manager.update(
                    episode_success=episode_success,
                    episode_length=episode_length,
                    collision_count=episode_collisions
                )
                
                rollout_stats['episodes'] += 1
                rollout_stats['successes'] += 1 if episode_success else 0
                rollout_stats['collisions'] += episode_collisions
                rollout_stats['avg_graph_size'] += np.mean(episode_graph_sizes)
                rollout_stats['avg_path_conf'] += np.mean(episode_path_confs)
                
                obs = self.env.reset()
                torch_obs = prepare_observation(obs, self.device)
                self.training_wrapper.reset_episode()
                
                episode_reward = 0
                episode_length = 0
                episode_success = False
                episode_collisions = 0
                episode_graph_sizes = []
                episode_path_confs = []
            else:
                obs = next_obs
                torch_obs = next_torch_obs
        
        with torch.no_grad():
            final_value = self.model.forward(
                torch_obs, 
                mode="value",
                time_delta=self.training_wrapper.get_time_delta()
            )['values'].cpu().item()
        
        self.buffer.compute_gae(final_value, self.gamma, self.lam)
        
        # Average metrics
        if rollout_stats['episodes'] > 0:
            rollout_stats['avg_graph_size'] /= rollout_stats['episodes']
            rollout_stats['avg_path_conf'] /= rollout_stats['episodes']
        
        return rollout_stats
    
    def update_policy(self) -> Dict[str, float]:
        batch = self.buffer.get()
        update_stats = defaultdict(float)
        
        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_data = {
            'observations': batch['observations'],
            'actions': batch['actions'],
            'rewards': batch['rewards'],
            'old_log_probs': batch['log_probs'],
            'advantages': advantages,
            'returns': batch['returns']
        }
        
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(len(advantages))
            
            for start in range(0, len(advantages), self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]
                
                mb_batch_data = {}
                for key in batch_data:
                    if key == 'observations':
                        mb_batch_data[key] = {
                            k: v[mb_indices] for k, v in batch_data[key].items()
                        }
                    else:
                        mb_batch_data[key] = batch_data[key][mb_indices]
                
                losses = self.training_wrapper.compute_losses(mb_batch_data)
                
                if self.use_amp:
                    self.optimizer.zero_grad()
                    self.scaler.scale(losses['total_loss']).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.zero_grad()
                    losses['total_loss'].backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Track statistics
                for key, value in losses.items():
                    update_stats[key] += value.item()
        
        # Average statistics
        num_updates = self.ppo_epochs * (len(advantages) // self.batch_size)
        for key in update_stats:
            update_stats[key] /= max(1, num_updates)
        
        return dict(update_stats)
    
    def train(self, total_timesteps: int):
        """Main training loop"""
        print(f"Starting unified training for {total_timesteps} timesteps...")
        
        timesteps_collected = 0
        update_count = 0
        
        while timesteps_collected < total_timesteps:
            # Check for curriculum update
            if update_count > 0 and update_count % self.config.get('curriculum_update_freq', 10) == 0:
                old_level = self.curriculum_manager.current_level
                if self.curriculum_manager.current_level != old_level:
                    self._update_environment_curriculum()
            
            # Collect rollouts
            rollout_stats = self.collect_rollouts(self.config['rollout_steps'])
            timesteps_collected += self.config['rollout_steps']
            
            update_stats = self.update_policy()
            update_count += 1
            
            if update_count % 10 == 0:
                torch.cuda.empty_cache()
            
            if update_count % self.config['log_freq'] == 0:
                self._log_training_stats(timesteps_collected, rollout_stats, update_stats)
                
                # Visualize graph periodically
                if update_count % (self.config['log_freq'] * 10) == 0:
                    self._visualize_graph(update_count)
            
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
            
            # Regular checkpoint
            if update_count % self.config['save_freq'] == 0:
                self._save_model(update_count, timesteps_collected)
        
        # Final evaluation
        print("\nTraining completed! Running final evaluation...")
        final_results = self._run_final_evaluation()
        self._save_final_results(final_results)
        
        # Cleanup
        self.env.close()
        self.val_env.close()
        self.test_env.close()
        
        return final_results
    
    def _visualize_graph(self, update_count: int):
        """Visualize the spatial memory graph"""
        save_path = os.path.join(
            self.config['save_dir'], 
            f'graphs/graph_update_{update_count}.png'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            visualize_spatial_memory_graph(self.model, save_path)
            print(f"Saved graph visualization to {save_path}")
        except Exception as e:
            print(f"Failed to visualize graph: {e}")
    
    def _log_training_stats(self, timesteps: int, rollout_stats: Dict, update_stats: Dict):
        curriculum_stats = self.curriculum_manager.get_progress_stats()
        
        print(f"\n--- Update {timesteps // self.config['rollout_steps']} (Timesteps: {timesteps}) ---")
        print(f"Curriculum Level: {curriculum_stats['current_level']} ({curriculum_stats['level_name']})")
        
        if len(self.episode_rewards) > 0:
            print(f"Episode Reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
            print(f"Success Rate: {np.mean(self.success_rates):.2%}")
            print(f"Avg Graph Size: {np.mean(self.graph_sizes):.1f} nodes")
            print(f"Path Confidence: {np.mean(self.path_confidences):.3f}")
        
        print(f"Policy Loss: {update_stats['policy_loss']:.4f}")
        print(f"Value Loss: {update_stats['value_loss']:.4f}")
        if 'counterfactual_loss' in update_stats:
            print(f"Counterfactual Loss: {update_stats['counterfactual_loss']:.4f}")
        if 'spatial_smoothness_loss' in update_stats:
            print(f"Spatial Smoothness Loss: {update_stats['spatial_smoothness_loss']:.4f}")
        
        if self.config.get('use_wandb', False):
            wandb.log({
                'timesteps': timesteps,
                'curriculum/level': curriculum_stats['current_level'],
                'episode_reward_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'success_rate': np.mean(self.success_rates) if self.success_rates else 0,
                'graph_size': np.mean(self.graph_sizes) if self.graph_sizes else 0,
                'path_confidence': np.mean(self.path_confidences) if self.path_confidences else 0,
                **update_stats})
    
    def _evaluate_on_split(self, split_name: str, env: EnhancedAI2ThorEnv, 
                          num_episodes: Optional[int] = None) -> Dict[str, float]:
        """Evaluate on a specific split"""
        if num_episodes is None:
            num_episodes = self.config.get('eval_episodes', 20)
        
        print(f"\nEvaluating on {split_name} split ({num_episodes} episodes)...")
        self.model.eval()
        
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_graph_sizes = []
        episode_path_confs = []
        
        for episode_idx in range(num_episodes):
            obs = env.reset()
            torch_obs = prepare_observation(obs, self.device)
            
            # Reset for evaluation episode
            self.training_wrapper.reset_episode()
            
            episode_reward = 0
            episode_length = 0
            graph_sizes = []
            path_confs = []
            
            while episode_length < self.config['max_episode_steps']:
                with torch.no_grad():
                    action, _, extra_info = self.training_wrapper.step(
                        torch_obs, deterministic=True
                    )
                
                next_obs, reward, done, info = env.step(action.cpu().item())
                
                episode_reward += reward
                episode_length += 1
                graph_sizes.append(extra_info['num_nodes'])
                path_confs.append(extra_info['path_confidence'].item())
                
                if done:
                    break
                
                torch_obs = prepare_observation(next_obs, self.device)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(info.get('success', False))
            episode_graph_sizes.append(np.mean(graph_sizes))
            episode_path_confs.append(np.mean(path_confs))
            
            if (episode_idx + 1) % 10 == 0:
                current_sr = sum(episode_successes) / len(episode_successes)
                print(f"  Completed {episode_idx + 1}/{num_episodes} episodes (SR: {current_sr:.2%})")
        
        self.model.train()
        
        metrics = {
            'success_rate': np.mean(episode_successes),
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'avg_graph_size': np.mean(episode_graph_sizes),
            'avg_path_confidence': np.mean(episode_path_confs),
            'num_episodes': num_episodes
        }
        
        print(f"\n{split_name.upper()} Results:")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Avg Graph Size: {metrics['avg_graph_size']:.1f} nodes")
        print(f"  Path Confidence: {metrics['avg_path_confidence']:.3f}")
        
        return metrics
    
    def _update_environment_curriculum(self):
        """Update environment based on curriculum progression"""
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        # Get new scenes
        new_scenes = self.curriculum_manager.get_current_scenes(self.all_train_scenes)
        
        # Recreate environment with new settings
        self.env.close()
        self.env = self._create_environment(
            new_scenes,
            self.config,
            goal_prob=curriculum_settings['goal_prob'],
            max_episode_steps=curriculum_settings['max_episode_steps']
        )
        
        print(f"Updated environment with {len(new_scenes)} scenes for curriculum level {self.curriculum_manager.current_level}")
    
    def _create_environment(self, scenes: List[str], config: Dict, 
                           goal_prob: float = None, max_episode_steps: int = None):
        return EnhancedAI2ThorEnv(
            scene_names=scenes,
            image_size=tuple(config['image_size']),
            max_episode_steps=max_episode_steps or config['max_episode_steps'],
            success_distance=config['success_distance'],
            context_size=config['context_size'],
            goal_prob=goal_prob if goal_prob is not None else config['goal_prob']
        )
    
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
    
    def _save_model(self, update_count: int, timesteps: int, is_best: bool = False):
        curriculum_stats = self.curriculum_manager.get_progress_stats()
        
        checkpoint_name = f'{self.dataset}_unified_{"best" if is_best else update_count}.pth'
        save_path = os.path.join(self.config['save_dir'], checkpoint_name)
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # Save full graph state
        graph_state_dict = {
            'node_features': self.model.spatial_memory_graph.graph_state.node_features if self.model.spatial_memory_graph.graph_state else None,
            'node_positions': self.model.spatial_memory_graph.graph_state.node_positions if self.model.spatial_memory_graph.graph_state else None,
            'edge_index': self.model.spatial_memory_graph.graph_state.edge_index if self.model.spatial_memory_graph.graph_state else None,
            'num_nodes': self.model.spatial_memory_graph.graph_state.num_nodes if self.model.spatial_memory_graph.graph_state else 0
        }
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': update_count,
            'timesteps': timesteps,
            'curriculum_level': curriculum_stats['current_level'],
            'curriculum_stats': curriculum_stats,
            'graph_state': graph_state_dict,
            'config': self.config,
            'success_rate': np.mean(self.success_rates) if self.success_rates else 0,
            'best_val_success_rate': self.best_val_success_rate
        }, save_path)
        
        print(f"Model saved to {save_path}")
        return save_path
    
    def _run_final_evaluation(self) -> Dict:
        final_results = {}
        
        # Load best model if available
        if self.best_val_checkpoint and os.path.exists(self.best_val_checkpoint):
            print(f"Loading best model from {self.best_val_checkpoint}")
            checkpoint = torch.load(self.best_val_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on all splits
        print("\n=== Training Set Evaluation ===")
        train_env = self._create_environment(
            self.splits['train'][:50],  # Subset for efficiency
            self.config,
            goal_prob=self.config.get('eval_goal_prob', 1.0)
        )
        train_metrics = self._evaluate_on_split('train', train_env, num_episodes=50)
        final_results['train'] = train_metrics
        train_env.close()
        
        print("\n=== Validation Set Evaluation ===")
        val_metrics = self._evaluate_on_split('val', self.val_env, num_episodes=50)
        final_results['val'] = val_metrics
        
        print("\n=== Test Set Evaluation ===")
        test_metrics = self._evaluate_on_split('test', self.test_env, num_episodes=100)
        final_results['test'] = test_metrics
        
        return final_results
    
    def _save_final_results(self, results: Dict):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(self.config['save_dir'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(
            results_dir,
            f'{self.dataset}_unified_advanced_{timestamp}_results.json'
        )
        
        # Get final graph statistics
        if self.model.spatial_memory_graph.graph_state:
            final_graph_stats = {
                'final_num_nodes': self.model.spatial_memory_graph.graph_state.num_nodes,
                'final_num_edges': self.model.spatial_memory_graph.graph_state.edge_index.size(1) // 2,
                'max_graph_size_seen': max(self.graph_sizes) if self.graph_sizes else 0
            }
        else:
            final_graph_stats = {}
        
        with open(results_file, 'w') as f:
            json.dump({
                'config': self.config,
                'final_results': results,
                'training_history': {
                    split: dict(self.results[split])
                    for split in ['train', 'val', 'test', 'curriculum', 'graph_metrics']
                },
                'curriculum_progression': {
                    'final_level': self.curriculum_manager.current_level,
                    'total_levels': len(self.curriculum_manager.levels),
                    'performance_history': self.curriculum_manager.performance_history
                },
                'graph_statistics': final_graph_stats,
                'best_val_success_rate': self.best_val_success_rate,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY - Unified Advanced NoMaD-RL")
        print("="*70)
        print(f"Dataset: {self.dataset}")
        print(f"Final Curriculum Level: {self.curriculum_manager.current_level}")
        print(f"Graph Statistics: {final_graph_stats}")
        print("-"*70)
        print(f"{'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10}")
        print("-"*70)
        
        metrics_to_show = [
            ('Success Rate (%)', 'success_rate', 100),
            ('Avg Reward', 'avg_reward', 1),
            ('Avg Episode Length', 'avg_length', 1),
            ('Avg Graph Size', 'avg_graph_size', 1),
            ('Path Confidence', 'avg_path_confidence', 1)
        ]
        
        for metric_name, metric_key, multiplier in metrics_to_show:
            train_val = results['train'].get(metric_key, 0) * multiplier
            val_val = results['val'].get(metric_key, 0) * multiplier
            test_val = results['test'].get(metric_key, 0) * multiplier
            print(f"{metric_name:<25} {train_val:>10.2f} {val_val:>10.2f} {test_val:>10.2f}")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Train Unified Advanced NoMaD-RL with Spatial Memory Graph ODE')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['ithor', 'robothor', 'combined'],
                       required=True, help='Dataset to use')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test'],
                       default='train', help='Mode to run')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['dataset'] = args.dataset
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False) and args.mode == 'train':
        wandb.init(
            project=config.get('wandb_project', 'nomad-rl-unified-advanced'),
            name=f"{args.dataset}_unified_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
    
    # Create trainer
    trainer = UnifiedAdvancedNoMaDTrainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore graph state
        if 'graph_state' in checkpoint and checkpoint['graph_state']['node_features'] is not None:
            graph_state = trainer.model.spatial_memory_graph.graph_state
            graph_state.node_features = checkpoint['graph_state']['node_features']
            graph_state.node_positions = checkpoint['graph_state']['node_positions']
            graph_state.edge_index = checkpoint['graph_state']['edge_index']
            graph_state.num_nodes = checkpoint['graph_state']['num_nodes']
        
        print(f"Resumed from checkpoint with success rate: {checkpoint.get('success_rate', 0):.2%}")
    
    # Run based on mode
    if args.mode == 'train':
        results = trainer.train(config['total_timesteps'])
    
    elif args.mode == 'eval':
        val_metrics = trainer._evaluate_on_split('val', trainer.val_env, num_episodes=100)
        print("\nValidation metrics:", val_metrics)
    
    elif args.mode == 'test':
        test_metrics = trainer._evaluate_on_split('test', trainer.test_env, num_episodes=200)
        print("\nTest metrics:", test_metrics)


if __name__ == '__main__':
    main()