import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import argparse
import yaml
import os
from collections import deque
from typing import Dict, List, Tuple, Optional
import time
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train')

from nomad_rl.environments.ai2thor_nomad_env import AI2ThorNoMaDEnv
from nomad_rl.models.enhanced_nomad_rl_model import (
    EnhancedNoMaDRL, MultiComponentRewardCalculator, 
    CurriculumManager, EvaluationMetrics, prepare_observation, PPOBuffer
)

class TwoStageTrainer:
    """Two-stage training for NoMaD-RL with curriculum learning"""
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.stage = config.get('training_stage', 1)
        
        # Initialize curriculum manager
        self.curriculum_manager = CurriculumManager(config)
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        # Environment with curriculum settings
        self.env = AI2ThorNoMaDEnv(
            scene_names=config['scene_names'],
            image_size=tuple(config['image_size']),
            max_episode_steps=config['max_episode_steps'],
            success_distance=config['success_distance'],
            context_size=config['context_size'],
            goal_prob=curriculum_settings['goal_prob']
        )
        
        # Enhanced model with LSTM
        self.model = EnhancedNoMaDRL(
            action_dim=self.env.action_space.n,
            encoding_size=config['encoding_size'],
            context_size=config['context_size'],
            mha_num_attention_heads=config['mha_num_attention_heads'],
            mha_num_attention_layers=config['mha_num_attention_layers'],
            mha_ff_dim_factor=config['mha_ff_dim_factor'],
            hidden_dim=config['hidden_dim'],
            lstm_hidden_size=config.get('lstm_hidden_size', 256),
            lstm_num_layers=config.get('lstm_num_layers', 2),
            use_auxiliary_heads=(self.stage == 1)  # Only in stage 1
        ).to(self.device)
        
        # Load checkpoint if transitioning stages
        if self.stage == 2 and config.get('stage1_checkpoint'):
            self._load_stage1_checkpoint(config['stage1_checkpoint'])
        
        # Multi-component reward calculator
        reward_config = config.copy()
        reward_config['training_stage'] = self.stage
        self.reward_calculator = MultiComponentRewardCalculator(reward_config)
        
        # Optimizer with different learning rates for stages
        stage_lr = config.get(f'stage{self.stage}_learning_rate', config['learning_rate'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(stage_lr))
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Buffer initialization
        image_size = tuple(config['image_size'])
        obs_shapes = {
            'rgb': (3, *image_size),
            'context': (3 * config['context_size'], *image_size),
            'goal_rgb': (3, *image_size),
            'goal_mask': (1,),
            'goal_position': (3,)
        }
        
        self.buffer = PPOBuffer(
            size=config['buffer_size'],
            obs_shape=obs_shapes,
            action_dim=self.env.action_space.n,
            device=self.device
        )
        
        # Metrics and logging
        self.evaluation_metrics = EvaluationMetrics()
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.auxiliary_losses = deque(maxlen=100)
        
        # Training parameters
        self.gamma = config['gamma']
        self.lam = config['lam']
        self.clip_ratio = config['clip_ratio']
        self.entropy_coef = config['entropy_coef']
        self.value_coef = config['value_coef']
        self.distance_coef = config['distance_coef']
        self.auxiliary_coef = config.get('auxiliary_coef', 0.1)
        self.max_grad_norm = config['max_grad_norm']
        self.ppo_epochs = config['ppo_epochs']
        self.batch_size = config['batch_size']
        
    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """Collect rollouts with LSTM hidden state management"""
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
            
            # Store transition
            self.buffer.store(
                obs=torch_obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )
            
            # Update metrics
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
                
                # Update curriculum
                self.curriculum_manager.update(episode_success)
                
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
                
                # Update environment with new curriculum settings
                curriculum_settings = self.curriculum_manager.get_current_settings()
                self.env.goal_prob = curriculum_settings['goal_prob']
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
    
    def update_policy(self) -> Dict[str, float]:
        """Update policy with auxiliary losses for stage 1"""
        batch = self.buffer.get()
        update_stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy_loss': 0,
            'distance_loss': 0,
            'auxiliary_loss': 0,
            'total_loss': 0,
            'approx_kl': 0,
            'clip_fraction': 0
        }
        
        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(len(advantages))
            
            for start in range(0, len(advantages), self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]
                
                mb_obs = {key: val[mb_indices] for key, val in batch['observations'].items()}
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['log_probs'][mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = batch['returns'][mb_indices]
                mb_old_values = batch['values'][mb_indices]
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        outputs = self.model.forward(mb_obs, mode="all")
                        log_probs = outputs['action_dist'].log_prob(mb_actions)
                        values = outputs['values']
                        entropy = outputs['action_dist'].entropy()
                        
                        # PPO losses
                        ratio = torch.exp(log_probs - mb_old_log_probs)
                        surr1 = ratio * mb_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        value_pred_clipped = mb_old_values + torch.clamp(
                            values - mb_old_values, -self.clip_ratio, self.clip_ratio
                        )
                        value_losses = (values - mb_returns).pow(2)
                        value_losses_clipped = (value_pred_clipped - mb_returns).pow(2)
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                        
                        entropy_loss = -entropy.mean()
                        
                        # Distance loss
                        distance_loss = torch.tensor(0.0, device=self.device)
                        if mb_obs['goal_mask'].sum() < len(mb_obs['goal_mask']):
                            goal_conditioned_mask = (mb_obs['goal_mask'].squeeze() == 0)
                            if goal_conditioned_mask.sum() > 0:
                                predicted_distances = outputs['distances'][goal_conditioned_mask]
                                target_distances = -mb_returns[goal_conditioned_mask]
                                distance_loss = F.mse_loss(
                                    predicted_distances.view(-1), 
                                    target_distances.view(-1)
                                )
                        
                        # Auxiliary losses (Stage 1 only)
                        auxiliary_loss = torch.tensor(0.0, device=self.device)
                        if self.stage == 1 and self.model.use_auxiliary_heads:
                            # Collision prediction loss (binary cross-entropy)
                            if 'collision_labels' in batch:
                                collision_pred = outputs['collision_prediction']
                                collision_labels = batch['collision_labels'][mb_indices]
                                collision_loss = F.binary_cross_entropy(
                                    collision_pred.squeeze(), 
                                    collision_labels.float()
                                )
                                auxiliary_loss += collision_loss
                            
                            # Exploration prediction loss
                            if 'exploration_labels' in batch:
                                exploration_pred = outputs['exploration_prediction']
                                exploration_labels = batch['exploration_labels'][mb_indices]
                                exploration_loss = F.mse_loss(
                                    exploration_pred.squeeze(),
                                    exploration_labels
                                )
                                auxiliary_loss += exploration_loss
                        
                        # Total loss
                        total_loss = (
                            policy_loss + 
                            self.value_coef * value_loss + 
                            self.entropy_coef * entropy_loss +
                            self.distance_coef * distance_loss +
                            self.auxiliary_coef * auxiliary_loss
                        )
                    
                    # Backward pass with mixed precision
                    self.optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                else:
                    # Standard precision training
                    outputs = self.model.forward(mb_obs, mode="all")
                    log_probs = outputs['action_dist'].log_prob(mb_actions)
                    values = outputs['values']
                    entropy = outputs['action_dist'].entropy()
                    
                    # Compute losses (same as above)
                    ratio = torch.exp(log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_pred_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values, -self.clip_ratio, self.clip_ratio
                    )
                    value_losses = (values - mb_returns).pow(2)
                    value_losses_clipped = (value_pred_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    
                    entropy_loss = -entropy.mean()
                    
                    # Distance and auxiliary losses
                    distance_loss = torch.tensor(0.0, device=self.device)
                    auxiliary_loss = torch.tensor(0.0, device=self.device)
                    
                    total_loss = (
                        policy_loss + 
                        self.value_coef * value_loss + 
                        self.entropy_coef * entropy_loss +
                        self.distance_coef * distance_loss +
                        self.auxiliary_coef * auxiliary_loss
                    )
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Track statistics
                with torch.no_grad():
                    approx_kl = ((log_probs - mb_old_log_probs).exp() - 1 - (log_probs - mb_old_log_probs)).mean()
                    clip_fraction = ((ratio - 1).abs() > self.clip_ratio).float().mean()
                
                update_stats['policy_loss'] += policy_loss.item()
                update_stats['value_loss'] += value_loss.item()
                update_stats['entropy_loss'] += entropy_loss.item()
                update_stats['distance_loss'] += distance_loss.item()
                update_stats['auxiliary_loss'] += auxiliary_loss.item()
                update_stats['total_loss'] += total_loss.item()
                update_stats['approx_kl'] += approx_kl.item()
                update_stats['clip_fraction'] += clip_fraction.item()
        
        # Average statistics
        num_updates = self.ppo_epochs * (len(advantages) // self.batch_size)
        for key in update_stats:
            update_stats[key] /= max(1, num_updates)
        
        if self.stage == 1:
            self.auxiliary_losses.append(update_stats['auxiliary_loss'])
        
        return update_stats
    
    def train(self, total_timesteps: int):
        """Main training loop with stage-specific logic"""
        print(f"Starting Stage {self.stage} training for {total_timesteps} timesteps...")
        print(f"Device: {self.device}")
        print(f"Scene names: {self.config['scene_names']}")
        print(f"Image size: {self.config['image_size']}")
        print(f"Using mixed precision: {self.use_amp}")
        
        timesteps_collected = 0
        update_count = 0
        best_success_rate = 0
        
        while timesteps_collected < total_timesteps:
            # Collect rollouts
            rollout_stats = self.collect_rollouts(self.config['rollout_steps'])
            timesteps_collected += self.config['rollout_steps']
            
            # Update policy
            update_stats = self.update_policy()
            update_count += 1
            
            # Clear GPU cache periodically
            if update_count % 10 == 0:
                torch.cuda.empty_cache()
            
            # Logging
            if update_count % self.config['log_freq'] == 0:
                self._log_training_stats(timesteps_collected, rollout_stats, update_stats)
            
            # Save checkpoint
            if update_count % self.config['save_freq'] == 0:
                self._save_model(update_count, timesteps_collected)
            
            # Evaluate
            if update_count % self.config.get('eval_freq', 100) == 0:
                eval_metrics = self._evaluate()
                if eval_metrics['success_rate'] > best_success_rate:
                    best_success_rate = eval_metrics['success_rate']
                    self._save_model(update_count, timesteps_collected, is_best=True)
        
        print(f"Stage {self.stage} training completed!")
        print(f"Best success rate: {best_success_rate:.2%}")
        self.env.close()
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy"""
        print("\nRunning evaluation...")
        self.model.eval()
        
        eval_episodes = self.config.get('eval_episodes', 10)
        eval_rewards = []
        eval_successes = []
        eval_lengths = []
        
        for _ in range(eval_episodes):
            obs = self.env.reset()
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
                
                next_obs, _, done, info = self.env.step(action.cpu().item())
                event = self.env.controller.last_event
                reward = self.reward_calculator.calculate_reward(event, next_obs, info, self.env)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                torch_obs = prepare_observation(next_obs, self.device)
            
            eval_rewards.append(episode_reward)
            eval_successes.append(info.get('success', False))
            eval_lengths.append(episode_length)
        
        self.model.train()
        
        eval_metrics = {
            'success_rate': np.mean(eval_successes),
            'avg_reward': np.mean(eval_rewards),
            'avg_length': np.mean(eval_lengths)
        }
        
        print(f"Evaluation - Success Rate: {eval_metrics['success_rate']:.2%}, "
              f"Avg Reward: {eval_metrics['avg_reward']:.2f}, "
              f"Avg Length: {eval_metrics['avg_length']:.1f}")
        
        return eval_metrics
    
    def _log_training_stats(self, timesteps: int, rollout_stats: Dict, update_stats: Dict):
        """Log training statistics"""
        print(f"\n--- Stage {self.stage} Update {timesteps // self.config['rollout_steps']} "
              f"(Timesteps: {timesteps}) ---")
        print(f"Curriculum Level: {self.curriculum_manager.current_level + 1}")
        
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
                'curriculum_level': self.curriculum_manager.current_level,
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
            
            # Add comprehensive metrics
            metrics = self.evaluation_metrics.compute_metrics()
            if metrics:
                log_dict.update(metrics)
            
            wandb.log(log_dict)
    
    def _save_model(self, update_count: int, timesteps: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_name = f'stage{self.stage}_{"best" if is_best else update_count}.pth'
        save_path = os.path.join(self.config['save_dir'], checkpoint_name)
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': update_count,
            'timesteps': timesteps,
            'stage': self.stage,
            'curriculum_level': self.curriculum_manager.current_level,
            'config': self.config,
            'success_rate': np.mean(self.success_rates) if self.success_rates else 0,
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def _load_stage1_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from stage 1 training"""
        print(f"Loading Stage 1 checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights, but not auxiliary heads if transitioning to stage 2
        model_state = checkpoint['model_state_dict']
        if self.stage == 2:
            # Remove auxiliary head weights
            model_state = {k: v for k, v in model_state.items() 
                          if not k.startswith('collision_predictor') and 
                          not k.startswith('exploration_predictor')}
        
        self.model.load_state_dict(model_state, strict=False)
        print(f"Successfully loaded Stage 1 model (Success Rate: {checkpoint.get('success_rate', 0):.2%})")


def main():
    parser = argparse.ArgumentParser(description='Two-Stage Training for NoMaD-RL')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                       help='Training stage (1 or 2)')
    parser.add_argument('--stage1-checkpoint', type=str, default=None,
                       help='Path to stage 1 checkpoint (required for stage 2)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['training_stage'] = args.stage
    if args.stage1_checkpoint:
        config['stage1_checkpoint'] = args.stage1_checkpoint
    
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'nomad-rl-two-stage'),
            name=f"{config.get('run_name', 'nomad_rl')}_stage{args.stage}",
            config=config
        )
    
    trainer = TwoStageTrainer(config)
    trainer.train(config[f'stage{args.stage}_timesteps'])


if __name__ == '__main__':
    main()