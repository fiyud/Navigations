import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import argparse
import yaml
import os
from collections import deque
from typing import Dict, List, Tuple
import time
import torch.nn.functional as F

import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train')

from nomad_rl.environments.ai2thor_nomad_env import AI2ThorNoMaDEnv
from nomad_rl.models.nomad_rl_model import NoMaDRL, prepare_observation, PPOBuffer

class NoMaDRLTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        self.env = AI2ThorNoMaDEnv(
            scene_names=config['scene_names'],
            image_size=tuple(config['image_size']),
            max_episode_steps=config['max_episode_steps'],
            success_distance=config['success_distance'],
            context_size=config['context_size'],
            goal_prob=config['goal_prob']
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
        
        if config.get('pretrained_vision_encoder'):
            self._load_pretrained_vision_encoder(config['pretrained_vision_encoder'])
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
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
            action_dim=self.env.action_space.n,  # Now 4
            device=self.device
        )
        
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.exploration_rewards = deque(maxlen=100)
        self.goal_conditioned_rewards = deque(maxlen=100)
        
        self.gamma = config['gamma']
        self.lam = config['lam']
        self.clip_ratio = config['clip_ratio']
        self.entropy_coef = config['entropy_coef']
        self.value_coef = config['value_coef']
        self.distance_coef = config['distance_coef']
        self.max_grad_norm = config['max_grad_norm']
        self.ppo_epochs = config['ppo_epochs']
        self.batch_size = config['batch_size']
        
    def _load_pretrained_vision_encoder(self, checkpoint_path: str):
        print(f"Loading pre-trained vision encoder from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            vision_encoder_state = {}
            for key, value in checkpoint.items():
                if key.startswith('vision_encoder.'):
                    vision_encoder_state[key] = value
            self.model.vision_encoder.load_state_dict(vision_encoder_state, strict=False)
            print("Successfully loaded pre-trained vision encoder")
            if self.config.get('freeze_vision_encoder', False):
                for param in self.model.vision_encoder.parameters():
                    param.requires_grad = False
                print("Froze vision encoder parameters")
        except Exception as e:
            print(f"Failed to load pre-trained vision encoder: {e}")
    
    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        obs = self.env.reset()
        torch_obs = prepare_observation(obs, self.device)
        
        episode_reward = 0
        episode_length = 0
        episode_success = False
        episode_goal_conditioned = self.env.is_goal_conditioned
        
        rollout_stats = {
            'total_reward': 0,
            'episodes': 0,
            'successes': 0,
            'exploration_episodes': 0,
            'goal_conditioned_episodes': 0
        }
        
        for step in range(num_steps):
            with torch.no_grad():
                outputs = self.model.forward(torch_obs, mode="all")
                action_dist = outputs['action_dist']
                value = outputs['values']
                
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
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
            if info.get('success', False):
                episode_success = True
            
            rollout_stats['total_reward'] += reward
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.success_rates.append(1.0 if episode_success else 0.0)
                
                if episode_goal_conditioned:
                    self.goal_conditioned_rewards.append(episode_reward)
                    rollout_stats['goal_conditioned_episodes'] += 1
                else:
                    self.exploration_rewards.append(episode_reward)
                    rollout_stats['exploration_episodes'] += 1
                
                rollout_stats['episodes'] += 1
                rollout_stats['successes'] += 1 if episode_success else 0
                
                obs = self.env.reset()
                torch_obs = prepare_observation(obs, self.device)
                episode_reward = 0
                episode_length = 0
                episode_success = False
                episode_goal_conditioned = self.env.is_goal_conditioned
            else:
                obs = next_obs
                torch_obs = next_torch_obs
        
        with torch.no_grad():
            final_value = self.model.forward(torch_obs, mode="value")['values'].cpu().item()
        
        self.buffer.compute_gae(final_value, self.gamma, self.lam)
        
        return rollout_stats
    
    def update_policy(self) -> Dict[str, float]:
        batch = self.buffer.get()
        update_stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy_loss': 0,
            'distance_loss': 0,
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
                
                log_probs, values, entropy = self.model.evaluate_actions(mb_obs, mb_actions)
                
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
                
                distance_loss = torch.tensor(0.0, device=self.device)
                if mb_obs['goal_mask'].sum() < len(mb_obs['goal_mask']):
                    goal_conditioned_mask = (mb_obs['goal_mask'].squeeze() == 0)
                    if goal_conditioned_mask.sum() > 0:
                        outputs = self.model.forward(mb_obs, mode="distance")
                        predicted_distances = outputs['distances'][goal_conditioned_mask]
                        target_distances = -mb_returns[goal_conditioned_mask]
                        predicted_distances = predicted_distances.view(-1)
                        target_distances = target_distances.view(-1)
                        distance_loss = F.mse_loss(predicted_distances, target_distances)
                
                total_loss = (policy_loss + 
                             self.value_coef * value_loss + 
                             self.entropy_coef * entropy_loss +
                             self.distance_coef * distance_loss)
                
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                with torch.no_grad():
                    approx_kl = ((log_probs - mb_old_log_probs).exp() - 1 - (log_probs - mb_old_log_probs)).mean()
                    clip_fraction = ((ratio - 1).abs() > self.clip_ratio).float().mean()
                
                update_stats['policy_loss'] += policy_loss.item()
                update_stats['value_loss'] += value_loss.item()
                update_stats['entropy_loss'] += entropy_loss.item()
                update_stats['distance_loss'] += distance_loss.item()
                update_stats['total_loss'] += total_loss.item()
                update_stats['approx_kl'] += approx_kl.item()
                update_stats['clip_fraction'] += clip_fraction.item()
        
        num_updates = self.ppo_epochs * (len(advantages) // self.batch_size)
        for key in update_stats:
            update_stats[key] /= num_updates
        
        return update_stats
    
    def train(self, total_timesteps: int):
        print(f"Starting training for {total_timesteps} timesteps...")
        print(f"Device: {self.device}")
        print(f"Scene names: {self.config['scene_names']}")
        
        timesteps_collected = 0
        update_count = 0
        
        while timesteps_collected < total_timesteps:
            rollout_stats = self.collect_rollouts(self.config['rollout_steps'])
            timesteps_collected += self.config['rollout_steps']
            
            update_stats = self.update_policy()
            update_count += 1
            
            if update_count % self.config['log_freq'] == 0:
                self._log_training_stats(timesteps_collected, rollout_stats, update_stats)
            
            if update_count % self.config['save_freq'] == 0:
                self._save_model(update_count)
        
        print("Training completed!")
        self.env.close()
    
    def _log_training_stats(self, timesteps: int, rollout_stats: Dict, update_stats: Dict):
        print(f"\n--- Update {timesteps // self.config['rollout_steps']} (Timesteps: {timesteps}) ---")
        
        if len(self.episode_rewards) > 0:
            print(f"Episode Reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
            print(f"Episode Length: {np.mean(self.episode_lengths):.1f}")
            print(f"Success Rate: {np.mean(self.success_rates):.2%}")
            
            if len(self.goal_conditioned_rewards) > 0:
                print(f"Goal-Conditioned Reward: {np.mean(self.goal_conditioned_rewards):.2f}")
            if len(self.exploration_rewards) > 0:
                print(f"Exploration Reward: {np.mean(self.exploration_rewards):.2f}")
        
        print(f"Policy Loss: {update_stats['policy_loss']:.4f}")
        print(f"Value Loss: {update_stats['value_loss']:.4f}")
        print(f"Entropy: {-update_stats['entropy_loss']:.4f}")
        print(f"Approx KL: {update_stats['approx_kl']:.4f}")
        
        if self.config.get('use_wandb', False):
            wandb.log({
                'timesteps': timesteps,
                'episode_reward_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'episode_length_mean': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'success_rate': np.mean(self.success_rates) if self.success_rates else 0,
                'goal_conditioned_reward': np.mean(self.goal_conditioned_rewards) if self.goal_conditioned_rewards else 0,
                'exploration_reward': np.mean(self.exploration_rewards) if self.exploration_rewards else 0,
                'policy_loss': update_stats['policy_loss'],
                'value_loss': update_stats['value_loss'],
                'entropy': -update_stats['entropy_loss'],
                'distance_loss': update_stats['distance_loss'],
                'approx_kl': update_stats['approx_kl'],
                'clip_fraction': update_stats['clip_fraction'],
            })
    
    def _save_model(self, update_count: int):
        save_path = os.path.join(self.config['save_dir'], f'nomad_rl_{update_count}.pth')
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': update_count,
            'config': self.config
        }, save_path)
        
        print(f"Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train NoMaD-RL in AI2Thor')
    parser.add_argument('--config', type=str, default='/home/tuandang/tuandang/quanganh/visualnav-transformer/train/config/nomad_rl.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'nomad-rl'),
            name=config.get('run_name', 'nomad_rl_run'),
            config=config
        )
    
    trainer = NoMaDRLTrainer(config)
    trainer.train(config['total_timesteps'])

if __name__ == '__main__':
    main()