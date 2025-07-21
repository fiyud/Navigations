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
from tqdm import tqdm

import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train/gru')

from model import (
    UnifiedAdvancedNoMaDRL, 
    UnifiedAdvancedTrainingWrapper,
    create_unified_model,
    visualize_spatial_memory_graph
)
# Check cac goc de dua ra quyet dinh, co the dung bell-man equations
from preprocess.grid_manager import GridBasedCurriculumManager
from environments import EnhancedAI2ThorEnv
from nomad_rl.models.nomad_rl_model import prepare_observation, PPOBuffer

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ':1.0'  # Change from :0 to :1.0
import matplotlib
matplotlib.use('Agg')

torch.autograd.set_detect_anomaly(True)

import torch.fx.traceback as fx_traceback
fx_traceback.has_preserved_node_meta = lambda: False

class UnifiedAdvancedNoMaDTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.dataset = config['dataset']
        self.splits = self._load_splits(config)
        
        # Initialize curriculum manager
        self.curriculum_manager = GridBasedCurriculumManager(config, self.dataset)
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        self.reset_graph_every_n_episodes = config.get('reset_graph_every_n_episodes', 50)
        self.episodes_since_graph_reset = 0

        # Get initial training scenes
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
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=100,
            gamma=0.9
        )
        
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
        # self.val_env = self._create_environment(
        #     self.splits['val'],
        #     config,
        #     goal_prob=config.get('eval_goal_prob', 1.0)
        # )
        
        # self.test_env = self._create_environment(
        #     self.splits['test'],
        #     config,
        #     goal_prob=config.get('eval_goal_prob', 1.0)
        # )
        
        self.val_env = None
        self.test_env = None
        self.val_scenes = self.splits['val']
        self.test_scenes = self.splits['test']

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

        self.use_parallel = config.get('use_parallel_collection', True)
        if self.use_parallel:
            self.collector = ParallelCollector(
                env_fn=lambda: self._create_environment(
                    current_scenes, config, 
                    goal_prob=curriculum_settings['goal_prob'],
                    max_episode_steps=curriculum_settings['max_episode_steps']
                ),
                model=self.model,
                device=self.device,
                num_workers=config.get('num_collectors', 2)
            )

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
            {'params': vision_params, 'lr': base_lr * 0.1},
            {'params': spatial_graph_params, 'lr': base_lr * 2},
            {'params': counterfactual_params, 'lr': base_lr},
            # {'params': value_params, 'lr': base_lr * 0.1},  # Lower LR for value network
            {'params': other_params, 'lr': base_lr}
        ]
                
        return optim.Adam(param_groups)
    
    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """Collect rollouts with parallel workers or single environment"""
        if self.use_parallel:
            return self._collect_rollouts_parallel(num_steps)
        else:
            return self._collect_rollouts_single(num_steps)
    
    def _collect_rollouts_parallel(self, num_steps: int) -> Dict[str, float]:
        """Collect rollouts using parallel workers"""
        # Update workers with latest model
        self.collector.update_model(self.model.state_dict())
        
        # Collect trajectories
        steps_collected = 0
        rollout_stats = defaultdict(float)
        
        while steps_collected < num_steps:
            trajectories = self.collector.get_trajectories(timeout=1.0)
            
            for trajectory in trajectories:
                for obs, action, reward, next_obs, done, info in trajectory:
                    # Process and store in buffer
                    torch_obs = prepare_observation(obs, self.device)
                    torch_next_obs = prepare_observation(next_obs, self.device)
                    
                    # Get value estimate
                    with torch.no_grad():
                        outputs = self.model.forward(torch_obs, mode="value")
                        value = outputs['values']
                    
                    # Store in buffer
                    self.buffer.store(
                        obs=torch_obs,
                        action=action,
                        reward=reward,
                        value=value,
                        log_prob=torch.tensor(0.0),  # Recompute during update
                        done=done
                    )
                    
                    steps_collected += 1
                    rollout_stats['total_reward'] += reward
                    
                    if done:
                        rollout_stats['episodes'] += 1
                        if info.get('success', False):
                            rollout_stats['successes'] += 1
        
        return rollout_stats

    def _collect_rollouts_single(self, num_steps: int) -> Dict[str, float]:
        """Collect rollouts with unified model"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        obs = self.env.reset()
        torch_obs = prepare_observation(obs, self.device)
        
        if hasattr(self.model.spatial_memory_graph, 'graph_state') and self.model.spatial_memory_graph.graph_state:
            print(f"Graph nodes before reset: {self.model.spatial_memory_graph.graph_state.num_nodes}")

        self.training_wrapper.reset_episode()
        
        if hasattr(self.model.spatial_memory_graph, 'graph_state') and self.model.spatial_memory_graph.graph_state:
            print(f"Graph nodes after reset: {self.model.spatial_memory_graph.graph_state.num_nodes}")

        episode_reward = 0
        episode_length = 0
        episode_success = False
        episode_collisions = 0
        episode_graph_sizes = []
        episode_path_confs = []
        
        episode_penalties = 0
        episode_exploration_rewards = 0
        episode_goal_rewards = 0
        action_counts = defaultdict(int)

        if not hasattr(self, 'goal_conditioned_episodes'):
            self.goal_conditioned_episodes = 0
            self.exploration_episodes = 0

        rollout_stats = defaultdict(float)
        rollout_stats['episodes'] = 0
        
        for step in range(num_steps):
            # with torch.no_grad():
            #     if self.use_amp:
            #         with autocast():
            #             action, log_prob, extra_info = self.training_wrapper.step(torch_obs)
            #             outputs = self.model.forward(torch_obs, mode="all", 
            #                                     time_delta=self.training_wrapper.get_time_delta())
            #     else:
            #         action, log_prob, extra_info = self.training_wrapper.step(torch_obs)
            #         outputs = self.model.forward(torch_obs, mode="all", 
            #                                 time_delta=self.training_wrapper.get_time_delta())
            #     value = outputs['values']
            
            with torch.no_grad():
                action, log_prob, extra_info = self.training_wrapper.step(torch_obs)
                outputs = self.model.forward(torch_obs, mode="all", 
                                        time_delta=self.training_wrapper.get_time_delta())
                value = outputs['values']

            action_item = action.cpu().item()
            action_counts[action_item] += 1

            next_obs, reward, done, info = self.env.step(action_item)
            next_torch_obs = prepare_observation(next_obs, self.device)

            if reward < 0:
                episode_penalties += reward
            if info.get('exploration_reward', 0) > 0:
                episode_exploration_rewards += info.get('exploration_reward', 0)
            if info.get('goal_reward', 0) > 0:
                episode_goal_rewards += info.get('goal_reward', 0)

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
                if self.env.is_goal_conditioned:
                    self.goal_conditioned_episodes += 1
                else:
                    self.exploration_episodes += 1

                total_episodes = self.goal_conditioned_episodes + self.exploration_episodes
                if total_episodes % 10 == 0:
                    print(f"\nEpisode Summary - Total: {total_episodes}, "
                        f"Goal: {self.goal_conditioned_episodes} ({self.goal_conditioned_episodes/total_episodes*100:.1f}%), "
                        f"Exploration: {self.exploration_episodes} ({self.exploration_episodes/total_episodes*100:.1f}%)")

                if self.config.get('use_wandb', False):
                    episode_metrics = {
                        'episode/reward': episode_reward,
                        'episode/length': episode_length,
                        'episode/success': float(episode_success),
                        'episode/collisions': episode_collisions,
                        'episode/penalties': episode_penalties,
                        'episode/exploration_rewards': episode_exploration_rewards,
                        'episode/goal_rewards': episode_goal_rewards,
                        'episode/avg_graph_size': np.mean(episode_graph_sizes),
                        'episode/avg_path_confidence': np.mean(episode_path_confs),
                        'episode/collision_rate': episode_collisions / max(1, episode_length),
                    }
                    
                    # Action distribution
                    for action_idx in range(self.env.action_space.n):
                        episode_metrics[f'episode/action_{action_idx}_freq'] = action_counts[action_idx] / episode_length
                
                    wandb.log(episode_metrics)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.success_rates.append(1.0 if episode_success else 0.0)
                self.graph_sizes.append(np.mean(episode_graph_sizes))
                self.path_confidences.append(np.mean(episode_path_confs))
                
                # Update curriculum
                print(f"Episode finished - Reward: {episode_reward:.2f}, "
                f"Success: {episode_success}, Length: {episode_length}, "
                f"Type: {'GOAL' if self.env.is_goal_conditioned else 'EXPLORE'}")
                
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
                rollout_stats['total_penalties'] += episode_penalties
                rollout_stats['total_exploration_rewards'] += episode_exploration_rewards
                rollout_stats['total_goal_rewards'] += episode_goal_rewards
                
                obs = self.env.reset()
                torch_obs = prepare_observation(obs, self.device)

                self.episodes_since_graph_reset += 1
    
                # Only reset graph periodically
                if self.episodes_since_graph_reset >= self.reset_graph_every_n_episodes:
                    print(f"Resetting spatial memory graph after {self.episodes_since_graph_reset} episodes")
                    self.model.reset_memory()
                    self.episodes_since_graph_reset = 0
                # self.training_wrapper.reset_episode()
                
                episode_reward = 0
                episode_length = 0
                episode_success = False
                episode_collisions = 0
                episode_graph_sizes = []
                episode_path_confs = []
                episode_penalties = 0
                episode_exploration_rewards = 0
                episode_goal_rewards = 0
                action_counts = defaultdict(int)
            else:
                obs = next_obs
                torch_obs = next_torch_obs
            
            if step % 30 == 0:
                torch.cuda.empty_cache()

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
        
        current_pos = self.env._get_agent_position()
        print(f"Step {step}: Position ({current_pos['x']:.2f}, {current_pos['z']:.2f})")

        return rollout_stats
    
    def pretrain_world_model(self, num_episodes: int = 50):
        """Pretrain world model with random actions like DayDreamer"""
        print("\n" + "="*60)
        print(f"Pretraining world model with {num_episodes} random episodes...")
        print("="*60)
        
        pretrain_buffer = PPOBuffer(
            size=self.config['buffer_size'],
            obs_shape={
                'rgb': (3, *self.config['image_size']),
                'context': (3 * self.config['context_size'], *self.config['image_size']),
                'goal_rgb': (3, *self.config['image_size']),
                'goal_mask': (1,),
                'goal_position': (3,)
            },
            action_dim=self.env.action_space.n,
            device=self.device
        )
        
        # Collect random experience
        for episode in range(num_episodes):
            obs = self.env.reset()
            torch_obs = prepare_observation(obs, self.device)
            self.training_wrapper.reset_episode()
            
            episode_steps = 0
            done = False
            
            while not done and episode_steps < self.config['max_episode_steps']:
                # Random action
                action = torch.tensor([self.env.action_space.sample()], device=self.device)
                
                # Get current features for world model training
                with torch.no_grad():
                    outputs = self.model.forward(torch_obs, mode="features")
                    value = torch.zeros(1, device=self.device)  # Dummy value
                
                # Step environment
                next_obs, reward, done, info = self.env.step(action.item())
                next_torch_obs = prepare_observation(next_obs, self.device)
                
                # Store transition
                pretrain_buffer.store(
                    obs=torch_obs,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=torch.tensor(0.0, device=self.device),
                    done=done
                )
                
                torch_obs = next_torch_obs
                episode_steps += 1
            
            print(f"Episode {episode + 1}/{num_episodes}: {episode_steps} steps")
            
            # Train world model every few episodes
            if (episode + 1) % 5 == 0 and pretrain_buffer.ptr > self.config['batch_size']:
                print(f"Training world model...")
                
                for _ in range(50):  # Multiple gradient steps
                    # Sample batch
                    indices = torch.randint(0, pretrain_buffer.ptr, (self.config['batch_size'],))
                    
                    batch_obs = {k: v[indices] for k, v in pretrain_buffer.observations.items()}
                    batch_actions = pretrain_buffer.actions[indices]
                    batch_rewards = pretrain_buffer.rewards[indices]
                    
                    # Compute world model loss
                    world_loss = self._compute_world_model_loss(batch_obs, batch_actions, batch_rewards)
                    
                    # Update only world model parameters
                    self.optimizer.zero_grad()
                    world_loss.backward()
                    
                    # Clip gradients
                    nn.utils.clip_grad_norm_(
                        self.model.spatial_memory_graph.parameters(), 
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                
                print(f"World model loss: {world_loss.item():.4f}")
        
        print("\nWorld model pretraining complete!")
        print("="*60)

    def _compute_world_model_loss(self, observations, actions, rewards):
        """Compute world model reconstruction and dynamics loss"""
        # Get features
        outputs = self.model.forward(observations, mode="features")
        features = outputs['features']
        
        # Reconstruction loss (if using decoder)
        recon_loss = 0.0
        if hasattr(self.model.spatial_memory_graph.graph_ode, 'decoder'):
            # Decode features
            decoded = self.model.spatial_memory_graph.graph_ode.decoder(features)
            target = observations['rgb'].flatten(1)  # Flatten spatial dims
            recon_loss = F.mse_loss(decoded, target)
        
        # Reward prediction loss
        reward_loss = 0.0
        if hasattr(self.model.spatial_memory_graph.graph_ode, 'reward_net'):
            pred_rewards = self.model.spatial_memory_graph.graph_ode.reward_net(features).squeeze()
            reward_loss = F.mse_loss(pred_rewards, rewards)
        
        # KL loss for RSSM (if using RSSM)
        kl_loss = 0.0
        if hasattr(self.model.spatial_memory_graph.graph_ode, 'rssm'):
            # Add KL divergence between prior and posterior
            kl_loss = 0.1  # Placeholder - implement proper KL computation
        
        total_loss = recon_loss + reward_loss + 0.8 * kl_loss
        
        return total_loss

    def update_policy(self) -> Dict[str, float]:
        batch = self.buffer.get()
        update_stats = defaultdict(float)
        
        returns = batch['returns']
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        batch['returns'] = returns
        
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
        
        print(f"  Running {self.ppo_epochs} PPO epochs with batch size {self.batch_size}")
        
        # Progress bar for PPO epochs
        epoch_pbar = tqdm(range(self.ppo_epochs), desc="  PPO Epochs", leave=False)
        
        for epoch in epoch_pbar:
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
                    self.scheduler.step()
                # Track statistics
                for key, value in losses.items():
                    update_stats[key] += value.item()
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'policy_loss': f"{update_stats['policy_loss']/(epoch+1):.4f}",
                'value_loss': f"{update_stats['value_loss']/(epoch+1):.4f}"
            })
        
        # Average statistics
        num_updates = self.ppo_epochs * (len(advantages) // self.batch_size)
        for key in update_stats:
            update_stats[key] /= max(1, num_updates)
        
        print(f"  Update complete - Policy loss: {update_stats['policy_loss']:.4f}, Value loss: {update_stats['value_loss']:.4f}")
        
        return dict(update_stats)
    
    def train(self, total_timesteps: int):
        print("="*80)
        print(f"Starting unified training for {total_timesteps} timesteps...")
        print(f"Dataset: {self.dataset}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*80)
        
        if self.config.get('pretrain_world_model', True):
            pretrain_episodes = self.config.get('pretrain_episodes', 50)
            self.pretrain_world_model(num_episodes=pretrain_episodes)

        timesteps_collected = 0
        update_count = 0
        
        pbar = tqdm(total=total_timesteps, desc="Training Progress", unit="timesteps")
        
        while timesteps_collected < total_timesteps:
            # Check for curriculum update
            if update_count > 0 and update_count % self.config.get('curriculum_update_freq', 10) == 0:
                print(f"\n[Update {update_count}] Checking curriculum progression...")
                old_level = self.curriculum_manager.current_level
                if self.curriculum_manager.current_level != old_level:
                    self._update_environment_curriculum()
            
            # Collect rollouts
            print(f"\n[Update {update_count}] Collecting rollouts...")
            rollout_start_time = time.time()
            rollout_stats = self.collect_rollouts(self.config['rollout_steps'])
            rollout_time = time.time() - rollout_start_time
            timesteps_collected += self.config['rollout_steps']
            
            # Update policy
            print(f"[Update {update_count}] Updating policy...")
            update_start_time = time.time()
            update_stats = self.update_policy()
            update_time = time.time() - update_start_time
            update_count += 1
            
            pbar.update(self.config['rollout_steps'])
            pbar.set_postfix({
                'reward': f"{np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f}",
                'success': f"{np.mean(self.success_rates) if self.success_rates else 0:.2%}",
                'update': update_count
            })
            
            print(f"[Update {update_count}] Rollout time: {rollout_time:.2f}s, Update time: {update_time:.2f}s")
            
            if update_count % 10 == 0:
                torch.cuda.empty_cache()
            
            if update_count % self.config['log_freq'] == 0:
                self._log_training_stats(timesteps_collected, rollout_stats, update_stats)
                
                if update_count % (self.config['log_freq'] * 10) == 0:
                    print(f"\n[Update {update_count}] Visualizing spatial memory graph...")
                    self._visualize_graph(update_count)
            
            if update_count % self.config.get('val_freq', 100) == 0:
                print(f"\n[Update {update_count}] Running validation...")
                val_metrics = self._evaluate_on_split('val', self.val_env)
                self.results['val']['timesteps'].append(timesteps_collected)
                
                if self.config.get('use_wandb', False):
                    val_wandb_metrics = {
                        f'val/{key}': value for key, value in val_metrics.items()
                    }
                    wandb.log(val_wandb_metrics, step=timesteps_collected)
                
                for key, value in val_metrics.items():
                    self.results['val'][key].append(value)
                
                if val_metrics['success_rate'] > self.best_val_success_rate:
                    print(f"New best validation success rate: {val_metrics['success_rate']:.2%}")
                    self.best_val_success_rate = val_metrics['success_rate']
                    
                    if self.config.get('use_wandb', False):
                        wandb.log({
                            'val/best_success_rate': self.best_val_success_rate,
                            'val/best_at_timestep': timesteps_collected
                        }, step=timesteps_collected)
                    
                    self.best_val_checkpoint = self._save_model(
                        update_count, timesteps_collected, is_best=True
                    )
            
            if update_count % self.config['save_freq'] == 0:
                print(f"\n[Update {update_count}] Saving checkpoint...")
                self._save_model(update_count, timesteps_collected)
        
        pbar.close()
        
        print("\n" + "="*80)
        print("Training completed! Running final evaluation...")
        print("="*80)
        final_results = self._run_final_evaluation()
        self._save_final_results(final_results)
        
        self.env.close()
        if self.val_env is not None:
            self.val_env.close()
        if self.test_env is not None:
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
        
        # Calculate detailed metrics
        metrics = {}
        
        if len(self.episode_rewards) > 0:
            # Basic metrics
            metrics['episode_reward_mean'] = np.mean(self.episode_rewards)
            metrics['episode_reward_std'] = np.std(self.episode_rewards)
            metrics['episode_reward_min'] = np.min(self.episode_rewards)
            metrics['episode_reward_max'] = np.max(self.episode_rewards)
            
            metrics['success_rate'] = np.mean(self.success_rates)
            metrics['episode_length_mean'] = np.mean(self.episode_lengths)
            metrics['episode_length_std'] = np.std(self.episode_lengths)
            
            # Graph metrics
            metrics['graph_size_mean'] = np.mean(self.graph_sizes)
            metrics['graph_size_max'] = np.max(self.graph_sizes) if self.graph_sizes else 0
            metrics['path_confidence_mean'] = np.mean(self.path_confidences)
            
            print(f"Episode Reward: {metrics['episode_reward_mean']:.2f} ± {metrics['episode_reward_std']:.2f}")
            print(f"Success Rate: {metrics['success_rate']:.2%}")
            print(f"Avg Graph Size: {metrics['graph_size_mean']:.1f} nodes")
            print(f"Path Confidence: {metrics['path_confidence_mean']:.3f}")
        
        # Loss metrics
        metrics['losses/policy_loss'] = update_stats['policy_loss']
        metrics['losses/value_loss'] = update_stats['value_loss']
        metrics['losses/entropy_loss'] = update_stats.get('entropy_loss', 0)
        metrics['losses/total_loss'] = update_stats['total_loss']
        
        # Auxiliary losses
        if 'counterfactual_loss' in update_stats:
            metrics['losses/counterfactual_loss'] = update_stats['counterfactual_loss']
        if 'spatial_smoothness_loss' in update_stats:
            metrics['losses/spatial_smoothness_loss'] = update_stats['spatial_smoothness_loss']
        
        print(f"Policy Loss: {metrics['losses/policy_loss']:.4f}")
        print(f"Value Loss: {metrics['losses/value_loss']:.4f}")
        
        # Curriculum metrics
        metrics['curriculum/level'] = curriculum_stats['current_level']
        metrics['curriculum/level_name'] = curriculum_stats['level_name']
        metrics['curriculum/episodes_at_level'] = curriculum_stats['episodes_at_level']
        metrics['curriculum/current_success_rate'] = curriculum_stats['current_success_rate']
        metrics['curriculum/grid_range_min'] = curriculum_stats['grid_range'][0]
        metrics['curriculum/grid_range_max'] = curriculum_stats['grid_range'][1]
        
        # Rollout statistics
        if rollout_stats:
            metrics['rollout/episodes'] = rollout_stats.get('episodes', 0)
            metrics['rollout/successes'] = rollout_stats.get('successes', 0)
            metrics['rollout/collisions'] = rollout_stats.get('collisions', 0)
            metrics['rollout/avg_graph_size'] = rollout_stats.get('avg_graph_size', 0)
            metrics['rollout/avg_path_conf'] = rollout_stats.get('avg_path_conf', 0)
            
            # Calculate rates
            if rollout_stats.get('episodes', 0) > 0:
                metrics['rollout/collision_rate'] = rollout_stats['collisions'] / rollout_stats['episodes']
                metrics['rollout/success_rate_batch'] = rollout_stats['successes'] / rollout_stats['episodes']
        
        # Training progress
        metrics['training/timesteps'] = timesteps
        metrics['training/episodes_total'] = len(self.episode_rewards)
        metrics['training/updates'] = timesteps // self.config['rollout_steps']
        
        # Learning rate (if using scheduler)
        metrics['training/learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        # Memory usage
        if torch.cuda.is_available():
            metrics['system/gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1e9  # GB
            metrics['system/gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1e9  # GB
        
        if self.config.get('use_wandb', False):
            wandb.log(metrics, step=timesteps)
    
    def _evaluate_on_split(self, split_name: str, env: EnhancedAI2ThorEnv, 
                      num_episodes: Optional[int] = None) -> Dict[str, float]:
        """Evaluate on a specific split"""
        if num_episodes is None:
            num_episodes = self.config.get('eval_episodes', 20)

        created_env = False
        if env is None:
            if split_name == 'val':
                if self.val_env is None:
                    print(f"Creating validation environment...")
                    self.val_env = self._create_environment(
                        self.val_scenes,
                        self.config,
                        goal_prob=self.config.get('eval_goal_prob', 1.0)
                    )
                    created_env = True
                env = self.val_env
            elif split_name == 'test':
                if self.test_env is None:
                    print(f"Creating test environment...")
                    self.test_env = self._create_environment(
                        self.test_scenes,
                        self.config,
                        goal_prob=self.config.get('eval_goal_prob', 1.0)
                    )
                    created_env = True
                env = self.test_env
        
        print(f"\nEvaluating on {split_name} split ({num_episodes} episodes)...")
        self.model.eval()
        
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_graph_sizes = []
        episode_path_confs = []
        
        # Progress bar for evaluation episodes
        eval_pbar = tqdm(range(num_episodes), desc=f"Evaluating {split_name}", leave=False)
        
        for episode_idx in eval_pbar:
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
            
            # Update progress bar
            current_sr = sum(episode_successes) / len(episode_successes)
            eval_pbar.set_postfix({
                'success_rate': f"{current_sr:.2%}",
                'avg_reward': f"{np.mean(episode_rewards):.2f}"
            })
        
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

        if created_env and split_name == 'test':
            env.close()
            if split_name == 'test':
                self.test_env = None
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
            splits_file = config.get('splits_file', '/home/tuandang/tuandang/quanganh/visualnav-transformer/train/ode/grid/config/splits/combined_splits.yaml')
        else:
            splits_file = config.get('splits_file', f'/home/tuandang/tuandang/quanganh/visualnav-transformer/train/ode/grid/config/splits/{dataset}_splits.yaml')
        
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
        """Run final evaluation on all splits"""
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

import multiprocessing as mp
import queue
from threading import Thread

class ParallelCollector:
    """Parallel data collection for faster training"""
    def __init__(self, env_fn, model, device, num_workers=2):
        self.env_fn = env_fn
        self.model = model
        self.device = device
        self.num_workers = num_workers
        
        # Queues for communication
        self.experience_queue = mp.Queue(maxsize=1000)
        self.model_queue = mp.Queue(maxsize=10)
        
        # Start worker processes
        self.workers = []
        for i in range(num_workers):
            worker = mp.Process(target=self._worker_loop, args=(i,))
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, worker_id):
        """Worker process that collects experience"""
        env = self.env_fn()
        model = create_unified_model(self.model.config).to('cpu')
        
        while True:
            # Update model if available
            try:
                model_state = self.model_queue.get_nowait()
                model.load_state_dict(model_state)
            except queue.Empty:
                pass
            
            # Collect episode
            obs = env.reset()
            done = False
            trajectory = []
            
            while not done:
                with torch.no_grad():
                    torch_obs = prepare_observation(obs, torch.device('cpu'))
                    action, log_prob, _ = model.get_action(torch_obs)
                
                next_obs, reward, done, info = env.step(action.item())
                trajectory.append((obs, action, reward, next_obs, done, info))
                obs = next_obs
            
            self.experience_queue.put(trajectory)
    
    def update_model(self, model_state):
        """Send updated model to workers"""
        for _ in range(self.num_workers):
            try:
                self.model_queue.put(model_state, block=False)
            except queue.Full:
                pass
    
    def get_trajectories(self, timeout=0.1):
        trajectories = []
        try:
            while True:
                traj = self.experience_queue.get(timeout=timeout)
                trajectories.append(traj)
        except queue.Empty:
            pass
        return trajectories
    
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

    trainer = UnifiedAdvancedNoMaDTrainer(config)

    # Initialize wandb if enabled
    if config.get('use_wandb', False) and args.mode == 'train':
        wandb.init(
            project=config.get('wandb_project', 'nomad-rl-unified-advanced'),
            name=f"{args.dataset}_unified_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                **config,
                'dataset': args.dataset,
                'model_params': sum(p.numel() for p in trainer.model.parameters()),
                'device': str(trainer.device),
                'curriculum_levels': len(trainer.curriculum_manager.levels) if hasattr(trainer, 'curriculum_manager') else 0
            },
            tags=[args.dataset, 'unified', 'spatial_ode', 'counterfactual'] if config.get('use_counterfactuals') else [args.dataset, 'unified', 'spatial_ode']
        )
        
        # Define metrics after wandb.init()
        wandb.define_metric("training/timesteps")
        wandb.define_metric("episode/*", step_metric="training/timesteps")
        wandb.define_metric("losses/*", step_metric="training/timesteps")
        wandb.define_metric("curriculum/*", step_metric="training/timesteps")
        wandb.define_metric("val/*", step_metric="training/timesteps")
        
        # Log initial curriculum info
        if hasattr(trainer, 'curriculum_manager'):
            wandb.log({
                'curriculum/initial_level': trainer.curriculum_manager.current_level,
                'curriculum/total_levels': len(trainer.curriculum_manager.levels)
            })

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