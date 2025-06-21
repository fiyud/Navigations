# NoMaD-RL: Deep Reinforcement Learning for Navigation in AI2Thor

This guide shows how to replace the diffusion policy in NoMaD with deep reinforcement learning using AI2Thor environments.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch torchvision
pip install ai2thor
pip install gym
pip install wandb
pip install matplotlib
pip install opencv-python
pip install efficientnet-pytorch
pip install pyyaml
pip install numpy

# Clone and install original NoMaD components
git clone https://github.com/robodhruv/visualnav-transformer.git
cd visualnav-transformer/train
pip install -e .
```

### Training

```bash
# Train NoMaD-RL model
python nomad_rl_trainer.py --config config/nomad_rl.yaml
```

### Evaluation

```bash
# Evaluate trained model
python nomad_rl_eval.py \
    --config config/nomad_rl.yaml \
    --checkpoint checkpoints/nomad_rl/nomad_rl_1000.pth \
    --episodes 20 \
    --plot

# Evaluate exploration capability
python nomad_rl_eval.py \
    --config config/nomad_rl.yaml \
    --checkpoint checkpoints/nomad_rl/nomad_rl_1000.pth \
    --episodes 20 \
    --exploration
```

## 🏗️ Architecture Overview

### Key Changes from Original NoMaD

1. **Environment**: Replaced real robot data with AI2Thor simulation
2. **Policy**: Replaced diffusion policy with PPO (Proximal Policy Optimization)
3. **Training**: Changed from supervised learning to reinforcement learning
4. **Preserved**: Vision encoder, goal masking, distance prediction

### Model Components

- **Vision Encoder**: EfficientNet-B0 + Transformer (from original NoMaD)
- **Goal Masking**: Binary mask for goal-conditioned vs exploration modes
- **Policy Network**: PPO policy for action selection
- **Value Network**: State value estimation for PPO
- **Distance Predictor**: Temporal distance estimation (preserved from NoMaD)

## 🎯 Key Features

### Unified Navigation + Exploration
- **Goal-Conditioned Mode**: Navigate to visual goals
- **Exploration Mode**: Discover environments without specific goals
- **Seamless Switching**: Single model handles both behaviors

### AI2Thor Integration
- **Realistic Environments**: Kitchen, living room, bedroom, bathroom scenes
- **Object Interaction**: Navigate around furniture and appliances
- **Physics Simulation**: Realistic collision detection and movement

### Advanced RL Training
- **PPO Algorithm**: Stable policy gradient method
- **GAE**: Generalized Advantage Estimation for better value learning
- **Curriculum Learning**: Automatic difficulty progression

## 📊 Training Configuration

### Environment Settings
```yaml
scene_names: [FloorPlan1, FloorPlan2, ...]  # AI2Thor scenes
image_size: [96, 96]                         # Input resolution
max_episode_steps: 500                       # Episode length
goal_prob: 0.5                              # Goal-conditioned probability
```

### Model Architecture
```yaml
encoding_size: 256                  # Feature dimension
context_size: 5                     # Temporal context
mha_num_attention_heads: 4         # Attention heads
hidden_dim: 512                    # Policy network size
```

### PPO Hyperparameters
```yaml
learning_rate: 3e-4
gamma: 0.99                        # Discount factor
lam: 0.95                         # GAE lambda
clip_ratio: 0.2                   # PPO clipping
entropy_coef: 0.01                # Exploration bonus
```

## 🎮 Action Space

The agent can perform 6 discrete actions in AI2Thor:
- `0`: Move Forward
- `1`: Move Backward  
- `2`: Turn Left
- `3`: Turn Right
- `4`: Look Up
- `5`: Look Down

## 🏆 Reward Design

### Goal-Conditioned Episodes
- **Success Reward**: +100 for reaching goal
- **Distance Reward**: Dense reward for getting closer to goal
- **Step Penalty**: -0.01 per step (encourages efficiency)
- **Collision Penalty**: -5.0 for hitting obstacles

### Exploration Episodes
- **Coverage Reward**: +5.0 for visiting new areas
- **Movement Reward**: +0.1 for successful forward movement
- **Step Penalty**: -0.01 per step
- **Collision Penalty**: -5.0 for hitting obstacles

## 📈 Performance Monitoring

### Training Metrics
- Episode reward (mean ± std)
- Success rate
- Episode length
- Policy loss, value loss, entropy
- Exploration coverage

### Evaluation Metrics
- Navigation success rate
- Average distance to goal
- Collision rate
- Exploration coverage
- Sample efficiency

## 🔧 Advanced Usage

### Using Pre-trained Vision Encoder

```yaml
# In config file
pretrained_vision_encoder: "path/to/nomad_checkpoint.pth"
freeze_vision_encoder: true  # Optional: freeze during RL training
```

### Multi-Scene Training

```yaml
scene_names:
  - FloorPlan1    # Kitchen
  - FloorPlan201  # Living room  
  - FloorPlan301  # Bedroom
  - FloorPlan401  # Bathroom
```

### Custom Reward Shaping

```python
def _calculate_reward(self, event, obs):
    reward = 0.0
    
    # Custom reward logic here
    if self.is_goal_conditioned:
        # Goal-conditioned rewards
        distance = self._distance_to_goal()
        reward -= distance * 0.1  # Distance penalty
    else:
        # Exploration rewards
        # Add custom exploration incentives
    
    return reward
```

## 🐛 Troubleshooting

### Common Issues

1. **AI2Thor Installation**
   ```bash
   # If AI2Thor fails to install
   pip install ai2thor --upgrade
   # Or use conda
   conda install -c conda-forge ai2thor
   ```

2. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Reduce `rollout_steps`
   - Use smaller `image_size`

3. **Training Instability**
   - Lower `learning_rate`
   - Increase `rollout_steps`
   - Adjust `clip_ratio`

4. **Poor Exploration**
   - Increase `entropy_coef`
   - Adjust exploration rewards
   - Use curiosity-driven rewards

### Performance Tips

- Use GPU for faster training
- Start with fewer scenes and gradually add more
- Pre-train vision encoder on real robot data
- Use curriculum learning for complex scenes

## 📚 References

- [Original NoMaD Paper](https://arxiv.org/abs/2310.07896)
- [AI2Thor Documentation](https://ai2thor.allenai.org/)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [ViNT Foundation Model](https://general-navigation-models.github.io/vint/)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional AI2Thor task types
- Multi-agent navigation
- Hierarchical RL integration
- Real-world transfer learning
- Advanced exploration strategies

## 📄 License

This project builds upon the original NoMaD codebase. Please refer to the original repository for licensing information.