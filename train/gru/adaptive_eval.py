import torch

class AdaptiveEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.base_state_dict = model.state_dict()
        
    def evaluate_with_adaptation(self, env, num_episodes=100):
        results = []
        
        for episode in range(num_episodes):
            # Reset to base parameters
            self.model.load_state_dict(self.base_state_dict)
            
            obs = env.reset()
            done = False
            episode_reward = 0
            adaptation_step = 0
            
            # Collect trajectory for adaptation
            recent_hidden = []
            recent_action_probs = []
            
            while not done:
                with torch.no_grad():
                    outputs = self.model(obs, mode="all")
                    action = outputs['action_dist'].sample()
                    
                    # Store for adaptation
                    recent_hidden.append(outputs['features'])
                    recent_action_probs.append(outputs['action_dist'].probs)
                
                # Adapt every k steps
                if len(recent_hidden) >= self.config.get('num_adaptation_steps', 6):
                    self._adapt_online(recent_hidden, recent_action_probs)
                    recent_hidden = []
                    recent_action_probs = []
                
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                obs = next_obs
            
            results.append({
                'reward': episode_reward,
                'success': info.get('success', False)
            })
        
        return results