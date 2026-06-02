"""
Centralized DDPG agent for vehicular offloading
A single agent maps the full system state to continuous actions for all followers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from typing import List, Tuple

from utils.replay_buffer import ReplayBuffer
from configs.common_presets import get_action_presets


class RunningStats:
    """Track running statistics for normalization"""
    def __init__(self, shape=()):
        self.n = 0
        self.mean = np.zeros(shape)
        self.variance = np.ones(shape)
        
    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.variance = np.zeros_like(x)
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.variance = self.variance + (x - old_mean) * (x - self.mean)
            
    def get_stats(self):
        if self.n > 1:
            std = np.sqrt(self.variance / (self.n - 1))
        else:
            std = np.ones_like(self.mean)
        return self.mean, std


class StableActor(nn.Module):
    """Actor network with stable initialization"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256], use_layer_norm: bool = True):
        super(StableActor, self).__init__()
        self.use_layer_norm = use_layer_norm
        
        # Build layers
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, action_dim)
        
        # Initialize weights for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stability"""
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        # Keep the default output bias neutral here; the DDPG agents override it
        # with a mode-aware action prior after construction.
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.hidden_layers(state)
        # Sigmoid for [0, 1] range
        actions = torch.sigmoid(self.output_layer(x))
        return actions


def _logit_clamped(prob: float, eps: float = 1e-4) -> float:
    """Map an action-space prior in (0, 1) to a numerically stable sigmoid bias."""
    p = float(np.clip(prob, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def initialize_ddpg_actor_for_mode(
    actor: StableActor,
    optimization_mode: str,
    config: dict | None = None,
) -> None:
    """Initialize DDPG actor heads with a mode-consistent action prior."""
    config = config or {}
    action_triplets = actor.output_layer.out_features // 3
    if action_triplets <= 0:
        return

    if optimization_mode == 'mixed':
        eta = float(np.clip(config.get('mixed_time_weight', 0.5), 0.0, 1.0))
        delta_action = (
            (1.0 - eta) * float(config.get('initial_delta_action_energy', 0.5))
            + eta * float(config.get('initial_delta_action_time', 0.5))
        )
        power_action = (
            (1.0 - eta) * float(config.get('initial_power_action_energy', 0.1))
            + eta * float(config.get('initial_power_action_time', 0.9))
        )
    elif optimization_mode == 'energy':
        delta_action = float(config.get('initial_delta_action_energy', 0.5))
        power_action = float(config.get('initial_power_action_energy', 0.1))
    else:
        delta_action = float(config.get('initial_delta_action_time', 0.5))
        power_action = float(config.get('initial_power_action_time', 0.9))

    delta_bias = _logit_clamped(delta_action)
    power_bias = _logit_clamped(power_action)

    bias_values = []
    for _ in range(action_triplets):
        bias_values.extend([delta_bias, power_bias, power_bias])

    actor.output_layer.bias.data = torch.tensor(
        bias_values,
        dtype=torch.float32,
        device=actor.output_layer.bias.device,
    )


class StableCritic(nn.Module):
    """Critic network with normalization and regularization"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256]):
        super(StableCritic, self).__init__()
        
        # First layer processes state
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.ln1 = nn.LayerNorm(hidden_sizes[0])
        
        # Second layer processes state representation + actions
        self.fc2 = nn.Linear(hidden_sizes[0] + action_dim, hidden_sizes[1])
        self.ln2 = nn.LayerNorm(hidden_sizes[1])
        
        # Output layer
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        
        # Output layer - initialize near zero
        nn.init.uniform_(self.fc3.weight, -1e-3, 1e-3)
        nn.init.constant_(self.fc3.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Process state
        x = F.relu(self.ln1(self.fc1(state)))
        
        # Concatenate with actions
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.ln2(self.fc2(x)))
        
        # Output Q-value
        q_value = self.fc3(x)
        
        return q_value


class DecorrelatedOUNoise:
    """Ornstein-Uhlenbeck noise with decorrelation for multiple agents"""
    
    def __init__(self, num_vehicles: int, action_dim_per_vehicle: int, 
                 mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2):
        self.num_vehicles = num_vehicles
        self.action_dim_per_vehicle = action_dim_per_vehicle
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        """Reset the internal state"""
        # Separate state for each vehicle
        self.states = [np.ones(self.action_dim_per_vehicle) * self.mu 
                      for _ in range(self.num_vehicles)]
    
    def sample(self) -> np.ndarray:
        """Sample decorrelated noise for all vehicles"""
        noise = []
        for i in range(self.num_vehicles):
            x = self.states[i]
            dx = self.theta * (self.mu - x) * self.dt + \
                 self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim_per_vehicle)
            self.states[i] = x + dx
            noise.extend(self.states[i])
        return np.array(noise)


class CentralizedDDPGAgent:
    """DDPG agent for vehicular offloading"""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu',
                 optimization_mode: str = 'time',
                 mixed_time_weight: float | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Load configuration with mode
        from configs.config_ddpg import get_ddpg_config
        config = get_ddpg_config(mode=optimization_mode)
        if mixed_time_weight is not None:
            config['mixed_time_weight'] = float(mixed_time_weight)
        self.config = config
        
        # Hyperparameters
        self.lr_actor = config['actor_lr']
        self.lr_critic = config['critic_lr']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.gradient_clip = config.get('grad_clip', 1.0)
        
        # Update frequencies
        self.update_every = config['update_every']
        self.num_updates = config['num_updates']
        
        # Networks
        self.actor = StableActor(state_dim, action_dim, config['actor_hidden_sizes'], config.get('use_layer_norm', True)).to(self.device)
        initialize_ddpg_actor_for_mode(self.actor, optimization_mode, config)
        self.actor_target = StableActor(state_dim, action_dim, config['actor_hidden_sizes'], config.get('use_layer_norm', True)).to(self.device)
        self.critic = StableCritic(state_dim, action_dim, config['critic_hidden_sizes']).to(self.device)
        self.critic_target = StableCritic(state_dim, action_dim, config['critic_hidden_sizes']).to(self.device)
        
        # Copy weights to targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Replay buffer
        self.memory = ReplayBuffer(config['buffer_size'], state_dim, action_dim, device)
        
        # Q-value normalization
        self.q_stats = RunningStats()
        
        # Exploration
        self.num_followers = action_dim // 3
        self.noise_type = config.get('noise_type', 'ou')
        
        if self.noise_type == 'ou':
            self.noise = DecorrelatedOUNoise(
                self.num_followers, 3,
                sigma=config['ou_sigma'],
                theta=config['ou_theta']
            )
        elif self.noise_type == 'gaussian':
            # Gaussian noise doesn't need initialization
            self.noise_std = config.get('ou_sigma', 0.1)  # Use ou_sigma for consistency
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
            
        self.noise_scale = config['noise_scale']
        self.noise_decay = config['noise_decay']
        self.min_noise = config['min_noise']
        
        # Epsilon for discrete exploration fallback
        self.use_preset_exploration = config.get('use_preset_exploration', False)
        self.epsilon = config.get('epsilon_start', 0.3)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.05)
        
        # Action presets - use common presets for fair comparison
        self.action_presets = get_action_presets() if self.use_preset_exploration else []
        
        # Training counters
        self.step_count = 0
    
    def reset_noise(self):
        """Reset noise generators"""
        if self.noise_type == 'ou':
            self.noise.reset()
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action with stable exploration"""
        
        # Epsilon-greedy with presets for stability
        if self.use_preset_exploration and add_noise and np.random.random() < self.epsilon:
            # Use preset action with individual variations to match continuous distribution
            preset_idx = np.random.randint(len(self.action_presets))
            preset = self.action_presets[preset_idx]
            action = []
            # Add individual noise to each vehicle to prevent distribution mismatch
            preset_noise_std = self.config.get('preset_noise_std', 0.05)
            for i in range(self.num_followers):
                # Each vehicle gets preset + individual noise
                vehicle_action = preset + np.random.normal(0, preset_noise_std, 3)
                vehicle_action = np.clip(vehicle_action, 0.0, 1.0)
                action.extend(vehicle_action)
            return np.array(action)
        
        # Use actor network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        # Add noise based on type
        if add_noise:
            if self.noise_type == 'ou':
                noise = self.noise.sample() * self.noise_scale
            elif self.noise_type == 'gaussian':
                # Gaussian noise like TD3
                noise = np.random.normal(0, self.noise_std * self.noise_scale, size=action.shape)
            else:
                noise = 0
                
            action = action + noise
            action = np.clip(action, 0.0, 1.0)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1
        
    def update_exploration(self):
        """Update exploration parameters - call after each episode"""
        if self.use_preset_exploration:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
    
    def train(self, num_updates: int = None) -> Tuple[float, float]:
        """Train networks"""
        if len(self.memory) < self.config['min_buffer_size']:
            return 0.0, 0.0
        
        # Use config value if not specified
        if num_updates is None:
            num_updates = self.config.get('num_updates', 1)
            
        actor_losses = []
        critic_losses = []
        
        for _ in range(num_updates):
            # Sample batch
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # Clip rewards (for fairness with other algorithms)
            clip_min = self.config.get('reward_clip_min', -25)
            clip_max = self.config.get('reward_clip_max', 0)
            rewards = torch.clamp(rewards, clip_min, clip_max)
            
            # Update Critic
            with torch.no_grad():
                target_actions = self.actor_target(next_states)
                target_q_values = self.critic_target(next_states, target_actions)
                
                # Update Q-value statistics (keeping for monitoring only)
                self.q_stats.update(target_q_values.mean().cpu().numpy())
                
                # Don't normalize Q-values - it can cause training instability
                
                target_values = rewards + (1 - dones) * self.gamma * target_q_values
            
            current_q_values = self.critic(states, actions)
            
            # Critic loss
            critic_loss = F.mse_loss(current_q_values, target_values)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
            self.critic_optimizer.step()
            
            critic_losses.append(critic_loss.item())
            
            # Update Actor
            predicted_actions = self.actor(states)
            actor_loss = -self.critic(states, predicted_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
            self.actor_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        return np.mean(actor_losses), np.mean(critic_losses)
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'noise_scale': self.noise_scale,
            'q_stats_mean': self.q_stats.mean,
            'q_stats_variance': self.q_stats.variance,
            'q_stats_n': self.q_stats.n
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.step_count = checkpoint.get('step_count', checkpoint.get('total_steps', 0))
        self.epsilon = checkpoint.get('epsilon', 0.1)
        self.noise_scale = checkpoint.get('noise_scale', 0.1)
        
        # Restore Q-stats
        if 'q_stats_mean' in checkpoint:
            self.q_stats.mean = checkpoint['q_stats_mean']
            self.q_stats.variance = checkpoint['q_stats_variance']
            self.q_stats.n = checkpoint['q_stats_n']
