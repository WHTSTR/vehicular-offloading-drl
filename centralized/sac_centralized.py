"""
Centralized Soft Actor-Critic (SAC) agent for vehicular offloading
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
from configs.config_sac import get_sac_config
from configs.common_presets import get_action_presets


class SACSquashedGaussianActor(nn.Module):
    """SAC actor with tanh squashing for bounded actions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256],
                 log_std_min: float = None, log_std_max: float = None):
        super(SACSquashedGaussianActor, self).__init__()
        
        self.log_std_min = log_std_min if log_std_min is not None else -20
        self.log_std_max = log_std_max if log_std_max is not None else 2
        
        # Build shared layers
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  # Added for consistency with DDPG
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_size, action_dim)
        self.log_std_head = nn.Linear(prev_size, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        # Mean head needs enough initial state sensitivity to separate V2V- and
        # V2I-favorable regimes early in training. Tiny output weights made SAC
        # start from an almost constant policy, unlike DDPG.
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, 0.0)

        # Keep the variance head tightly initialized so the policy does not
        # begin with excessive stochasticity on the [0, 1] action range.
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns mean and log_std in unbounded space"""
        x = self.shared_layers(state)
        
        # Unbounded mean and log_std
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action with tanh squashing to [0,1]"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from Gaussian
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action_tanh = torch.tanh(z)
        
        # Transform from [-1, 1] to [0, 1]
        action = 0.5 * (action_tanh + 1.0)
        
        # Compute log probability with the Jacobian correction
        log_prob = normal.log_prob(z)
        # Jacobian: |d/dz[0.5 * (tanh(z) + 1)]| = 0.5 * (1 - tanh²(z))
        log_prob -= torch.log(0.5 * (1 - action_tanh.pow(2) + 1e-6))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Also return the transformed mean for evaluation
        mean_action = 0.5 * (torch.tanh(mean) + 1.0)
        
        return action, log_prob, mean_action
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for deployment (no gradients needed)"""
        with torch.no_grad():
            if deterministic:
                mean, _ = self.forward(state)
                action = 0.5 * (torch.tanh(mean) + 1.0)
                return action
            else:
                action, _, _ = self.sample(state)
                return action


def initialize_sac_actor_for_mode(
    actor: SACSquashedGaussianActor,
    optimization_mode: str,
    config: dict | None = None,
) -> None:
    """Initialize SAC heads with a mode-consistent action prior.

    The vehicular actions are structured as [delta, p_v2v, p_v2i] triplets.
    Time-oriented runs benefit from starting near the high-power regime, while
    energy-oriented runs should start from lower transmit powers.
    """
    action_triplets = actor.mean_head.out_features // 3
    if action_triplets <= 0:
        return

    config = config or {}
    if optimization_mode == 'mixed':
        eta = float(np.clip(config.get('mixed_time_weight', 0.5), 0.0, 1.0))
        delta_bias_time = config.get('initial_delta_bias_time', 0.0)
        delta_bias_energy = config.get('initial_delta_bias_energy', 0.0)
        power_bias_time = config.get('initial_power_bias_time', 1.1)  # ~0.90
        power_bias_energy = config.get('initial_power_bias_energy', -1.1)  # ~0.10
        delta_bias = (1.0 - eta) * delta_bias_energy + eta * delta_bias_time
        power_bias = (1.0 - eta) * power_bias_energy + eta * power_bias_time
    elif optimization_mode == 'energy':
        delta_bias = config.get('initial_delta_bias_energy', 0.0)
        power_bias = config.get('initial_power_bias_energy', -1.1)  # ~0.10
    else:
        delta_bias = config.get('initial_delta_bias_time', 0.0)
        power_bias = config.get('initial_power_bias_time', 1.1)  # ~0.90
    log_std_bias = config.get('initial_log_std_bias', -2.0)

    mean_biases = []
    log_std_biases = []
    for _ in range(action_triplets):
        mean_biases.extend([delta_bias, power_bias, power_bias])
        log_std_biases.extend([log_std_bias, log_std_bias, log_std_bias])

    actor.mean_head.bias.data = torch.tensor(
        mean_biases, dtype=torch.float32, device=actor.mean_head.bias.device
    )
    actor.log_std_head.bias.data = torch.tensor(
        log_std_biases, dtype=torch.float32, device=actor.log_std_head.bias.device
    )


class SACCritic(nn.Module):
    """Twin Q-networks for SAC"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256]):
        super(SACCritic, self).__init__()
        
        # Q1 network
        self.q1_layers = []
        prev_size = state_dim + action_dim
        
        for hidden_size in hidden_sizes:
            self.q1_layers.append(nn.Linear(prev_size, hidden_size))
            self.q1_layers.append(nn.LayerNorm(hidden_size))  # Added for consistency
            self.q1_layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.q1_layers.append(nn.Linear(prev_size, 1))
        self.q1 = nn.Sequential(*self.q1_layers)
        
        # Q2 network (independent)
        self.q2_layers = []
        prev_size = state_dim + action_dim
        
        for hidden_size in hidden_sizes:
            self.q2_layers.append(nn.Linear(prev_size, hidden_size))
            self.q2_layers.append(nn.LayerNorm(hidden_size))  # Added for consistency
            self.q2_layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.q2_layers.append(nn.Linear(prev_size, 1))
        self.q2 = nn.Sequential(*self.q2_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for net in [self.q1, self.q2]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both Q-networks"""
        state_action = torch.cat([state, action], dim=1)
        q1_value = self.q1(state_action)
        q2_value = self.q2(state_action)
        return q1_value, q2_value
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q1 only (for actor loss)"""
        state_action = torch.cat([state, action], dim=1)
        return self.q1(state_action)


class CentralizedSACAgent:
    """Soft Actor-Critic agent with bounded action handling"""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu',
                 optimization_mode: str = 'time', mixed_time_weight: float | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.num_followers = action_dim // 3  # Each follower has 3 action components
        
        # Load configuration with mode
        config = get_sac_config(mode=optimization_mode)
        if mixed_time_weight is not None:
            config['mixed_time_weight'] = float(mixed_time_weight)
        self.config = config
        
        # Hyperparameters
        self.lr_actor = config['actor_lr']
        self.lr_critic = config['critic_lr']
        self.lr_alpha = config['alpha_lr']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.grad_clip = config.get('grad_clip', 1.0)
        self.actor_update_every = max(1, int(config.get('actor_update_every', 1)))
        self.actor_use_min_q = bool(config.get('actor_use_min_q', False))
        
        # Temperature parameter (alpha)
        self.automatic_entropy_tuning = config['automatic_entropy_tuning']
        if self.automatic_entropy_tuning:
            self.target_entropy = config['target_entropy']
            if self.target_entropy is None:
                self.target_entropy = -action_dim  # Default: -dim(A)
            initial_alpha = max(config.get('alpha', 0.1), 1e-6)
            self.log_alpha = torch.tensor(
                [np.log(initial_alpha)],
                requires_grad=True,
                device=self.device,
                dtype=torch.float32,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        else:
            self.alpha = config['alpha']
        
        # Networks
        self.actor = SACSquashedGaussianActor(
            state_dim, action_dim, config['actor_hidden_sizes'],
            log_std_min=config.get('log_std_min', -20),
            log_std_max=config.get('log_std_max', 2)
        ).to(self.device)
        initialize_sac_actor_for_mode(self.actor, optimization_mode, config)
        self.critic = SACCritic(state_dim, action_dim, config['critic_hidden_sizes']).to(self.device)
        self.critic_target = SACCritic(state_dim, action_dim, config['critic_hidden_sizes']).to(self.device)
        
        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Replay buffer
        self.memory = ReplayBuffer(config['buffer_size'], state_dim, action_dim, device)
        
        # Counters
        self.step_count = 0
        
        # Continuous SAC should rely on its stochastic policy for exploration.
        self.use_preset_exploration = config.get('use_preset_exploration', False)
        self.action_presets = get_action_presets() if self.use_preset_exploration else []
        
        # Training counters
        self.update_every = config.get('update_every', 1)
        
        # Exploration parameters
        self.epsilon = config.get('epsilon_start', 0.3)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        self.epsilon_decay = config['epsilon_decay']
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action using the SAC policy"""
        
        # Epsilon-greedy with presets for exploration
        if self.use_preset_exploration and not evaluate and np.random.random() < self.epsilon:
            # Use preset action with individual variations
            preset_idx = np.random.randint(len(self.action_presets))
            preset = self.action_presets[preset_idx]
            action = []
            # Add individual noise to prevent distribution mismatch
            preset_noise_std = self.config.get('preset_noise_std', 0.05)
            for i in range(self.num_followers):
                vehicle_action = preset + np.random.normal(0, preset_noise_std, 3)
                vehicle_action = np.clip(vehicle_action, 0.0, 1.0)
                action.extend(vehicle_action)
            return np.array(action)
        
        # Use actor network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action from policy
        action = self.actor.get_action(state_tensor, deterministic=evaluate)
        action = action.cpu().numpy()[0]
        
        # The action is already in [0, 1] from the network
        # Clip to the valid range as a safeguard
        action = np.clip(action, 0.0, 1.0)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1
        
        # No epsilon decay needed for SAC
        
    def update_exploration(self):
        """Update exploration parameters - SAC uses entropy instead of epsilon"""
        if self.use_preset_exploration:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, num_updates: int = None) -> Tuple[float, float, float, float]:
        """Train SAC networks"""
        if len(self.memory) < self.config['min_buffer_size']:
            return 0.0, 0.0, 0.0, 0.0
        
        # Use config value if not specified
        if num_updates is None:
            num_updates = self.config.get('num_updates', 1)
            
        actor_losses, critic_losses, alpha_losses, alpha_values = [], [], [], []
        
        for update_idx in range(num_updates):
            # Sample batch
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # Clip rewards (for fairness with DDPG)
            clip_min = self.config.get('reward_clip_min', -25)
            clip_max = self.config.get('reward_clip_max', 0)
            rewards = torch.clamp(rewards, clip_min, clip_max)
            
            # Current alpha value
            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha
        
            # Update Critic
            with torch.no_grad():
                # Sample next actions and their log probs
                next_actions, next_log_probs, _ = self.actor.sample(next_states)
                
                # Compute target Q-values
                target_q1, target_q2 = self.critic_target(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2)
                target_value = target_q - alpha * next_log_probs
                target_q_value = rewards + (1 - dones) * self.gamma * target_value
            
            # Current Q-values
            current_q1, current_q2 = self.critic(states, actions)
        
            # Critic loss
            critic_loss = F.mse_loss(current_q1, target_q_value) + F.mse_loss(current_q2, target_q_value)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()
        
            if (update_idx % self.actor_update_every) == 0:
                # Critic-heavy updates can help SAC when the actor is chasing
                # a critic that still underestimates the boundary-optimal
                # regime.
                new_actions, log_probs, _ = self.actor.sample(states)
                if self.actor_use_min_q:
                    q1_new, q2_new = self.critic(states, new_actions)
                    q_new_actions = torch.min(q1_new, q2_new)
                else:
                    q_new_actions = self.critic.q1_forward(states, new_actions)
                actor_loss = (alpha * log_probs - q_new_actions).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.actor_optimizer.step()

                # Update temperature (alpha) if automatic tuning is enabled
                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                else:
                    alpha_loss = torch.tensor(0.0)

                actor_losses.append(actor_loss.item())
                alpha_losses.append(alpha_loss.item())
                alpha_values.append(float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha))
            
            # Soft update target network
            self._soft_update(self.critic, self.critic_target)
            
            # Collect losses
            critic_losses.append(critic_loss.item())
        
        # Return average losses
        return (
            float(np.mean(actor_losses)) if actor_losses else 0.0,
            float(np.mean(critic_losses)) if critic_losses else 0.0,
            float(np.mean(alpha_losses)) if alpha_losses else 0.0,
            float(np.mean(alpha_values)) if alpha_values else (
                float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
            ),
        )
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'automatic_entropy_tuning': self.automatic_entropy_tuning,
        }
        if self.automatic_entropy_tuning:
            payload['alpha_optimizer'] = self.alpha_optimizer.state_dict()
            payload['log_alpha'] = self.log_alpha
        torch.save(payload, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.automatic_entropy_tuning and 'alpha_optimizer' in checkpoint and 'log_alpha' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.log_alpha = checkpoint['log_alpha']
        self.step_count = checkpoint.get('step_count', checkpoint.get('total_steps', 0))
        self.epsilon = checkpoint.get('epsilon', 0.1)
