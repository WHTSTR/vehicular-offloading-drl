"""
Centralized DQN agent for vehicular offloading
A single agent maps the full system state to a discrete action preset per follower.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from typing import List, Tuple

from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from configs.common_presets import get_action_presets


class CentralizedQNetwork(nn.Module):
    """Q-network for centralized control with independent preset selection per follower"""
    
    def __init__(self, state_size: int, num_actions: int, num_followers: int,
                 hidden_sizes: List[int], use_layer_norm: bool = True):
        super().__init__()
        
        self.num_actions = num_actions
        self.num_followers = num_followers
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        # Output Q-values for each follower's preset selection
        self.q_head = nn.Linear(prev_size, num_followers * num_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state):
        """Forward pass - returns Q-values shaped as (batch, num_followers, num_actions)"""
        features = self.feature_extractor(state)
        q_values = self.q_head(features)
        # Reshape to (batch_size, num_followers, num_actions)
        batch_size = state.shape[0]
        q_values = q_values.view(batch_size, self.num_followers, self.num_actions)
        return q_values


class CentralizedDQNAgent:
    """Centralized DQN agent with independent preset selection per follower"""
    
    def __init__(self, state_size: int, num_followers: int, device: str = 'cpu', 
                 optimization_mode: str = 'time'):
        self.device = device
        self.state_size = state_size
        self.num_followers = num_followers
        
        # Load configuration with mode
        from configs.config_dqn import get_dqn_config
        config = get_dqn_config(mode=optimization_mode)
        self.config = config
        
        # Action presets - use common presets for fair comparison
        self.action_presets = np.array(get_action_presets())
        self.num_actions = len(self.action_presets)
        
        # Networks with independent Q-values per follower
        self.q_network = CentralizedQNetwork(
            state_size, 
            self.num_actions,
            num_followers,
            config['hidden_sizes'],
            config['use_layer_norm']
        ).to(device)
        
        self.target_network = CentralizedQNetwork(
            state_size,
            self.num_actions,
            num_followers,
            config['hidden_sizes'],
            config['use_layer_norm']
        ).to(device)
        
        # Copy weights to target
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        
        # Replay buffer - store preset indices
        buffer_cls = PrioritizedReplayBuffer if config.get('prioritized_replay', False) else ReplayBuffer
        if buffer_cls is PrioritizedReplayBuffer:
            self.memory = buffer_cls(
                config['buffer_size'],
                state_size,
                num_followers,
                alpha=config.get('per_alpha', 0.6),
                device=device,
            )
        else:
            self.memory = buffer_cls(
                config['buffer_size'],
                state_size,
                num_followers,
                device
            )
        
        # Training parameters
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.update_every = config['update_every']
        self.target_update_every = config['target_update_every']
        self.num_updates = config.get('num_updates', 1)
        
        # Exploration parameters
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_schedule = config.get('epsilon_schedule', 'multiplicative')
        self.epsilon_linear_decay_episodes = config.get('epsilon_linear_decay_episodes', 250)
        self.post_warmup_episode_count = 0
        self.step_count = 0
        
        # Reward scaling
        self.double_dqn = bool(config['double_dqn'])
        self.loss_type = config.get('loss_type', 'huber')
        self.prioritized_replay = config.get('prioritized_replay', False)
        self.per_beta = config.get('per_beta', 0.4)
        self.per_priority_eps = config.get('per_priority_eps', 1e-6)
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select action allowing different presets for each follower
        Returns: (continuous_actions, preset_indices)
        """
        preset_indices = np.zeros(self.num_followers, dtype=np.int32)
        
        if explore and np.random.random() < self.epsilon:
            # Random preset selection for each follower independently
            for i in range(self.num_followers):
                preset_indices[i] = np.random.randint(self.num_actions)
        else:
            # Greedy action selection
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)  # Shape: (1, num_followers, num_actions)
                # Select best action for each follower
                preset_indices = q_values.argmax(dim=2).cpu().numpy()[0]  # Shape: (num_followers,)
            self.q_network.train()
        
        # Convert preset indices to continuous actions
        continuous_actions = []
        for i in range(self.num_followers):
            preset = self.action_presets[preset_indices[i]]
            continuous_actions.extend(preset)
        
        return np.array(continuous_actions), preset_indices
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition - action should be preset indices, not continuous values"""
        # Extract preset indices if continuous action is provided
        if action.shape[0] == self.num_followers * 3:
            # Need to convert back to preset indices
            preset_indices = self._continuous_to_preset_indices(action)
        else:
            preset_indices = action
            
        self.memory.add(state, preset_indices, reward, next_state, done)
        self.step_count += 1
    
    def _continuous_to_preset_indices(self, continuous_action: np.ndarray) -> np.ndarray:
        """Convert continuous action back to preset indices (approximate)"""
        preset_indices = np.zeros(self.num_followers, dtype=np.int32)
        
        for i in range(self.num_followers):
            follower_action = continuous_action[i*3:(i+1)*3]
            # Find closest preset
            distances = np.sum((self.action_presets - follower_action)**2, axis=1)
            preset_indices[i] = np.argmin(distances)
            
        return preset_indices
    
    def train(self, num_updates: int = None) -> Tuple[float, float]:
        """Train the Q-network with independent Q-values per follower"""
        if len(self.memory) < self.config['min_buffer_size']:
            return 0.0, 0.0
            
        if num_updates is None:
            num_updates = self.num_updates
            
        total_loss = 0.0
        max_q = 0.0
        
        for _ in range(num_updates):
            # Sample batch
            if self.prioritized_replay and isinstance(self.memory, PrioritizedReplayBuffer):
                states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(
                    self.batch_size, beta=self.per_beta
                )
            else:
                states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
                weights = None
                indices = None
            
            # Clip rewards (for fairness with DDPG)
            clip_min = self.config.get('reward_clip_min', -25)
            clip_max = self.config.get('reward_clip_max', 0)
            rewards = torch.clamp(rewards, clip_min, clip_max)
            
            # Current Q-values for selected actions
            current_q_values = self.q_network(states)  # Shape: (batch, num_followers, num_actions)
            
            # Gather Q-values for selected actions
            actions_long = actions.long()  # Convert to long for indexing
            current_q_selected = current_q_values.gather(2, actions_long.unsqueeze(2)).squeeze(2)
            
            # Target Q-values
            with torch.no_grad():
                next_q_values_target = self.target_network(next_states)
                if self.double_dqn:
                    next_q_values_online = self.q_network(next_states)
                    next_actions = next_q_values_online.argmax(dim=2, keepdim=True)
                    next_q_max = next_q_values_target.gather(2, next_actions).squeeze(2)
                else:
                    next_q_max = next_q_values_target.max(dim=2)[0]
                # Compute a Bellman target for each follower independently.
                targets = rewards + (1 - dones) * self.gamma * next_q_max
            
            # Compute loss
            td_error = current_q_selected - targets
            if self.loss_type == 'huber':
                per_sample_loss = F.smooth_l1_loss(
                    current_q_selected, targets, reduction='none'
                ).mean(dim=1, keepdim=True)
            else:
                per_sample_loss = F.mse_loss(
                    current_q_selected, targets, reduction='none'
                ).mean(dim=1, keepdim=True)

            if weights is not None:
                loss = (per_sample_loss * weights).mean()
            else:
                loss = per_sample_loss.mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

            if indices is not None:
                priorities = td_error.detach().abs().mean(dim=1).cpu().numpy() + self.per_priority_eps
                self.memory.update_priorities(indices, priorities)
            
            total_loss += loss.item()
            max_q += current_q_values.max().item()
        
        return total_loss / num_updates, max_q / num_updates
    
    def update_target_network(self):
        """Hard update of target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """Update exploration rate"""
        if self.epsilon_schedule == 'linear':
            self.post_warmup_episode_count += 1
            progress = min(
                1.0,
                self.post_warmup_episode_count / max(1, self.epsilon_linear_decay_episodes),
            )
            self.epsilon = self.config['epsilon_start'] + (
                self.epsilon_end - self.config['epsilon_start']
            ) * progress
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(), 
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
                'q_network_config': {
                    'state_size': self.state_size,
                    'num_actions': self.num_actions,
                    'num_followers': self.num_followers,
                    'hidden_sizes': self.config['hidden_sizes'],
                    'use_layer_norm': self.config['use_layer_norm']
                }
            }, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Check if old format (single preset for all followers)
        if 'q_network_config' not in checkpoint:
            print("Warning: Loading old DQN checkpoint format. Creating new network architecture.")
            # Don't load the incompatible weights
            return
            
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint.get('step_count', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
