"""
Decentralized DQN agent for vehicular offloading
Each follower independently selects a discrete action preset from its own local observation, using a single shared policy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple
import os

from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from configs.config_dqn import get_dqn_config
from configs.common_presets import get_action_presets


class DecentralizedQNetwork(nn.Module):
    """Q-network for individual agent in decentralized setting"""
    
    def __init__(self, state_size: int, num_actions: int, hidden_sizes: List[int], 
                 use_layer_norm: bool = True):
        super().__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev_size, num_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state):
        """Forward pass"""
        features = self.feature_extractor(state)
        q_values = self.q_head(features)
        return q_values


class DecentralizedDQNAgent:
    """Single DQN agent for one follower in decentralized setting"""
    
    def __init__(self, state_size: int, agent_id: int, device: str = 'cpu',
                 optimization_mode: str = 'time'):
        self.device = device
        self.state_size = state_size
        self.agent_id = agent_id
        
        # Load configuration with mode
        config = get_dqn_config(mode=optimization_mode)
        self.config = config
        
        # Action presets - use common presets for fair comparison
        self.action_presets = np.array(get_action_presets())
        self.num_actions = len(self.action_presets)
        
        # Networks
        self.q_network = DecentralizedQNetwork(
            state_size, 
            self.num_actions,
            config['hidden_sizes'],
            config['use_layer_norm']
        ).to(device)
        
        self.target_network = DecentralizedQNetwork(
            state_size,
            self.num_actions,
            config['hidden_sizes'],
            config['use_layer_norm']
        ).to(device)
        
        # Copy weights to target
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        
        # Training parameters
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.update_every = config['update_every']
        self.target_update_every = config['target_update_every']
        self.grad_clip = config.get('grad_clip', 1.0)
        self.double_dqn = bool(config['double_dqn'])
        self.loss_type = config.get('loss_type', 'huber')
        self.prioritized_replay = config.get('prioritized_replay', False)
        self.per_beta = config.get('per_beta', 0.4)
        self.per_priority_eps = config.get('per_priority_eps', 1e-6)
        
        # Exploration (individual epsilon for each agent)
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_schedule = config.get('epsilon_schedule', 'multiplicative')
        self.epsilon_linear_decay_episodes = config.get('epsilon_linear_decay_episodes', 250)
        self.post_warmup_episode_count = 0
        
        # Counters
        self.step_count = 0
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[np.ndarray, int]:
        """
        Select action using epsilon-greedy policy
        Returns action vector and preset index
        """
        if explore and np.random.random() < self.epsilon:
            # Random preset
            preset_idx = np.random.randint(self.num_actions)
        else:
            # Greedy action selection
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            self.q_network.train()
            preset_idx = q_values.argmax(dim=1).item()
        
        # Convert preset index to action vector
        action = self.action_presets[preset_idx].copy()
        
        return action, preset_idx
    
    def update_epsilon(self):
        """Decay epsilon after episode"""
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
        self.episode_count += 1
    
    def train(self, memory: ReplayBuffer, num_updates: int = 1):
        """Train the Q-network using shared follower replay."""
        if len(memory) < self.config['min_buffer_size']:
            return
        
        for _ in range(num_updates):
            if self.prioritized_replay and isinstance(memory, PrioritizedReplayBuffer):
                states, actions, rewards, next_states, dones, weights, indices = memory.sample(
                    self.batch_size, beta=self.per_beta
                )
            else:
                states, actions, rewards, next_states, dones = memory.sample(self.batch_size)
                weights = None
                indices = None
            
            # Find preset indices from continuous actions
            preset_indices = self._actions_to_preset_indices(actions.cpu().numpy())
            preset_indices = torch.LongTensor(preset_indices).to(self.device)
            
            # Compute current Q-values
            current_q_values = self.q_network(states).gather(1, preset_indices.unsqueeze(1))
            
            # Clip rewards (for fairness with DDPG)
            clip_min = self.config.get('reward_clip_min', -25)
            clip_max = self.config.get('reward_clip_max', 0)
            rewards = torch.clamp(rewards, clip_min, clip_max)
            
            # Compute target Q-values
            with torch.no_grad():
                if self.double_dqn:
                    next_online_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                    next_q_values = self.target_network(next_states).gather(1, next_online_actions)
                else:
                    next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
            # Compute loss
            td_error = current_q_values - target_q_values
            if self.loss_type == 'huber':
                per_sample_loss = F.smooth_l1_loss(
                    current_q_values, target_q_values, reduction='none'
                )
            else:
                per_sample_loss = F.mse_loss(
                    current_q_values, target_q_values, reduction='none'
                )

            if weights is not None:
                loss = (per_sample_loss * weights).mean()
            else:
                loss = per_sample_loss.mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
            self.optimizer.step()

            if indices is not None:
                priorities = td_error.detach().abs().cpu().numpy().reshape(-1) + self.per_priority_eps
                memory.update_priorities(indices, priorities)
    
    def _actions_to_preset_indices(self, actions: np.ndarray) -> List[int]:
        """Convert continuous actions to preset indices"""
        indices = []
        for action in actions:
            # Find closest preset
            distances = np.sum((self.action_presets - action) ** 2, axis=1)
            preset_idx = np.argmin(distances)
            indices.append(preset_idx)
        return indices
    
    def update_target_network(self):
        """Update target network - explicit call for consistency"""
        self._update_target_network()
    
    def _update_target_network(self):
        """Hard update of target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str):
        """Save agent state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'agent_id': self.agent_id,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']


class MultiAgentDQN:
    """
    Shared-policy decentralized DQN manager.

    One follower Q-policy is applied to all followers, while each follower still
    acts on its own local observation.
    """
    
    def __init__(self, local_state_size: int, num_followers: int, device: str = 'cpu',
                 optimization_mode: str = 'time'):
        self.num_followers = num_followers
        self.device = device

        # One follower policy shared across all follower vehicles.
        self.shared_agent = DecentralizedDQNAgent(
            local_state_size, 0, device, optimization_mode
        )

        # Shared replay buffer over all follower transitions.
        config = get_dqn_config(mode=optimization_mode)
        self.config = config  # Store config for reuse
        buffer_cls = PrioritizedReplayBuffer if config.get('prioritized_replay', False) else ReplayBuffer
        if buffer_cls is PrioritizedReplayBuffer:
            self.memory = buffer_cls(
                config['buffer_size'],
                local_state_size,
                3,
                alpha=config.get('per_alpha', 0.6),
                device=device,
            )
        else:
            self.memory = buffer_cls(
                config['buffer_size'],
                local_state_size,
                3,
                device
            )
        self.update_every = config['update_every']
    
    def select_actions(self, local_states: List[np.ndarray], explore: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Select actions for all agents
        Returns concatenated action vector and list of preset indices
        """
        actions = []
        preset_indices = []
        
        for state in local_states:
            action, preset_idx = self.shared_agent.select_action(state, explore)
            actions.append(action)
            preset_indices.append(preset_idx)
        
        # Concatenate actions for environment
        full_action = np.concatenate(actions)
        
        return full_action, preset_indices
    
    def step(self, local_states: List[np.ndarray], action: np.ndarray, 
             reward: float, next_local_states: List[np.ndarray], done):
        """Store follower transitions in one shared replay and train one policy."""
        # Split concatenated action into individual agent actions
        agent_actions = []
        for i in range(self.num_followers):
            agent_action = action[i*3:(i+1)*3]
            agent_actions.append(agent_action)

        if np.isscalar(done):
            dones = [done] * self.num_followers
        else:
            dones = list(done)

        for i in range(self.num_followers):
            self.memory.add(
                local_states[i],
                agent_actions[i],
                reward,
                next_local_states[i],
                dones[i],
            )

        self.shared_agent.step_count += 1

        if len(self.memory) >= self.config['min_buffer_size']:
            if self.shared_agent.step_count % self.update_every == 0:
                gradient_steps = self.config.get('num_updates', 1)
                self.shared_agent.train(self.memory, num_updates=gradient_steps)
    
    def update_exploration(self):
        """Update exploration parameters - called by train_evaluate_all"""
        self.shared_agent.update_epsilon()
    
    def update_target_network(self):
        """Update the shared target network."""
        self.shared_agent.update_target_network()
    
    def save(self, base_path: str):
        """Save the shared follower policy."""
        os.makedirs(base_path, exist_ok=True)
        self.shared_agent.save(os.path.join(base_path, 'shared_agent.pth'))
    
    def load(self, base_path: str):
        """Load the shared follower policy checkpoint."""
        shared_path = os.path.join(base_path, 'shared_agent.pth')
        if os.path.exists(shared_path):
            self.shared_agent.load(shared_path)
            return

        raise FileNotFoundError(
            f"No shared DQN checkpoint found in {base_path}. Expected shared_agent.pth."
        )
