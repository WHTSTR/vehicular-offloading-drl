"""
Decentralized DDPG agent for vehicular offloading
Each follower acts independently on its own local observation, using a single policy shared across the homogeneous followers (parameter sharing).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from typing import List, Tuple

from utils.replay_buffer import ReplayBuffer
from configs.config_ddpg import get_ddpg_config
from centralized.ddpg_centralized import (
    StableActor,
    StableCritic,
    DecorrelatedOUNoise,
    RunningStats,
    initialize_ddpg_actor_for_mode,
)
from configs.common_presets import get_action_presets


class DecentralizedDDPGAgent:
    """Single DDPG agent for one follower in decentralized setting"""
    
    def __init__(self, state_dim: int, action_dim: int, agent_id: int, device: str = 'cpu',
                 optimization_mode: str = 'time',
                 mixed_time_weight: float | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.device = device
        
        # Load configuration with mode
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
        
        # Networks
        self.actor = StableActor(state_dim, action_dim, config['actor_hidden_sizes'], config.get('use_layer_norm', True)).to(device)
        initialize_ddpg_actor_for_mode(self.actor, optimization_mode, config)
        self.actor_target = StableActor(state_dim, action_dim, config['actor_hidden_sizes'], config.get('use_layer_norm', True)).to(device)
        self.critic = StableCritic(state_dim, action_dim, config['critic_hidden_sizes']).to(device)
        self.critic_target = StableCritic(state_dim, action_dim, config['critic_hidden_sizes']).to(device)
        
        # Copy weights to targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Q-value normalization
        self.q_stats = RunningStats()
        
        # Exploration noise (individual for each agent)
        if config['noise_type'] == 'ou':
            self.noise = DecorrelatedOUNoise(
                1, action_dim,  # Single agent, 3D action
                sigma=config['ou_sigma'],
                theta=config['ou_theta']
            )
        self.noise_scale = config['noise_scale']
        self.noise_decay = config['noise_decay']
        self.min_noise = config['min_noise']
        
        # Counters
        self.step_count = 0
        
        # Epsilon for preset exploration (use config values for fairness)
        self.use_preset_exploration = config.get('use_preset_exploration', False)
        self.epsilon = config.get('epsilon_start', 0.3)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.05)
        
        # Action presets - use common presets for fair comparison
        self.action_presets = get_action_presets() if self.use_preset_exploration else []
    
    def reset_noise(self):
        """Reset noise generator"""
        if hasattr(self, 'noise'):
            self.noise.reset()
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action with exploration (epsilon-greedy + noise)"""
        
        # Epsilon-greedy with presets for stability
        if self.use_preset_exploration and explore and np.random.random() < self.epsilon:
            # Use preset action
            preset_idx = np.random.randint(len(self.action_presets))
            action = np.array(self.action_presets[preset_idx])
            # Add small variation
            noise_std = self.config.get('preset_noise_std', 0.05)
            noise = np.random.normal(0, noise_std, 3)
            action = action + noise
            action = np.clip(action, 0.0, 1.0)
            return action
        
        # Use actor network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        # Add exploration noise
        if explore and hasattr(self, 'noise'):
            noise = self.noise.sample()
            action = action + self.noise_scale * noise
            action = np.clip(action, 0.0, 1.0)
        
        return action
    
    def train(self, memory, num_updates: int = 1) -> Tuple[float, float]:
        """Train networks using either per-agent or shared replay."""
        if hasattr(memory, 'agent_len'):
            ready = memory.agent_len(self.agent_id) >= self.config['min_buffer_size']
        else:
            ready = len(memory) >= self.config['min_buffer_size']
        if not ready:
            return 0.0, 0.0
        
        actor_losses = []
        critic_losses = []
        
        for _ in range(num_updates):
            # Keep each follower agent on its own experience stream. Mixing
            # replay across followers pushes separate agents toward the same
            # compromise policy.
            if hasattr(memory, 'agent_len'):
                states, actions, rewards, next_states, dones = memory.sample(
                    self.batch_size, agent_idx=self.agent_id
                )
            else:
                states, actions, rewards, next_states, dones = memory.sample(self.batch_size)
            
            # Clip rewards for stability
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
    
    def update_exploration(self):
        """Update exploration parameters - call after each episode"""
        if self.use_preset_exploration:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save agent state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'agent_id': self.agent_id,
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
        }
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.step_count = checkpoint.get('step_count', 0)
        self.epsilon = checkpoint.get('epsilon', 0.1)
        self.noise_scale = checkpoint.get('noise_scale', 0.1)
        
        # Restore Q-stats
        if 'q_stats_mean' in checkpoint:
            self.q_stats.mean = checkpoint['q_stats_mean']
            self.q_stats.variance = checkpoint['q_stats_variance']
            self.q_stats.n = checkpoint['q_stats_n']


class MultiAgentDDPG:
    """
    Shared-policy decentralized DDPG manager.

    All followers solve the same local decision problem with the same action
    space, so one follower policy is applied to each follower's local
    observation instead of training slot-specific networks.
    """
    
    def __init__(self, local_state_size: int, num_followers: int, device: str = 'cpu',
                 optimization_mode: str = 'time',
                 mixed_time_weight: float | None = None):
        self.num_followers = num_followers
        self.device = device
        
        # One follower policy shared across all follower vehicles.
        self.shared_agent = DecentralizedDDPGAgent(
            local_state_size,
            3,
            0,
            device,
            optimization_mode,
            mixed_time_weight=mixed_time_weight,
        )

        # Shared replay buffer over all follower transitions.
        config = get_ddpg_config(mode=optimization_mode)
        if mixed_time_weight is not None:
            config['mixed_time_weight'] = float(mixed_time_weight)
        self.config = config  # Store config for reuse
        self.memory = ReplayBuffer(
            config['buffer_size'],
            local_state_size,
            3,
            device,
        )
        
        self.update_every = config['update_every']
    
    def reset_noise(self):
        """Reset shared exploration noise at the start of each episode."""
        self.shared_agent.reset_noise()
    
    def update_exploration(self):
        """Update exploration parameters for the shared follower policy."""
        self.shared_agent.update_exploration()
    
    def select_actions(self, local_states: List[np.ndarray], explore: bool = True) -> np.ndarray:
        """
        Select actions for all agents
        Returns concatenated action vector
        """
        actions = []
        
        for state in local_states:
            action = self.shared_agent.select_action(state, explore)
            actions.append(action)
        
        # Concatenate actions for environment
        full_action = np.concatenate(actions)
        
        return full_action
    
    def step(self, local_states: List[np.ndarray], actions: np.ndarray, 
             reward: float, next_local_states: List[np.ndarray], done):
        """Store follower transitions in one shared replay and train one policy."""
        # Split concatenated actions back to individual actions
        individual_actions = []
        for i in range(self.num_followers):
            start_idx = i * 3
            end_idx = (i + 1) * 3
            individual_actions.append(actions[start_idx:end_idx])
        
        if np.isscalar(done):
            dones = [done] * self.num_followers
        else:
            dones = list(done)

        for i in range(self.num_followers):
            self.memory.add(
                local_states[i],
                individual_actions[i],
                reward,
                next_local_states[i],
                dones[i],
            )

        self.shared_agent.step_count += 1

        if len(self.memory) >= self.config['min_buffer_size']:
            if self.shared_agent.step_count % self.update_every == 0:
                self.shared_agent.train(self.memory, self.config['num_updates'])
    
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
            f"No shared DDPG checkpoint found in {base_path}. Expected shared_agent.pth."
        )
