"""
Decentralized SAC agent for vehicular offloading
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
from configs.config_sac import get_sac_config
from centralized.sac_centralized import SACSquashedGaussianActor, SACCritic, initialize_sac_actor_for_mode
from configs.common_presets import get_action_presets


class DecentralizedSACAgent:
    """Single SAC agent for one follower in decentralized setting"""
    
    def __init__(self, state_dim: int, action_dim: int, agent_id: int, device: str = 'cpu',
                 optimization_mode: str = 'time', mixed_time_weight: float | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.device = device
        
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
        ).to(device)
        initialize_sac_actor_for_mode(self.actor, optimization_mode, config)
        
        self.critic = SACCritic(
            state_dim, action_dim, config['critic_hidden_sizes']
        ).to(device)
        
        self.critic_target = SACCritic(
            state_dim, action_dim, config['critic_hidden_sizes']
        ).to(device)
        
        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Counters
        self.step_count = 0
        
        # Epsilon for preset exploration (for consistency with centralized)
        self.use_preset_exploration = config.get('use_preset_exploration', False)
        self.epsilon = config.get('epsilon_start', 0.3)
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config.get('epsilon_min', 0.1)
        
        # Action presets - use common presets for fair comparison
        self.action_presets = get_action_presets() if self.use_preset_exploration else []
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action using SAC policy with epsilon-greedy exploration"""
        
        # Epsilon-greedy with presets for stability
        if self.use_preset_exploration and not evaluate and np.random.random() < self.epsilon:
            # Use preset action
            preset_idx = np.random.randint(len(self.action_presets))
            action = np.array(self.action_presets[preset_idx])
            # Add small variation
            noise = np.random.normal(0, 0.05, 3)
            action = action + noise
            action = np.clip(action, 0.0, 1.0)
            return action
        
        # Use actor network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                action = self.actor.get_action(state_tensor, deterministic=True)
                action = action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action, _, _ = self.actor.sample(state_tensor)
                action = action.cpu().numpy()[0]
        
        return action
    
    def train(self, memory, num_updates: int = 1) -> Tuple[float, float, float, float]:
        """Train networks using either per-agent or shared replay."""
        if hasattr(memory, 'agent_len'):
            ready = memory.agent_len(self.agent_id) >= self.config['min_buffer_size']
        else:
            ready = len(memory) >= self.config['min_buffer_size']
        if not ready:
            return 0.0, 0.0, 0.0, 0.0
        
        actor_losses = []
        critic_losses = []
        alpha_losses = []
        alpha_values = []
        
        for update_idx in range(num_updates):
            # Keep each follower agent on its own experience stream. Mixing
            # replay across followers makes separate agents regress toward the
            # same average behavior.
            if hasattr(memory, 'agent_len'):
                states, actions, rewards, next_states, dones = memory.sample(
                    self.batch_size, agent_idx=self.agent_id
                )
            else:
                states, actions, rewards, next_states, dones = memory.sample(self.batch_size)
            
            # Clip rewards (for fairness with DDPG)
            clip_min = self.config.get('reward_clip_min', -25)
            clip_max = self.config.get('reward_clip_max', 0)
            rewards = torch.clamp(rewards, clip_min, clip_max)
            
            # Current alpha value
            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha
            
            # Update critic
            with torch.no_grad():
                # Sample actions and log probs from current policy
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
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()
            
            critic_losses.append(critic_loss.item())
            
            if (update_idx % self.actor_update_every) == 0:
                # Update actor less frequently than the critic when configured.
                # This helps when the actor is chasing a critic that is still
                # too soft or underfit around the boundary-optimal regime.
                new_actions, log_probs, _ = self.actor.sample(states)
                if self.actor_use_min_q:
                    q1_new, q2_new = self.critic(states, new_actions)
                    q_values = torch.min(q1_new, q2_new)
                else:
                    q_values = self.critic.q1_forward(states, new_actions)
                actor_loss = (alpha * log_probs - q_values).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.actor_optimizer.step()

                actor_losses.append(actor_loss.item())

                # Update temperature (alpha) if automatic tuning is enabled
                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()

                    alpha_losses.append(alpha_loss.item())
                else:
                    alpha_losses.append(0.0)
                alpha_values.append(alpha.item() if self.automatic_entropy_tuning else self.alpha)
            
            # Soft update target network
            self._soft_update(self.critic, self.critic_target)
        
        return (
            float(np.mean(actor_losses)) if actor_losses else 0.0,
            float(np.mean(critic_losses)) if critic_losses else 0.0,
            float(np.mean(alpha_losses)) if alpha_losses else 0.0,
            float(np.mean(alpha_values)) if alpha_values else float(alpha.item() if self.automatic_entropy_tuning else self.alpha),
        )
    
    def update_exploration(self):
        """Update exploration parameters - call after each episode"""
        if self.use_preset_exploration:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
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
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
        }
        
        if self.automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha.data.cpu().numpy()
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.step_count = checkpoint.get('step_count', 0)
        self.epsilon = checkpoint.get('epsilon', 0.1)
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha.data = torch.tensor(checkpoint['log_alpha']).to(self.device)
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])


class MultiAgentSAC:
    """
    Shared-policy decentralized SAC manager.

    One follower policy is applied to all followers, with different actions
    emerging from different local observations.
    """
    
    def __init__(self, local_state_size: int, num_followers: int, device: str = 'cpu',
                 optimization_mode: str = 'time', mixed_time_weight: float | None = None):
        self.num_followers = num_followers
        self.device = device
        
        # One follower policy shared across all follower vehicles.
        self.shared_agent = DecentralizedSACAgent(
            local_state_size, 3, 0, device, optimization_mode,
            mixed_time_weight=mixed_time_weight,
        )

        # Shared replay buffer over all follower transitions.
        config = get_sac_config(mode=optimization_mode)
        if mixed_time_weight is not None:
            config['mixed_time_weight'] = float(mixed_time_weight)
        self.config = config  # Store config for reuse
        self.memory = ReplayBuffer(
            config['buffer_size'],
            local_state_size,
            3,
            device
        )
        
        self.update_every = config['update_every']
    
    def select_actions(self, local_states: List[np.ndarray], evaluate: bool = False) -> np.ndarray:
        """
        Select actions for all agents
        Returns concatenated action vector
        """
        actions = []
        
        for state in local_states:
            action = self.shared_agent.select_action(state, evaluate)
            actions.append(action)
        
        # Concatenate actions for environment
        full_action = np.concatenate(actions)
        
        return full_action
    
    def update_exploration(self):
        """Update exploration parameters for the shared follower policy."""
        self.shared_agent.update_exploration()
    
    def step(self, local_states: List[np.ndarray], actions: np.ndarray, 
             reward: float, next_local_states: List[np.ndarray], done):
        """Store experience and train agents"""
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
            f"No shared SAC checkpoint found in {base_path}. Expected shared_agent.pth."
        )
