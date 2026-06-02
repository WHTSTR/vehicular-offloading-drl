"""
Replay buffer implementations for experience replay
"""

import numpy as np
import torch
import random
from typing import Tuple, List, Optional


class ReplayBuffer:
    """Basic replay buffer for single agent"""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add a transition to the buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size


class MultiAgentReplayBuffer:
    """Replay buffer for multiple agents (decentralized training)"""
    
    def __init__(self, buffer_size: int, num_agents: int, state_dim: int, 
                 action_dim: int, device: str = 'cpu'):
        self.num_agents = num_agents
        self.buffers = [
            ReplayBuffer(buffer_size, state_dim, action_dim, device)
            for _ in range(num_agents)
        ]
    
    def add(self, states: List[np.ndarray], actions: List[np.ndarray], 
            reward, next_states: List[np.ndarray], done):
        """
        Add transitions for all agents
        Note: reward/done may be scalars or per-agent sequences.
        """
        if np.isscalar(reward):
            rewards = [reward] * self.num_agents
        else:
            rewards = list(reward)

        if np.isscalar(done):
            dones = [done] * self.num_agents
        else:
            dones = list(done)

        for i in range(self.num_agents):
            self.buffers[i].add(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def sample(self, batch_size: int, agent_idx: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Sample from a specific agent's buffer or from all agents
        """
        if agent_idx is not None:
            return self.buffers[agent_idx].sample(batch_size)
        else:
            # Sample from random agents
            samples = []
            for _ in range(batch_size):
                agent = random.randint(0, self.num_agents - 1)
                while len(self.buffers[agent]) == 0:
                    agent = random.randint(0, self.num_agents - 1)
                sample = self.buffers[agent].sample(1)
                samples.append(sample)
            
            # Concatenate samples
            states = torch.cat([s[0] for s in samples], dim=0)
            actions = torch.cat([s[1] for s in samples], dim=0)
            rewards = torch.cat([s[2] for s in samples], dim=0)
            next_states = torch.cat([s[3] for s in samples], dim=0)
            dones = torch.cat([s[4] for s in samples], dim=0)
            
            return states, actions, rewards, next_states, dones

    def agent_len(self, agent_idx: int) -> int:
        """Number of transitions stored for a specific agent."""
        return len(self.buffers[agent_idx])

    def min_agent_len(self) -> int:
        """Minimum number of transitions available across all agent buffers."""
        return min(len(buffer) for buffer in self.buffers) if self.buffers else 0

    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay buffer"""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, 
                 alpha: float = 0.6, device: str = 'cpu'):
        super().__init__(buffer_size, state_dim, action_dim, device)
        self.alpha = alpha
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """Add with max priority"""
        self.priorities[self.ptr] = self.max_priority
        super().add(state, action, reward, next_state, done)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """Sample with priorities"""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        
        # Get samples
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
