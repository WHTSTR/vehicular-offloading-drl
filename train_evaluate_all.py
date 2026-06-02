#!/usr/bin/env python3
"""
Combined training and evaluation for both centralized and decentralized algorithms
Trains and evaluates all algorithms: C-DQN, C-DDPG, C-SAC, D-DQN, D-DDPG, D-SAC
"""

import os
import sys
import time
import json
import csv
import argparse
import random
from contextlib import contextmanager
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any

try:
    import seaborn as sns  # noqa: F401
except ImportError:
    sns = None

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from environment import PaperVehicularEnvironment
from configs.config_base import get_paper_config
from baselines import BaselineAlgorithms
from oracle_baselines import DifferentialEvolutionOracle, PresetBranchAndBoundOracle

# Import centralized agents
from centralized.dqn_centralized import CentralizedDQNAgent
from centralized.ddpg_centralized import CentralizedDDPGAgent
from centralized.sac_centralized import CentralizedSACAgent

# Import decentralized agents
from decentralized.dqn_decentralized import MultiAgentDQN
from decentralized.ddpg_decentralized import MultiAgentDDPG
from decentralized.sac_decentralized import MultiAgentSAC


class CombinedTrainEvaluate:
    """Comprehensive trainer and evaluator for both centralized and decentralized algorithms"""
    
    def __init__(self, optimization_target: str = 'mixed', episodes: int = 5000,
                 eval_interval: int = 50, device: str = 'cpu', seed: int = 42,
                 num_vehicles: int = 5, cpu_cycles_per_bit: int = 10,
                 chunk_overhead_cycles: float = 1e6,
                 redundancy_ratio_range: tuple = (0.5, 0.5),
                 leader_selection_mode: str = 'episode_start_scored',
                 leader_reselection_interval_slots: int = 10,
                 base_station_position: tuple[float, float] | None = None,
                 initial_position_mode: str | None = None,
                 initial_position_x_range: tuple[float, float] | None = None,
                 initial_position_jitter: float | None = None,
                 time_budget: float | None = None,
                 follower_time_budget: float | None = None,
                 leader_time_budget: float | None = None,
                 energy_budget: float | None = None,
                 follower_energy_budget: float | None = None,
                 leader_energy_budget: float | None = None,
                 chunk_size: float | None = None,
                 time_slot_duration: float | None = None,
                 leader_power_multiplier: float | None = None,
                 leader_uplink_resource_factor: float | None = None,
                 min_active_power_ratio: float | None = None,
                 mixed_time_weight: float = 0.5,
                 reward_mode: str = 'paper_counted_normalized',
                 constraint_penalty_value: float = 10.0,
                 energy_reward_normalization_factor: float = 4.0,
                 checkpoint_eval_episodes: int = 25,
                 use_fixed_eval_seeds: bool = True,
                 fixed_eval_seed_base: int = 100000):
        self.optimization_target = optimization_target
        self.episodes = episodes
        self.eval_interval = eval_interval
        self.device = device
        self.seed = seed
        self.num_vehicles = num_vehicles
        self.cpu_cycles_per_bit = cpu_cycles_per_bit
        self.chunk_overhead_cycles = chunk_overhead_cycles
        self.redundancy_ratio_range = redundancy_ratio_range
        self.leader_selection_mode = leader_selection_mode
        self.leader_reselection_interval_slots = leader_reselection_interval_slots
        self.base_station_position = base_station_position
        self.initial_position_mode = initial_position_mode
        self.initial_position_x_range = initial_position_x_range
        self.initial_position_jitter = initial_position_jitter
        self.time_budget = time_budget
        self.follower_time_budget = follower_time_budget
        self.leader_time_budget = leader_time_budget
        self.energy_budget = energy_budget
        self.follower_energy_budget = follower_energy_budget
        self.leader_energy_budget = leader_energy_budget
        self.chunk_size = chunk_size
        self.time_slot_duration = time_slot_duration
        self.leader_power_multiplier = leader_power_multiplier
        self.leader_uplink_resource_factor = leader_uplink_resource_factor
        self.min_active_power_ratio = min_active_power_ratio
        self.mixed_time_weight = mixed_time_weight
        self.reward_mode = reward_mode
        self.constraint_penalty_value = constraint_penalty_value
        self.energy_reward_normalization_factor = energy_reward_normalization_factor
        self.checkpoint_eval_episodes = checkpoint_eval_episodes
        self.use_fixed_eval_seeds = use_fixed_eval_seeds
        self.fixed_eval_seed_base = fixed_eval_seed_base
        
        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if optimization_target == 'mixed':
            eta_tag = f"eta_{int(round(self.mixed_time_weight * 100)):03d}"
            self.save_dir = f"results/combined_mixed_mode/{eta_tag}/run_{timestamp}_{os.getpid()}"
        else:
            self.save_dir = f"results/combined_{optimization_target}_mode/run_{timestamp}_{os.getpid()}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Get configuration
        self.config = get_paper_config()
        self.config['environment']['optimization_target'] = optimization_target
        
        # Update config with command-line parameters
        self.config['environment']['num_vehicles'] = self.num_vehicles
        self.config['environment']['cpu_cycles_per_bit'] = self.cpu_cycles_per_bit
        self.config['environment']['chunk_overhead_cycles'] = self.chunk_overhead_cycles
        self.config['environment']['redundancy_lower'] = self.redundancy_ratio_range[0]
        self.config['environment']['redundancy_upper'] = self.redundancy_ratio_range[1]
        self.config['environment']['leader_selection_mode'] = self.leader_selection_mode
        self.config['environment']['leader_reselection_interval_slots'] = self.leader_reselection_interval_slots
        if self.base_station_position is not None:
            self.config['environment']['base_station_position'] = list(self.base_station_position)
        if self.initial_position_mode is not None:
            self.config['environment']['initial_position_mode'] = self.initial_position_mode
        if self.initial_position_x_range is not None:
            self.config['environment']['initial_position_x_range'] = list(self.initial_position_x_range)
        if self.initial_position_jitter is not None:
            self.config['environment']['initial_position_jitter'] = self.initial_position_jitter
        self.config['environment']['mixed_time_weight'] = self.mixed_time_weight
        self.config['environment']['reward_mode'] = self.reward_mode
        self.config['environment']['constraint_penalty_value'] = self.constraint_penalty_value
        self.config['environment']['energy_reward_normalization_factor'] = self.energy_reward_normalization_factor
        if self.time_budget is not None:
            self.config['environment']['time_budget'] = self.time_budget
        if self.follower_time_budget is not None:
            self.config['environment']['follower_time_budget'] = self.follower_time_budget
        if self.leader_time_budget is not None:
            self.config['environment']['leader_time_budget'] = self.leader_time_budget
        if self.energy_budget is not None:
            self.config['environment']['energy_budget'] = self.energy_budget
        if self.leader_power_multiplier is not None:
            self.config['environment']['leader_power_multiplier'] = self.leader_power_multiplier
        if self.leader_uplink_resource_factor is not None:
            self.config['environment']['leader_uplink_resource_factor'] = self.leader_uplink_resource_factor
        if self.min_active_power_ratio is not None:
            self.config['environment']['min_active_power_ratio'] = self.min_active_power_ratio
        if self.follower_energy_budget is not None:
            self.config['environment']['follower_energy_budget'] = self.follower_energy_budget
        elif self.follower_time_budget is not None:
            max_power = float(self.config['environment']['max_power'])
            joint_power_constraint = bool(self.config['environment'].get('joint_power_constraint', True))
            follower_power_ceiling = max_power if joint_power_constraint else 2.0 * max_power
            self.config['environment']['follower_energy_budget'] = self.follower_time_budget * follower_power_ceiling
        if self.leader_energy_budget is not None:
            self.config['environment']['leader_energy_budget'] = self.leader_energy_budget
        elif self.follower_time_budget is not None or self.leader_time_budget is not None:
            max_power = float(self.config['environment']['max_power'])
            leader_power_multiplier = float(self.config['environment'].get('leader_power_multiplier', 1.0))
            leader_cpu_frequency = float(self.config['environment']['leader_cpu_frequency'])
            cpu_power_constant = float(self.config['environment']['cpu_power_constant'])
            effective_leader_time_budget = (
                self.leader_time_budget
                if self.leader_time_budget is not None
                else float(self.config['environment']['leader_time_budget'])
            )
            leader_upload_power_ceiling = max_power * leader_power_multiplier
            leader_cpu_power = cpu_power_constant * (leader_cpu_frequency ** 2)
            self.config['environment']['leader_energy_budget'] = (
                effective_leader_time_budget * (leader_upload_power_ceiling + leader_cpu_power)
            )
        if self.chunk_size is not None:
            self.config['environment']['chunk_size'] = self.chunk_size
        if self.time_slot_duration is not None:
            self.config['environment']['time_slot_duration'] = self.time_slot_duration
        self.config['evaluation'] = {
            'checkpoint_eval_episodes': self.checkpoint_eval_episodes,
            'use_fixed_eval_seeds': self.use_fixed_eval_seeds,
            'fixed_eval_seed_base': self.fixed_eval_seed_base,
        }
        
        # Set environment seed for reproducibility
        self.config['environment']['seed'] = self.seed
        
        # Create environment
        self.env = PaperVehicularEnvironment(self.config['environment'])
        self.num_followers = self.env.num_followers
        
        # Get state dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Get local state size for decentralized agents
        self.env.reset()
        dummy_local_state = self.env.get_local_observation(0)
        self.local_state_size = len(dummy_local_state)
        
        # Initialize metrics storage for all algorithms.
        self.training_metrics = {
            alg_name: self._empty_training_metric_store()
            for alg_name in ['C-DQN', 'C-DDPG', 'C-SAC', 'D-DQN', 'D-DDPG', 'D-SAC']
        }
        
        self.best_models = {
            'C-DQN': {'path': None, 'reward': -float('inf')},
            'C-DDPG': {'path': None, 'reward': -float('inf')},
            'C-SAC': {'path': None, 'reward': -float('inf')},
            'D-DQN': {'path': None, 'reward': -float('inf')},
            'D-DDPG': {'path': None, 'reward': -float('inf')},
            'D-SAC': {'path': None, 'reward': -float('inf')}
        }
        
        # Store trained agents
        self.agents = {}
        self.state_dim = state_dim
        self.action_dim = action_dim

    def _reset_global_rngs(self, seed: int) -> None:
        """Reset all global RNGs so algorithm training is independent of run order."""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _reseed_for_algorithm(self, alg_name: str) -> None:
        """Give each algorithm the same clean starting RNG state before training."""
        self._reset_global_rngs(self.seed)
        self.env.seed(self.seed)

    def _empty_training_metric_store(self) -> Dict[str, List[float]]:
        """Metric arrays saved for every training run."""
        return {
            'rewards': [],
            'penalizeds': [],
            'objective_costs': [],
            'times': [],
            'energies': [],
            'mixeds': [],
            'normalized_times': [],
            'normalized_energies': [],
            'constraint_violations': [],
            'constraint_penalties': [],
            'any_constraint_violations': [],
            'follower_time_violations': [],
            'leader_time_violations': [],
            'follower_energy_violations': [],
            'leader_energy_violations': [],
            'eval_rewards': [],
            'eval_penalizeds': [],
            'eval_objective_costs': [],
            'eval_times': [],
            'eval_energies': [],
            'eval_mixeds': [],
            'eval_normalized_times': [],
            'eval_normalized_energies': [],
            'eval_constraint_violations': [],
            'eval_constraint_penalties': [],
            'eval_any_constraint_violations': [],
            'eval_follower_time_violations': [],
            'eval_leader_time_violations': [],
            'eval_follower_energy_violations': [],
            'eval_leader_energy_violations': [],
        }

    def _new_episode_metrics(self) -> Dict[str, float]:
        """Accumulator for one environment episode."""
        return {
            'reward': 0.0,
            'penalized': 0.0,
            'objective_cost': 0.0,
            'time': 0.0,
            'energy': 0.0,
            'mixed': 0.0,
            'normalized_time': 0.0,
            'normalized_energy': 0.0,
            'constraint_violations': 0.0,
            'constraint_penalty': 0.0,
            'any_constraint_violations': 0.0,
            'follower_time_violations': 0.0,
            'leader_time_violations': 0.0,
            'follower_energy_violations': 0.0,
            'leader_energy_violations': 0.0,
        }

    def _accumulate_episode_metrics(self, episode_metrics: Dict[str, float], reward: float, info: Dict[str, Any]) -> None:
        """Accumulate one environment step into episode-level diagnostics."""
        episode_metrics['reward'] += float(reward)
        episode_metrics['penalized'] += float(info.get('penalized_objective', -reward))
        episode_metrics['objective_cost'] += float(info.get('objective_cost', info.get('mixed_cost', 0.0)))
        episode_metrics['time'] += float(info.get('time_cost', 0.0))
        episode_metrics['energy'] += float(info.get('energy_cost', 0.0))
        episode_metrics['mixed'] += float(info.get('mixed_cost', 0.0))
        episode_metrics['normalized_time'] += float(info.get('normalized_time_cost', 0.0))
        episode_metrics['normalized_energy'] += float(info.get('normalized_energy_cost', 0.0))
        episode_metrics['constraint_violations'] += float(info.get('constraint_violations', 0.0))
        episode_metrics['constraint_penalty'] += float(info.get('constraint_penalty_applied', 0.0))
        episode_metrics['any_constraint_violations'] += float(int(bool(info.get('constraint_violated', False))))
        episode_metrics['follower_time_violations'] += float(info.get('follower_time_violations', 0.0))
        episode_metrics['leader_time_violations'] += float(info.get('leader_time_violation', 0.0))
        episode_metrics['follower_energy_violations'] += float(info.get('follower_energy_violations', 0.0))
        episode_metrics['leader_energy_violations'] += float(info.get('leader_energy_violation', 0.0))

    def _append_training_episode_metrics(self, alg_name: str, episode_metrics: Dict[str, float]) -> None:
        """Append one finished training episode to stored arrays."""
        store = self.training_metrics[alg_name]
        store['rewards'].append(float(episode_metrics['reward']))
        store['penalizeds'].append(float(episode_metrics['penalized']))
        store['objective_costs'].append(float(episode_metrics['objective_cost']))
        store['times'].append(float(episode_metrics['time']))
        store['energies'].append(float(episode_metrics['energy']))
        store['mixeds'].append(float(episode_metrics['mixed']))
        store['normalized_times'].append(float(episode_metrics['normalized_time']))
        store['normalized_energies'].append(float(episode_metrics['normalized_energy']))
        store['constraint_violations'].append(float(episode_metrics['constraint_violations']))
        store['constraint_penalties'].append(float(episode_metrics['constraint_penalty']))
        store['any_constraint_violations'].append(float(episode_metrics['any_constraint_violations']))
        store['follower_time_violations'].append(float(episode_metrics['follower_time_violations']))
        store['leader_time_violations'].append(float(episode_metrics['leader_time_violations']))
        store['follower_energy_violations'].append(float(episode_metrics['follower_energy_violations']))
        store['leader_energy_violations'].append(float(episode_metrics['leader_energy_violations']))

    def _append_eval_metrics(self, alg_name: str, eval_metrics: Dict[str, float]) -> None:
        """Append one checkpoint evaluation summary to stored arrays."""
        store = self.training_metrics[alg_name]
        store['eval_rewards'].append(float(eval_metrics['reward']))
        store['eval_penalizeds'].append(float(eval_metrics['penalized']))
        store['eval_objective_costs'].append(float(eval_metrics['objective_cost']))
        store['eval_times'].append(float(eval_metrics['time']))
        store['eval_energies'].append(float(eval_metrics['energy']))
        store['eval_mixeds'].append(float(eval_metrics['mixed']))
        store['eval_normalized_times'].append(float(eval_metrics['normalized_time']))
        store['eval_normalized_energies'].append(float(eval_metrics['normalized_energy']))
        store['eval_constraint_violations'].append(float(eval_metrics['constraint_violations']))
        store['eval_constraint_penalties'].append(float(eval_metrics['constraint_penalty']))
        store['eval_any_constraint_violations'].append(float(eval_metrics['any_constraint_violations']))
        store['eval_follower_time_violations'].append(float(eval_metrics['follower_time_violations']))
        store['eval_leader_time_violations'].append(float(eval_metrics['leader_time_violations']))
        store['eval_follower_energy_violations'].append(float(eval_metrics['follower_energy_violations']))
        store['eval_leader_energy_violations'].append(float(eval_metrics['leader_energy_violations']))

    def _checkpoint_dir(self, alg_name: str, episode: int) -> str:
        """Directory for a specific evaluation checkpoint."""
        return os.path.join(
            self.save_dir,
            "checkpoints",
            alg_name.lower(),
            f"episode_{episode:05d}",
        )

    def _write_checkpoint_meta(self, checkpoint_dir: str, alg_name: str, episode: int,
                               eval_metrics: Dict[str, float]) -> None:
        """Persist minimal metadata alongside a saved checkpoint."""
        meta = {
            "algorithm": alg_name,
            "episode": int(episode),
            "eval_reward": float(eval_metrics["reward"]),
            "eval_penalized": float(eval_metrics["penalized"]),
            "eval_objective_cost": float(eval_metrics["objective_cost"]),
            "eval_time": float(eval_metrics["time"]),
            "eval_energy": float(eval_metrics["energy"]),
            "eval_mixed": float(eval_metrics["mixed"]),
            "eval_normalized_time": float(eval_metrics["normalized_time"]),
            "eval_normalized_energy": float(eval_metrics["normalized_energy"]),
            "eval_constraint_violations": float(eval_metrics["constraint_violations"]),
            "eval_constraint_penalty": float(eval_metrics["constraint_penalty"]),
            "eval_any_constraint_violations": float(eval_metrics["any_constraint_violations"]),
            "eval_follower_time_violations": float(eval_metrics["follower_time_violations"]),
            "eval_leader_time_violations": float(eval_metrics["leader_time_violations"]),
            "eval_follower_energy_violations": float(eval_metrics["follower_energy_violations"]),
            "eval_leader_energy_violations": float(eval_metrics["leader_energy_violations"]),
            "optimization_target": self.optimization_target,
            "mixed_time_weight": float(self.mixed_time_weight),
            "reward_mode": self.reward_mode,
            "constraint_penalty_value": float(self.constraint_penalty_value),
            "leader_reselection_interval_slots": int(self.leader_reselection_interval_slots),
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    def _evaluation_seed(self, episode_idx: int, stream: str) -> int:
        """Deterministic holdout seed for evaluation-only rollouts."""
        stream_offsets = {
            'checkpoint': 0,
            'baseline': 10000,
            'final': 20000,
            'report': 30000,
            'surface': 40000,
        }
        if stream not in stream_offsets:
            raise ValueError(f"Unknown evaluation stream: {stream}")
        return int(self.fixed_eval_seed_base + stream_offsets[stream] + episode_idx)

    def reset_for_evaluation(self, episode_idx: int, stream: str = 'checkpoint'):
        """Reset the environment using deterministic holdout seeds when enabled."""
        if self.use_fixed_eval_seeds:
            self.env.seed(self._evaluation_seed(episode_idx, stream))
        return self.env.reset()

    def _result_metric_key(self) -> str:
        return 'mixed' if self.optimization_target == 'mixed' else self.optimization_target

    def _capture_rng_state(self) -> Dict[str, Any]:
        """Capture global RNG state so evaluation does not perturb training."""
        state: Dict[str, Any] = {
            'numpy': np.random.get_state(),
            'python_random': random.getstate(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: Dict[str, Any]) -> None:
        """Restore previously captured global RNG state."""
        np.random.set_state(state['numpy'])
        random.setstate(state['python_random'])
        torch.set_rng_state(state['torch'])
        if torch.cuda.is_available() and 'torch_cuda' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda'])

    @contextmanager
    def _preserve_training_rng(self):
        """Run evaluation without consuming training RNG streams."""
        state = self._capture_rng_state()
        try:
            yield
        finally:
            self._restore_rng_state(state)

    def _warmup_complete(self, learner: Any) -> bool:
        """Whether replay-based training has enough experience to start updating."""
        if not hasattr(learner, 'memory'):
            return True
        if hasattr(learner, 'config'):
            min_buffer_size = learner.config.get('min_buffer_size', 0)
        elif hasattr(learner, 'shared_agent'):
            min_buffer_size = learner.shared_agent.config.get('min_buffer_size', 0)
        elif hasattr(learner, 'agents') and learner.agents:
            min_buffer_size = learner.agents[0].config.get('min_buffer_size', 0)
        else:
            min_buffer_size = 0
        if hasattr(learner.memory, 'min_agent_len'):
            return learner.memory.min_agent_len() >= min_buffer_size
        return len(learner.memory) >= min_buffer_size

    def _aligned_decentralized_next_states(
        self,
        acting_follower_indices: List[int],
        done: bool,
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """
        Align decentralized replay transitions to the physical follower that acted.

        After periodic leader reselection, follower slot i in the next slot may
        correspond to a different physical vehicle. For replay storage, we need
        the next observation of the same acting vehicle. If that vehicle becomes
        the leader on the next slot, its follower transition terminates.
        """
        next_follower_set = set(self.env.follower_indices)
        aligned_next_states: List[np.ndarray] = []
        aligned_dones: List[bool] = []

        for vehicle_idx in acting_follower_indices:
            role_changed = vehicle_idx not in next_follower_set
            agent_done = bool(done or role_changed)
            aligned_dones.append(agent_done)

            if agent_done:
                aligned_next_states.append(np.zeros(self.local_state_size, dtype=np.float32))
            else:
                aligned_next_states.append(
                    self.env.get_local_observation_for_vehicle(vehicle_idx)
                )

        return aligned_next_states, aligned_dones

    def _training_metric_key(self) -> str:
        return {
            'time': 'times',
            'energy': 'energies',
            'mixed': 'mixeds',
        }[self._result_metric_key()]

    def _objective_display_name(self) -> str:
        if self._result_metric_key() == 'mixed':
            return f"Mixed Objective (eta={self.mixed_time_weight:.2f})"
        if self._result_metric_key() == 'time':
            return "Time"
        return "Energy"

    def _objective_axis_label(self) -> str:
        if self._result_metric_key() == 'mixed':
            return "Mixed Objective (normalized)"
        if self._result_metric_key() == 'time':
            return "Time (s)"
        return "Energy (J)"

    @staticmethod
    def _summarize_action_component(values: np.ndarray, distribution: np.ndarray | None = None) -> dict:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            summary = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'distribution': np.asarray([], dtype=np.float32),
            }
        else:
            summary = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'distribution': values.copy(),
            }
        if distribution is not None:
            summary['distribution'] = np.asarray(distribution, dtype=np.float32).reshape(-1)
        return summary
    
    def analyze_actions(self, actions: np.ndarray, num_followers: int) -> dict:
        """
        Analyze action distributions.

        Power summaries use the executed active-link powers seen by the
        environment, not the raw network outputs. Raw power statistics are kept separately.
        """
        # Reshape actions for per-follower analysis
        actions_reshaped = actions.reshape(-1, num_followers, 3)
        
        # Extract components
        v2v_fractions = np.clip(actions_reshaped[:, :, 0], 0.0, 1.0)  # delta
        raw_v2v_powers = np.clip(actions_reshaped[:, :, 1], 0.0, 1.0)  # p_v2v / p_max
        raw_v2i_powers = np.clip(actions_reshaped[:, :, 2], 0.0, 1.0)  # p_v2i / p_max
        min_active_power_ratio = float(getattr(self.env, 'min_active_power_ratio', 0.0))

        v2v_active_mask = v2v_fractions > 0.0
        v2i_active_mask = v2v_fractions < 1.0
        executed_v2v_powers = np.where(
            v2v_active_mask,
            np.maximum(raw_v2v_powers, min_active_power_ratio),
            0.0,
        )
        executed_v2i_powers = np.where(
            v2i_active_mask,
            np.maximum(raw_v2i_powers, min_active_power_ratio),
            0.0,
        )
        
        analysis = {
            'v2v_fraction': self._summarize_action_component(v2v_fractions),
            'v2v_power': self._summarize_action_component(
                executed_v2v_powers[v2v_active_mask],
                distribution=executed_v2v_powers,
            ),
            'v2i_power': self._summarize_action_component(
                executed_v2i_powers[v2i_active_mask],
                distribution=executed_v2i_powers,
            ),
            'v2v_power_raw': self._summarize_action_component(raw_v2v_powers),
            'v2i_power_raw': self._summarize_action_component(raw_v2i_powers),
            'v2v_active_fraction': {
                'mean': float(np.mean(v2v_active_mask)),
                'std': float(np.std(v2v_active_mask.astype(np.float32))),
            },
            'v2i_active_fraction': {
                'mean': float(np.mean(v2i_active_mask)),
                'std': float(np.std(v2i_active_mask.astype(np.float32))),
            },
            'min_active_power_ratio': min_active_power_ratio,
        }
        
        return analysis
    
    def train_centralized_algorithm(self, algorithm: str, alg_name: str) -> Any:
        """Train a single centralized algorithm"""
        print(f"\n{'='*60}")
        print(f"Training {alg_name} (Centralized {algorithm})")
        print(f"{'='*60}")
        self._reseed_for_algorithm(alg_name)
        
        # Create agent with optimization mode
        if algorithm == "DQN":
            agent = CentralizedDQNAgent(self.state_dim, self.num_followers, self.device, 
                                       optimization_mode=self.optimization_target)
        elif algorithm == "DDPG":
            agent = CentralizedDDPGAgent(self.state_dim, self.action_dim, self.device,
                                        optimization_mode=self.optimization_target,
                                        mixed_time_weight=self.mixed_time_weight)
        elif algorithm == "SAC":
            agent = CentralizedSACAgent(self.state_dim, self.action_dim, self.device,
                                       optimization_mode=self.optimization_target,
                                       mixed_time_weight=self.mixed_time_weight)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Training metrics
        episode_rewards = []
        best_eval_reward = -float('inf')
        
        # Training loop
        for episode in range(self.episodes):
            state = self.env.reset()
            episode_metrics = self._new_episode_metrics()
            done = False
            step_count = 0
            
            # Reset noise for DDPG
            if algorithm == "DDPG" and hasattr(agent, 'reset_noise'):
                agent.reset_noise()
            
            while not done:
                # Select action
                if algorithm == "DQN":
                    action, _ = agent.select_action(state, explore=True)
                elif algorithm == "DDPG":
                    action = agent.select_action(state, add_noise=True)
                elif algorithm == "SAC":
                    action = agent.select_action(state, evaluate=False)
                
                # Step environment
                next_state, reward, done, info = self.env.step(action)
                
                # Track metrics
                self._accumulate_episode_metrics(episode_metrics, reward, info)
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train if ready
                if self._warmup_complete(agent):
                    if step_count % agent.update_every == 0:
                        if algorithm == "DQN":
                            agent.train(agent.config.get('num_updates', 1))
                            # Update target network for DQN
                            if agent.step_count % agent.target_update_every == 0:
                                agent.update_target_network()
                        else:
                            agent.train()  # DDPG/SAC use num_updates internally
                
                state = next_state
                step_count += 1
            
            # Update exploration parameters after episode
            if self._warmup_complete(agent):
                if hasattr(agent, 'update_exploration'):
                    agent.update_exploration()
                elif algorithm == "DQN" and hasattr(agent, 'update_epsilon'):
                    agent.update_epsilon()
                
            # Record metrics
            episode_rewards.append(episode_metrics['reward'])
            self._append_training_episode_metrics(alg_name, episode_metrics)
            
            # Evaluation
            if (episode + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate_centralized_agent_metrics(
                    agent,
                    algorithm,
                    num_episodes=self.checkpoint_eval_episodes,
                    stream='checkpoint',
                )
                self._append_eval_metrics(alg_name, eval_metrics)

                # Save each evaluated checkpoint.
                checkpoint_dir = self._checkpoint_dir(alg_name, episode + 1)
                checkpoint_path = os.path.join(checkpoint_dir, f'{alg_name.lower()}_model.pth')
                agent.save(checkpoint_path)
                self._write_checkpoint_meta(checkpoint_dir, alg_name, episode + 1, eval_metrics)
                
                # Save best model
                if eval_metrics['reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['reward']
                    model_path = os.path.join(self.save_dir, f'{alg_name.lower()}_best_model.pth')
                    agent.save(model_path)
                    self.best_models[alg_name]['path'] = model_path
                    self.best_models[alg_name]['reward'] = eval_metrics['reward']
                
                print(f"Episode {episode + 1}/{self.episodes}")
                print(f"  Training reward (last {self.eval_interval}): {np.mean(episode_rewards[-self.eval_interval:]):.2f}")
                print(f"  Evaluation reward: {eval_metrics['reward']:.2f}")
                print(f"  Eval penalized objective: {eval_metrics['penalized']:.2f}")
                print(f"  Eval violation count: {eval_metrics['constraint_violations']:.2f}")
                print(f"  Best eval reward: {best_eval_reward:.2f}")
        
        # Save final model
        final_path = os.path.join(self.save_dir, f'{alg_name.lower()}_final_model.pth')
        agent.save(final_path)
        
        print(f"\n{alg_name} training completed!")
        print(f"Best evaluation reward: {best_eval_reward:.2f}")
        
        return agent
    
    def train_decentralized_algorithm(self, algorithm: str, alg_name: str) -> Any:
        """Train a single decentralized algorithm"""
        print(f"\n{'='*60}")
        print(f"Training {alg_name} (Decentralized {algorithm})")
        print(f"{'='*60}")
        print(f"Number of agents: {self.num_followers}")
        self._reseed_for_algorithm(alg_name)
        
        # Create multi-agent system with optimization mode
        if algorithm == "DQN":
            multi_agent = MultiAgentDQN(self.local_state_size, self.num_followers, self.device,
                                      optimization_mode=self.optimization_target)
        elif algorithm == "DDPG":
            multi_agent = MultiAgentDDPG(self.local_state_size, self.num_followers, self.device,
                                       optimization_mode=self.optimization_target,
                                       mixed_time_weight=self.mixed_time_weight)
        elif algorithm == "SAC":
            multi_agent = MultiAgentSAC(self.local_state_size, self.num_followers, self.device,
                                      optimization_mode=self.optimization_target,
                                      mixed_time_weight=self.mixed_time_weight)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Training metrics
        episode_rewards = []
        best_eval_reward = -float('inf')
        
        # Training loop
        for episode in range(self.episodes):
            state = self.env.reset()
            episode_metrics = self._new_episode_metrics()
            done = False
            
            # Reset noise for DDPG
            if algorithm == "DDPG" and hasattr(multi_agent, 'reset_noise'):
                multi_agent.reset_noise()
            
            # Get initial local observations
            local_states = [self.env.get_local_observation(i) for i in range(self.num_followers)]
            
            while not done:
                acting_follower_indices = list(self.env.follower_indices)

                # Select actions
                if algorithm == "DQN":
                    action, _ = multi_agent.select_actions(local_states, explore=True)
                elif algorithm == "DDPG":
                    action = multi_agent.select_actions(local_states, explore=True)
                elif algorithm == "SAC":
                    action = multi_agent.select_actions(local_states, evaluate=False)
                
                # Step environment
                next_state, reward, done, info = self.env.step(action)
                
                # Track metrics
                self._accumulate_episode_metrics(episode_metrics, reward, info)
                
                # Slot-indexed next observations are used for the next action
                # selection step. Replay storage must instead stay aligned to the
                # physical follower that actually acted before any reselection.
                next_local_states = [self.env.get_local_observation(i) for i in range(self.num_followers)]
                transition_next_local_states, agent_dones = self._aligned_decentralized_next_states(
                    acting_follower_indices, done
                )
                
                # Store transition and train
                multi_agent.step(
                    local_states,
                    action,
                    reward,
                    transition_next_local_states,
                    agent_dones,
                )
                
                # Update target network for DQN.
                if algorithm == "DQN" and hasattr(multi_agent, 'shared_agent'):
                    if multi_agent.shared_agent.step_count % multi_agent.shared_agent.target_update_every == 0:
                        multi_agent.update_target_network()
                
                local_states = next_local_states
            
            # Update exploration parameters after episode
            if self._warmup_complete(multi_agent) and hasattr(multi_agent, 'update_exploration'):
                multi_agent.update_exploration()
            
            # Record metrics
            episode_rewards.append(episode_metrics['reward'])
            self._append_training_episode_metrics(alg_name, episode_metrics)
            
            # Evaluation
            if (episode + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate_decentralized_agent_metrics(
                    multi_agent,
                    algorithm,
                    num_episodes=self.checkpoint_eval_episodes,
                    stream='checkpoint',
                )
                self._append_eval_metrics(alg_name, eval_metrics)

                # Save each evaluated checkpoint.
                checkpoint_dir = self._checkpoint_dir(alg_name, episode + 1)
                multi_agent.save(checkpoint_dir)
                self._write_checkpoint_meta(checkpoint_dir, alg_name, episode + 1, eval_metrics)
                
                # Save best model
                if eval_metrics['reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['reward']
                    model_dir = os.path.join(self.save_dir, f'{alg_name.lower()}_best_model')
                    multi_agent.save(model_dir)
                    self.best_models[alg_name]['path'] = model_dir
                    self.best_models[alg_name]['reward'] = eval_metrics['reward']
                
                print(f"Episode {episode + 1}/{self.episodes}")
                print(f"  Training reward (last {self.eval_interval}): {np.mean(episode_rewards[-self.eval_interval:]):.2f}")
                print(f"  Evaluation reward: {eval_metrics['reward']:.2f}")
                print(f"  Eval penalized objective: {eval_metrics['penalized']:.2f}")
                print(f"  Eval violation count: {eval_metrics['constraint_violations']:.2f}")
                print(f"  Best eval reward: {best_eval_reward:.2f}")
                
                # Print exploration status for DQN
                if algorithm == "DQN":
                    if hasattr(multi_agent, 'shared_agent'):
                        print(f"  Shared epsilon: {multi_agent.shared_agent.epsilon:.3f}")
        
        # Save final model
        final_dir = os.path.join(self.save_dir, f'{alg_name.lower()}_final_model')
        multi_agent.save(final_dir)
        
        print(f"\n{alg_name} training completed!")
        print(f"Best evaluation reward: {best_eval_reward:.2f}")
        
        return multi_agent
    
    def evaluate_centralized_agent_metrics(self, agent: Any, algorithm: str, num_episodes: int = 10,
                                           stream: str = 'checkpoint') -> Dict[str, float]:
        """Evaluate centralized agent and return mean diagnostics."""
        episode_metrics_all: List[Dict[str, float]] = []
        with self._preserve_training_rng():
            for episode_idx in range(num_episodes):
                state = self.reset_for_evaluation(episode_idx, stream=stream)
                episode_metrics = self._new_episode_metrics()
                done = False
                
                while not done:
                    if algorithm == "DQN":
                        action, _ = agent.select_action(state, explore=False)
                    elif algorithm == "DDPG":
                        action = agent.select_action(state, add_noise=False)
                    elif algorithm == "SAC":
                        action = agent.select_action(state, evaluate=True)
                    
                    next_state, reward, done, info = self.env.step(action)
                    state = next_state
                    self._accumulate_episode_metrics(episode_metrics, reward, info)
                
                episode_metrics_all.append(episode_metrics)
        
        return {
            key: float(np.mean([ep[key] for ep in episode_metrics_all]))
            for key in episode_metrics_all[0].keys()
        }
    
    def evaluate_centralized_agent(self, agent: Any, algorithm: str, num_episodes: int = 10,
                                   stream: str = 'checkpoint') -> float:
        """Evaluate centralized agent performance."""
        return self.evaluate_centralized_agent_metrics(
            agent, algorithm, num_episodes=num_episodes, stream=stream
        )['reward']
    
    def evaluate_decentralized_agent_metrics(self, multi_agent: Any, algorithm: str, num_episodes: int = 10,
                                             stream: str = 'checkpoint') -> Dict[str, float]:
        """Evaluate decentralized multi-agent performance and return mean diagnostics."""
        episode_metrics_all: List[Dict[str, float]] = []
        with self._preserve_training_rng():
            for episode_idx in range(num_episodes):
                self.reset_for_evaluation(episode_idx, stream=stream)
                episode_metrics = self._new_episode_metrics()
                done = False
                local_states = [self.env.get_local_observation(i) for i in range(self.num_followers)]
                
                while not done:
                    if algorithm == "DQN":
                        action, _ = multi_agent.select_actions(local_states, explore=False)
                    elif algorithm == "DDPG":
                        action = multi_agent.select_actions(local_states, explore=False)
                    elif algorithm == "SAC":
                        action = multi_agent.select_actions(local_states, evaluate=True)
                    
                    _, reward, done, info = self.env.step(action)
                    local_states = [self.env.get_local_observation(i) for i in range(self.num_followers)]
                    self._accumulate_episode_metrics(episode_metrics, reward, info)
                
                episode_metrics_all.append(episode_metrics)
        
        return {
            key: float(np.mean([ep[key] for ep in episode_metrics_all]))
            for key in episode_metrics_all[0].keys()
        }

    def evaluate_decentralized_agent(self, multi_agent: Any, algorithm: str, num_episodes: int = 10,
                                     stream: str = 'checkpoint') -> float:
        """Evaluate decentralized multi-agent performance."""
        return self.evaluate_decentralized_agent_metrics(
            multi_agent, algorithm, num_episodes=num_episodes, stream=stream
        )['reward']
    
    def evaluate_baselines(
        self,
        num_episodes: int = 100,
        include_de_search: bool = False,
        de_maxiter: int = 25,
        de_popsize: int = 8,
        de_polish: bool = True,
        de_seed: int = 42,
        de_num_restarts: int = 2,
        de_powell_maxiter: int = 60,
        de_local_refine_top_k: int = 3,
        de_selection_mode: str = "penalized_objective",
        include_bnb_search: bool = False,
        bnb_max_nodes: int = 50000,
        bnb_incumbent_passes: int = 2,
        bnb_seed: int = 42,
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Evaluate baseline algorithms"""
        print(f"\nEvaluating Baseline Algorithms ({num_episodes} episodes each)...")
        
        baselines = BaselineAlgorithms(self.env.num_followers, self.env.max_power)
        baseline_methods = {
            'All-Leader': baselines.all_leader_strategy,
            'All-Base': baselines.all_base_station_strategy,
            'Balanced': baselines.balanced_strategy
        }
        baseline_search_objects = {}
        if include_de_search:
            de_search = DifferentialEvolutionOracle(
                self.env,
                maxiter=de_maxiter,
                popsize=de_popsize,
                polish=de_polish,
                seed=de_seed,
                num_restarts=de_num_restarts,
                powell_maxiter=de_powell_maxiter,
                local_refine_top_k=de_local_refine_top_k,
                selection_mode=de_selection_mode,
            )
            baseline_methods['DE-Search'] = de_search.select_action
            baseline_search_objects['DE-Search'] = de_search
        if include_bnb_search:
            bnb_search = PresetBranchAndBoundOracle(
                self.env,
                max_nodes=bnb_max_nodes,
                incumbent_passes=bnb_incumbent_passes,
                seed=bnb_seed,
            )
            baseline_methods['BnB-Search'] = bnb_search.select_action
            baseline_search_objects['BnB-Search'] = bnb_search
        
        results = {}
        metric_key = self._result_metric_key()
        
        with self._preserve_training_rng():
            for name, method in baseline_methods.items():
                print(f"  Evaluating {name}...", end='', flush=True)
                times = []
                energies = []
                mixeds = []
                rewards = []
                penalizeds = []
                objective_costs = []
                normalized_times = []
                normalized_energies = []
                constraint_violations = []
                constraint_penalties = []
                any_constraint_violations = []
                follower_time_violations = []
                leader_time_violations = []
                follower_energy_violations = []
                leader_energy_violations = []
                actions_collected = []
                
                # Timestep-level data collection
                timestep_times = []
                timestep_energies = []
                timestep_actions = []
                
                for episode_idx in range(num_episodes):
                    state = self.reset_for_evaluation(episode_idx, stream='baseline')
                    episode_metrics = self._new_episode_metrics()
                    done = False
                    
                    # Per-timestep data for this episode
                    ep_timestep_times = []
                    ep_timestep_energies = []
                    ep_timestep_actions = []
                    
                    while not done:
                        action = method(state)
                        actions_collected.append(action.copy())
                        next_state, reward, done, info = self.env.step(action)
                        
                        # Collect timestep data
                        step_time = info.get('time_cost', 0)
                        step_energy = info.get('energy_cost', 0)
                        ep_timestep_times.append(step_time)
                        ep_timestep_energies.append(step_energy)
                        ep_timestep_actions.append(action.copy())
                        
                        state = next_state
                        self._accumulate_episode_metrics(episode_metrics, reward, info)
                    
                    rewards.append(episode_metrics['reward'])
                    penalizeds.append(episode_metrics['penalized'])
                    objective_costs.append(episode_metrics['objective_cost'])
                    times.append(episode_metrics['time'])
                    energies.append(episode_metrics['energy'])
                    mixeds.append(episode_metrics['mixed'])
                    normalized_times.append(episode_metrics['normalized_time'])
                    normalized_energies.append(episode_metrics['normalized_energy'])
                    constraint_violations.append(episode_metrics['constraint_violations'])
                    constraint_penalties.append(episode_metrics['constraint_penalty'])
                    any_constraint_violations.append(episode_metrics['any_constraint_violations'])
                    follower_time_violations.append(episode_metrics['follower_time_violations'])
                    leader_time_violations.append(episode_metrics['leader_time_violations'])
                    follower_energy_violations.append(episode_metrics['follower_energy_violations'])
                    leader_energy_violations.append(episode_metrics['leader_energy_violations'])
                    
                    # Store timestep data
                    timestep_times.append(ep_timestep_times)
                    timestep_energies.append(ep_timestep_energies)
                    timestep_actions.append(ep_timestep_actions)
                
                results[name] = {
                    'reward': (np.mean(rewards), np.std(rewards)),
                    'penalized_objective': (np.mean(penalizeds), np.std(penalizeds)),
                    'objective_cost': (np.mean(objective_costs), np.std(objective_costs)),
                    'time': (np.mean(times), np.std(times)),
                    'energy': (np.mean(energies), np.std(energies)),
                    'mixed': (np.mean(mixeds), np.std(mixeds)),
                    'normalized_time': (np.mean(normalized_times), np.std(normalized_times)),
                    'normalized_energy': (np.mean(normalized_energies), np.std(normalized_energies)),
                    'constraint_violations': (np.mean(constraint_violations), np.std(constraint_violations)),
                    'constraint_penalty': (np.mean(constraint_penalties), np.std(constraint_penalties)),
                    'any_constraint_violations': (np.mean(any_constraint_violations), np.std(any_constraint_violations)),
                    'follower_time_violations': (np.mean(follower_time_violations), np.std(follower_time_violations)),
                    'leader_time_violations': (np.mean(leader_time_violations), np.std(leader_time_violations)),
                    'follower_energy_violations': (np.mean(follower_energy_violations), np.std(follower_energy_violations)),
                    'leader_energy_violations': (np.mean(leader_energy_violations), np.std(leader_energy_violations)),
                    'actions': np.array(actions_collected),
                    'timestep_times': timestep_times,
                    'timestep_energies': timestep_energies,
                    'timestep_actions': timestep_actions
                }
                results[name]['action_analysis'] = self.analyze_actions(results[name]['actions'], self.num_followers)
                search_object = baseline_search_objects.get(name)
                if search_object is not None and hasattr(search_object, 'aggregate_search_stats'):
                    results[name]['search_stats'] = dict(search_object.aggregate_search_stats)
                
                print(f" {self._objective_display_name()}: {results[name][metric_key][0]:.2f}±{results[name][metric_key][1]:.2f}")
        
        return results
    
    def final_evaluation(self, num_episodes: int = 100) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Final evaluation using best models"""
        print(f"\n{'='*80}")
        print(f"FINAL EVALUATION (Using Best Models)")
        print(f"{'='*80}")
        
        results = {}
        metric_key = self._result_metric_key()
        
        # Evaluate each available algorithm's best model
        ordered_algorithms = ['C-DQN', 'C-DDPG', 'C-SAC', 'D-DQN', 'D-DDPG', 'D-SAC']
        for alg_name in [name for name in ordered_algorithms if name in self.agents]:
            print(f"\nEvaluating {alg_name}...")
            
            # Determine if centralized or decentralized
            is_centralized = alg_name.startswith('C-')
            algorithm = alg_name[2:]  # Remove prefix
            
            # Load best model if available
            if self.best_models[alg_name]['path'] and os.path.exists(self.best_models[alg_name]['path']):
                print(f"  Loading best model (eval reward: {self.best_models[alg_name]['reward']:.2f})")
                self.agents[alg_name].load(self.best_models[alg_name]['path'])
            else:
                print(f"  Warning: No best model found, using final model")
            
            # Evaluate
            times = []
            energies = []
            mixeds = []
            rewards = []
            penalizeds = []
            objective_costs = []
            normalized_times = []
            normalized_energies = []
            constraint_violations = []
            constraint_penalties = []
            any_constraint_violations = []
            follower_time_violations = []
            leader_time_violations = []
            follower_energy_violations = []
            leader_energy_violations = []
            actions_collected = []
            
            # Timestep-level data collection
            timestep_times = []
            timestep_energies = []
            timestep_actions = []
            
            for episode_idx in range(num_episodes):
                state = self.reset_for_evaluation(episode_idx, stream='final')
                episode_metrics = self._new_episode_metrics()
                done = False
                
                # Per-timestep data for this episode
                ep_timestep_times = []
                ep_timestep_energies = []
                ep_timestep_actions = []
                
                # Get initial local observations for decentralized
                if not is_centralized:
                    local_states = [self.env.get_local_observation(i) for i in range(self.num_followers)]
                
                while not done:
                    # Select action without exploration
                    if is_centralized:
                        if algorithm == "DQN":
                            action, _ = self.agents[alg_name].select_action(state, explore=False)
                        elif algorithm == "DDPG":
                            action = self.agents[alg_name].select_action(state, add_noise=False)
                        elif algorithm == "SAC":
                            action = self.agents[alg_name].select_action(state, evaluate=True)
                    else:
                        if algorithm == "DQN":
                            action, _ = self.agents[alg_name].select_actions(local_states, explore=False)
                        elif algorithm == "DDPG":
                            action = self.agents[alg_name].select_actions(local_states, explore=False)
                        elif algorithm == "SAC":
                            action = self.agents[alg_name].select_actions(local_states, evaluate=True)
                    
                    # Store action for analysis
                    actions_collected.append(action.copy())
                    
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Collect timestep data
                    step_time = info.get('time_cost', 0)
                    step_energy = info.get('energy_cost', 0)
                    ep_timestep_times.append(step_time)
                    ep_timestep_energies.append(step_energy)
                    ep_timestep_actions.append(action.copy())
                    
                    if is_centralized:
                        state = next_state
                    else:
                        local_states = [self.env.get_local_observation(i) for i in range(self.num_followers)]
                    self._accumulate_episode_metrics(episode_metrics, reward, info)
                
                rewards.append(episode_metrics['reward'])
                penalizeds.append(episode_metrics['penalized'])
                objective_costs.append(episode_metrics['objective_cost'])
                times.append(episode_metrics['time'])
                energies.append(episode_metrics['energy'])
                mixeds.append(episode_metrics['mixed'])
                normalized_times.append(episode_metrics['normalized_time'])
                normalized_energies.append(episode_metrics['normalized_energy'])
                constraint_violations.append(episode_metrics['constraint_violations'])
                constraint_penalties.append(episode_metrics['constraint_penalty'])
                any_constraint_violations.append(episode_metrics['any_constraint_violations'])
                follower_time_violations.append(episode_metrics['follower_time_violations'])
                leader_time_violations.append(episode_metrics['leader_time_violations'])
                follower_energy_violations.append(episode_metrics['follower_energy_violations'])
                leader_energy_violations.append(episode_metrics['leader_energy_violations'])
                
                # Store timestep data
                timestep_times.append(ep_timestep_times)
                timestep_energies.append(ep_timestep_energies)
                timestep_actions.append(ep_timestep_actions)
            
            results[alg_name] = {
                'reward': (np.mean(rewards), np.std(rewards)),
                'penalized_objective': (np.mean(penalizeds), np.std(penalizeds)),
                'objective_cost': (np.mean(objective_costs), np.std(objective_costs)),
                'time': (np.mean(times), np.std(times)),
                'energy': (np.mean(energies), np.std(energies)),
                'mixed': (np.mean(mixeds), np.std(mixeds)),
                'normalized_time': (np.mean(normalized_times), np.std(normalized_times)),
                'normalized_energy': (np.mean(normalized_energies), np.std(normalized_energies)),
                'constraint_violations': (np.mean(constraint_violations), np.std(constraint_violations)),
                'constraint_penalty': (np.mean(constraint_penalties), np.std(constraint_penalties)),
                'any_constraint_violations': (np.mean(any_constraint_violations), np.std(any_constraint_violations)),
                'follower_time_violations': (np.mean(follower_time_violations), np.std(follower_time_violations)),
                'leader_time_violations': (np.mean(leader_time_violations), np.std(leader_time_violations)),
                'follower_energy_violations': (np.mean(follower_energy_violations), np.std(follower_energy_violations)),
                'leader_energy_violations': (np.mean(leader_energy_violations), np.std(leader_energy_violations)),
                'actions': np.array(actions_collected),
                'timestep_times': timestep_times,
                'timestep_energies': timestep_energies,
                'timestep_actions': timestep_actions
            }
            
            print(f"  Performance: {self._objective_display_name()}={results[alg_name][metric_key][0]:.2f}±{results[alg_name][metric_key][1]:.2f}")
            
            # Analyze and print action distributions
            action_analysis = self.analyze_actions(results[alg_name]['actions'], self.num_followers)
            print(f"\n  Action Analysis:")
            print(f"    V2V Fraction: {action_analysis['v2v_fraction']['mean']:.3f}±{action_analysis['v2v_fraction']['std']:.3f}")
            print(f"    V2V Power (executed active): {action_analysis['v2v_power']['mean']:.3f}±{action_analysis['v2v_power']['std']:.3f}")
            print(f"    V2I Power (executed active): {action_analysis['v2i_power']['mean']:.3f}±{action_analysis['v2i_power']['std']:.3f}")
            
            # Store action analysis in results
            results[alg_name]['action_analysis'] = action_analysis
        
        return results
    
    def plot_results(self, baseline_results: Dict, rl_results: Dict):
        """Create comprehensive visualization of results"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(24, 16))
        metric_key = self._result_metric_key()
        training_metric_key = self._training_metric_key()
        objective_axis_label = self._objective_axis_label()
        objective_display_name = self._objective_display_name()
        
        # Define colors for consistency
        colors_cent = {'C-DQN': 'blue', 'C-DDPG': 'green', 'C-SAC': 'red'}
        colors_decent = {'D-DQN': 'lightblue', 'D-DDPG': 'lightgreen', 'D-SAC': 'lightcoral'}
        
        # 1. Training curves - Centralized
        ax1 = plt.subplot(3, 3, 1)
        for alg in ['C-DQN', 'C-DDPG', 'C-SAC']:
            metric_data = self.training_metrics[alg][training_metric_key]
            if len(metric_data) > 0:
                window = min(50, len(metric_data))
                if len(metric_data) >= window:
                    ma = np.convolve(metric_data, np.ones(window)/window, mode='valid')
                    ax1.plot(range(window-1, len(metric_data)), ma, label=alg, 
                            linewidth=2, color=colors_cent[alg])
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel(objective_axis_label)
        ax1.set_title(f'Centralized Training Progress')
        ax1.legend()
        
        # 2. Training curves - Decentralized
        ax2 = plt.subplot(3, 3, 2)
        for alg in ['D-DQN', 'D-DDPG', 'D-SAC']:
            metric_data = self.training_metrics[alg][training_metric_key]
            if len(metric_data) > 0:
                window = min(50, len(metric_data))
                if len(metric_data) >= window:
                    ma = np.convolve(metric_data, np.ones(window)/window, mode='valid')
                    ax2.plot(range(window-1, len(metric_data)), ma, label=alg, 
                            linewidth=2, color=colors_decent[alg])
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel(objective_axis_label)
        ax2.set_title(f'Decentralized Training Progress')
        ax2.legend()
        
        # 3. Algorithm comparison
        ax3 = plt.subplot(3, 3, 3)
        algorithms = list(baseline_results.keys()) + list(rl_results.keys())
        means = []
        stds = []
        colors = []
        
        for alg in algorithms:
            if alg in baseline_results:
                mean, std = baseline_results[alg][metric_key]
                colors.append('gray')
            else:
                mean, std = rl_results[alg][metric_key]
                if alg.startswith('C-'):
                    colors.append(colors_cent[alg])
                else:
                    colors.append(colors_decent[alg])
            means.append(mean)
            stds.append(std)
        
        bars = ax3.bar(algorithms, means, yerr=stds, color=colors, capsize=5, edgecolor='black')
        
        # Highlight best
        best_idx = np.argmin(means)
        bars[best_idx].set_color('gold')
        
        ax3.set_ylabel(objective_axis_label)
        ax3.set_title(f'{objective_display_name} Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(means)*0.01,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Centralized vs Decentralized Direct Comparison
        ax4 = plt.subplot(3, 3, 4)
        cent_algs = ['C-DQN', 'C-DDPG', 'C-SAC']
        decent_algs = ['D-DQN', 'D-DDPG', 'D-SAC']
        base_algs = ['DQN', 'DDPG', 'SAC']
        
        x = np.arange(len(base_algs))
        width = 0.35
        
        cent_means = [rl_results[f'C-{alg}'][metric_key][0] for alg in base_algs]
        cent_stds = [rl_results[f'C-{alg}'][metric_key][1] for alg in base_algs]
        decent_means = [rl_results[f'D-{alg}'][metric_key][0] for alg in base_algs]
        decent_stds = [rl_results[f'D-{alg}'][metric_key][1] for alg in base_algs]
        
        bars1 = ax4.bar(x - width/2, cent_means, width, yerr=cent_stds, 
                        label='Centralized', color='skyblue', capsize=5)
        bars2 = ax4.bar(x + width/2, decent_means, width, yerr=decent_stds, 
                        label='Decentralized', color='lightgreen', capsize=5)
        
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel(objective_axis_label)
        ax4.set_title(f'Centralized vs Decentralized {objective_display_name}')
        ax4.set_xticks(x)
        ax4.set_xticklabels(base_algs)
        ax4.legend()
        
        # 5. Action distribution - V2V Fraction (Centralized)
        ax5 = plt.subplot(3, 3, 5)
        for alg in ['C-DQN', 'C-DDPG', 'C-SAC']:
            if alg in rl_results and 'action_analysis' in rl_results[alg]:
                data = rl_results[alg]['action_analysis']['v2v_fraction']['distribution']
                ax5.hist(data, bins=30, alpha=0.6, label=alg, density=True, color=colors_cent[alg])
        
        ax5.set_xlabel('V2V Fraction (δ)')
        ax5.set_ylabel('Density')
        ax5.set_title('V2V Fraction Distribution (Centralized)')
        ax5.legend()
        ax5.set_xlim(0, 1)
        
        # 6. Action distribution - V2V Fraction (Decentralized)
        ax6 = plt.subplot(3, 3, 6)
        for alg in ['D-DQN', 'D-DDPG', 'D-SAC']:
            if alg in rl_results and 'action_analysis' in rl_results[alg]:
                data = rl_results[alg]['action_analysis']['v2v_fraction']['distribution']
                ax6.hist(data, bins=30, alpha=0.6, label=alg, density=True, color=colors_decent[alg])
        
        ax6.set_xlabel('V2V Fraction (δ)')
        ax6.set_ylabel('Density')
        ax6.set_title('V2V Fraction Distribution (Decentralized)')
        ax6.legend()
        ax6.set_xlim(0, 1)
        
        # 7. Action means comparison - Centralized
        ax7 = plt.subplot(3, 3, 7)
        action_types = ['V2V Fraction', 'V2V Power (active)', 'V2I Power (active)']
        action_keys = ['v2v_fraction', 'v2v_power', 'v2i_power']
        
        x = np.arange(len(action_types))
        width = 0.25
        
        for i, alg in enumerate(['C-DQN', 'C-DDPG', 'C-SAC']):
            if alg in rl_results and 'action_analysis' in rl_results[alg]:
                means = [rl_results[alg]['action_analysis'][key]['mean'] for key in action_keys]
                stds = [rl_results[alg]['action_analysis'][key]['std'] for key in action_keys]
                ax7.bar(x + i*width, means, width, yerr=stds, label=alg, 
                       capsize=5, color=colors_cent[alg])
        
        ax7.set_xlabel('Action Component')
        ax7.set_ylabel('Mean Value')
        ax7.set_title('Action Means (Centralized, executed active-link powers)')
        ax7.set_xticks(x + width)
        ax7.set_xticklabels(action_types)
        ax7.legend()
        ax7.set_ylim(0, 1.2)
        
        # 8. Action means comparison - Decentralized
        ax8 = plt.subplot(3, 3, 8)
        
        for i, alg in enumerate(['D-DQN', 'D-DDPG', 'D-SAC']):
            if alg in rl_results and 'action_analysis' in rl_results[alg]:
                means = [rl_results[alg]['action_analysis'][key]['mean'] for key in action_keys]
                stds = [rl_results[alg]['action_analysis'][key]['std'] for key in action_keys]
                ax8.bar(x + i*width, means, width, yerr=stds, label=alg, 
                       capsize=5, color=colors_decent[alg])
        
        ax8.set_xlabel('Action Component')
        ax8.set_ylabel('Mean Value')
        ax8.set_title('Action Means (Decentralized, executed active-link powers)')
        ax8.set_xticks(x + width)
        ax8.set_xticklabels(action_types)
        ax8.legend()
        ax8.set_ylim(0, 1.2)
        
        # 9. Paradigm comparison summary
        ax9 = plt.subplot(3, 3, 9)
        
        # Calculate average performance for each paradigm
        cent_avg = np.mean([rl_results[alg][metric_key][0] 
                           for alg in ['C-DQN', 'C-DDPG', 'C-SAC']])
        decent_avg = np.mean([rl_results[alg][metric_key][0] 
                             for alg in ['D-DQN', 'D-DDPG', 'D-SAC']])
        baseline_avg = np.mean([baseline_results[alg][metric_key][0] 
                               for alg in baseline_results.keys()])
        
        paradigms = ['Baseline', 'Centralized', 'Decentralized']
        avgs = [baseline_avg, cent_avg, decent_avg]
        colors_par = ['gray', 'skyblue', 'lightgreen']
        
        bars = ax9.bar(paradigms, avgs, color=colors_par, edgecolor='black')
        ax9.set_ylabel(f'Average {objective_display_name}')
        ax9.set_title('Paradigm Average Objective')
        
        # Add value labels
        for bar, avg in zip(bars, avgs):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + max(avgs)*0.01,
                    f'{avg:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'combined_comprehensive_results.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional specialized plots
        self.plot_action_heatmaps(rl_results)
        timestep_data = self.plot_timestep_analysis(baseline_results, rl_results)
        self.plot_summary_table(baseline_results, rl_results)
        
        # Store timestep data for later use
        self.timestep_data = timestep_data
    
    def plot_action_heatmaps(self, rl_results: Dict):
        """Create heatmaps comparing centralized and decentralized action distributions"""
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        
        # Top row: Centralized
        for idx, alg in enumerate(['C-DQN', 'C-DDPG', 'C-SAC']):
            if alg not in rl_results or 'action_analysis' not in rl_results[alg]:
                continue
            
            # Extract action data
            analysis = rl_results[alg]['action_analysis']
            
            # Create 2D histogram for V2V fraction vs V2V power
            v2v_frac = analysis['v2v_fraction']['distribution']
            v2v_power = analysis['v2v_power']['distribution']
            
            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(v2v_frac, v2v_power, bins=20)
            
            # Plot heatmap
            im = axes[0, idx].imshow(H.T, origin='lower', aspect='auto', 
                                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                    cmap='YlOrRd')
            
            axes[0, idx].set_xlabel('V2V Fraction (δ)')
            axes[0, idx].set_ylabel('V2V Power')
            axes[0, idx].set_title(f'{alg} Action Distribution')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[0, idx])
            cbar.set_label('Frequency')
        
        # Bottom row: Decentralized
        for idx, alg in enumerate(['D-DQN', 'D-DDPG', 'D-SAC']):
            if alg not in rl_results or 'action_analysis' not in rl_results[alg]:
                continue
            
            # Extract action data
            analysis = rl_results[alg]['action_analysis']
            
            # Create 2D histogram for V2V fraction vs V2V power
            v2v_frac = analysis['v2v_fraction']['distribution']
            v2v_power = analysis['v2v_power']['distribution']
            
            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(v2v_frac, v2v_power, bins=20)
            
            # Plot heatmap
            im = axes[1, idx].imshow(H.T, origin='lower', aspect='auto', 
                                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                    cmap='YlOrRd')
            
            axes[1, idx].set_xlabel('V2V Fraction (δ)')
            axes[1, idx].set_ylabel('V2V Power')
            axes[1, idx].set_title(f'{alg} Action Distribution')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, idx])
            cbar.set_label('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'combined_action_heatmaps.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_timestep_analysis(self, baseline_results: Dict, rl_results: Dict):
        """Create time-slot level analysis plots"""
        print("\nCreating timestep analysis plots...")
        
        # Combine all results
        all_results = {**baseline_results, **rl_results}
        
        # Find the maximum episode length across all algorithms
        max_timesteps = 0
        for name, results in all_results.items():
            if 'timestep_times' in results:
                for episode_times in results['timestep_times']:
                    max_timesteps = max(max_timesteps, len(episode_times))
        
        # Prepare data for each timestep
        timestep_data = {name: {'times': [], 'energies': [], 'v2v_fractions': [], 'v2v_powers': [], 'v2i_powers': []} 
                        for name in all_results.keys()}
        
        for timestep in range(max_timesteps):
            for name, results in all_results.items():
                if 'timestep_times' not in results:
                    continue
                
                # Collect data for this timestep across episodes
                times_at_t = []
                energies_at_t = []
                actions_at_t = []
                
                for ep_idx, episode_times in enumerate(results['timestep_times']):
                    if timestep < len(episode_times):
                        times_at_t.append(results['timestep_times'][ep_idx][timestep])
                        energies_at_t.append(results['timestep_energies'][ep_idx][timestep])
                        actions_at_t.append(results['timestep_actions'][ep_idx][timestep])
                
                if times_at_t:
                    timestep_data[name]['times'].append((np.mean(times_at_t), np.std(times_at_t)))
                    timestep_data[name]['energies'].append((np.mean(energies_at_t), np.std(energies_at_t)))
                    
                    # Analyze actions at this timestep
                    if actions_at_t:
                        actions_array = np.array(actions_at_t)
                        actions_reshaped = actions_array.reshape(-1, self.num_followers, 3)
                        timestep_data[name]['v2v_fractions'].append(
                            (np.mean(actions_reshaped[:, :, 0]), np.std(actions_reshaped[:, :, 0])))
                        timestep_data[name]['v2v_powers'].append(
                            (np.mean(actions_reshaped[:, :, 1]), np.std(actions_reshaped[:, :, 1])))
                        timestep_data[name]['v2i_powers'].append(
                            (np.mean(actions_reshaped[:, :, 2]), np.std(actions_reshaped[:, :, 2])))
        
        # Create figure for paradigm comparison
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        
        # Colors for different algorithms
        colors = {
            'All-Leader': 'red', 'All-Base': 'blue', 'Balanced': 'green',
            'C-DQN': 'darkblue', 'C-DDPG': 'darkgreen', 'C-SAC': 'darkred',
            'D-DQN': 'lightblue', 'D-DDPG': 'lightgreen', 'D-SAC': 'lightcoral'
        }
        
        # Plot time costs
        for name in all_results.keys():
            if timestep_data[name]['times']:
                timesteps = range(len(timestep_data[name]['times']))
                means = [x[0] for x in timestep_data[name]['times']]
                
                if name.startswith('C-'):
                    linestyle = '-'
                elif name.startswith('D-'):
                    linestyle = '--'
                else:
                    linestyle = ':'
                
                ax1.plot(timesteps, means, label=name, color=colors.get(name, 'gray'), 
                        linewidth=2, linestyle=linestyle)
        
        ax1.set_xlabel('Time Slot')
        ax1.set_ylabel('Time Cost (s)')
        ax1.set_title('Time Cost Evolution: Centralized (solid) vs Decentralized (dashed)')
        ax1.legend(ncol=3)
        ax1.grid(True, alpha=0.3)
        
        # Plot energy costs
        for name in all_results.keys():
            if timestep_data[name]['energies']:
                timesteps = range(len(timestep_data[name]['energies']))
                means = [x[0] for x in timestep_data[name]['energies']]
                
                if name.startswith('C-'):
                    linestyle = '-'
                elif name.startswith('D-'):
                    linestyle = '--'
                else:
                    linestyle = ':'
                
                ax2.plot(timesteps, means, label=name, color=colors.get(name, 'gray'), 
                        linewidth=2, linestyle=linestyle)
        
        ax2.set_xlabel('Time Slot')
        ax2.set_ylabel('Energy Cost (J)')
        ax2.set_title('Energy Cost Evolution: Centralized (solid) vs Decentralized (dashed)')
        ax2.legend(ncol=3)
        ax2.grid(True, alpha=0.3)
        
        # Plot V2V fraction evolution
        for name in all_results.keys():
            if timestep_data[name]['v2v_fractions']:
                timesteps = range(len(timestep_data[name]['v2v_fractions']))
                means = [x[0] for x in timestep_data[name]['v2v_fractions']]
                
                if name.startswith('C-'):
                    linestyle = '-'
                elif name.startswith('D-'):
                    linestyle = '--'
                else:
                    linestyle = ':'
                
                ax3.plot(timesteps, means, label=name, color=colors.get(name, 'gray'), 
                        linewidth=2, linestyle=linestyle)
        
        ax3.set_xlabel('Time Slot')
        ax3.set_ylabel('V2V Fraction (δ)')
        ax3.set_title('V2V Offloading Strategy Evolution: Centralized (solid) vs Decentralized (dashed)')
        ax3.legend(ncol=3)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.1, 1.1)
        
        # Plot V2V power evolution
        for name in all_results.keys():
            if timestep_data[name]['v2v_powers']:
                timesteps = range(len(timestep_data[name]['v2v_powers']))
                means = [x[0] for x in timestep_data[name]['v2v_powers']]
                
                if name.startswith('C-'):
                    linestyle = '-'
                elif name.startswith('D-'):
                    linestyle = '--'
                else:
                    linestyle = ':'
                
                ax4.plot(timesteps, means, label=name, color=colors.get(name, 'gray'), 
                        linewidth=2, linestyle=linestyle)
        
        ax4.set_xlabel('Time Slot')
        ax4.set_ylabel('V2V Power')
        ax4.set_title('V2V Power Allocation Evolution: Centralized (solid) vs Decentralized (dashed)')
        ax4.legend(ncol=3)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-0.1, 1.1)
        
        # Plot V2I power evolution
        for name in all_results.keys():
            if timestep_data[name]['v2i_powers']:
                timesteps = range(len(timestep_data[name]['v2i_powers']))
                means = [x[0] for x in timestep_data[name]['v2i_powers']]
                
                if name.startswith('C-'):
                    linestyle = '-'
                elif name.startswith('D-'):
                    linestyle = '--'
                else:
                    linestyle = ':'
                
                ax5.plot(timesteps, means, label=name, color=colors.get(name, 'gray'), 
                        linewidth=2, linestyle=linestyle)
        
        ax5.set_xlabel('Time Slot')
        ax5.set_ylabel('V2I Power')
        ax5.set_title('V2I Power Allocation Evolution: Centralized (solid) vs Decentralized (dashed)')
        ax5.legend(ncol=3)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(-0.1, 1.1)
        
        # Empty subplot for better layout
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'combined_timestep_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save timestep data as CSV files
        print("  - Saving timestep data as CSV files...")
        
        # Get all algorithm names (ordered)
        algorithm_names = sorted(all_results.keys())
        
        # Save timestep times CSV
        with open(os.path.join(self.save_dir, 'timestep_times.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            # Header: timestep, alg1_mean, alg1_std, alg2_mean, alg2_std, ...
            header = ['timestep']
            for name in algorithm_names:
                header.extend([f'{name}_mean', f'{name}_std'])
            writer.writerow(header)
            
            # Write data for each timestep
            for t in range(max_timesteps):
                row = [t]
                for name in algorithm_names:
                    if t < len(timestep_data[name]['times']):
                        mean, std = timestep_data[name]['times'][t]
                        row.extend([mean, std])
                    else:
                        row.extend(['', ''])
                writer.writerow(row)
        
        # Save timestep energies CSV
        with open(os.path.join(self.save_dir, 'timestep_energies.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestep']
            for name in algorithm_names:
                header.extend([f'{name}_mean', f'{name}_std'])
            writer.writerow(header)
            
            for t in range(max_timesteps):
                row = [t]
                for name in algorithm_names:
                    if t < len(timestep_data[name]['energies']):
                        mean, std = timestep_data[name]['energies'][t]
                        row.extend([mean, std])
                    else:
                        row.extend(['', ''])
                writer.writerow(row)
        
        # Save timestep V2V fraction CSV
        with open(os.path.join(self.save_dir, 'timestep_v2v_fraction.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestep']
            for name in algorithm_names:
                header.extend([f'{name}_mean', f'{name}_std'])
            writer.writerow(header)
            
            for t in range(max_timesteps):
                row = [t]
                for name in algorithm_names:
                    if t < len(timestep_data[name]['v2v_fractions']):
                        mean, std = timestep_data[name]['v2v_fractions'][t]
                        row.extend([mean, std])
                    else:
                        row.extend(['', ''])
                writer.writerow(row)
        
        # Save timestep V2V power CSV
        with open(os.path.join(self.save_dir, 'timestep_v2v_power.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestep']
            for name in algorithm_names:
                header.extend([f'{name}_mean', f'{name}_std'])
            writer.writerow(header)
            
            for t in range(max_timesteps):
                row = [t]
                for name in algorithm_names:
                    if t < len(timestep_data[name]['v2v_powers']):
                        mean, std = timestep_data[name]['v2v_powers'][t]
                        row.extend([mean, std])
                    else:
                        row.extend(['', ''])
                writer.writerow(row)
        
        # Save timestep V2I power CSV
        with open(os.path.join(self.save_dir, 'timestep_v2i_power.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestep']
            for name in algorithm_names:
                header.extend([f'{name}_mean', f'{name}_std'])
            writer.writerow(header)
            
            for t in range(max_timesteps):
                row = [t]
                for name in algorithm_names:
                    if t < len(timestep_data[name]['v2i_powers']):
                        mean, std = timestep_data[name]['v2i_powers'][t]
                        row.extend([mean, std])
                    else:
                        row.extend(['', ''])
                writer.writerow(row)
        
        print("Timestep analysis plots and CSV files saved!")
        
        # Return timestep_data for use in save_all_plot_data
        return timestep_data
    
    def plot_summary_table(self, baseline_results: Dict, rl_results: Dict):
        """Create comprehensive summary table"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        metric_key = self._result_metric_key()
        
        # Create summary data
        all_results = {**baseline_results, **rl_results}
        algorithms = list(all_results.keys())
        summary_data = []
        
        # Add header row
        header = ['Algorithm', 'Paradigm', 'Objective', 'Time (s)', 'Energy (J)', 'V2V Frac', 'V2V Power (active)', 'V2I Power (active)']
        
        for name in algorithms:
            results = all_results[name]
            
            # Determine paradigm
            if name in baseline_results:
                paradigm = 'Baseline'
            elif name.startswith('C-'):
                paradigm = 'Centralized'
            else:
                paradigm = 'Decentralized'
            
            row = [name, paradigm]
            
            mean, std = results[metric_key]
            row.append(f"{mean:.3f}±{std:.3f}")

            # Add performance metrics
            for metric in ['time', 'energy']:
                mean, std = results[metric]
                row.append(f"{mean:.2f}±{std:.2f}")
            
            # Get action stats if available
            if 'action_analysis' in results:
                analysis = results['action_analysis']
                row.append(f"{analysis['v2v_fraction']['mean']:.3f}±{analysis['v2v_fraction']['std']:.3f}")
                row.append(f"{analysis['v2v_power']['mean']:.3f}±{analysis['v2v_power']['std']:.3f}")
                row.append(f"{analysis['v2i_power']['mean']:.3f}±{analysis['v2i_power']['std']:.3f}")
            else:
                # Fixed baseline strategies
                if name == 'All-Leader':
                    row.extend(["1.000±0.000", "1.000±0.000", "0.000±0.000"])
                elif name == 'All-Base':
                    row.extend(["0.000±0.000", "0.000±0.000", "1.000±0.000"])
                elif name == 'Balanced':
                    row.extend(["0.500±0.000", "1.000±0.000", "1.000±0.000"])
                else:
                    row.extend(["-", "-", "-"])
            
            summary_data.append(row)
        
        # Create table
        table = ax.table(cellText=summary_data,
                        colLabels=header,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color code by paradigm
        for i, row in enumerate(summary_data):
            if row[1] == 'Baseline':
                color = '#f0f0f0'
            elif row[1] == 'Centralized':
                color = '#e6f2ff'
            else:  # Decentralized
                color = '#f0fff0'
            
            for j in range(len(row)):
                table[(i + 1, j)].set_facecolor(color)
        
        # Highlight best in each column
        for j in range(2, 5):  # Objective, Time, and Energy columns
            vals = []
            for i, row in enumerate(summary_data):
                try:
                    val = float(row[j].split('±')[0])
                    vals.append((i, val))
                except:
                    pass
            
            if vals:
                best_idx = min(vals, key=lambda x: x[1])[0]
                table[(best_idx + 1, j)].set_facecolor('#ffff90')
        
        plt.title('Combined Algorithm Performance Summary', fontsize=16, pad=20)
        plt.savefig(os.path.join(self.save_dir, 'combined_summary_table.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_all_plot_data(self, baseline_results: Dict, rl_results: Dict):
        """Save all plot data for LaTeX visualization"""
        print("\nSaving all plot data for LaTeX...")
        metric_key = self._result_metric_key()
        
        # 1. Save training curves data
        print("  - Saving training curves data...")
        with open(os.path.join(self.save_dir, 'training_rewards.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'C-DQN', 'C-DDPG', 'C-SAC', 'D-DQN', 'D-DDPG', 'D-SAC'])
            max_len = max(len(self.training_metrics[alg]['rewards']) 
                         for alg in self.training_metrics.keys())
            for i in range(max_len):
                row = [i]
                for alg in ['C-DQN', 'C-DDPG', 'C-SAC', 'D-DQN', 'D-DDPG', 'D-SAC']:
                    if i < len(self.training_metrics[alg]['rewards']):
                        row.append(self.training_metrics[alg]['rewards'][i])
                    else:
                        row.append('')
                writer.writerow(row)
        
        # 2. Save algorithm comparison data
        print("  - Saving algorithm comparison data...")
        all_results = {**baseline_results, **rl_results}
        with open(os.path.join(self.save_dir, 'algorithm_comparison.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['algorithm', 'paradigm', 'objective_mean', 'objective_std', 'time_mean', 'time_std', 'energy_mean', 'energy_std'])
            for name, results in all_results.items():
                if name in baseline_results:
                    paradigm = 'baseline'
                elif name.startswith('C-'):
                    paradigm = 'centralized'
                else:
                    paradigm = 'decentralized'
                
                writer.writerow([
                    name, paradigm,
                    results[metric_key][0], results[metric_key][1],
                    results['time'][0], results['time'][1],
                    results['energy'][0], results['energy'][1]
                ])
        
        # 3. Save paradigm comparison data
        print("  - Saving paradigm comparison data...")
        with open(os.path.join(self.save_dir, 'paradigm_comparison.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['algorithm_base', 'cent_objective', 'cent_time', 'cent_energy', 'decent_objective', 'decent_time', 'decent_energy'])
            for base in ['DQN', 'DDPG', 'SAC']:
                cent = rl_results[f'C-{base}']
                decent = rl_results[f'D-{base}']
                writer.writerow([
                    base,
                    cent[metric_key][0],
                    cent['time'][0], cent['energy'][0],
                    decent[metric_key][0],
                    decent['time'][0], decent['energy'][0]
                ])
        
        # 4. Save comprehensive NPZ file
        print("  - Saving comprehensive NPZ file...")
        npz_data = {
            # Training metrics
            **{f"training_{alg}_{metric}": np.array(data) 
               for alg, metrics in self.training_metrics.items()
               for metric, data in metrics.items()},
            
            # Final evaluation results
            **{f"final_{name}_{metric}_mean": results[metric][0]
               for name, results in all_results.items()
               for metric in ['mixed', 'time', 'energy']},
            **{f"final_{name}_{metric}_std": results[metric][1]
               for name, results in all_results.items()
               for metric in ['mixed', 'time', 'energy']},
        }
        
        # Add timestep data if available
        if hasattr(self, 'timestep_data') and self.timestep_data:
            print("  - Adding timestep data to NPZ file...")
            for alg_name, alg_data in self.timestep_data.items():
                for data_type in ['times', 'energies', 'v2v_fractions', 'v2v_powers', 'v2i_powers']:
                    if alg_data[data_type]:
                        # Convert list of tuples to numpy arrays
                        means = np.array([x[0] for x in alg_data[data_type]])
                        stds = np.array([x[1] for x in alg_data[data_type]])
                        npz_data[f"timestep_{alg_name}_{data_type}_mean"] = means
                        npz_data[f"timestep_{alg_name}_{data_type}_std"] = stds
        
        np.savez_compressed(os.path.join(self.save_dir, 'combined_all_plot_data.npz'), **npz_data)
        
        print("  - All plot data saved successfully!")
        print(f"  - Files saved in: {self.save_dir}")
    
    def plot_training_rewards(self):
        """Plot training rewards for all algorithms"""
        print("\nPlotting training rewards...")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Define colors for each algorithm
        colors = {
            'C-DQN': 'blue',
            'C-DDPG': 'green',
            'C-SAC': 'red',
            'D-DQN': 'cyan',
            'D-DDPG': 'lime',
            'D-SAC': 'orange'
        }
        
        # Plot raw rewards on the left
        ax1.set_title('Training Rewards (Raw)', fontsize=14)
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        for alg in ['C-DQN', 'C-DDPG', 'C-SAC', 'D-DQN', 'D-DDPG', 'D-SAC']:
            if alg in self.training_metrics:
                rewards = self.training_metrics[alg]['rewards']
                episodes = range(len(rewards))
                linestyle = '-' if alg.startswith('C-') else '--'
                ax1.plot(episodes, rewards, color=colors[alg], label=alg, 
                        linewidth=1.5, linestyle=linestyle, alpha=0.7)
        
        ax1.legend(loc='lower right')
        
        # Plot smoothed rewards on the right (moving average)
        ax2.set_title('Training Rewards (Smoothed, window=25)', fontsize=14)
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Reward', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        window_size = 25
        for alg in ['C-DQN', 'C-DDPG', 'C-SAC', 'D-DQN', 'D-DDPG', 'D-SAC']:
            if alg in self.training_metrics:
                rewards = self.training_metrics[alg]['rewards']
                if len(rewards) >= window_size:
                    # Calculate moving average
                    smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    episodes = range(window_size-1, len(rewards))
                    linestyle = '-' if alg.startswith('C-') else '--'
                    ax2.plot(episodes, smoothed, color=colors[alg], label=alg,
                            linewidth=2, linestyle=linestyle)
        
        ax2.legend(loc='lower right')
        
        # Add suptitle
        plt.suptitle(f'Training Progress - {self._objective_display_name()}', fontsize=16)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.save_dir, 'training_rewards.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.save_dir, 'training_rewards.pdf'), bbox_inches='tight')
        print(f"  - Training rewards plot saved to: {os.path.join(self.save_dir, 'training_rewards.png')}")
        
        # Close the figure to free memory
        plt.close()
    
    def save_results(self, baseline_results: Dict, rl_results: Dict):
        """Save all results to files"""
        # Convert numpy types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_native(v) for v in obj)
            else:
                return obj
        
        # Prepare comprehensive results
        all_results = {
            'optimization_target': self.optimization_target,
            'episodes': self.episodes,
            'eval_interval': self.eval_interval,
            'seed': self.seed,
            'device': self.device,
            'timestamp': datetime.now().isoformat(),
            'config_snapshot': convert_to_native(self.config),
            'baseline_results': convert_to_native(baseline_results),
            'rl_results': convert_to_native(rl_results),
            'best_models': convert_to_native(self.best_models),
            'training_summary': {
                alg: {
                    'final_reward': float(self.training_metrics[alg]['rewards'][-1]) if self.training_metrics[alg]['rewards'] else 0,
                    'best_eval_reward': float(max(self.training_metrics[alg]['eval_rewards'])) if self.training_metrics[alg]['eval_rewards'] else 0,
                    'mean_last_100_rewards': (
                        float(np.mean(self.training_metrics[alg]['rewards'][-100:]))
                        if len(self.training_metrics[alg]['rewards']) >= 100
                        else (
                            float(np.mean(self.training_metrics[alg]['rewards']))
                            if self.training_metrics[alg]['rewards']
                            else 0.0
                        )
                    ),
                }
                for alg in self.training_metrics.keys()
            }
        }
        
        # Save JSON results
        with open(os.path.join(self.save_dir, 'combined_comprehensive_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)

        with open(os.path.join(self.save_dir, 'experiment_config.json'), 'w') as f:
            json.dump(convert_to_native(self.config), f, indent=4)
        
        # Save training curves data
        np.savez(os.path.join(self.save_dir, 'combined_training_data.npz'),
                 **{f"{alg}_{metric}": np.array(data) 
                    for alg, metrics in self.training_metrics.items()
                    for metric, data in metrics.items()})
        
        # Save all plot data for LaTeX
        self.save_all_plot_data(baseline_results, rl_results)
        
        # Plot training rewards
        self.plot_training_rewards()
        
        print(f"\nAll results saved to: {self.save_dir}")
    
    def run(self):
        """Run the complete training and evaluation pipeline"""
        print("="*80)
        print(f"COMBINED CENTRALIZED AND DECENTRALIZED TRAINING AND EVALUATION")
        print(f"Optimization Target: {self._objective_display_name()}")
        print(f"Episodes: {self.episodes}")
        print(f"Device: {self.device}")
        print(f"Seed: {self.seed}")
        if self.optimization_target == 'mixed':
            print(f"Mixed objective weight eta: {self.mixed_time_weight:.2f}")
        print("="*80)
        
        start_time = time.time()
        
        # Train all centralized algorithms
        print("\n" + "="*80)
        print("TRAINING CENTRALIZED ALGORITHMS")
        print("="*80)
        
        for algorithm in ['DQN', 'DDPG', 'SAC']:
            alg_name = f'C-{algorithm}'
            self.agents[alg_name] = self.train_centralized_algorithm(algorithm, alg_name)
        
        # Train all decentralized algorithms
        print("\n" + "="*80)
        print("TRAINING DECENTRALIZED ALGORITHMS")
        print("="*80)
        
        for algorithm in ['DQN', 'DDPG', 'SAC']:
            alg_name = f'D-{algorithm}'
            self.agents[alg_name] = self.train_decentralized_algorithm(algorithm, alg_name)
        
        # Evaluate baselines
        print("\n" + "="*80)
        print("EVALUATING BASELINES")
        print("="*80)
        baseline_results = self.evaluate_baselines(num_episodes=100)
        
        # Final evaluation with best models
        print("\n" + "="*80)
        print("FINAL EVALUATION")
        print("="*80)
        rl_results = self.final_evaluation(num_episodes=100)
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.plot_results(baseline_results, rl_results)
        
        # Save results
        self.save_results(baseline_results, rl_results)
        
        # Print final summary
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        print(f"\nOptimization Target: {self._objective_display_name()}")
        print(f"Total Time: {total_time/60:.1f} minutes")
        
        # Find best performers
        all_results = {**baseline_results, **rl_results}
        metric_key = self._result_metric_key()
        best_alg = min(all_results.items(), key=lambda x: x[1][metric_key][0])
        
        print(f"\nBest Algorithm Overall: {best_alg[0]}")
        print(f"  {self._objective_display_name()}: {best_alg[1][metric_key][0]:.3f}±{best_alg[1][metric_key][1]:.3f}")
        
        # Compare paradigms
        best_baseline = min(baseline_results.items(), key=lambda x: x[1][metric_key][0])
        best_centralized = min(((k, v) for k, v in rl_results.items() if k.startswith('C-')), 
                              key=lambda x: x[1][metric_key][0])
        best_decentralized = min(((k, v) for k, v in rl_results.items() if k.startswith('D-')), 
                                key=lambda x: x[1][metric_key][0])
        
        print(f"\nBest by Paradigm:")
        print(f"  Baseline: {best_baseline[0]} = {best_baseline[1][metric_key][0]:.3f}")
        print(f"  Centralized: {best_centralized[0]} = {best_centralized[1][metric_key][0]:.3f}")
        print(f"  Decentralized: {best_decentralized[0]} = {best_decentralized[1][metric_key][0]:.3f}")
        
        # Calculate improvements
        baseline_val = best_baseline[1][metric_key][0]
        cent_improvement = (baseline_val - best_centralized[1][metric_key][0]) / baseline_val * 100
        decent_improvement = (baseline_val - best_decentralized[1][metric_key][0]) / baseline_val * 100
        
        print(f"\nImprovements over best baseline:")
        print(f"  Centralized: {cent_improvement:.1f}%")
        print(f"  Decentralized: {decent_improvement:.1f}%")
        
        # Detailed comparison table
        print("  * Power columns report executed active-link powers after environment flooring.")
        print(f"\n{'Algorithm':<12} {'Paradigm':<15} {'Objective':<15} {'Time (s)':<15} {'Energy (J)':<15} {'V2V Frac':<12} {'V2V Pow*':<12} {'V2I Pow*':<12}")
        print("-"*122)
        
        for name, results in all_results.items():
            if name in baseline_results:
                paradigm = "Baseline"
            elif name.startswith('C-'):
                paradigm = "Centralized"
            else:
                paradigm = "Decentralized"
            
            objective_str = f"{results[metric_key][0]:.3f}±{results[metric_key][1]:.3f}"
            time_str = f"{results['time'][0]:.2f}±{results['time'][1]:.2f}"
            energy_str = f"{results['energy'][0]:.2f}±{results['energy'][1]:.2f}"
            
            # Get action stats if available
            if 'action_analysis' in results:
                v2v_frac = f"{results['action_analysis']['v2v_fraction']['mean']:.3f}±{results['action_analysis']['v2v_fraction']['std']:.3f}"
                v2v_power = f"{results['action_analysis']['v2v_power']['mean']:.3f}±{results['action_analysis']['v2v_power']['std']:.3f}"
                v2i_power = f"{results['action_analysis']['v2i_power']['mean']:.3f}±{results['action_analysis']['v2i_power']['std']:.3f}"
            else:
                # Fixed baseline strategies
                if name == 'All-Leader':
                    v2v_frac = "1.000±0.000"
                    v2v_power = "1.000±0.000"
                    v2i_power = "0.000±0.000"
                elif name == 'All-Base':
                    v2v_frac = "0.000±0.000"
                    v2v_power = "0.000±0.000"
                    v2i_power = "1.000±0.000"
                elif name == 'Balanced':
                    v2v_frac = "0.500±0.000"
                    v2v_power = "1.000±0.000"
                    v2i_power = "1.000±0.000"
                else:
                    v2v_frac = "-"
                    v2v_power = "-"
                    v2i_power = "-"
            
            print(f"{name:<12} {paradigm:<15} {objective_str:<15} {time_str:<15} {energy_str:<15} {v2v_frac:<12} {v2v_power:<12} {v2i_power:<12}")


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate all centralized and decentralized algorithms')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--optimization-target', type=str, 
                       choices=['mixed'], default='mixed',
                       help='Unified optimization target')
    parser.add_argument('--eval-interval', type=int, default=50,
                       help='Evaluation interval in episodes')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--num-vehicles', type=int, default=5,
                       help='Number of vehicles (default: 5 as per paper config)')
    parser.add_argument('--cpu-cycles-per-bit', type=int, default=10,
                       help='CPU cycles per bit for deduplication (default: 10)')
    parser.add_argument('--chunk-overhead-cycles', type=float, default=1e6,
                       help='Chunk overhead cycles for deduplication (default: 1e6)')
    parser.add_argument('--redundancy-ratio-range', type=float, nargs=2, 
                       default=[0.5, 0.5],
                       help='Min and max redundancy ratio (default: 0.5 0.5)')
    parser.add_argument('--leader-selection-mode', type=str,
                       choices=['fixed_zero', 'episode_start_scored'],
                       default='episode_start_scored',
                       help='Leader selection mode')
    parser.add_argument('--leader-reselection-interval-slots', type=int, default=10,
                       help='Slots between leader reevaluations; 0 disables mid-episode reselection')
    parser.add_argument('--mixed-time-weight', type=float, default=0.5,
                       help='Eta in the unified objective: 1.0=time-only, 0.0=energy-only')
    parser.add_argument('--reward-mode', type=str,
                       choices=['canonical_flat_normalized', 'paper_counted_normalized', 'paper_excess_normalized', 'paper_counted_raw', 'custom'],
                       default='paper_counted_normalized',
                       help='Reward formulation to use')
    parser.add_argument('--constraint-penalty-value', type=float, default=10.0,
                       help='Constraint penalty coefficient / flat penalty value')
    parser.add_argument('--energy-reward-normalization-factor', type=float, default=4.0,
                       help='Energy-mode reward scaling factor')
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = CombinedTrainEvaluate(
        optimization_target=args.optimization_target,
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        device=args.device,
        seed=args.seed,
        num_vehicles=args.num_vehicles,
        cpu_cycles_per_bit=args.cpu_cycles_per_bit,
        chunk_overhead_cycles=args.chunk_overhead_cycles,
        redundancy_ratio_range=tuple(args.redundancy_ratio_range),
        leader_selection_mode=args.leader_selection_mode,
        leader_reselection_interval_slots=args.leader_reselection_interval_slots,
        mixed_time_weight=args.mixed_time_weight,
        reward_mode=args.reward_mode,
        constraint_penalty_value=args.constraint_penalty_value,
        energy_reward_normalization_factor=args.energy_reward_normalization_factor
    )
    
    trainer.run()


if __name__ == "__main__":
    main()
