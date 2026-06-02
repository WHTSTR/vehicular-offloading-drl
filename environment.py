"""
Vehicular edge-offloading environment implementing the paper's system model.
"""

import numpy as np
from typing import Tuple, List, Optional

try:
    import gym
    from gym import spaces
except ImportError:
    class _FallbackEnv:
        """Minimal fallback when gym is unavailable."""

        pass

    class _FallbackBox:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _FallbackSpaces:
        Box = _FallbackBox

    class _FallbackGymModule:
        Env = _FallbackEnv

    gym = _FallbackGymModule()
    spaces = _FallbackSpaces()


class PaperVehicularEnvironment(gym.Env):
    """
    Vehicular edge computing environment implementing the system model from the paper
    
    Key features:
    - Chunk-based data segmentation (Section III.C)
    - Discrete-time model with one chunk per vehicle per slot
    - Log-normal shadowing channel model (Section III.D)
    - Score-based leader selection at the start of the upload session with optional periodic reselection
    - Deduplication with CPU processing time/energy (Section III.F)
    
    """
    
    def __init__(self, config: dict):
        super().__init__()

        self.seed_value = config.get('seed')
        if self.seed_value is not None:
            self.seed(self.seed_value)
        
        # All parameters come from config - no defaults here
        # Network parameters
        self.num_vehicles = config['num_vehicles']  # N(t) in paper
        self.num_time_slots = config['num_time_slots']  # T in paper
        self.time_slot_duration = config['time_slot_duration']  # Δt
        
        # Chunk parameters (Section III.C)
        self.chunks_per_vehicle = config['chunks_per_vehicle']  # K_i
        self.chunk_size = config['chunk_size']  # d_{i,k} in bits
        
        # Leader selection (Section III.E)
        self.leader_selection_mode = config.get('leader_selection_mode', 'episode_start_scored')
        self.leader_selection_cpu_weight = config.get('leader_selection_cpu_weight', 0.0)
        self.leader_selection_v2v_aggregation = config.get('leader_selection_v2v_aggregation', 'sum')
        if self.leader_selection_v2v_aggregation not in {'sum', 'mean'}:
            raise ValueError("leader_selection_v2v_aggregation must be 'sum' or 'mean'")
        self.leader_reselection_interval_slots = int(config.get('leader_reselection_interval_slots', 0))
        self.leader_idx = 0  # Updated at reset() when score-based selection is enabled
        self.follower_indices = [i for i in range(self.num_vehicles) if i != self.leader_idx]
        self.num_followers = self.num_vehicles - 1
        
        # Channel parameters (Section III.D)
        self.bandwidth_v2v = config['bandwidth_v2v']  # B^{v2v}
        self.bandwidth_v2i = config['bandwidth_v2i']  # B^{v2i}
        self.noise_power_density = config['noise_power_density']  # N_0
        
        # Path loss parameters
        self.reference_distance = config['reference_distance']  # d_0
        self.path_loss_constant_v2v = config['path_loss_constant_v2v']  # p_0^{v2v}
        self.path_loss_constant_v2i = config['path_loss_constant_v2i']  # p_0^{v2i}
        self.path_loss_exponent_v2v = config['path_loss_exponent_v2v']  # γ^{v2v}
        self.path_loss_exponent_v2i = config['path_loss_exponent_v2i']  # γ^{v2i}
        
        # Fading correlation parameter
        self.fading_correlation = config['fading_correlation']  # ρ - correlation between time slots
        
        # Vehicle movement parameters
        self.vehicle_speed_min = config['vehicle_speed_min']  # Minimum speed m/s
        self.vehicle_speed_max = config['vehicle_speed_max']  # Maximum speed m/s
        self.num_lanes = config['num_lanes']  # Number of lanes
        self.lane_width = config['lane_width']  # Width of each lane
        self.road_length = config['road_length']  # Total road length
        self.base_station_position = np.array(config['base_station_position'])
        self.initial_position_mode = config.get('initial_position_mode', 'random_uniform')
        if self.initial_position_mode not in {'random_uniform', 'ordered_platoon'}:
            raise ValueError("initial_position_mode must be 'random_uniform' or 'ordered_platoon'")
        initial_position_x_range = config.get('initial_position_x_range', [0.0, 50.0])
        if len(initial_position_x_range) != 2:
            raise ValueError("initial_position_x_range must contain exactly two values")
        self.initial_position_x_min = float(initial_position_x_range[0])
        self.initial_position_x_max = float(initial_position_x_range[1])
        if self.initial_position_x_max <= self.initial_position_x_min:
            raise ValueError("initial_position_x_range must have max > min")
        self.initial_position_jitter = float(config.get('initial_position_jitter', 0.0))
        if self.initial_position_jitter < 0.0:
            raise ValueError("initial_position_jitter must be non-negative")
        
        # Each vehicle will have its own speed
        self.vehicle_speeds = None  # Will be initialized in reset()
        
        # Deduplication parameters (Section III.F)
        self.cpu_cycles_per_bit = config['cpu_cycles_per_bit']  # C_1
        self.chunk_overhead_cycles = config['chunk_overhead_cycles']  # C_2
        self.leader_cpu_frequency = config['leader_cpu_frequency']  # f_{i*} in Hz
        self.cpu_power_constant = config['cpu_power_constant']  # κ
        
        # Redundancy parameters
        self.redundancy_lower = config['redundancy_lower']
        self.redundancy_upper = config['redundancy_upper']
        
        # Power constraints
        self.max_power = config['max_power']  # p_max
        self.min_active_power_ratio = float(config.get('min_active_power_ratio', 1e-3))
        if self.min_active_power_ratio < 0.0 or self.min_active_power_ratio > 1.0:
            raise ValueError("min_active_power_ratio must lie in [0, 1]")
        self.min_active_power = self.max_power * self.min_active_power_ratio
        
        # Power constraint type
        self.joint_power_constraint = config.get('joint_power_constraint', True)  # Joint V2V+V2I power budget

        # Time and energy budgets
        self.time_budget = config['time_budget']  # T_max per slot
        follower_time_budget_explicit = 'follower_time_budget' in config
        leader_time_budget_explicit = 'leader_time_budget' in config
        self.follower_time_budget = float(config.get('follower_time_budget', self.time_budget))
        self.leader_time_budget = float(config.get('leader_time_budget', self.time_budget))

        legacy_energy_budget = float(config['energy_budget'])  # E_max per slot
        follower_energy_budget = config.get('follower_energy_budget')
        if follower_energy_budget is None:
            if follower_time_budget_explicit or leader_time_budget_explicit:
                follower_power_ceiling = self.max_power if self.joint_power_constraint else 2.0 * self.max_power
                follower_energy_budget = self.follower_time_budget * follower_power_ceiling
            else:
                follower_energy_budget = legacy_energy_budget
        leader_energy_budget = config.get('leader_energy_budget')
        if leader_energy_budget is None:
            if follower_time_budget_explicit or leader_time_budget_explicit:
                leader_upload_power_ceiling = self.max_power * float(config.get('leader_power_multiplier', 1.0))
                leader_cpu_power = self.cpu_power_constant * (self.leader_cpu_frequency ** 2)
                leader_energy_budget = self.leader_time_budget * (leader_upload_power_ceiling + leader_cpu_power)
            else:
                leader_energy_budget = legacy_energy_budget

        self.follower_energy_budget = float(follower_energy_budget)
        self.leader_energy_budget = float(leader_energy_budget)
        self.energy_budget = self.leader_energy_budget

        self.mixed_time_weight = float(config.get('mixed_time_weight', 0.5))
        if not 0.0 <= self.mixed_time_weight <= 1.0:
            raise ValueError("mixed_time_weight must lie in [0, 1]")
        self.reward_mode = config.get('reward_mode', 'canonical_flat_normalized')
        self.constraint_penalty_mode = config.get('constraint_penalty_mode', 'flat_any_violation')
        self.constraint_penalty_value = config.get('constraint_penalty_value', 10.0)
        self.energy_reward_normalization_factor = config.get('energy_reward_normalization_factor', 4.0)
        self.active_constraint_penalty_mode = self.constraint_penalty_mode
        self.active_energy_reward_normalization_factor = self.energy_reward_normalization_factor
        self._apply_reward_mode_defaults()
        
        # Optimization target
        self.optimization_target = config['optimization_target']
        
        # Leader upload handling
        self.ignore_leader_upload = config['ignore_leader_upload']
        
        # Leader capabilities
        self.leader_power_multiplier = config.get('leader_power_multiplier', 1.0)  # Default to same as followers
        self.leader_uplink_resource_factor = float(config.get('leader_uplink_resource_factor', 1.0))
        if self.leader_uplink_resource_factor <= 0.0:
            raise ValueError("leader_uplink_resource_factor must be positive")
        
        # Action space: for each follower [δ, p^{v2v}, p^{v2i}]
        # δ_{i,t} ∈ [0,1], p_i^{v2v}[t] and p_i^{v2i}[t] subject to p_max
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_followers * 3,),
            dtype=np.float32
        )
        
        # State space:
        # - Channel gains h_{ij}[t] and g_i[t]
        # - Vehicle distances
        # - Current time slot
        # - Previous-slot local time/energy utilization for each follower
        state_dim = (
            2 * self.num_followers +  # V2V and V2I channel gains
            self.num_vehicles +       # Distances from BS
            1 +                       # Current time slot
            2 * self.num_followers    # Previous time/energy utilization for each follower
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.current_slot = 0
        self.vehicle_positions = None
        self.channel_gains_v2v = None  # h_{ij}[t]
        self.channel_gains_v2i = None  # g_i[t]
        self.shadowing_v2v = None
        self.shadowing_v2i = None
        self.redundancy_ratios = None  # β_{i,t} - still used internally for simulation
        
        # Track which chunks have been transmitted
        self.chunks_transmitted = None
        
        # Storage for previous fading coefficients (for correlation)
        self.prev_fading_v2i = None
        self.prev_fading_v2v = None

        self.last_system_reward = 0.0
        self.prev_follower_time_utilization = None
        self.prev_follower_energy_utilization = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the environment RNG used by numpy-based simulation state."""
        if seed is None:
            return [self.seed_value] if self.seed_value is not None else []

        self.seed_value = int(seed)
        np.random.seed(self.seed_value)
        return [self.seed_value]

    def _apply_reward_mode_defaults(self) -> None:
        """Resolve a named reward mode into concrete scaling and penalty behavior."""
        if self.reward_mode == 'canonical_flat_normalized':
            self.active_constraint_penalty_mode = 'flat_any_violation'
            self.active_energy_reward_normalization_factor = self.energy_reward_normalization_factor
        elif self.reward_mode == 'paper_counted_normalized':
            self.active_constraint_penalty_mode = 'per_violation'
            self.active_energy_reward_normalization_factor = self.energy_reward_normalization_factor
        elif self.reward_mode == 'paper_excess_normalized':
            self.active_constraint_penalty_mode = 'normalized_excess'
            self.active_energy_reward_normalization_factor = self.energy_reward_normalization_factor
        elif self.reward_mode == 'paper_counted_raw':
            self.active_constraint_penalty_mode = 'per_violation'
            self.active_energy_reward_normalization_factor = 1.0
        elif self.reward_mode == 'custom':
            self.active_constraint_penalty_mode = self.constraint_penalty_mode
            self.active_energy_reward_normalization_factor = self.energy_reward_normalization_factor
        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")

    def _get_effective_mixed_time_weight(self) -> float:
        """Map the objective target to the unified eta-parameterized weight."""
        if self.optimization_target == 'time':
            return 1.0
        if self.optimization_target == 'energy':
            return 0.0
        if self.optimization_target == 'mixed':
            return self.mixed_time_weight
        raise ValueError(f"Unknown optimization target: {self.optimization_target}")
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_slot = 0
        
        # Initialize vehicle positions using the configured geometry.
        self.vehicle_positions = np.zeros((self.num_vehicles, 2))
        # Initialize vehicle speeds randomly
        self.vehicle_speeds = np.random.uniform(self.vehicle_speed_min, 
                                               self.vehicle_speed_max, 
                                               self.num_vehicles)

        self.vehicle_positions[:, 0] = self._sample_initial_x_positions()
        for i in range(self.num_vehicles):
            lane = np.random.randint(0, self.num_lanes)
            self.vehicle_positions[i, 1] = lane * self.lane_width
        
        # Initialize channel conditions
        self._initialize_channel_conditions()

        # Select the leader once at the beginning of the upload session.
        self._select_episode_leader()
        
        # Initialize redundancy ratios for all vehicles and chunks.
        # The current leader's rows are unused during that episode, but keeping
        # per-vehicle storage makes leader changes across episodes well defined.
        self.redundancy_ratios = np.random.uniform(
            self.redundancy_lower,
            self.redundancy_upper,
            (self.num_vehicles, self.chunks_per_vehicle)
        )
        
        # Track chunk progress per physical vehicle.
        self.chunks_transmitted = np.zeros(self.num_vehicles, dtype=int)
        self.prev_follower_time_utilization = np.zeros(self.num_vehicles, dtype=np.float32)
        self.prev_follower_energy_utilization = np.zeros(self.num_vehicles, dtype=np.float32)
        
        return self._get_state()

    @staticmethod
    def _normalize_utilization(raw_value: float, clip_max: float = 2.0) -> float:
        """Clip a utilization ratio and normalize it into [0, 1]."""
        return float(np.clip(raw_value, 0.0, clip_max) / clip_max)

    def _sample_initial_x_positions(self) -> np.ndarray:
        """Sample initial x-positions according to the configured geometry."""
        if self.initial_position_mode == 'random_uniform':
            return np.random.uniform(
                self.initial_position_x_min,
                self.initial_position_x_max,
                self.num_vehicles,
            )

        x_positions = np.linspace(
            self.initial_position_x_min,
            self.initial_position_x_max,
            self.num_vehicles,
        )
        if self.initial_position_jitter > 0.0:
            x_positions = np.clip(
                x_positions + np.random.uniform(
                    -self.initial_position_jitter,
                    self.initial_position_jitter,
                    self.num_vehicles,
                ),
                self.initial_position_x_min,
                self.initial_position_x_max,
            )
        return np.sort(x_positions)

    def _select_episode_leader(self) -> None:
        """Select the leader once per reset() and derive the fixed follower map."""
        if self.leader_selection_mode == 'fixed_zero':
            self.leader_idx = 0
        elif self.leader_selection_mode == 'closest_bs':
            self.leader_idx = self._closest_bs_leader_selection()
        elif self.leader_selection_mode == 'episode_start_scored':
            self.leader_idx = self._score_based_leader_selection()
        else:
            raise ValueError(f"Unknown leader selection mode: {self.leader_selection_mode}")

        self.follower_indices = [idx for idx in range(self.num_vehicles) if idx != self.leader_idx]

    def _closest_bs_leader_selection(self) -> int:
        """Choose the vehicle currently closest to the base station."""
        distances = [
            np.linalg.norm(self.vehicle_positions[idx] - self.base_station_position)
            for idx in range(self.num_vehicles)
        ]
        return int(np.argmin(distances))

    def _score_based_leader_selection(self) -> int:
        """Apply the communication score to choose the episode leader."""
        candidate_scores = []
        for candidate_idx in range(self.num_vehicles):
            total_v2v_rate = 0.0
            for peer_idx in range(self.num_vehicles):
                if peer_idx == candidate_idx:
                    continue
                total_v2v_rate += self._calculate_v2v_rate(peer_idx, candidate_idx, self.max_power)

            if self.leader_selection_v2v_aggregation == 'mean':
                v2v_term = total_v2v_rate / max(self.num_vehicles - 1, 1)
            else:
                v2v_term = total_v2v_rate
            v2i_rate = self._calculate_v2i_rate(candidate_idx, self.max_power)
            cpu_term = self.leader_selection_cpu_weight * self.leader_cpu_frequency
            candidate_scores.append(v2v_term + v2i_rate + cpu_term)

        return int(np.argmax(candidate_scores))

    def _maybe_reselect_leader_for_next_slot(self) -> Tuple[bool, int, List[int]]:
        """
        Re-evaluate the leader at configured slot boundaries.

        This is called after the current slot is executed and before the next
        state is exposed to the agents, so the next observation is consistent
        with the next slot's leader assignment.
        """
        if self.leader_selection_mode not in {'episode_start_scored', 'closest_bs'}:
            return False, self.leader_idx, list(self.follower_indices)

        if self.leader_reselection_interval_slots <= 0:
            return False, self.leader_idx, list(self.follower_indices)

        if self.current_slot <= 0 or self.current_slot >= self.num_time_slots:
            return False, self.leader_idx, list(self.follower_indices)

        if self.current_slot % self.leader_reselection_interval_slots != 0:
            return False, self.leader_idx, list(self.follower_indices)

        previous_leader_idx = self.leader_idx
        if self.leader_selection_mode == 'closest_bs':
            self.leader_idx = self._closest_bs_leader_selection()
        else:
            self.leader_idx = self._score_based_leader_selection()
        self.follower_indices = [idx for idx in range(self.num_vehicles) if idx != self.leader_idx]
        leader_changed = self.leader_idx != previous_leader_idx
        return leader_changed, self.leader_idx, list(self.follower_indices)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one time slot according to paper's model"""
        active_leader_idx = self.leader_idx
        active_follower_indices = list(self.follower_indices)

        # Update vehicle positions
        self._update_vehicle_positions()

        # Update channel conditions
        self._update_channel_conditions()

        reward, info = self.evaluate_action(action)

        follower_time_budget = max(float(self.follower_time_budget), 1e-12)
        follower_energy_budget = max(float(self.follower_energy_budget), 1e-12)
        for local_idx, follower_idx in enumerate(active_follower_indices):
            follower_time = float(info['per_follower_times'][local_idx])
            follower_energy = float(info['per_follower_energies'][local_idx])
            self.prev_follower_time_utilization[follower_idx] = self._normalize_utilization(
                follower_time / follower_time_budget
            )
            self.prev_follower_energy_utilization[follower_idx] = self._normalize_utilization(
                follower_energy / follower_energy_budget
            )
        # Leader is not queried through the follower local state.
        self.prev_follower_time_utilization[active_leader_idx] = 0.0
        self.prev_follower_energy_utilization[active_leader_idx] = 0.0

        # Mark one chunk processed for each currently active vehicle.
        for follower_idx in active_follower_indices:
            if self.chunks_transmitted[follower_idx] < self.chunks_per_vehicle:
                self.chunks_transmitted[follower_idx] += 1
        if self.chunks_transmitted[active_leader_idx] < self.chunks_per_vehicle:
            self.chunks_transmitted[active_leader_idx] += 1

        self.current_slot += 1
        done = self.current_slot >= self.num_time_slots
        leader_reselected = False
        next_leader_idx = active_leader_idx
        next_follower_indices = list(active_follower_indices)
        if not done:
            leader_reselected, next_leader_idx, next_follower_indices = self._maybe_reselect_leader_for_next_slot()

        info = dict(info)
        info.update({
            'current_slot': self.current_slot,
            'leader_idx': active_leader_idx,
            'follower_indices': list(active_follower_indices),
            'leader_reselected_for_next_slot': leader_reselected,
            'next_leader_idx': next_leader_idx,
            'next_follower_indices': list(next_follower_indices),
            'vehicle_distances': [np.linalg.norm(pos - self.base_station_position)
                                for pos in self.vehicle_positions],
        })

        return self._get_state(), reward, done, info

    def evaluate_action_post_update(self, action: np.ndarray) -> Tuple[float, dict]:
        """
        Evaluate an action on the same post-update slot snapshot used by step(),
        but restore the environment afterwards so search methods can score many
        candidates without advancing the real episode.
        """
        saved_vehicle_positions = None if self.vehicle_positions is None else self.vehicle_positions.copy()
        saved_vehicle_speeds = None if self.vehicle_speeds is None else self.vehicle_speeds.copy()
        saved_channel_gains_v2v = None if self.channel_gains_v2v is None else self.channel_gains_v2v.copy()
        saved_channel_gains_v2i = None if self.channel_gains_v2i is None else self.channel_gains_v2i.copy()
        saved_prev_fading_v2v = None if self.prev_fading_v2v is None else self.prev_fading_v2v.copy()
        saved_prev_fading_v2i = None if self.prev_fading_v2i is None else self.prev_fading_v2i.copy()
        saved_last_system_reward = self.last_system_reward
        saved_rng_state = np.random.get_state()

        try:
            self._update_vehicle_positions()
            self._update_channel_conditions()
            return self.evaluate_action(action)
        finally:
            self.vehicle_positions = saved_vehicle_positions
            self.vehicle_speeds = saved_vehicle_speeds
            self.channel_gains_v2v = saved_channel_gains_v2v
            self.channel_gains_v2i = saved_channel_gains_v2i
            self.prev_fading_v2v = saved_prev_fading_v2v
            self.prev_fading_v2i = saved_prev_fading_v2i
            self.last_system_reward = saved_last_system_reward
            np.random.set_state(saved_rng_state)

    def evaluate_action(self, action: np.ndarray) -> Tuple[float, dict]:
        """Evaluate the current-slot effect of an action without advancing the environment."""
        active_leader_idx = self.leader_idx
        active_follower_indices = list(self.follower_indices)

        # Parse actions for each follower
        offloading_fractions = []  # δ_{i,t}
        v2v_powers = []            # p_i^{v2v}[t]
        v2i_powers = []            # p_i^{v2i}[t]
        active_link_power_floor_hits = 0

        for i in range(self.num_followers):
            delta = action[i*3]
            p_v2v = action[i*3 + 1] * self.max_power
            p_v2i = action[i*3 + 2] * self.max_power

            if self.joint_power_constraint:
                total_power = p_v2v + p_v2i
                if total_power > self.max_power:
                    scale = self.max_power / total_power
                    p_v2v *= scale
                    p_v2i *= scale
            else:
                p_v2v = min(p_v2v, self.max_power)
                p_v2i = min(p_v2i, self.max_power)

            if delta > 0.0 and p_v2v < self.min_active_power:
                p_v2v = self.min_active_power
                active_link_power_floor_hits += 1
            if delta < 1.0 and p_v2i < self.min_active_power:
                p_v2i = self.min_active_power
                active_link_power_floor_hits += 1

            offloading_fractions.append(delta)
            v2v_powers.append(p_v2v)
            v2i_powers.append(p_v2i)

        follower_times = []
        follower_energies = []
        total_v2v_data = 0.0
        leader_chunk_data = self.chunk_size
        total_dedup_input_data = 0.0
        follower_unique_data = 0.0
        total_unique_data = leader_chunk_data
        active_dedup_chunks = 0

        for i, follower_idx in enumerate(active_follower_indices):
            chunk_idx = min(self.chunks_transmitted[follower_idx], self.chunks_per_vehicle - 1)
            beta = self.redundancy_ratios[follower_idx, chunk_idx]

            t_v2v = 0.0
            e_v2v = 0.0
            if offloading_fractions[i] > 0 and v2v_powers[i] > 0:
                v2v_data = offloading_fractions[i] * self.chunk_size
                v2v_rate = self._calculate_v2v_rate(follower_idx, active_leader_idx, v2v_powers[i])
                if v2v_rate > 0:
                    t_v2v = v2v_data / v2v_rate
                    e_v2v = v2v_powers[i] * t_v2v
                    if active_dedup_chunks == 0:
                        total_dedup_input_data += leader_chunk_data
                        active_dedup_chunks = 1
                    total_v2v_data += v2v_data
                    total_dedup_input_data += v2v_data
                    follower_unique_data += (1 - beta) * v2v_data
                    total_unique_data = leader_chunk_data + follower_unique_data
                    active_dedup_chunks += 1

            t_v2i = 0.0
            e_v2i = 0.0
            if (1 - offloading_fractions[i]) > 0 and v2i_powers[i] > 0:
                v2i_data = (1 - offloading_fractions[i]) * self.chunk_size
                v2i_rate = self._calculate_v2i_rate(follower_idx, v2i_powers[i])
                if v2i_rate > 0:
                    t_v2i = v2i_data / v2i_rate
                    e_v2i = v2i_powers[i] * t_v2i

            follower_times.append(max(t_v2v, t_v2i))
            follower_energies.append(e_v2v + e_v2i)

        t_dedup = 0.0
        e_dedup = 0.0
        if total_dedup_input_data > 0:
            cpu_cycles = (
                self.cpu_cycles_per_bit * total_dedup_input_data
                + self.chunk_overhead_cycles * active_dedup_chunks
            )
            t_dedup = cpu_cycles / self.leader_cpu_frequency
            e_dedup = self.cpu_power_constant * (self.leader_cpu_frequency ** 2) * t_dedup

        t_leader_upload = 0.0
        e_leader_upload = 0.0
        if total_unique_data > 0 and not self.ignore_leader_upload:
            leader_upload_power = self.max_power * self.leader_power_multiplier
            leader_v2i_rate = self._calculate_v2i_rate(
                active_leader_idx,
                leader_upload_power,
                bandwidth_scale=self.leader_uplink_resource_factor,
            )
            if leader_v2i_rate > 0:
                t_leader_upload = total_unique_data / leader_v2i_rate
                e_leader_upload = leader_upload_power * t_leader_upload

        time_cost = sum(follower_times) + t_dedup + t_leader_upload
        energy_cost = sum(follower_energies) + e_dedup + e_leader_upload

        time_normalizer = (
            self.num_followers * self.follower_time_budget + self.leader_time_budget
            if (self.follower_time_budget > 0 and self.leader_time_budget > 0)
            else 1.0
        )
        energy_normalizer = (
            self.num_followers * self.follower_energy_budget + self.leader_energy_budget
            if (self.follower_energy_budget > 0 and self.leader_energy_budget > 0)
            else 1.0
        )
        normalized_time_cost = time_cost / time_normalizer
        normalized_energy_cost = energy_cost / energy_normalizer
        effective_mixed_time_weight = self._get_effective_mixed_time_weight()
        mixed_cost = (
            effective_mixed_time_weight * normalized_time_cost
            + (1.0 - effective_mixed_time_weight) * normalized_energy_cost
        )

        if self.optimization_target == 'time':
            base_reward = -time_cost
            objective_cost = time_cost
        elif self.optimization_target == 'energy':
            base_reward = -energy_cost * self.active_energy_reward_normalization_factor
            objective_cost = energy_cost
        else:
            base_reward = -mixed_cost
            objective_cost = mixed_cost

        reward = base_reward

        follower_time_violations = sum(t > self.follower_time_budget for t in follower_times)
        leader_time_violation = int(t_dedup + t_leader_upload > self.leader_time_budget)
        follower_energy_violations = sum(e > self.follower_energy_budget for e in follower_energies)
        leader_energy_violation = int(e_dedup + e_leader_upload > self.leader_energy_budget)
        constraint_violation_count = (
            follower_time_violations
            + leader_time_violation
            + follower_energy_violations
            + leader_energy_violation
        )
        constraint_violated = constraint_violation_count > 0
        follower_time_budget = max(float(self.follower_time_budget), 1e-12)
        leader_time_budget = max(float(self.leader_time_budget), 1e-12)
        follower_energy_budget = max(float(self.follower_energy_budget), 1e-12)
        leader_energy_budget = max(float(self.leader_energy_budget), 1e-12)
        follower_time_excess = sum(
            max((float(t) - follower_time_budget) / follower_time_budget, 0.0)
            for t in follower_times
        )
        leader_time_excess = max(
            (float(t_dedup + t_leader_upload) - leader_time_budget) / leader_time_budget,
            0.0,
        )
        follower_energy_excess = sum(
            max((float(e) - follower_energy_budget) / follower_energy_budget, 0.0)
            for e in follower_energies
        )
        leader_energy_excess = max(
            (float(e_dedup + e_leader_upload) - leader_energy_budget) / leader_energy_budget,
            0.0,
        )
        constraint_excess = (
            follower_time_excess
            + leader_time_excess
            + follower_energy_excess
            + leader_energy_excess
        )

        constraint_penalty_applied = 0.0
        if self.active_constraint_penalty_mode == 'flat_any_violation':
            if constraint_violated:
                constraint_penalty_applied = float(self.constraint_penalty_value)
        elif self.active_constraint_penalty_mode == 'per_violation':
            if constraint_violated:
                constraint_penalty_applied = float(self.constraint_penalty_value) * constraint_violation_count
        elif self.active_constraint_penalty_mode == 'normalized_excess':
            constraint_penalty_applied = float(self.constraint_penalty_value) * constraint_excess
        else:
            raise ValueError(f"Unknown constraint penalty mode: {self.active_constraint_penalty_mode}")

        if constraint_penalty_applied > 0.0:
            reward -= constraint_penalty_applied

        info = {
            'objective_cost': objective_cost,
            'penalized_objective': -reward,
            'time_cost': time_cost,
            'energy_cost': energy_cost,
            'normalized_time_cost': normalized_time_cost,
            'normalized_energy_cost': normalized_energy_cost,
            'mixed_cost': mixed_cost,
            'mixed_time_weight': effective_mixed_time_weight,
            'follower_times': follower_times,
            'follower_energies': follower_energies,
            'dedup_time': t_dedup,
            'dedup_energy': e_dedup,
            'leader_upload_time': t_leader_upload,
            'leader_upload_energy': e_leader_upload,
            'v2v_data_received': total_v2v_data,
            'leader_chunk_data': leader_chunk_data,
            'dedup_input_data': total_dedup_input_data,
            'follower_unique_data': follower_unique_data,
            'unique_data': total_unique_data,
            'redundancy_removed': max(0.0, total_dedup_input_data - total_unique_data),
            'base_reward': base_reward,
            'constraint_violated': constraint_violated,
            'constraint_violations': constraint_violation_count,
            'constraint_excess': constraint_excess,
            'constraint_penalty_applied': constraint_penalty_applied,
            'reward_mode': self.reward_mode,
            'constraint_penalty_mode': self.active_constraint_penalty_mode,
            'energy_reward_normalization_factor': self.active_energy_reward_normalization_factor,
            'follower_time_budget': self.follower_time_budget,
            'leader_time_budget': self.leader_time_budget,
            'follower_energy_budget': self.follower_energy_budget,
            'leader_energy_budget': self.leader_energy_budget,
            'per_follower_times': [float(t) for t in follower_times],
            'per_follower_energies': [float(e) for e in follower_energies],
            'active_link_power_floor_hits': active_link_power_floor_hits,
            'follower_time_violations': int(follower_time_violations),
            'leader_time_violation': leader_time_violation,
            'follower_energy_violations': int(follower_energy_violations),
            'leader_energy_violation': leader_energy_violation,
            'follower_time_excess': follower_time_excess,
            'leader_time_excess': leader_time_excess,
            'follower_energy_excess': follower_energy_excess,
            'leader_energy_excess': leader_energy_excess,
        }

        self.last_system_reward = reward
        return reward, info
    
    def _initialize_channel_conditions(self):
        """Initialize channel conditions for first time slot"""
        # Initialize channel gain matrices
        self.channel_gains_v2v = np.zeros((self.num_vehicles, self.num_vehicles))
        self.channel_gains_v2i = np.zeros(self.num_vehicles)
        
        # Initialize fading coefficients for first time slot
        # |f[0]|^2 follows exponential distribution with mean 1
        self.prev_fading_v2i = np.random.exponential(1.0, self.num_vehicles)
        self.prev_fading_v2v = np.random.exponential(1.0, (self.num_vehicles, self.num_vehicles))
        
        self._update_channel_conditions(initial=True)
    
    def _update_channel_conditions(self, initial=False):
        """Update channel gains based on current positions (Section III.D)"""
        
        if initial:
            # Use the pre-generated initial fading values
            current_fading_v2i = self.prev_fading_v2i
            current_fading_v2v = self.prev_fading_v2v
        else:
            # Generate correlated fading coefficients
            # f[t] = ρ * f[t-1] + sqrt(1-ρ²) * new_random
            ρ = self.fading_correlation
            noise_scale = np.sqrt(1 - ρ**2)
            
            # V2I fading correlation
            new_fading_v2i = np.random.exponential(1.0, self.num_vehicles)
            current_fading_v2i = ρ * self.prev_fading_v2i + noise_scale * new_fading_v2i
            # Ensure positive values (as it represents |f|^2)
            current_fading_v2i = np.maximum(current_fading_v2i, 0.01)
            self.prev_fading_v2i = current_fading_v2i
            
            # V2V fading correlation  
            new_fading_v2v = np.random.exponential(1.0, (self.num_vehicles, self.num_vehicles))
            current_fading_v2v = ρ * self.prev_fading_v2v + noise_scale * new_fading_v2v
            current_fading_v2v = np.maximum(current_fading_v2v, 0.01)
            self.prev_fading_v2v = current_fading_v2v
        
        # V2I channels: g_i[t] = |f_i[t]|^2 * p_0^{v2i} * (d_0/d_i[t])^{γ^{v2i}}
        for i in range(self.num_vehicles):
            distance = np.linalg.norm(self.vehicle_positions[i] - self.base_station_position)
            distance = max(distance, self.reference_distance)  # Avoid division by zero
            
            # Path loss component
            path_loss = self.path_loss_constant_v2i * \
                       (self.reference_distance / distance) ** self.path_loss_exponent_v2i
            
            # Complete channel gain (no shadowing as per paper)
            self.channel_gains_v2i[i] = current_fading_v2i[i] * path_loss
        
        # V2V channels: h_{ij}[t] = |f_{ij}[t]|^2 * p_0^{v2v} * (d_0/d_{ij}[t])^{γ^{v2v}}
        for i in range(self.num_vehicles):
            for j in range(self.num_vehicles):
                if i != j:
                    distance = np.linalg.norm(self.vehicle_positions[i] - self.vehicle_positions[j])
                    distance = max(distance, self.reference_distance)
                    
                    # Path loss component
                    path_loss = self.path_loss_constant_v2v * \
                               (self.reference_distance / distance) ** self.path_loss_exponent_v2v
                    
                    # Complete channel gain (no shadowing as per paper)
                    self.channel_gains_v2v[i, j] = current_fading_v2v[i, j] * path_loss
    
    def _calculate_v2i_rate(self, vehicle_idx: int, power: float, bandwidth_scale: float = 1.0) -> float:
        """Calculate V2I transmission rate (Section III.D)"""
        if power <= 0 or bandwidth_scale <= 0:
            return 0
        
        # R_i^{v2i}[t] = B^{v2i} * log2(1 + p_i^{v2i}[t] * g_i[t] / (N_0 * B^{v2i}))
        effective_bandwidth = self.bandwidth_v2i * bandwidth_scale
        noise_power = self.noise_power_density * effective_bandwidth
        snr = power * self.channel_gains_v2i[vehicle_idx] / noise_power
        rate = effective_bandwidth * np.log2(1 + snr)
        
        return max(0, rate)
    
    def _calculate_v2v_rate(self, sender_idx: int, receiver_idx: int, power: float) -> float:
        """Calculate V2V transmission rate (Section III.D)"""
        if power <= 0:
            return 0
        
        # R_{ij}^{v2v}[t] = B^{v2v} * log2(1 + p_i^{v2v}[t] * h_{ij}[t] / (N_0 * B^{v2v}))
        noise_power = self.noise_power_density * self.bandwidth_v2v
        snr = power * self.channel_gains_v2v[sender_idx, receiver_idx] / noise_power
        rate = self.bandwidth_v2v * np.log2(1 + snr)
        
        return max(0, rate)
    
    def _update_vehicle_positions(self):
        """Update vehicle positions as they move"""
        # Each vehicle moves at its own speed
        for i in range(self.num_vehicles):
            distance_traveled = self.vehicle_speeds[i] * self.time_slot_duration
            self.vehicle_positions[i, 0] += distance_traveled
            
            # Wrap around if vehicle reaches end of road
            if self.vehicle_positions[i, 0] >= self.road_length:
                self.vehicle_positions[i, 0] -= self.road_length
                # Optionally change lane when wrapping
                new_lane = np.random.randint(0, self.num_lanes)
                self.vehicle_positions[i, 1] = new_lane * self.lane_width
                # Also update speed slightly when wrapping (±10% variation)
                speed_variation = 0.1 * (self.vehicle_speed_max - self.vehicle_speed_min)
                new_speed = self.vehicle_speeds[i] + np.random.uniform(-speed_variation, speed_variation)
                self.vehicle_speeds[i] = np.clip(new_speed, self.vehicle_speed_min, self.vehicle_speed_max)
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        state = []
        
        # Channel gains for followers (log scale for stability)
        for i in self.follower_indices:
            # V2I channel gain
            state.append(np.log10(self.channel_gains_v2i[i] + 1e-10))
            # V2V channel gain to leader
            state.append(np.log10(self.channel_gains_v2v[i, self.leader_idx] + 1e-10))
        
        # Vehicle distances from base station (normalized)
        for i in range(self.num_vehicles):
            distance = np.linalg.norm(self.vehicle_positions[i] - self.base_station_position)
            state.append(distance / 1000)  # Normalize to km
        
        # Current time slot (normalized)
        state.append(self.current_slot / self.num_time_slots)

        # Previous-slot local utilization feedback for each current follower.
        for i in self.follower_indices:
            state.append(float(self.prev_follower_time_utilization[i]))
            state.append(float(self.prev_follower_energy_utilization[i]))

        
        return np.array(state, dtype=np.float32)
    
    def get_info(self) -> dict:
        """Get detailed environment information"""
        # Calculate average rates
        avg_v2i_rate = np.mean([self._calculate_v2i_rate(i, self.max_power) 
                               for i in self.follower_indices])
        avg_v2v_rate = np.mean([self._calculate_v2v_rate(i, self.leader_idx, self.max_power) 
                               for i in self.follower_indices])
        
        # Calculate average redundancy for current chunks
        avg_redundancy = np.mean([self.redundancy_ratios[i, min(self.chunks_transmitted[i], 
                                                                self.chunks_per_vehicle - 1)]
                                 for i in self.follower_indices])
        
        return {
            'avg_distance': np.mean([np.linalg.norm(pos - self.base_station_position) 
                                   for pos in self.vehicle_positions]),
            'avg_v2i_rate_mbps': avg_v2i_rate / 1e6,
            'avg_v2v_rate_mbps': avg_v2v_rate / 1e6,
            'avg_redundancy': avg_redundancy,
            'chunks_progress': self.chunks_transmitted / self.chunks_per_vehicle,
            'current_slot': self.current_slot,
            'vehicle_positions': self.vehicle_positions.copy(),
            'leader_idx': self.leader_idx,
            'follower_indices': list(self.follower_indices),
            'leader_reselection_interval_slots': self.leader_reselection_interval_slots,
            'road_length': self.road_length,
            'num_lanes': self.num_lanes,
            'optimization_target': self.optimization_target
        }
    
    def get_local_observation_for_vehicle(self, vehicle_idx: int) -> np.ndarray:
        """
        Get local observation for a specific physical follower vehicle.

        This keeps replay transitions aligned to the vehicle that actually took
        the action, even if follower-slot identities change after leader
        reselection.
        """
        if vehicle_idx == self.leader_idx:
            raise ValueError("Leader vehicle does not have a follower local observation")

        state = []

        # Own channel conditions
        state.append(np.log10(self.channel_gains_v2i[vehicle_idx] + 1e-10))
        state.append(np.log10(self.channel_gains_v2v[vehicle_idx, self.leader_idx] + 1e-10))
        
        # Own distance from base station (normalized)
        distance = np.linalg.norm(self.vehicle_positions[vehicle_idx] - self.base_station_position)
        state.append(distance / 1000)  # Normalize to km
        
        # Leader information
        leader_distance = np.linalg.norm(self.vehicle_positions[self.leader_idx] - self.base_station_position)
        state.append(leader_distance / 1000)
        
        # Current time slot (normalized)
        state.append(self.current_slot / self.num_time_slots)

        # Previous-slot local constraint utilization feedback.
        state.append(float(self.prev_follower_time_utilization[vehicle_idx]))
        state.append(float(self.prev_follower_energy_utilization[vehicle_idx]))
        
        # The lagged global reward is intentionally excluded from the local
        # state so each follower reacts to its own current conditions.
        
        return np.array(state, dtype=np.float32)

    def get_local_observation(self, follower_idx: int) -> np.ndarray:
        """
        Get local observation for a follower slot (for decentralized agents).
        """
        vehicle_idx = self.follower_indices[follower_idx]
        return self.get_local_observation_for_vehicle(vehicle_idx)
    
    def get_centralized_state(self) -> np.ndarray:
        """
        Get full system state for centralized agent
        Same as _get_state() but exposed publicly
        """
        return self._get_state()
