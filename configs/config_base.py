"""
Base configuration for the paper experiments.
"""

def get_paper_config():
    """Get configuration for paper-based environment with all parameters"""
    
    # Single integrated objective controlled by the weight eta. The
    # environment also accepts 'time' and 'energy' as objective targets.
    optimization_target = 'mixed'
    mixed_time_weight = 0.5  # eta in the unified objective
    
    # The active leader uploads at twice the per-vehicle uplink power budget.
    leader_power_multiplier = 2.0
    
    # Base environment configuration matching paper's system model
    environment_config = {
        # Network parameters (Table I notation from paper)
        'num_vehicles': 5,  # N(t) - Total vehicles (1 leader + 4 followers)
        'num_time_slots': 30,  # T - Total time slots
        'time_slot_duration': 1.0,  # Δt - Duration of each slot in seconds
        
        # Communication parameters (Section III.D)
        'bandwidth_v2v': 10e6,  # B^{v2v} - 10 MHz for V2V
        'bandwidth_v2i': 20e6,  # B^{v2i} - 20 MHz for V2I
        'noise_power_density': 4e-21,  # N_0 - Noise power spectral density (-174 dBm/Hz)
        'max_power': 0.2,  # p_max - Maximum transmission power (200mW)
        
        # Channel model parameters (Section III.D)
        'reference_distance': 1.0,  # d_0 - Reference distance in meters
        'path_loss_constant_v2v': 2e-5,  # p_0^{v2v} - Path loss constant for V2V
        'path_loss_constant_v2i': 2e-5,  # p_0^{v2i} - Path loss constant for V2I
        'path_loss_exponent_v2v': 3.5,   # γ^{v2v} - Path loss exponent for V2V
        'path_loss_exponent_v2i': 3.5,   # γ^{v2i} - Path loss exponent for V2I
        'fading_correlation': 0.8,  # ρ - Correlation coefficient for fading between time slots (0-1)
        
        # Location and movement parameters
        'base_station_position': [225, 0],  # Base station at road start
        'vehicle_speed_min': 10,  # Minimum vehicle speed in m/s (36 km/h)
        'vehicle_speed_max': 15,  # Maximum vehicle speed in m/s (54 km/h)
        'num_lanes': 3,  # Number of lanes on the road
        'lane_width': 3.5,  # Width of each lane in meters
        'road_length': 1000,  # Length of the road in meters
        
        # Chunk-based data parameters (Section III.C)
        'chunks_per_vehicle': 30,  # K_i - Number of chunks per vehicle
        'chunk_size': 20e6,  # d_{i,k} - Size of each chunk in bits (20 Mb)
        
        # Redundancy parameters (Section III.F)
        'redundancy_lower': 0.5,  # β_{i,k} minimum - 50% redundancy
        'redundancy_upper': 0.5,  # β_{i,k} maximum - 50% redundancy
        
        # Deduplication parameters (Section III.F)
        'cpu_cycles_per_bit': 10,  # C_1 - CPU cycles per bit
        'chunk_overhead_cycles': 1e6,  # C_3 - Per-chunk overhead cycles
        'leader_cpu_frequency': 2.8e9,  # f_{i*} - Leader CPU frequency (2.8 GHz)
        'cpu_power_constant': 1e-27,  # κ - Hardware-dependent constant
        'leader_selection_mode': 'episode_start_scored',  # Leader selected by per-episode score
        'leader_selection_cpu_weight': 0.0,  # ζ - CPU weight in leader score
        'leader_selection_v2v_aggregation': 'sum',  # V2V score aggregation
        'leader_reselection_interval_slots': 10,  # 0 disables mid-episode reselection; 10 means reevaluate at slots 10 and 20
        
        # Constraints (Section IV)
        'time_budget': 2.0,  # T_max - Maximum time budget per slot
        'energy_budget': 1.0,  # E_max - Maximum energy budget per slot (1 Joule)
        'mixed_time_weight': mixed_time_weight,  # eta in the unified time-energy objective
        'reward_mode': 'paper_counted_normalized',  # Reward shaping mode
        'constraint_penalty_mode': 'flat_any_violation',  # Constraint penalty mode
        'constraint_penalty_value': 0.1,  # Lambda_cons / flat penalty magnitude
        'energy_reward_normalization_factor': 4.0,  # Energy-mode reward normalization
        
        # Optimization target
        'optimization_target': optimization_target,
        
        # Leader upload handling
        'ignore_leader_upload': False,  # Whether to ignore leader's upload to BS
        
        # Power constraint type
        'joint_power_constraint': False,  # False: separate constraints (p_v2v ≤ p_max, p_v2i ≤ p_max)
                                         # True: joint constraint (p_v2v + p_v2i ≤ p_max)
        'min_active_power_ratio': 1e-3,  # Active links are floored to this fraction of p_max to avoid zero-power data loss
        
        # Leader upload power
        'leader_power_multiplier': leader_power_multiplier,  # Leader upload power is leader_power_multiplier * p_max
        'leader_uplink_resource_factor': 1.0,  # Effective BS scheduling/resource multiplier for the active leader upload
    }
    
    
    # Training configuration
    training_config = {
        'num_episodes': 5000,
        'eval_interval': 50,
        'save_interval': 100,
    }
    
    # Complete configuration
    config = {
        'environment': environment_config,
        'training': training_config,
        'seed': 42,
    }
    
    return config
