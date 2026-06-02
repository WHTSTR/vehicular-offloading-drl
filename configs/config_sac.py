"""
SAC-specific configuration parameters
"""

def get_sac_config(mode='time'):
    """Get SAC hyperparameters."""
    config = {
        # Network architecture
        'actor_hidden_sizes': [256, 256],
        'critic_hidden_sizes': [256, 256],
        'use_layer_norm': True,
        
        # Training parameters
        'actor_lr': 1e-4,
        'critic_lr': 2e-4,
        'alpha_lr': 1e-4,
        'batch_size': 256,  # Increased from 128 for reduced variance
        'gamma': 0.99,
        'tau': 0.005,
        
        # SAC-specific
        'alpha': 3e-4,
        'automatic_entropy_tuning': True,
        'target_entropy': None,  # If None, uses -dim(A)
        
        # Experience replay
        'buffer_size': 500000,
        'min_buffer_size': 1000,
        
        # Training schedule
        'update_every': 4,
        'num_updates': 12,
        'actor_update_every': 4,
        'actor_use_min_q': True,
        
        # Gradient clipping
        'grad_clip': 1.0,
        
        # Action limits
        'action_bounds': [0.0, 1.0],
        
        # Epsilon for preset exploration
        'use_preset_exploration': True,
        'epsilon_start': 0.1,
        'epsilon_decay': 0.999,
        'epsilon_min': 0.01,

        # Log std bounds for stochastic policy
        'log_std_min': -20,
        'log_std_max': -0.5,

        # Initialization priors for SAC actor heads. These are useful in RL
        # because the initial policy determines the early replay distribution.
        'initial_delta_bias_time': 0.0,
        'initial_power_bias_time': 1.1,
        'initial_delta_bias_energy': 0.0,
        'initial_power_bias_energy': -1.1,
        'initial_log_std_bias': -2.0,
        
        # Preset variation noise (for exploration)
        'preset_noise_std': 0.05,
        
        # Shared learner-side safety guard for critic stability
        'reward_clip_min': -25,
        'reward_clip_max': 0,
    }
    return config
