"""
DDPG-specific configuration parameters
"""

def get_ddpg_config(mode='time'):
    """Get DDPG hyperparameters."""
    config = {
        # Network architecture
        'actor_hidden_sizes': [256, 256],
        'critic_hidden_sizes': [256, 256],
        'use_layer_norm': True,
        
        # Training parameters
        'actor_lr': 5e-5,
        'critic_lr': 1e-4,
        'batch_size': 256,
        'gamma': 0.99,
        'tau': 0.005,
        
        # Noise parameters
        'noise_type': 'ou',
        'ou_theta': 0.1,
        'ou_sigma': 0.05,
        'ou_dt': 1e-2,
        'noise_scale': 0.1,
        'noise_decay': 0.9995,
        'min_noise': 0.01,
        
        # Experience replay
        'buffer_size': 500000,
        'min_buffer_size': 1000,
        
        # Training schedule
        'update_every': 4,
        'num_updates': 12,
        
        # Gradient clipping
        'grad_clip': 1.0,
        
        # Epsilon-greedy exploration
        'use_preset_exploration': True,
        'epsilon_start': 0.1,
        'epsilon_decay': 0.999,
        'epsilon_min': 0.01,

        # Preset variation noise (for decentralized agents)
        'preset_noise_std': 0.05,

        # Initial action priors (mode-aware)
        'initial_delta_action_time': 0.5,
        'initial_power_action_time': 0.9,
        'initial_delta_action_energy': 0.5,
        'initial_power_action_energy': 0.1,
        
        # Reward clipping
        'reward_clip_min': -25,
        'reward_clip_max': 0,
    }
    return config
