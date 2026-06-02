"""
DQN-specific configuration parameters
"""

def get_dqn_config(mode='time'):
    """Get DQN hyperparameters."""
    config = {
        # Network architecture
        'hidden_sizes': [256, 256],
        'use_layer_norm': True,
        
        # Training parameters
        'learning_rate': 5e-5,
        'batch_size': 256,
        'gamma': 0.99,
        'double_dqn': True,
        'loss_type': 'huber',
        
        # Exploration
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.999,
        'epsilon_schedule': 'linear',
        'epsilon_linear_decay_episodes': 250,
        
        # Experience replay
        'buffer_size': 500000,
        'min_buffer_size': 3000,
        'prioritized_replay': False,
        'per_alpha': 0.6,
        'per_beta': 0.4,
        'per_priority_eps': 1e-6,
        
        # Training schedule
        'update_every': 4,
        'target_update_every': 25,
        'num_updates': 4,

        # Gradient clipping
        'grad_clip': 1.0,
        
        # Shared learner-side safety guard for critic stability
        'reward_clip_min': -25,
        'reward_clip_max': 0,
    }
    return config
