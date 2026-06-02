"""
Common action presets for all algorithms - ensures fair comparison
"""

def get_action_presets():
    """Get common action presets used by all algorithms"""
    return [
        # Extreme scenarios
        [0.0, 0.0, 1.0],  # V2V blocked
        [1.0, 1.0, 0.0],  # V2I poor

        # V2I-favorable (near infrastructure)
        [0.1, 1.0, 1.0],  # High-power
        [0.2, 1.0, 1.0],  # High-power
        [0.3, 1.0, 1.0],  # High-power
        [0.1, 0.6, 0.8],  # Moderate-power
        [0.2, 0.6, 0.8],  # Moderate-power
        [0.3, 0.6, 0.8],  # Moderate-power
        [0.1, 0.3, 0.6],  # Energy-lean
        [0.2, 0.3, 0.6],  # Energy-lean
        [0.3, 0.3, 0.6],  # Energy-lean

        # Balanced channels (diversity gain)
        [0.4, 1.0, 1.0],  # High-power
        [0.5, 1.0, 1.0],  # High-power
        [0.6, 1.0, 1.0],  # High-power
        [0.4, 0.7, 0.7],  # Moderate-power
        [0.5, 0.7, 0.7],  # Moderate-power
        [0.6, 0.7, 0.7],  # Moderate-power
        [0.4, 0.4, 0.5],  # Energy-lean
        [0.5, 0.4, 0.4],  # Energy-lean
        [0.6, 0.5, 0.4],  # Energy-lean

        # Energy-critical
        [0.2, 0.3, 0.5],  # V2I-lean bridge
        [0.3, 0.4, 0.6],  # V2I bias, graduated power
        [0.4, 0.5, 0.6],  # Slight V2I bias
        [0.5, 0.5, 0.5],  # Uniform low power
        [0.6, 0.6, 0.5],  # Slight V2V bias
        [0.7, 0.6, 0.4],  # V2V bias, graduated power
        [0.8, 0.5, 0.3],  # V2V-lean bridge

        # V2V-favorable (convoy/platooning)
        [0.7, 1.0, 1.0],  # High-power
        [0.8, 1.0, 1.0],  # High-power
        [0.9, 1.0, 1.0],  # High-power
        [0.7, 0.8, 0.6],  # Moderate-power
        [0.8, 0.8, 0.6],  # Moderate-power
        [0.9, 0.8, 0.6],  # Moderate-power
        [0.7, 0.6, 0.3],  # Energy-lean
        [0.8, 0.6, 0.3],  # Energy-lean
        [0.9, 0.6, 0.3],  # Energy-lean

        # Ultra-low-power (energy-minimal strategies)
        [0.2, 0.05, 0.15],  # Low V2V, low V2I
        [0.3, 0.05, 0.20],  # Low V2V, slightly more V2I
        [0.4, 0.10, 0.15],  # Moderate split, minimal power
        [0.2, 0.10, 0.20],  # Low split, low power
        [0.3, 0.10, 0.10],  # Symmetric minimal
        [0.5, 0.05, 0.10],  # V2V-biased split, minimal power
        [0.2, 0.15, 0.25],  # Slightly above minimal
        [0.4, 0.15, 0.20],  # Balanced split, low power
        [0.3, 0.20, 0.20],  # Upper edge of low-power region
    ]
