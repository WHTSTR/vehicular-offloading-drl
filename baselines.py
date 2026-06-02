import numpy as np

class BaselineAlgorithms:
    """
    Baseline offloading strategies for comparison with the DRL methods.
    """
    
    def __init__(self, num_followers: int, max_power: float):
        self.num_followers = num_followers
        self.max_power = max_power
        
    def _create_action(self, offloading_fractions: np.ndarray, 
                      v2v_power_ratios: np.ndarray, 
                      v2i_power_ratios: np.ndarray) -> np.ndarray:
        """
        Create action vector from components.
        With separate power budgets for V2V and V2I.
        """
        action = np.zeros(self.num_followers * 3)
        
        for i in range(self.num_followers):
            # Offloading fraction
            action[i * 3] = offloading_fractions[i]
            
            # Power allocation - separate budgets for V2V and V2I
            # Each can use up to 1.0 (full power) independently
            action[i * 3 + 1] = np.clip(v2v_power_ratios[i], 0.0, 1.0)
            action[i * 3 + 2] = np.clip(v2i_power_ratios[i], 0.0, 1.0)
            
        return action
        
    def all_leader_strategy(self, state: np.ndarray) -> np.ndarray:
        """
        All vehicles transmit their data to the leader vehicle over V2V links.
        This strategy maximizes deduplication benefits but may suffer from 
        congestion at the leader.
        """
        offloading_fractions = np.ones(self.num_followers)  # δ = 1 for all
        v2v_power_ratios = np.ones(self.num_followers)  # Use full power for V2V
        v2i_power_ratios = np.zeros(self.num_followers)  # No V2I transmission
        
        return self._create_action(offloading_fractions, v2v_power_ratios, v2i_power_ratios)
        
    def all_base_station_strategy(self, state: np.ndarray) -> np.ndarray:
        """
        All vehicles directly transmit to the base station over V2I links.
        This strategy avoids V2V overhead but misses deduplication opportunities.
        """
        offloading_fractions = np.zeros(self.num_followers)  # δ = 0 for all
        v2v_power_ratios = np.zeros(self.num_followers)  # No V2V transmission
        v2i_power_ratios = np.ones(self.num_followers)  # Use full power for V2I
        
        return self._create_action(offloading_fractions, v2v_power_ratios, v2i_power_ratios)
        
    def balanced_strategy(self, state: np.ndarray) -> np.ndarray:
        """
        Half of the data is sent over V2V, half over V2I.
        This strategy provides a middle ground between the two extremes.
        """
        offloading_fractions = np.full(self.num_followers, 0.5)  # δ = 0.5 for all
        v2v_power_ratios = np.full(self.num_followers, 1.0)  # Full power on both links (the joint power cap splits it evenly)
        v2i_power_ratios = np.full(self.num_followers, 1.0)
        
        return self._create_action(offloading_fractions, v2v_power_ratios, v2i_power_ratios)
