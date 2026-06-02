"""
Open-ended horizontal U-turn environment for transfer experiments.

This keeps the state/action interfaces identical to the straight-road paper
environment so previously trained models can be evaluated zero-shot.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from environment import PaperVehicularEnvironment


class UTurnVehicularEnvironment(PaperVehicularEnvironment):
    """Paper environment with vehicle motion constrained to an open-ended U-turn."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.road_geometry = config.get("road_geometry", "uturn_horizontal")
        if self.road_geometry != "uturn_horizontal":
            raise ValueError("UTurnVehicularEnvironment expects road_geometry='uturn_horizontal'")

        self.uturn_radius = float(config.get("uturn_radius", 25.0))
        self.uturn_leg_length = float(config.get("uturn_leg_length", 154.48009183012758))
        self.uturn_open_ended = bool(config.get("uturn_open_ended", True))
        if self.uturn_radius <= 0.0:
            raise ValueError("uturn_radius must be positive")
        if self.uturn_leg_length <= 0.0:
            raise ValueError("uturn_leg_length must be positive")

        self.uturn_total_arc_length = 2.0 * self.uturn_leg_length + np.pi * self.uturn_radius
        self.vehicle_arc_positions = None
        self.vehicle_lane_indices = None

    def _centerline_position(self, s: float) -> np.ndarray:
        """Map path length to the centerline position on the U-turn road."""
        if s <= self.uturn_leg_length:
            return np.array([s, 2.0 * self.uturn_radius], dtype=np.float64)
        if s <= self.uturn_leg_length + np.pi * self.uturn_radius:
            arc_len = s - self.uturn_leg_length
            theta = arc_len / self.uturn_radius
            x = self.uturn_leg_length + self.uturn_radius * np.sin(theta)
            y = self.uturn_radius + self.uturn_radius * np.cos(theta)
            return np.array([x, y], dtype=np.float64)

        lower_leg_progress = s - self.uturn_leg_length - np.pi * self.uturn_radius
        return np.array([self.uturn_leg_length - lower_leg_progress, 0.0], dtype=np.float64)

    def _centerline_tangent(self, s: float) -> np.ndarray:
        """Unit tangent aligned with the direction of travel."""
        if s <= self.uturn_leg_length:
            return np.array([1.0, 0.0], dtype=np.float64)
        if s <= self.uturn_leg_length + np.pi * self.uturn_radius:
            arc_len = s - self.uturn_leg_length
            theta = arc_len / self.uturn_radius
            tangent = np.array([np.cos(theta), -np.sin(theta)], dtype=np.float64)
            return tangent / np.linalg.norm(tangent)
        return np.array([-1.0, 0.0], dtype=np.float64)

    def _centerline_normal(self, s: float) -> np.ndarray:
        """Right-hand normal of the travel direction."""
        tangent = self._centerline_tangent(s)
        normal = np.array([tangent[1], -tangent[0]], dtype=np.float64)
        return normal / np.linalg.norm(normal)

    def _lane_offset(self, lane_idx: int) -> float:
        return (float(lane_idx) - 0.5 * (self.num_lanes - 1)) * self.lane_width

    def _vehicle_position_from_arc(self, arc_position: float, lane_idx: int) -> np.ndarray:
        center = self._centerline_position(arc_position)
        return center + self._lane_offset(lane_idx) * self._centerline_normal(arc_position)

    def _refresh_vehicle_positions_from_arc(self) -> None:
        for i in range(self.num_vehicles):
            self.vehicle_positions[i] = self._vehicle_position_from_arc(
                float(self.vehicle_arc_positions[i]),
                int(self.vehicle_lane_indices[i]),
            )

    def reset(self) -> np.ndarray:
        """Reset environment to the initial U-turn state."""
        self.current_slot = 0
        self.vehicle_positions = np.zeros((self.num_vehicles, 2), dtype=np.float64)
        self.vehicle_arc_positions = self._sample_initial_x_positions().astype(np.float64)
        self.vehicle_speeds = np.random.uniform(
            self.vehicle_speed_min,
            self.vehicle_speed_max,
            self.num_vehicles,
        )
        self.vehicle_lane_indices = np.random.randint(0, self.num_lanes, size=self.num_vehicles)
        self._refresh_vehicle_positions_from_arc()

        self._initialize_channel_conditions()
        self._select_episode_leader()

        self.redundancy_ratios = np.random.uniform(
            self.redundancy_lower,
            self.redundancy_upper,
            (self.num_vehicles, self.chunks_per_vehicle),
        )
        self.chunks_transmitted = np.zeros(self.num_vehicles, dtype=int)
        self.prev_follower_time_utilization = np.zeros(self.num_vehicles, dtype=np.float32)
        self.prev_follower_energy_utilization = np.zeros(self.num_vehicles, dtype=np.float32)
        return self._get_state()

    def _update_vehicle_positions(self):
        """Advance each vehicle along the U-turn centerline."""
        for i in range(self.num_vehicles):
            self.vehicle_arc_positions[i] += self.vehicle_speeds[i] * self.time_slot_duration

            if not self.uturn_open_ended and self.vehicle_arc_positions[i] >= self.uturn_total_arc_length:
                self.vehicle_arc_positions[i] -= self.uturn_total_arc_length
                self.vehicle_lane_indices[i] = np.random.randint(0, self.num_lanes)
                speed_variation = 0.1 * (self.vehicle_speed_max - self.vehicle_speed_min)
                new_speed = self.vehicle_speeds[i] + np.random.uniform(-speed_variation, speed_variation)
                self.vehicle_speeds[i] = np.clip(new_speed, self.vehicle_speed_min, self.vehicle_speed_max)

        self._refresh_vehicle_positions_from_arc()

    def get_info(self) -> dict:
        info = super().get_info()
        info.update(
            {
                "road_geometry": self.road_geometry,
                "uturn_radius": self.uturn_radius,
                "uturn_leg_length": self.uturn_leg_length,
                "uturn_open_ended": self.uturn_open_ended,
                "vehicle_arc_positions": self.vehicle_arc_positions.copy(),
                "vehicle_lane_indices": self.vehicle_lane_indices.copy(),
            }
        )
        return info

    def evaluate_action_post_update(self, action):
        """Extend parent to also save/restore U-turn arc state."""
        saved_arc = self.vehicle_arc_positions.copy()
        saved_lanes = self.vehicle_lane_indices.copy()
        try:
            return super().evaluate_action_post_update(action)
        finally:
            self.vehicle_arc_positions = saved_arc
            self.vehicle_lane_indices = saved_lanes
