"""
Continuous-search benchmarks (differential evolution) over the action space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import differential_evolution, minimize

from baselines import BaselineAlgorithms
from configs.common_presets import get_action_presets


class DifferentialEvolutionOracle:
    """
    Continuous per-slot search baseline.

    The current simulator's action affects only the current slot cost, so a
    slot-wise numerical search is a valid strong baseline under this model.
    """

    def __init__(
        self,
        env,
        maxiter: int = 25,
        popsize: int = 8,
        polish: bool = True,
        seed: int = 42,
        local_refine: bool = True,
        powell_maxiter: int = 60,
        num_restarts: int = 2,
        use_guidance_surrogate: bool = True,
        guidance_linear_scale: float = 1.5,
        guidance_quadratic_scale: float = 3.0,
        local_refine_top_k: int = 3,
        block_refine: bool = True,
        block_refine_passes: int = 1,
        block_powell_maxiter: int = 25,
        coordinate_polish: bool = True,
        coordinate_step_schedule: tuple[float, ...] = (0.2, 0.1, 0.05),
        init_jitter: float = 0.08,
        selection_mode: str = "penalized_objective",
    ):
        self.env = env
        self.maxiter = int(maxiter)
        self.popsize = int(popsize)
        self.polish = bool(polish)
        self.seed = int(seed)
        self.local_refine = bool(local_refine)
        self.powell_maxiter = int(powell_maxiter)
        self.num_restarts = int(num_restarts)
        self.use_guidance_surrogate = bool(use_guidance_surrogate)
        self.guidance_linear_scale = float(guidance_linear_scale)
        self.guidance_quadratic_scale = float(guidance_quadratic_scale)
        self.local_refine_top_k = int(local_refine_top_k)
        self.block_refine = bool(block_refine)
        self.block_refine_passes = int(block_refine_passes)
        self.block_powell_maxiter = int(block_powell_maxiter)
        self.coordinate_polish = bool(coordinate_polish)
        self.coordinate_step_schedule = tuple(float(step) for step in coordinate_step_schedule)
        self.init_jitter = float(init_jitter)
        self.selection_mode = str(selection_mode)
        self.bounds = [(0.0, 1.0)] * (self.env.num_followers * 3)
        self._slot_counter = 0
        self._rng = np.random.default_rng(seed)
        self._baselines = BaselineAlgorithms(self.env.num_followers, self.env.max_power)
        self.last_search_stats = {}
        self.aggregate_search_stats = {
            "slots": 0,
            "de_restarts": 0,
            "powell_runs": 0,
            "block_powell_runs": 0,
            "coordinate_evals": 0,
            "candidates_considered": 0,
        }

    @dataclass
    class _Candidate:
        action: np.ndarray
        true_objective: float
        search_objective: float
        objective_cost: float
        time_cost: float
        energy_cost: float
        constraint_excess: float
        constraint_violations: int
        follower_time_violations: int
        leader_time_violations: int
        feasible: bool

    def _evaluate_info(self, x: np.ndarray) -> dict:
        _, info = self.env.evaluate_action_post_update(np.asarray(x, dtype=np.float32))
        return info

    def _true_objective(self, x: np.ndarray) -> float:
        return float(self._evaluate_info(x)["penalized_objective"])

    def _candidate_from_action(
        self,
        x: np.ndarray,
        search_objective: Optional[float] = None,
    ) -> "DifferentialEvolutionOracle._Candidate":
        action = np.asarray(x, dtype=np.float32)
        info = self._evaluate_info(action)
        if search_objective is None:
            search_objective = float(info["penalized_objective"])
        return self._Candidate(
            action=action,
            true_objective=float(info["penalized_objective"]),
            search_objective=float(search_objective),
            objective_cost=float(info.get("objective_cost", info.get("mixed_cost", 0.0))),
            time_cost=float(info.get("time_cost", 0.0)),
            energy_cost=float(info.get("energy_cost", 0.0)),
            constraint_excess=float(info.get("constraint_excess", 0.0)),
            constraint_violations=int(info.get("constraint_violations", 0)),
            follower_time_violations=int(info.get("follower_time_violations", 0)),
            leader_time_violations=int(info.get("leader_time_violation", 0)),
            feasible=not bool(info.get("constraint_violated", False)),
        )

    def _constraint_excess(self, info: dict) -> float:
        follower_time_budget = max(
            float(getattr(self.env, "follower_time_budget", self.env.time_budget)),
            1e-12,
        )
        leader_time_budget = max(
            float(getattr(self.env, "leader_time_budget", self.env.time_budget)),
            1e-12,
        )
        follower_energy_budget = max(
            float(getattr(self.env, "follower_energy_budget", self.env.energy_budget)),
            1e-12,
        )
        leader_energy_budget = max(
            float(getattr(self.env, "leader_energy_budget", self.env.energy_budget)),
            1e-12,
        )
        follower_times = info.get("follower_times", [])
        follower_energies = info.get("follower_energies", [])
        leader_time = float(info.get("dedup_time", 0.0) + info.get("leader_upload_time", 0.0))
        leader_energy = float(info.get("dedup_energy", 0.0) + info.get("leader_upload_energy", 0.0))

        excess = 0.0
        for value in follower_times:
            excess += max((float(value) - follower_time_budget) / follower_time_budget, 0.0)
        excess += max((leader_time - leader_time_budget) / leader_time_budget, 0.0)
        for value in follower_energies:
            excess += max((float(value) - follower_energy_budget) / follower_energy_budget, 0.0)
        excess += max((leader_energy - leader_energy_budget) / leader_energy_budget, 0.0)
        return excess

    def _search_penalty(self, info: dict) -> float:
        excess = self._constraint_excess(info)
        if excess <= 0.0:
            return 0.0
        base = float(self.env.constraint_penalty_value)
        return base * (
            self.guidance_linear_scale * excess
            + self.guidance_quadratic_scale * (excess ** 2)
        )

    def _search_objective(self, x: np.ndarray) -> float:
        info = self._evaluate_info(x)
        if not self.use_guidance_surrogate:
            return float(info["penalized_objective"])
        # Search on the true counted-penalty objective, then add smooth excess
        # guidance so the solver can distinguish "slightly infeasible" from
        # "heavily infeasible" candidates.
        return float(info["penalized_objective"] + self._search_penalty(info))

    def _heuristic_candidates(self) -> list[np.ndarray]:
        state = self.env._get_state()
        candidates = [
            self._baselines.all_leader_strategy(state).astype(np.float32),
            self._baselines.all_base_station_strategy(state).astype(np.float32),
            self._baselines.balanced_strategy(state).astype(np.float32),
            np.full(self.env.num_followers * 3, 0.5, dtype=np.float32),
        ]
        midpoint = np.zeros(self.env.num_followers * 3, dtype=np.float32)
        midpoint[0::3] = 0.5
        midpoint[1::3] = 0.4
        midpoint[2::3] = 0.4
        candidates.append(midpoint)
        return candidates

    def _initial_population(self, slot_seed: int) -> np.ndarray:
        dim = len(self.bounds)
        population_size = max(8, self.popsize * dim)
        rng = np.random.default_rng(slot_seed)
        heuristics = self._heuristic_candidates()
        population = []
        for candidate in heuristics:
            population.append(np.asarray(candidate, dtype=np.float64))
            population.append(
                np.clip(
                    np.asarray(candidate, dtype=np.float64) + rng.normal(0.0, self.init_jitter, dim),
                    0.0,
                    1.0,
                )
            )
        while len(population) < population_size:
            population.append(rng.uniform(0.0, 1.0, dim))
        return np.asarray(population[:population_size], dtype=np.float64)

    def _deduplicate_candidates(self, candidates: list["DifferentialEvolutionOracle._Candidate"]) -> list["DifferentialEvolutionOracle._Candidate"]:
        unique = {}
        for candidate in candidates:
            key = tuple(np.round(candidate.action.astype(np.float64), 4))
            previous = unique.get(key)
            if previous is None or candidate.true_objective < previous.true_objective:
                unique[key] = candidate
        return list(unique.values())

    def _selection_key(self, candidate: "DifferentialEvolutionOracle._Candidate") -> tuple[float, ...]:
        if self.selection_mode == "penalized_objective":
            return (
                candidate.true_objective,
                candidate.search_objective,
                candidate.time_cost,
                candidate.energy_cost,
            )
        if self.selection_mode == "feasible_first":
            if candidate.feasible:
                return (
                    0.0,
                    candidate.objective_cost,
                    candidate.time_cost,
                    candidate.energy_cost,
                    candidate.search_objective,
                )
            return (
                1.0,
                candidate.constraint_excess,
                candidate.objective_cost,
                candidate.time_cost,
                candidate.energy_cost,
                candidate.search_objective,
            )
        if self.selection_mode == "follower_time_first":
            return (
                candidate.follower_time_violations,
                candidate.constraint_violations,
                candidate.constraint_excess,
                candidate.objective_cost,
                candidate.time_cost,
                candidate.energy_cost,
                candidate.search_objective,
            )
        raise ValueError(f"Unknown DE selection_mode: {self.selection_mode}")

    def _powell_refine(self, x0: np.ndarray) -> tuple[np.ndarray, float]:
        result = minimize(
            self._search_objective,
            x0=np.asarray(x0, dtype=np.float64),
            method="Powell",
            bounds=self.bounds,
            options={"maxiter": self.powell_maxiter, "disp": False},
        )
        x = np.clip(np.asarray(result.x, dtype=np.float32), 0.0, 1.0)
        return x, float(self._true_objective(x))

    def _block_powell_refine(self, x0: np.ndarray) -> tuple[np.ndarray, float, int]:
        best_x = np.asarray(x0, dtype=np.float32).copy()
        best_obj = float(self._true_objective(best_x))
        runs = 0

        for _ in range(max(1, self.block_refine_passes)):
            info = self._evaluate_info(best_x)
            follower_times = np.asarray(info.get("follower_times", [0.0] * self.env.num_followers), dtype=np.float64)
            order = list(np.argsort(-follower_times))
            improved = False

            for follower_slot in order:
                sl = slice(3 * follower_slot, 3 * (follower_slot + 1))
                x_block0 = np.asarray(best_x[sl], dtype=np.float64)

                def block_objective(block_x: np.ndarray) -> float:
                    trial = best_x.copy()
                    trial[sl] = np.clip(np.asarray(block_x, dtype=np.float32), 0.0, 1.0)
                    return float(self._search_objective(trial))

                result = minimize(
                    block_objective,
                    x0=x_block0,
                    method="Powell",
                    bounds=[(0.0, 1.0)] * 3,
                    options={"maxiter": self.block_powell_maxiter, "disp": False},
                )
                runs += 1
                trial = best_x.copy()
                trial[sl] = np.clip(np.asarray(result.x, dtype=np.float32), 0.0, 1.0)
                trial_obj = float(self._true_objective(trial))
                if trial_obj + 1e-12 < best_obj:
                    best_x = trial
                    best_obj = trial_obj
                    improved = True

            if not improved:
                break

        return best_x, best_obj, runs

    def _coordinate_polish(self, x0: np.ndarray) -> tuple[np.ndarray, float, int]:
        best_x = np.asarray(x0, dtype=np.float32).copy()
        best_obj = float(self._true_objective(best_x))
        evals = 1

        for step in self.coordinate_step_schedule:
            improved = True
            while improved:
                improved = False
                for dim in range(len(best_x)):
                    current_value = float(best_x[dim])
                    for delta in (-step, step):
                        trial_value = float(np.clip(current_value + delta, 0.0, 1.0))
                        if abs(trial_value - current_value) < 1e-12:
                            continue
                        trial = best_x.copy()
                        trial[dim] = trial_value
                        trial_obj = float(self._true_objective(trial))
                        evals += 1
                        if trial_obj + 1e-12 < best_obj:
                            best_x = trial
                            best_obj = trial_obj
                            improved = True
        return best_x, best_obj, evals

    def select_action(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        del state
        candidates: list[DifferentialEvolutionOracle._Candidate] = []

        for candidate in self._heuristic_candidates():
            action = np.asarray(candidate, dtype=np.float32)
            search_obj = float(self._search_objective(action))
            candidates.append(self._candidate_from_action(action, search_objective=search_obj))

        self._slot_counter += 1
        slot_seed = self.seed + self._slot_counter * 1000
        for restart_idx in range(max(1, self.num_restarts)):
            init_population = self._initial_population(slot_seed + restart_idx)
            result = differential_evolution(
                self._search_objective,
                bounds=self.bounds,
                maxiter=self.maxiter,
                popsize=self.popsize,
                polish=self.polish,
                seed=slot_seed + restart_idx,
                init=init_population,
                updating="deferred",
                workers=1,
            )
            de_x = np.clip(np.asarray(result.x, dtype=np.float32), 0.0, 1.0)
            de_search_obj = float(self._search_objective(de_x))
            candidates.append(self._candidate_from_action(de_x, search_objective=de_search_obj))
            self.aggregate_search_stats["de_restarts"] += 1

        if self.local_refine:
            unique_candidates = self._deduplicate_candidates(candidates)
            unique_candidates.sort(key=lambda candidate: (candidate.true_objective, candidate.search_objective))
            top_candidates = unique_candidates[:max(1, self.local_refine_top_k)]
            coordinate_evals = 0

            for candidate in top_candidates:
                refined_x, refined_obj = self._powell_refine(candidate.action)
                self.aggregate_search_stats["powell_runs"] += 1
                candidates.append(
                    self._candidate_from_action(refined_x, search_objective=float(refined_obj))
                )
                if self.block_refine:
                    block_x, block_obj, block_runs = self._block_powell_refine(refined_x)
                    self.aggregate_search_stats["block_powell_runs"] += int(block_runs)
                    candidates.append(
                        self._candidate_from_action(block_x, search_objective=float(block_obj))
                    )
                if self.coordinate_polish:
                    polished_x, polished_obj, polish_evals = self._coordinate_polish(refined_x)
                    coordinate_evals += polish_evals
                    candidates.append(
                        self._candidate_from_action(polished_x, search_objective=float(polished_obj))
                    )
            self.aggregate_search_stats["coordinate_evals"] += int(coordinate_evals)
            self.aggregate_search_stats["candidates_considered"] += int(len(top_candidates))
            candidates_considered = len(top_candidates)
        else:
            candidates_considered = len(candidates)

        unique_candidates = self._deduplicate_candidates(candidates)
        best_candidate = min(unique_candidates, key=self._selection_key)
        best_x = best_candidate.action

        self.last_search_stats = {
            "slot": self._slot_counter,
            "best_penalized_objective": float(best_candidate.true_objective),
            "best_objective_cost": float(best_candidate.objective_cost),
            "best_time_cost": float(best_candidate.time_cost),
            "best_energy_cost": float(best_candidate.energy_cost),
            "best_constraint_excess": float(best_candidate.constraint_excess),
            "best_constraint_violations": int(best_candidate.constraint_violations),
            "best_feasible": bool(best_candidate.feasible),
            "selection_mode": self.selection_mode,
            "feasible_candidates": int(sum(candidate.feasible for candidate in unique_candidates)),
            "candidates_considered": int(candidates_considered),
            "de_restarts": int(max(1, self.num_restarts)),
            "local_refine": bool(self.local_refine),
            "block_refine": bool(self.block_refine),
            "coordinate_polish": bool(self.coordinate_polish),
        }
        self.aggregate_search_stats["slots"] += 1

        return np.asarray(best_x, dtype=np.float32)


@dataclass
class _FollowerPresetMetrics:
    delta: float
    action: np.ndarray
    follower_time: float
    follower_energy: float
    v2v_data: float
    unique_v2v_data: float
    active_v2v: int
    follower_time_violation: int
    follower_energy_violation: int
    follower_time_excess: float
    follower_energy_excess: float
    local_objective: float
    local_penalized: float
    local_violation_count: int


class PresetBranchAndBoundOracle:
    """
    Discrete branch-and-bound search over the common 36-preset library.

    This is an anytime heuristic:
    - exact on the explored subtree
    - returns the best incumbent if the node budget is reached
    - evaluates the true penalized objective at every complete leaf
    """

    def __init__(
        self,
        env,
        max_nodes: int = 50000,
        incumbent_passes: int = 2,
        seed: int = 42,
    ):
        self.env = env
        self.max_nodes = int(max_nodes)
        self.incumbent_passes = int(incumbent_passes)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._baselines = BaselineAlgorithms(self.env.num_followers, self.env.max_power)
        self._presets = np.asarray(get_action_presets(), dtype=np.float32)
        self._slot_counter = 0
        self.last_search_stats = {}
        self.aggregate_search_stats = {
            "slots": 0,
            "nodes_expanded": 0,
            "nodes_pruned": 0,
            "leaves_evaluated": 0,
            "hit_node_budget": 0,
        }

    def _build_action_from_indices(self, preset_indices: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.num_followers * 3, dtype=np.float32)
        for follower_slot, preset_idx in enumerate(preset_indices):
            action[follower_slot * 3:(follower_slot + 1) * 3] = self._presets[int(preset_idx)]
        return action

    def _evaluate_info(self, action: np.ndarray) -> dict:
        _, info = self.env.evaluate_action_post_update(np.asarray(action, dtype=np.float32))
        return info

    def _objective_from_totals(
        self,
        follower_time_sum: float,
        follower_energy_sum: float,
        total_v2v_data: float,
        follower_unique_data: float,
        active_v2v_count: int,
    ) -> tuple[float, float, float, float]:
        leader_chunk_data = float(self.env.chunk_size)
        total_dedup_input_data = 0.0
        active_dedup_chunks = 0
        total_unique_data = leader_chunk_data
        if active_v2v_count > 0:
            total_dedup_input_data = leader_chunk_data + total_v2v_data
            active_dedup_chunks = active_v2v_count + 1
            total_unique_data = leader_chunk_data + follower_unique_data

        t_dedup = 0.0
        e_dedup = 0.0
        if total_dedup_input_data > 0.0:
            cpu_cycles = (
                self.env.cpu_cycles_per_bit * total_dedup_input_data
                + self.env.chunk_overhead_cycles * active_dedup_chunks
            )
            t_dedup = cpu_cycles / self.env.leader_cpu_frequency
            e_dedup = self.env.cpu_power_constant * (self.env.leader_cpu_frequency ** 2) * t_dedup

        t_upload = 0.0
        e_upload = 0.0
        if total_unique_data > 0.0 and not self.env.ignore_leader_upload:
            leader_upload_power = self.env.max_power * self.env.leader_power_multiplier
            leader_v2i_rate = self.env._calculate_v2i_rate(self.env.leader_idx, leader_upload_power)
            if leader_v2i_rate > 0:
                t_upload = total_unique_data / leader_v2i_rate
                e_upload = leader_upload_power * t_upload

        total_time = follower_time_sum + t_dedup + t_upload
        total_energy = follower_energy_sum + e_dedup + e_upload

        if self.env.optimization_target == "time":
            objective = total_time
        elif self.env.optimization_target == "energy":
            objective = total_energy
        else:
            time_normalizer = (
                self.env.num_followers * self.env.follower_time_budget + self.env.leader_time_budget
                if (self.env.follower_time_budget > 0 and self.env.leader_time_budget > 0)
                else 1.0
            )
            energy_normalizer = (
                self.env.num_followers * self.env.follower_energy_budget + self.env.leader_energy_budget
                if (self.env.follower_energy_budget > 0 and self.env.leader_energy_budget > 0)
                else 1.0
            )
            weight = self.env._get_effective_mixed_time_weight()
            objective = (
                weight * (total_time / time_normalizer)
                + (1.0 - weight) * (total_energy / energy_normalizer)
            )

        return float(objective), float(t_dedup + t_upload), float(e_dedup + e_upload), float(total_unique_data)

    def _penalty_from_counts(self, violation_count: int, any_violation: bool) -> float:
        if not any_violation:
            return 0.0
        if self.env.active_constraint_penalty_mode == "flat_any_violation":
            return float(self.env.constraint_penalty_value)
        if self.env.active_constraint_penalty_mode == "per_violation":
            return float(self.env.constraint_penalty_value) * float(violation_count)
        if self.env.active_constraint_penalty_mode == "normalized_excess":
            return 0.0
        raise ValueError(f"Unknown constraint penalty mode: {self.env.active_constraint_penalty_mode}")

    def _penalty_from_excess(self, excess: float) -> float:
        if excess <= 0.0:
            return 0.0
        if self.env.active_constraint_penalty_mode == "normalized_excess":
            return float(self.env.constraint_penalty_value) * float(excess)
        return 0.0

    def _compute_follower_metrics(self) -> list[list[_FollowerPresetMetrics]]:
        active_leader_idx = self.env.leader_idx
        metrics: list[list[_FollowerPresetMetrics]] = []

        for follower_slot, follower_idx in enumerate(self.env.follower_indices):
            chunk_idx = min(self.env.chunks_transmitted[follower_idx], self.env.chunks_per_vehicle - 1)
            beta = float(self.env.redundancy_ratios[follower_idx, chunk_idx])
            follower_metrics: list[_FollowerPresetMetrics] = []

            for preset in self._presets:
                delta = float(preset[0])
                p_v2v = float(preset[1]) * self.env.max_power
                p_v2i = float(preset[2]) * self.env.max_power

                if self.env.joint_power_constraint:
                    total_power = p_v2v + p_v2i
                    if total_power > self.env.max_power:
                        scale = self.env.max_power / total_power
                        p_v2v *= scale
                        p_v2i *= scale
                else:
                    p_v2v = min(p_v2v, self.env.max_power)
                    p_v2i = min(p_v2i, self.env.max_power)

                if delta > 0.0 and p_v2v < self.env.min_active_power:
                    p_v2v = self.env.min_active_power
                if delta < 1.0 and p_v2i < self.env.min_active_power:
                    p_v2i = self.env.min_active_power

                t_v2v = 0.0
                e_v2v = 0.0
                v2v_data = 0.0
                unique_v2v_data = 0.0
                active_v2v = 0
                if delta > 0.0 and p_v2v > 0.0:
                    v2v_data = delta * self.env.chunk_size
                    v2v_rate = self.env._calculate_v2v_rate(follower_idx, active_leader_idx, p_v2v)
                    if v2v_rate > 0.0:
                        t_v2v = v2v_data / v2v_rate
                        e_v2v = p_v2v * t_v2v
                        unique_v2v_data = (1.0 - beta) * v2v_data
                        active_v2v = 1

                t_v2i = 0.0
                e_v2i = 0.0
                if delta < 1.0 and p_v2i > 0.0:
                    v2i_data = (1.0 - delta) * self.env.chunk_size
                    v2i_rate = self.env._calculate_v2i_rate(follower_idx, p_v2i)
                    if v2i_rate > 0.0:
                        t_v2i = v2i_data / v2i_rate
                        e_v2i = p_v2i * t_v2i

                follower_time = max(t_v2v, t_v2i)
                follower_energy = e_v2v + e_v2i

                if self.env.optimization_target == "time":
                    local_objective = follower_time
                elif self.env.optimization_target == "energy":
                    local_objective = follower_energy
                else:
                    time_normalizer = (
                        self.env.num_followers * self.env.follower_time_budget + self.env.leader_time_budget
                        if (self.env.follower_time_budget > 0 and self.env.leader_time_budget > 0)
                        else 1.0
                    )
                    energy_normalizer = (
                        self.env.num_followers * self.env.follower_energy_budget + self.env.leader_energy_budget
                        if (self.env.follower_energy_budget > 0 and self.env.leader_energy_budget > 0)
                        else 1.0
                    )
                    weight = self.env._get_effective_mixed_time_weight()
                    local_objective = (
                        weight * (follower_time / time_normalizer)
                        + (1.0 - weight) * (follower_energy / energy_normalizer)
                    )

                follower_time_violation = int(follower_time > self.env.follower_time_budget)
                follower_energy_violation = int(follower_energy > self.env.follower_energy_budget)
                time_budget = max(float(self.env.follower_time_budget), 1e-12)
                energy_budget = max(float(self.env.follower_energy_budget), 1e-12)
                follower_time_excess = max((follower_time - time_budget) / time_budget, 0.0)
                follower_energy_excess = max((follower_energy - energy_budget) / energy_budget, 0.0)
                local_violation_count = follower_time_violation + follower_energy_violation
                local_excess = follower_time_excess + follower_energy_excess
                local_penalty = (
                    self._penalty_from_excess(local_excess)
                    if self.env.active_constraint_penalty_mode == "normalized_excess"
                    else self._penalty_from_counts(
                        local_violation_count,
                        any_violation=local_violation_count > 0,
                    )
                )

                follower_metrics.append(
                    _FollowerPresetMetrics(
                        delta=delta,
                        action=np.asarray(preset, dtype=np.float32),
                        follower_time=float(follower_time),
                        follower_energy=float(follower_energy),
                        v2v_data=float(v2v_data),
                        unique_v2v_data=float(unique_v2v_data),
                        active_v2v=active_v2v,
                        follower_time_violation=follower_time_violation,
                        follower_energy_violation=follower_energy_violation,
                        follower_time_excess=float(follower_time_excess),
                        follower_energy_excess=float(follower_energy_excess),
                        local_objective=float(local_objective),
                        local_penalized=float(local_objective + local_penalty),
                        local_violation_count=local_violation_count,
                    )
                )

            metrics.append(follower_metrics)

        return metrics

    def _evaluate_preset_indices(self, preset_indices: np.ndarray, follower_metrics: list[list[_FollowerPresetMetrics]]) -> float:
        follower_time_sum = 0.0
        follower_energy_sum = 0.0
        total_v2v_data = 0.0
        follower_unique_data = 0.0
        active_v2v_count = 0
        follower_violation_count = 0
        follower_excess = 0.0

        for follower_slot, preset_idx in enumerate(preset_indices):
            metrics = follower_metrics[follower_slot][int(preset_idx)]
            follower_time_sum += metrics.follower_time
            follower_energy_sum += metrics.follower_energy
            total_v2v_data += metrics.v2v_data
            follower_unique_data += metrics.unique_v2v_data
            active_v2v_count += metrics.active_v2v
            follower_violation_count += metrics.local_violation_count
            follower_excess += metrics.follower_time_excess + metrics.follower_energy_excess

        objective_cost, leader_time, leader_energy, _ = self._objective_from_totals(
            follower_time_sum,
            follower_energy_sum,
            total_v2v_data,
            follower_unique_data,
            active_v2v_count,
        )
        leader_time_violation = int(leader_time > self.env.leader_time_budget)
        leader_energy_violation = int(leader_energy > self.env.leader_energy_budget)
        violation_count = follower_violation_count + leader_time_violation + leader_energy_violation
        if self.env.active_constraint_penalty_mode == "normalized_excess":
            time_budget = max(float(self.env.leader_time_budget), 1e-12)
            energy_budget = max(float(self.env.leader_energy_budget), 1e-12)
            leader_excess = max((leader_time - time_budget) / time_budget, 0.0)
            leader_excess += max((leader_energy - energy_budget) / energy_budget, 0.0)
            penalty = self._penalty_from_excess(follower_excess + leader_excess)
        else:
            penalty = self._penalty_from_counts(violation_count, any_violation=violation_count > 0)
        return float(objective_cost + penalty)

    def _coordinate_descent_incumbent(self, follower_metrics: list[list[_FollowerPresetMetrics]]) -> tuple[np.ndarray, float]:
        best_local = np.array(
            [int(np.argmin([m.local_penalized for m in metrics])) for metrics in follower_metrics],
            dtype=np.int32,
        )
        candidates = [
            best_local.copy(),
            np.zeros(self.env.num_followers, dtype=np.int32),
            np.full(self.env.num_followers, 1, dtype=np.int32),
        ]

        baseline_actions = [
            self._baselines.all_leader_strategy(self.env._get_state()),
            self._baselines.all_base_station_strategy(self.env._get_state()),
            self._baselines.balanced_strategy(self.env._get_state()),
        ]
        for action in baseline_actions:
            indices = []
            for follower_slot in range(self.env.num_followers):
                preset = np.asarray(action[follower_slot * 3:(follower_slot + 1) * 3], dtype=np.float32)
                distances = np.sum((self._presets - preset) ** 2, axis=1)
                indices.append(int(np.argmin(distances)))
            candidates.append(np.asarray(indices, dtype=np.int32))

        best_indices = None
        best_objective = float("inf")
        for indices in candidates:
            objective = self._evaluate_preset_indices(indices, follower_metrics)
            if objective < best_objective:
                best_objective = objective
                best_indices = indices.copy()

        for _ in range(max(1, self.incumbent_passes)):
            improved = False
            for follower_slot in range(self.env.num_followers):
                current_best_idx = int(best_indices[follower_slot])
                current_best_obj = best_objective
                trial = best_indices.copy()
                for preset_idx in range(len(self._presets)):
                    if preset_idx == current_best_idx:
                        continue
                    trial[follower_slot] = preset_idx
                    objective = self._evaluate_preset_indices(trial, follower_metrics)
                    if objective + 1e-12 < current_best_obj:
                        current_best_obj = objective
                        current_best_idx = preset_idx
                if current_best_idx != int(best_indices[follower_slot]):
                    best_indices[follower_slot] = current_best_idx
                    best_objective = current_best_obj
                    improved = True
            if not improved:
                break

        return best_indices.astype(np.int32), float(best_objective)

    def _search_order(self, follower_metrics: list[list[_FollowerPresetMetrics]]) -> list[int]:
        spreads = []
        for follower_slot, metrics in enumerate(follower_metrics):
            local_penalizeds = np.asarray([m.local_penalized for m in metrics], dtype=np.float64)
            spreads.append((float(np.ptp(local_penalizeds)), follower_slot))
        spreads.sort(key=lambda item: item[0], reverse=True)
        return [follower_slot for _, follower_slot in spreads]

    def select_action(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        del state
        follower_metrics = self._compute_follower_metrics()
        min_local_objectives = np.asarray(
            [min(metric.local_objective for metric in metrics) for metrics in follower_metrics],
            dtype=np.float64,
        )
        min_local_violation_counts = np.asarray(
            [min(metric.local_violation_count for metric in metrics) for metrics in follower_metrics],
            dtype=np.int32,
        )
        branch_orders = [
            np.asarray(
                sorted(
                    range(len(metrics)),
                    key=lambda preset_idx: metrics[preset_idx].local_penalized,
                ),
                dtype=np.int32,
            )
            for metrics in follower_metrics
        ]

        incumbent_indices, incumbent_objective = self._coordinate_descent_incumbent(follower_metrics)
        search_order = self._search_order(follower_metrics)
        inverse_order = {ordered_slot: depth for depth, ordered_slot in enumerate(search_order)}

        assigned = np.full(self.env.num_followers, -1, dtype=np.int32)
        for follower_slot in range(self.env.num_followers):
            assigned[follower_slot] = -1

        best_indices = incumbent_indices.copy()
        best_objective = float(incumbent_objective)

        nodes_expanded = 0
        nodes_pruned = 0
        leaves_evaluated = 0
        hit_node_budget = False

        def lower_bound(
            depth: int,
            follower_time_sum: float,
            follower_energy_sum: float,
            total_v2v_data: float,
            follower_unique_data: float,
            active_v2v_count: int,
            assigned_follower_violations: int,
        ) -> float:
            objective_lb, leader_time, leader_energy, _ = self._objective_from_totals(
                follower_time_sum,
                follower_energy_sum,
                total_v2v_data,
                follower_unique_data,
                active_v2v_count,
            )
            remaining_slots = search_order[depth:]
            objective_lb += float(np.sum(min_local_objectives[remaining_slots]))

            leader_violation_count = int(leader_time > self.env.leader_time_budget) + int(leader_energy > self.env.leader_energy_budget)
            if self.env.active_constraint_penalty_mode == "per_violation":
                violation_lb = assigned_follower_violations + leader_violation_count + int(np.sum(min_local_violation_counts[remaining_slots]))
                penalty_lb = self._penalty_from_counts(violation_lb, any_violation=violation_lb > 0)
            elif self.env.active_constraint_penalty_mode == "normalized_excess":
                penalty_lb = 0.0
            else:
                guaranteed_any = (
                    assigned_follower_violations > 0
                    or leader_violation_count > 0
                    or any(int(min_local_violation_counts[idx]) > 0 for idx in remaining_slots)
                )
                penalty_lb = self._penalty_from_counts(1 if guaranteed_any else 0, any_violation=guaranteed_any)

            return float(objective_lb + penalty_lb)

        def dfs(
            depth: int,
            follower_time_sum: float,
            follower_energy_sum: float,
            total_v2v_data: float,
            follower_unique_data: float,
            active_v2v_count: int,
            assigned_follower_violations: int,
        ) -> None:
            nonlocal nodes_expanded, nodes_pruned, leaves_evaluated, hit_node_budget, best_indices, best_objective

            if self.max_nodes > 0 and nodes_expanded >= self.max_nodes:
                hit_node_budget = True
                return

            bound = lower_bound(
                depth,
                follower_time_sum,
                follower_energy_sum,
                total_v2v_data,
                follower_unique_data,
                active_v2v_count,
                assigned_follower_violations,
            )
            if bound >= best_objective - 1e-12:
                nodes_pruned += 1
                return

            nodes_expanded += 1
            if depth == self.env.num_followers:
                leaves_evaluated += 1
                objective = self._evaluate_preset_indices(assigned, follower_metrics)
                if objective + 1e-12 < best_objective:
                    best_objective = objective
                    best_indices = assigned.copy()
                return

            follower_slot = search_order[depth]
            children = []
            for preset_idx in branch_orders[follower_slot]:
                metrics = follower_metrics[follower_slot][int(preset_idx)]
                child_bound = lower_bound(
                    depth + 1,
                    follower_time_sum + metrics.follower_time,
                    follower_energy_sum + metrics.follower_energy,
                    total_v2v_data + metrics.v2v_data,
                    follower_unique_data + metrics.unique_v2v_data,
                    active_v2v_count + metrics.active_v2v,
                    assigned_follower_violations + metrics.local_violation_count,
                )
                if child_bound < best_objective - 1e-12:
                    children.append((child_bound, int(preset_idx), metrics))
                else:
                    nodes_pruned += 1

            children.sort(key=lambda item: item[0])
            for _, preset_idx, metrics in children:
                if self.max_nodes > 0 and nodes_expanded >= self.max_nodes:
                    hit_node_budget = True
                    return
                assigned[follower_slot] = preset_idx
                dfs(
                    depth + 1,
                    follower_time_sum + metrics.follower_time,
                    follower_energy_sum + metrics.follower_energy,
                    total_v2v_data + metrics.v2v_data,
                    follower_unique_data + metrics.unique_v2v_data,
                    active_v2v_count + metrics.active_v2v,
                    assigned_follower_violations + metrics.local_violation_count,
                )
                assigned[follower_slot] = -1
                if hit_node_budget and self.max_nodes > 0 and nodes_expanded >= self.max_nodes:
                    return

        dfs(
            depth=0,
            follower_time_sum=0.0,
            follower_energy_sum=0.0,
            total_v2v_data=0.0,
            follower_unique_data=0.0,
            active_v2v_count=0,
            assigned_follower_violations=0,
        )

        self._slot_counter += 1
        self.last_search_stats = {
            "slot": self._slot_counter,
            "objective": float(best_objective),
            "nodes_expanded": int(nodes_expanded),
            "nodes_pruned": int(nodes_pruned),
            "leaves_evaluated": int(leaves_evaluated),
            "hit_node_budget": bool(hit_node_budget),
            "search_order": list(search_order),
            "incumbent_indices": best_indices.astype(int).tolist(),
        }
        self.aggregate_search_stats["slots"] += 1
        self.aggregate_search_stats["nodes_expanded"] += int(nodes_expanded)
        self.aggregate_search_stats["nodes_pruned"] += int(nodes_pruned)
        self.aggregate_search_stats["leaves_evaluated"] += int(leaves_evaluated)
        self.aggregate_search_stats["hit_node_budget"] += int(hit_node_budget)

        return self._build_action_from_indices(best_indices)
