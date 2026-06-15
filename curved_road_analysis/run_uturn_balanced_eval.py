#!/usr/bin/env python3
"""
Evaluate only the Balanced baseline on the U-turn scenario.

This evaluates the fixed Balanced baseline on a given U-turn geometry
without running DE-Search or the DRL models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from baselines import BaselineAlgorithms
from run_paper_main_single_eta import build_trainer, canonical_eta, write_manifest
from run_uturn_transfer_eval import load_source_config, swap_to_uturn_environment


def convert_to_native(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return [convert_to_native(v) for v in obj]
    return obj


def evaluate_balanced(trainer, num_episodes: int) -> dict[str, Any]:
    balanced = BaselineAlgorithms(trainer.env.num_followers, trainer.env.max_power).balanced_strategy

    rewards = []
    penalizeds = []
    objective_costs = []
    times = []
    energies = []
    mixeds = []
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

    timestep_times = []
    timestep_energies = []
    timestep_actions = []

    with trainer._preserve_training_rng():
        for episode_idx in range(num_episodes):
            state = trainer.reset_for_evaluation(episode_idx, stream="baseline")
            episode_metrics = trainer._new_episode_metrics()
            done = False

            ep_timestep_times = []
            ep_timestep_energies = []
            ep_timestep_actions = []

            while not done:
                action = balanced(state)
                actions_collected.append(action.copy())
                next_state, reward, done, info = trainer.env.step(action)

                ep_timestep_times.append(info.get("time_cost", 0.0))
                ep_timestep_energies.append(info.get("energy_cost", 0.0))
                ep_timestep_actions.append(action.copy())

                state = next_state
                trainer._accumulate_episode_metrics(episode_metrics, reward, info)

            rewards.append(episode_metrics["reward"])
            penalizeds.append(episode_metrics["penalized"])
            objective_costs.append(episode_metrics["objective_cost"])
            times.append(episode_metrics["time"])
            energies.append(episode_metrics["energy"])
            mixeds.append(episode_metrics["mixed"])
            normalized_times.append(episode_metrics["normalized_time"])
            normalized_energies.append(episode_metrics["normalized_energy"])
            constraint_violations.append(episode_metrics["constraint_violations"])
            constraint_penalties.append(episode_metrics["constraint_penalty"])
            any_constraint_violations.append(episode_metrics["any_constraint_violations"])
            follower_time_violations.append(episode_metrics["follower_time_violations"])
            leader_time_violations.append(episode_metrics["leader_time_violations"])
            follower_energy_violations.append(episode_metrics["follower_energy_violations"])
            leader_energy_violations.append(episode_metrics["leader_energy_violations"])

            timestep_times.append(ep_timestep_times)
            timestep_energies.append(ep_timestep_energies)
            timestep_actions.append(ep_timestep_actions)

    results = {
        "reward": (np.mean(rewards), np.std(rewards)),
        "penalized_objective": (np.mean(penalizeds), np.std(penalizeds)),
        "objective_cost": (np.mean(objective_costs), np.std(objective_costs)),
        "time": (np.mean(times), np.std(times)),
        "energy": (np.mean(energies), np.std(energies)),
        "mixed": (np.mean(mixeds), np.std(mixeds)),
        "normalized_time": (np.mean(normalized_times), np.std(normalized_times)),
        "normalized_energy": (np.mean(normalized_energies), np.std(normalized_energies)),
        "constraint_violations": (np.mean(constraint_violations), np.std(constraint_violations)),
        "constraint_penalty": (np.mean(constraint_penalties), np.std(constraint_penalties)),
        "any_constraint_violations": (np.mean(any_constraint_violations), np.std(any_constraint_violations)),
        "follower_time_violations": (np.mean(follower_time_violations), np.std(follower_time_violations)),
        "leader_time_violations": (np.mean(leader_time_violations), np.std(leader_time_violations)),
        "follower_energy_violations": (np.mean(follower_energy_violations), np.std(follower_energy_violations)),
        "leader_energy_violations": (np.mean(leader_energy_violations), np.std(leader_energy_violations)),
        "actions": np.array(actions_collected),
        "timestep_times": timestep_times,
        "timestep_energies": timestep_energies,
        "timestep_actions": timestep_actions,
    }
    results["action_analysis"] = trainer.analyze_actions(results["actions"], trainer.num_followers)
    return results


def compact_summary(results: dict[str, Any]) -> dict[str, float]:
    return {
        "objective": float(results["objective_cost"][0]),
        "objective_std": float(results["objective_cost"][1]),
        "time": float(results["time"][0]),
        "time_std": float(results["time"][1]),
        "energy": float(results["energy"][0]),
        "energy_std": float(results["energy"][1]),
        "violations": float(results["constraint_violations"][0]),
        "violations_std": float(results["constraint_violations"][1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate only the Balanced baseline on a U-turn scenario.")
    parser.add_argument("--source-run-dir", type=str, required=True, help="Existing straight-road run directory")
    parser.add_argument("--eta", type=float, default=None, help="Optional eta override; by default inferred from source run")
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5000, help="Metadata only; no retraining is performed")
    parser.add_argument("--results-group", type=str, default="uturn_balanced_eval")
    parser.add_argument("--uturn-radius", type=float, default=10.0)
    parser.add_argument("--uturn-leg-length", type=float, default=178.04203673205103)
    parser.add_argument("--base-station-x", type=float, default=0.0)
    parser.add_argument("--base-station-y", type=float, default=10.0)
    parser.add_argument("--closed-loop", action="store_true", help="Wrap vehicles after one full U-turn instead of using an open-ended lower leg")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    source_run_dir = os.path.abspath(args.source_run_dir)
    source_manifest, source_config = load_source_config(source_run_dir)
    eta = canonical_eta(args.eta if args.eta is not None else source_manifest["eta"])
    env_cfg = source_config["environment"]

    trainer = build_trainer(
        eta=eta,
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        checkpoint_eval_episodes=args.checkpoint_eval_episodes,
        seed=int(source_manifest["seed"]),
        device=args.device,
        chunk_size_mb=float(env_cfg["chunk_size"]) / 1e6,
        follower_time_budget=float(env_cfg["follower_time_budget"]),
        leader_time_budget=float(env_cfg["leader_time_budget"]),
        num_vehicles=int(source_manifest["num_vehicles"]),
        beta=float(source_manifest["beta"]),
    )
    swap_to_uturn_environment(
        trainer,
        source_config=source_config,
        uturn_radius=args.uturn_radius,
        uturn_leg_length=args.uturn_leg_length,
        uturn_open_ended=not args.closed_loop,
        base_station_x=args.base_station_x,
        base_station_y=args.base_station_y,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    eta_tag = f"eta_{int(round(eta * 100)):03d}"
    run_dir = os.path.join(
        PARENT_DIR,
        "results",
        args.results_group,
        eta_tag,
        f"run_{timestamp}_{os.getpid()}",
    )
    os.makedirs(run_dir, exist_ok=True)
    trainer.save_dir = run_dir

    start_time = time.time()
    balanced_results = evaluate_balanced(trainer, num_episodes=args.final_eval_episodes)
    compact = compact_summary(balanced_results)

    payload = {
        "eta": eta,
        "source_run_dir": source_run_dir,
        "uturn_geometry": {
            "road_geometry": "uturn_horizontal",
            "uturn_radius": args.uturn_radius,
            "uturn_leg_length": args.uturn_leg_length,
            "uturn_open_ended": not args.closed_loop,
            "base_station_position": [args.base_station_x, args.base_station_y],
        },
        "final_eval_episodes": args.final_eval_episodes,
        "seed": int(source_manifest["seed"]),
        "device": args.device,
        "config_snapshot": convert_to_native(trainer.config),
        "balanced_results": convert_to_native(balanced_results),
        "compact": compact,
    }
    with open(os.path.join(run_dir, "uturn_balanced_results.json"), "w", encoding="utf-8") as handle:
        json.dump(convert_to_native(payload), handle, indent=2)
    with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump(convert_to_native(trainer.config), handle, indent=2)

    manifest = {
        "eta": eta,
        "source_run_dir": source_run_dir,
        "final_eval_episodes": args.final_eval_episodes,
        "seed": int(source_manifest["seed"]),
        "device": args.device,
        "results_dir": run_dir,
        "uturn_radius": args.uturn_radius,
        "uturn_leg_length": args.uturn_leg_length,
        "uturn_open_ended": not args.closed_loop,
        "base_station_position": [args.base_station_x, args.base_station_y],
        "elapsed_minutes": (time.time() - start_time) / 60.0,
    }
    write_manifest(os.path.join(run_dir, "uturn_balanced_manifest.json"), manifest)

    print(f"Balanced-only U-turn evaluation complete. Results saved in: {run_dir}")
    print(f"  Mixed Objective (eta={eta:.2f}): {compact['objective']:.3f}±{compact['objective_std']:.3f}")
    print(f"  Time: {compact['time']:.2f}±{compact['time_std']:.2f}")
    print(f"  Energy: {compact['energy']:.3f}±{compact['energy_std']:.3f}")
    print(f"  Violations: {compact['violations']:.2f}±{compact['violations_std']:.2f}")


if __name__ == "__main__":
    main()
