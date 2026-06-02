#!/usr/bin/env python3
"""
Lambda (constraint-penalty) sweep for the six DRL algorithms.

This script trains/evaluates only DRL methods:
- C-DQN, C-DDPG, C-SAC
- D-DQN, D-DDPG, D-SAC

It does not run fixed baselines or DE, so it is suitable for lambda selection.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch

from train_evaluate_all import CombinedTrainEvaluate


DEFAULT_LAMBDAS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
ORDERED_ALGORITHMS = ["C-DQN", "C-DDPG", "C-SAC", "D-DQN", "D-DDPG", "D-SAC"]


def canonical_eta(value: float) -> float:
    rounded = round(float(value), 2)
    supported = [0.0, 0.5, 1.0]
    for candidate in supported:
        if abs(candidate - rounded) < 1e-9:
            return candidate
    raise ValueError(f"Unsupported eta={value}. Use one of {supported}.")


def build_trainer(
    eta: float,
    lmbda: float,
    episodes: int,
    eval_interval: int,
    checkpoint_eval_episodes: int,
    seed: int,
    device: str,
    chunk_size_mb: float,
    follower_time_budget: float,
    leader_time_budget: float,
    num_vehicles: int,
    beta: float,
) -> CombinedTrainEvaluate:
    return CombinedTrainEvaluate(
        optimization_target="mixed",
        mixed_time_weight=eta,
        episodes=episodes,
        eval_interval=eval_interval,
        checkpoint_eval_episodes=checkpoint_eval_episodes,
        device=device,
        seed=seed,
        num_vehicles=num_vehicles,
        redundancy_ratio_range=(beta, beta),
        time_budget=2.0,
        follower_time_budget=follower_time_budget,
        leader_time_budget=leader_time_budget,
        chunk_size=float(chunk_size_mb) * 1e6,
        leader_power_multiplier=2.0,
        min_active_power_ratio=1e-3,
        reward_mode="paper_counted_normalized",
        constraint_penalty_value=float(lmbda),
        base_station_position=(225.0, 0.0),
        leader_selection_mode="episode_start_scored",
        leader_reselection_interval_slots=10,
        initial_position_mode="random_uniform",
        initial_position_x_range=(0.0, 50.0),
        initial_position_jitter=0.0,
        use_fixed_eval_seeds=True,
        fixed_eval_seed_base=100000,
    )


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


def summarize_result(result: dict[str, Any]) -> dict[str, float]:
    return {
        "objective": float(result["objective_cost"][0]),
        "time": float(result["time"][0]),
        "energy": float(result["energy"][0]),
        "violations": float(result["constraint_violations"][0]),
        "follower_time_violations": float(result["follower_time_violations"][0]),
        "leader_time_violations": float(result["leader_time_violations"][0]),
        "follower_energy_violations": float(result["follower_energy_violations"][0]),
        "leader_energy_violations": float(result["leader_energy_violations"][0]),
    }


def save_training_npz(trainer: CombinedTrainEvaluate, run_dir: str) -> None:
    payload: dict[str, np.ndarray] = {}
    for alg_name in ORDERED_ALGORITHMS:
        for metric_name, values in trainer.training_metrics[alg_name].items():
            payload[f"{alg_name}_{metric_name}"] = np.array(values)
    np.savez(os.path.join(run_dir, "combined_training_data.npz"), **payload)


def write_manifest(path: str, manifest: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def train_all_drl(trainer: CombinedTrainEvaluate) -> None:
    trainer.agents["C-DQN"] = trainer.train_centralized_algorithm("DQN", "C-DQN")
    trainer.agents["C-DDPG"] = trainer.train_centralized_algorithm("DDPG", "C-DDPG")
    trainer.agents["C-SAC"] = trainer.train_centralized_algorithm("SAC", "C-SAC")
    trainer.agents["D-DQN"] = trainer.train_decentralized_algorithm("DQN", "D-DQN")
    trainer.agents["D-DDPG"] = trainer.train_decentralized_algorithm("DDPG", "D-DDPG")
    trainer.agents["D-SAC"] = trainer.train_decentralized_algorithm("SAC", "D-SAC")


def write_summary_md(path: str, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Lambda Sweep Summary",
        "",
        "| Lambda | Algorithm | Objective | Time | Energy | Violations |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lmbda = row["lambda"]
        for alg_name in ORDERED_ALGORITHMS:
            metrics = row["compact"][alg_name]
            lines.append(
                f"| `{lmbda:.2f}` | `{alg_name}` | `{metrics['objective']:.3f}` | "
                f"`{metrics['time']:.2f}` | `{metrics['energy']:.3f}` | `{metrics['violations']:.2f}` |"
            )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lambda sweep for all six DRL algorithms.")
    parser.add_argument("--eta", type=float, required=True, help="One of 0.0, 0.5, 1.0")
    parser.add_argument("--episodes", type=int, default=2500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=25)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size-mb", type=float, default=20.0)
    parser.add_argument("--follower-time-budget", type=float, default=1.0)
    parser.add_argument("--leader-time-budget", type=float, default=2.0)
    parser.add_argument("--num-vehicles", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--results-group", type=str, default="paper_lambda_sweep_all_drl")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDAS,
        help="Lambda values to sweep",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    eta = canonical_eta(args.eta)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    eta_tag = f"eta_{int(round(eta * 100)):03d}"
    root_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        args.results_group,
        eta_tag,
        f"run_{timestamp}_{os.getpid()}",
    )
    os.makedirs(root_dir, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for lmbda in args.lambdas:
        lambda_tag = str(lmbda).replace(".", "p")
        run_dir = os.path.join(root_dir, f"lambda_{lambda_tag}")
        os.makedirs(run_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Lambda sweep run: eta={eta:.2f}, lambda={lmbda:.2f}")
        print("=" * 80)

        start_time = time.time()
        trainer = build_trainer(
            eta=eta,
            lmbda=lmbda,
            episodes=args.episodes,
            eval_interval=args.eval_interval,
            checkpoint_eval_episodes=args.checkpoint_eval_episodes,
            seed=args.seed,
            device=args.device,
            chunk_size_mb=args.chunk_size_mb,
            follower_time_budget=args.follower_time_budget,
            leader_time_budget=args.leader_time_budget,
            num_vehicles=args.num_vehicles,
            beta=args.beta,
        )
        trainer.save_dir = run_dir

        train_all_drl(trainer)
        rl_results = trainer.final_evaluation(num_episodes=args.final_eval_episodes)

        results_payload = {
            "eta": eta,
            "lambda": lmbda,
            "optimization_target": trainer.optimization_target,
            "episodes": trainer.episodes,
            "eval_interval": trainer.eval_interval,
            "checkpoint_eval_episodes": trainer.checkpoint_eval_episodes,
            "final_eval_episodes": args.final_eval_episodes,
            "seed": trainer.seed,
            "device": trainer.device,
            "config_snapshot": convert_to_native(trainer.config),
            "rl_results": convert_to_native(rl_results),
            "best_models": convert_to_native(trainer.best_models),
        }
        with open(os.path.join(run_dir, "combined_comprehensive_results.json"), "w", encoding="utf-8") as handle:
            json.dump(results_payload, handle, indent=2)
        with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as handle:
            json.dump(convert_to_native(trainer.config), handle, indent=2)
        save_training_npz(trainer, run_dir)

        manifest = {
            "eta": eta,
            "lambda": lmbda,
            "episodes": args.episodes,
            "eval_interval": args.eval_interval,
            "checkpoint_eval_episodes": args.checkpoint_eval_episodes,
            "final_eval_episodes": args.final_eval_episodes,
            "seed": args.seed,
            "device": args.device,
            "chunk_size_mb": args.chunk_size_mb,
            "follower_time_budget": args.follower_time_budget,
            "leader_time_budget": args.leader_time_budget,
            "num_vehicles": args.num_vehicles,
            "beta": args.beta,
            "save_dir": run_dir,
            "elapsed_minutes": (time.time() - start_time) / 60.0,
        }
        write_manifest(os.path.join(run_dir, "lambda_sweep_manifest.json"), manifest)

        summary_rows.append(
            {
                "eta": eta,
                "lambda": lmbda,
                "run_dir": run_dir,
                "compact": {alg_name: summarize_result(rl_results[alg_name]) for alg_name in ORDERED_ALGORITHMS},
            }
        )

    summary_json = os.path.join(root_dir, "summary.json")
    summary_md = os.path.join(root_dir, "summary.md")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(convert_to_native(summary_rows), handle, indent=2)
    write_summary_md(summary_md, summary_rows)

    print(f"\nLambda sweep complete. Summary saved to: {summary_json}")


if __name__ == "__main__":
    main()
