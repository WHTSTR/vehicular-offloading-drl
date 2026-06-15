#!/usr/bin/env python3
"""
Zero-shot transfer evaluation of straight-road models on the U-turn scenario.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from run_paper_main_single_eta import (
    build_trainer,
    canonical_eta,
    get_de_settings,
    instantiate_single_agent,
    recover_best_model_info,
    write_manifest,
)
from uturn_environment import UTurnVehicularEnvironment


ORDERED_ALGORITHMS = ["C-DQN", "C-DDPG", "C-SAC", "D-DQN", "D-DDPG", "D-SAC"]
PAPER_TOP3 = ["C-SAC", "D-DDPG", "D-SAC"]


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


def load_source_config(source_run_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = os.path.join(source_run_dir, "paper_run_manifest.json")
    config_path = os.path.join(source_run_dir, "experiment_config.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing source config: {config_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return manifest, config


def swap_to_uturn_environment(
    trainer,
    source_config: dict[str, Any],
    uturn_radius: float,
    uturn_leg_length: float,
    uturn_open_ended: bool,
    base_station_x: float,
    base_station_y: float,
):
    trainer.config = copy.deepcopy(source_config)
    trainer.config["environment"]["road_geometry"] = "uturn_horizontal"
    trainer.config["environment"]["uturn_radius"] = float(uturn_radius)
    trainer.config["environment"]["uturn_leg_length"] = float(uturn_leg_length)
    trainer.config["environment"]["uturn_open_ended"] = bool(uturn_open_ended)
    trainer.config["environment"]["base_station_position"] = [float(base_station_x), float(base_station_y)]
    trainer.env = UTurnVehicularEnvironment(trainer.config["environment"])
    trainer.num_followers = trainer.env.num_followers
    trainer.state_dim = trainer.env.observation_space.shape[0]
    trainer.action_dim = trainer.env.action_space.shape[0]
    trainer.env.reset()
    trainer.local_state_size = len(trainer.env.get_local_observation(0))


def write_summary_md(path: str, compact_rows: dict[str, dict[str, float]]) -> None:
    lines = [
        "# U-turn Transfer Evaluation",
        "",
        "| Algorithm | Objective | Time | Energy | Violations | FT | LT | FE | LE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for alg_name in [name for name in ORDERED_ALGORITHMS if name in compact_rows]:
        row = compact_rows[alg_name]
        lines.append(
            f"| `{alg_name}` | `{row['objective']:.3f}` | `{row['time']:.2f}` | `{row['energy']:.3f}` | "
            f"`{row['violations']:.2f}` | `{row['follower_time_violations']:.2f}` | "
            f"`{row['leader_time_violations']:.2f}` | `{row['follower_energy_violations']:.2f}` | "
            f"`{row['leader_energy_violations']:.2f}` |"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_algorithms(args: argparse.Namespace) -> List[str]:
    specified = sum(
        [
            1 if args.only_algorithm is not None else 0,
            1 if args.algorithm_set is not None else 0,
            1 if args.algorithms else 0,
        ]
    )
    if specified > 1:
        raise ValueError("Use only one of --only-algorithm, --algorithm-set, or --algorithms")

    if args.only_algorithm is not None:
        return [args.only_algorithm]
    if args.algorithms:
        return list(args.algorithms)
    if args.algorithm_set == "paper3":
        return list(PAPER_TOP3)
    if args.algorithm_set == "all6":
        return list(ORDERED_ALGORITHMS)
    return list(PAPER_TOP3)


def clear_algorithm_agent(trainer, alg_name: str) -> None:
    agent = trainer.agents.pop(alg_name, None)
    if agent is not None:
        del agent
    gc.collect()
    if str(trainer.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_selected_algorithms(trainer, selected_algorithms: List[str], num_episodes: int) -> dict[str, Any]:
    rl_results: dict[str, Any] = {}
    for alg_name in selected_algorithms:
        instantiate_single_agent(trainer, alg_name)
        try:
            single_result = trainer.final_evaluation(num_episodes=num_episodes)
            rl_results[alg_name] = single_result[alg_name]
        finally:
            clear_algorithm_agent(trainer, alg_name)
    return rl_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate straight-road models zero-shot on the U-turn scenario.")
    parser.add_argument("--source-run-dir", type=str, required=True, help="Existing straight-road run directory")
    parser.add_argument("--eta", type=float, default=None, help="Optional eta override; by default inferred from source run")
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5000, help="Metadata only; no retraining is performed")
    parser.add_argument("--results-group", type=str, default="uturn_transfer_eval")
    parser.add_argument("--only-algorithm", choices=ORDERED_ALGORITHMS, default=None)
    parser.add_argument("--algorithm-set", choices=["paper3", "all6"], default=None)
    parser.add_argument("--algorithms", nargs="+", choices=ORDERED_ALGORITHMS, default=None)
    parser.add_argument("--include-baselines", action=argparse.BooleanOptionalAction, default=True, help="Evaluate fixed baselines and DE (default: on)")
    parser.add_argument("--de-mode", choices=["paper", "screen", "sweep"], default="paper")
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
    selected_algorithms = parse_algorithms(args)

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

    recover_best_model_info(trainer, source_run_dir)
    for alg_name in selected_algorithms:
        path = trainer.best_models[alg_name]["path"]
        if not path:
            raise FileNotFoundError(f"Could not recover best model path for {alg_name} from {source_run_dir}")

    start_time = time.time()
    baseline_results = None
    if args.include_baselines:
        baseline_results = trainer.evaluate_baselines(
            num_episodes=args.final_eval_episodes,
            **get_de_settings(args.de_mode, int(source_manifest["seed"])),
        )
    rl_results = evaluate_selected_algorithms(
        trainer,
        selected_algorithms=selected_algorithms,
        num_episodes=args.final_eval_episodes,
    )

    compact_rows = {alg_name: summarize_result(rl_results[alg_name]) for alg_name in selected_algorithms}
    payload = {
        "eta": eta,
        "source_run_dir": source_run_dir,
        "selected_algorithms": selected_algorithms,
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
        "baseline_results": convert_to_native(baseline_results) if baseline_results is not None else None,
        "rl_results": convert_to_native(rl_results),
        "best_models": convert_to_native({alg: trainer.best_models[alg] for alg in selected_algorithms}),
        "compact": compact_rows,
    }
    with open(os.path.join(run_dir, "uturn_transfer_results.json"), "w", encoding="utf-8") as handle:
        json.dump(convert_to_native(payload), handle, indent=2)
    with open(os.path.join(run_dir, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump(convert_to_native(trainer.config), handle, indent=2)
    write_summary_md(os.path.join(run_dir, "summary.md"), compact_rows)

    manifest = {
        "eta": eta,
        "source_run_dir": source_run_dir,
        "final_eval_episodes": args.final_eval_episodes,
        "seed": int(source_manifest["seed"]),
        "device": args.device,
        "results_dir": run_dir,
        "include_baselines": args.include_baselines,
        "de_mode": args.de_mode if args.include_baselines else None,
        "only_algorithm": args.only_algorithm,
        "selected_algorithms": selected_algorithms,
        "uturn_radius": args.uturn_radius,
        "uturn_leg_length": args.uturn_leg_length,
        "uturn_open_ended": not args.closed_loop,
        "base_station_position": [args.base_station_x, args.base_station_y],
        "elapsed_minutes": (time.time() - start_time) / 60.0,
    }
    write_manifest(os.path.join(run_dir, "uturn_transfer_manifest.json"), manifest)
    print(f"U-turn transfer evaluation complete. Results saved in: {run_dir}")


if __name__ == "__main__":
    main()
