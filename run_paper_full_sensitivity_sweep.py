#!/usr/bin/env python3
"""
Sensitivity sweeps across eta values and a chosen sweep dimension.

For each (eta, sweep_value) combination this script:
1. trains all centralized and decentralized DRL methods,
2. evaluates the fixed baselines and DE-Search,
3. saves the training and evaluation outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import torch

from train_evaluate_all import CombinedTrainEvaluate


PAPER_LAMBDAS = {
    0.0: 0.1,
    0.5: 0.1,
    1.0: 0.1,
}

DEFAULT_ETAS = [0.0, 0.5, 1.0]
DEFAULT_SWEEP_VALUES = {
    "beta": [0.3, 0.4, 0.5, 0.6, 0.7],
    "num_vehicles": [3, 4, 5, 6, 7],
}

DE_PRESETS = {
    "paper": {
        "de_maxiter": 20,
        "de_popsize": 8,
        "de_polish": True,
        "de_num_restarts": 2,
        "de_powell_maxiter": 60,
        "de_local_refine_top_k": 2,
        "de_selection_mode": "follower_time_first",
    },
    "screen": {
        "de_maxiter": 10,
        "de_popsize": 6,
        "de_polish": True,
        "de_num_restarts": 1,
        "de_powell_maxiter": 25,
        "de_local_refine_top_k": 1,
        "de_selection_mode": "follower_time_first",
    },
    "sweep": {
        "de_maxiter": 6,
        "de_popsize": 4,
        "de_polish": True,
        "de_num_restarts": 1,
        "de_powell_maxiter": 15,
        "de_local_refine_top_k": 1,
        "de_selection_mode": "follower_time_first",
    },
}

def canonical_eta(value: float) -> float:
    rounded = round(float(value), 2)
    for candidate in PAPER_LAMBDAS:
        if abs(candidate - rounded) < 1e-9:
            return candidate
    raise ValueError(f"Unsupported eta={value}. Use one of {sorted(PAPER_LAMBDAS)}.")


def get_de_settings(mode: str, seed: int) -> Dict[str, Any]:
    if mode not in DE_PRESETS:
        raise ValueError(f"Unsupported DE preset '{mode}'. Use one of {sorted(DE_PRESETS)}.")
    settings = dict(DE_PRESETS[mode])
    settings["include_de_search"] = True
    settings["de_seed"] = seed
    return settings


def parse_float_list(raw: str | None, default_values: List[float]) -> List[float]:
    if not raw:
        return list(default_values)
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def build_trainer(
    eta: float,
    lambda_value: float,
    sweep_name: str,
    sweep_value: float,
    episodes: int,
    eval_interval: int,
    checkpoint_eval_episodes: int,
    seed: int,
    device: str,
) -> CombinedTrainEvaluate:
    extra_kwargs: Dict[str, Any] = {}
    if sweep_name == "beta":
        extra_kwargs["redundancy_ratio_range"] = (float(sweep_value), float(sweep_value))
    elif sweep_name == "num_vehicles":
        extra_kwargs["num_vehicles"] = int(round(sweep_value))
    else:
        raise ValueError(f"Unknown sweep: {sweep_name}")

    trainer = CombinedTrainEvaluate(
        optimization_target="mixed",
        mixed_time_weight=eta,
        episodes=episodes,
        eval_interval=eval_interval,
        checkpoint_eval_episodes=checkpoint_eval_episodes,
        device=device,
        seed=seed,
        time_budget=2.0,
        follower_time_budget=1.0,
        leader_time_budget=2.0,
        chunk_size=20e6,
        leader_power_multiplier=2.0,
        min_active_power_ratio=1e-3,
        reward_mode="paper_counted_normalized",
        constraint_penalty_value=lambda_value,
        base_station_position=(225.0, 0.0),
        leader_selection_mode="episode_start_scored",
        leader_reselection_interval_slots=10,
        initial_position_mode="random_uniform",
        initial_position_x_range=(0.0, 50.0),
        initial_position_jitter=0.0,
        use_fixed_eval_seeds=True,
        fixed_eval_seed_base=100000,
        **extra_kwargs,
    )
    return trainer


def retarget_save_dir(trainer: CombinedTrainEvaluate, sweep_name: str, eta: float, sweep_value: float) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    eta_tag = f"eta_{int(round(eta * 100)):03d}"
    value_tag = str(int(round(sweep_value))) if sweep_name == "num_vehicles" else str(sweep_value).replace(".", "p")
    run_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        "paper_full_sensitivity",
        sweep_name,
        eta_tag,
        f"{sweep_name}_{value_tag}",
        f"run_{timestamp}_{os.getpid()}",
    )
    os.makedirs(run_dir, exist_ok=True)
    trainer.save_dir = run_dir
    return run_dir


def compact_result_block(result_block: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    compact: Dict[str, Dict[str, float]] = {}
    for name, metrics in result_block.items():
        compact[name] = {
            "objective_cost": float(metrics["objective_cost"][0]),
            "objective_cost_std": float(metrics["objective_cost"][1]),
            "time": float(metrics["time"][0]),
            "time_std": float(metrics["time"][1]),
            "energy": float(metrics["energy"][0]),
            "energy_std": float(metrics["energy"][1]),
            "constraint_violations": float(metrics["constraint_violations"][0]),
            "constraint_violations_std": float(metrics["constraint_violations"][1]),
            "follower_time_violations": float(metrics["follower_time_violations"][0]),
            "leader_time_violations": float(metrics["leader_time_violations"][0]),
            "follower_energy_violations": float(metrics["follower_energy_violations"][0]),
            "leader_energy_violations": float(metrics["leader_energy_violations"][0]),
        }
    return compact


def write_markdown_report(path: str, summary: Dict[str, Any]) -> None:
    lines = [
        f"# Full paper sensitivity sweep: {summary['config']['sweep']}",
        "",
        f"- etas: `{summary['config']['etas']}`",
        f"- values: `{summary['config']['values']}`",
        f"- episodes: `{summary['config']['episodes']}`",
        "",
        "| Eta | Value | Best centralized | Best decentralized | DE objective |",
        "| --- | ---: | --- | --- | ---: |",
    ]

    for combo in summary["combinations"]:
        rl = combo["rl_results"]
        best_cent_name = min(
            ["C-DQN", "C-DDPG", "C-SAC"],
            key=lambda name: rl[name]["objective_cost"],
        )
        best_decent_name = min(
            ["D-DQN", "D-DDPG", "D-SAC"],
            key=lambda name: rl[name]["objective_cost"],
        )
        lines.append(
            f"| {combo['eta']} | {combo['value']} | "
            f"{best_cent_name} ({rl[best_cent_name]['objective_cost']:.3f}) | "
            f"{best_decent_name} ({rl[best_decent_name]['objective_cost']:.3f}) | "
            f"{combo['baseline_results']['DE-Search']['objective_cost']:.3f} |"
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_combo(
    sweep_name: str,
    eta: float,
    sweep_value: float,
    episodes: int,
    eval_interval: int,
    checkpoint_eval_episodes: int,
    final_eval_episodes: int,
    seed: int,
    device: str,
    de_mode: str,
) -> Dict[str, Any]:
    lambda_value = PAPER_LAMBDAS[eta]
    trainer = build_trainer(
        eta=eta,
        lambda_value=lambda_value,
        sweep_name=sweep_name,
        sweep_value=sweep_value,
        episodes=episodes,
        eval_interval=eval_interval,
        checkpoint_eval_episodes=checkpoint_eval_episodes,
        seed=seed,
        device=device,
    )
    run_dir = retarget_save_dir(trainer, sweep_name, eta, sweep_value)
    start_time = time.time()

    for algorithm in ["DQN", "DDPG", "SAC"]:
        alg_name = f"C-{algorithm}"
        trainer.agents[alg_name] = trainer.train_centralized_algorithm(algorithm, alg_name)

    for algorithm in ["DQN", "DDPG", "SAC"]:
        alg_name = f"D-{algorithm}"
        trainer.agents[alg_name] = trainer.train_decentralized_algorithm(algorithm, alg_name)

    baseline_results = trainer.evaluate_baselines(
        num_episodes=final_eval_episodes,
        **get_de_settings(de_mode, seed),
    )
    rl_results = trainer.final_evaluation(num_episodes=final_eval_episodes)
    trainer.plot_results(baseline_results, rl_results)
    trainer.save_results(baseline_results, rl_results)

    manifest = {
        "sweep": sweep_name,
        "eta": eta,
        "value": sweep_value,
        "lambda": lambda_value,
        "episodes": episodes,
        "eval_interval": eval_interval,
        "checkpoint_eval_episodes": checkpoint_eval_episodes,
        "final_eval_episodes": final_eval_episodes,
        "seed": seed,
        "device": device,
        "de_mode": de_mode,
        "de_settings": get_de_settings(de_mode, seed),
        "save_dir": run_dir,
        "elapsed_minutes": (time.time() - start_time) / 60.0,
    }
    with open(os.path.join(run_dir, "paper_sweep_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "eta": eta,
        "value": sweep_value,
        "lambda": lambda_value,
        "save_dir": run_dir,
        "elapsed_minutes": manifest["elapsed_minutes"],
        "baseline_results": compact_result_block(baseline_results),
        "rl_results": compact_result_block(rl_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full paper sensitivity sweep.")
    parser.add_argument("--sweep", choices=["beta", "num_vehicles"], required=True, help="Sweep dimension")
    parser.add_argument("--etas", type=str, default="0,0.5,1.0", help="Comma-separated eta values")
    parser.add_argument("--values", type=str, default=None, help="Comma-separated sweep values")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    parser.add_argument("--eval-interval", type=int, default=100, help="Checkpoint eval interval")
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=25, help="Checkpoint eval episodes")
    parser.add_argument("--final-eval-episodes", type=int, default=100, help="Final evaluation episodes")
    parser.add_argument("--de-mode", choices=sorted(DE_PRESETS), default="sweep", help="DE preset: use lighter mode for large sweeps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    args = parser.parse_args()

    etas = [canonical_eta(value) for value in parse_float_list(args.etas, DEFAULT_ETAS)]
    values = parse_float_list(args.values, DEFAULT_SWEEP_VALUES[args.sweep])

    summary: Dict[str, Any] = {
        "config": {
            "sweep": args.sweep,
            "etas": etas,
            "values": values,
            "episodes": args.episodes,
            "eval_interval": args.eval_interval,
            "checkpoint_eval_episodes": args.checkpoint_eval_episodes,
            "final_eval_episodes": args.final_eval_episodes,
            "de_mode": args.de_mode,
            "seed": args.seed,
            "device": args.device,
            "lambda_by_eta": {str(eta): PAPER_LAMBDAS[eta] for eta in etas},
        },
        "combinations": [],
    }

    for eta in etas:
        for value in values:
            combo_summary = run_combo(
                sweep_name=args.sweep,
                eta=eta,
                sweep_value=value,
                episodes=args.episodes,
                eval_interval=args.eval_interval,
                checkpoint_eval_episodes=args.checkpoint_eval_episodes,
                final_eval_episodes=args.final_eval_episodes,
                seed=args.seed,
                device=args.device,
                de_mode=args.de_mode,
            )
            summary["combinations"].append(combo_summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        "paper_full_sensitivity",
        args.sweep,
    )
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{args.sweep}_summary_{timestamp}.json")
    md_path = os.path.join(output_dir, f"{args.sweep}_summary_{timestamp}.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_markdown_report(md_path, summary)

    print(f"Full sweep complete. Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
