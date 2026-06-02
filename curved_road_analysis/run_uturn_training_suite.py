#!/usr/bin/env python3
"""
Train or fine-tune DRL algorithms on the U-turn scenario.

This script is separate from the straight-road training pipeline. It supports:
  - scratch training on the chosen U-turn geometry
  - fine-tuning from straight-road best models
  - a selected subset of algorithms or all six
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
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
from run_uturn_transfer_eval import load_source_config, swap_to_uturn_environment


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


def parse_algorithms(args: argparse.Namespace) -> List[str]:
    if args.algorithms:
        algs = list(args.algorithms)
    elif args.algorithm_set == "paper3":
        algs = list(PAPER_TOP3)
    elif args.algorithm_set == "all6":
        algs = list(ORDERED_ALGORITHMS)
    else:
        raise ValueError(f"Unsupported algorithm set: {args.algorithm_set}")

    for alg in algs:
        if alg not in ORDERED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {alg}")
    return algs


def algorithm_family(alg_name: str) -> str:
    return alg_name.split("-")[1]


def is_centralized(alg_name: str) -> bool:
    return alg_name.startswith("C-")


def recover_source_best_models(trainer, source_run_dir: str, selected_algorithms: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    recover_best_model_info(trainer, source_run_dir)
    source_info: Dict[str, Dict[str, Any]] = {}
    for alg_name in selected_algorithms:
        info = dict(trainer.best_models[alg_name])
        if not info["path"]:
            raise FileNotFoundError(f"Could not recover best model path for {alg_name} from {source_run_dir}")
        source_info[alg_name] = info
    return source_info


def clear_algorithm_agent(trainer, alg_name: str) -> None:
    agent = trainer.agents.pop(alg_name, None)
    if agent is not None:
        del agent
    gc.collect()
    if str(trainer.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def instantiate_and_optionally_load(
    trainer,
    alg_name: str,
    source_best_models: Dict[str, Dict[str, Any]] | None,
) -> Any:
    instantiate_single_agent(trainer, alg_name)
    agent = trainer.agents[alg_name]
    if source_best_models is not None:
        agent.load(source_best_models[alg_name]["path"])
    return agent


def evaluate_selected_algorithms(trainer, selected_algorithms: List[str], num_episodes: int) -> Dict[str, Any]:
    rl_results: Dict[str, Any] = {}
    for alg_name in selected_algorithms:
        instantiate_single_agent(trainer, alg_name)
        try:
            single_result = trainer.final_evaluation(num_episodes=num_episodes)
            rl_results[alg_name] = single_result[alg_name]
        finally:
            clear_algorithm_agent(trainer, alg_name)
    return rl_results


def train_existing_centralized_algorithm(trainer, agent, algorithm: str, alg_name: str) -> Any:
    print(f"\n{'='*60}")
    print(f"Training {alg_name} (Centralized {algorithm})")
    print(f"{'='*60}")
    print(f"State dimension: {trainer.state_dim}")
    print(f"Action dimension: {trainer.action_dim}")
    trainer._reseed_for_algorithm(alg_name)

    episode_rewards = []
    best_eval_reward = -float("inf")

    for episode in range(trainer.episodes):
        state = trainer.env.reset()
        episode_metrics = trainer._new_episode_metrics()
        done = False
        step_count = 0

        if algorithm == "DDPG" and hasattr(agent, "reset_noise"):
            agent.reset_noise()

        while not done:
            if algorithm == "DQN":
                action, _ = agent.select_action(state, explore=True)
            elif algorithm == "DDPG":
                action = agent.select_action(state, add_noise=True)
            elif algorithm == "SAC":
                action = agent.select_action(state, evaluate=False)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            next_state, reward, done, info = trainer.env.step(action)
            trainer._accumulate_episode_metrics(episode_metrics, reward, info)

            agent.store_transition(state, action, reward, next_state, done)

            if trainer._warmup_complete(agent):
                if step_count % agent.update_every == 0:
                    if algorithm == "DQN":
                        agent.train(agent.config.get("num_updates", 1))
                        if agent.step_count % agent.target_update_every == 0:
                            agent.update_target_network()
                    else:
                        agent.train()

            state = next_state
            step_count += 1

        if trainer._warmup_complete(agent):
            if hasattr(agent, "update_exploration"):
                agent.update_exploration()
            elif algorithm == "DQN" and hasattr(agent, "update_epsilon"):
                agent.update_epsilon()

        episode_rewards.append(episode_metrics["reward"])
        trainer._append_training_episode_metrics(alg_name, episode_metrics)

        if (episode + 1) % trainer.eval_interval == 0:
            eval_metrics = trainer.evaluate_centralized_agent_metrics(
                agent,
                algorithm,
                num_episodes=trainer.checkpoint_eval_episodes,
                stream="checkpoint",
            )
            trainer._append_eval_metrics(alg_name, eval_metrics)

            checkpoint_dir = trainer._checkpoint_dir(alg_name, episode + 1)
            checkpoint_path = os.path.join(checkpoint_dir, f"{alg_name.lower()}_model.pth")
            agent.save(checkpoint_path)
            trainer._write_checkpoint_meta(checkpoint_dir, alg_name, episode + 1, eval_metrics)

            if eval_metrics["reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["reward"]
                model_path = os.path.join(trainer.save_dir, f"{alg_name.lower()}_best_model.pth")
                agent.save(model_path)
                trainer.best_models[alg_name]["path"] = model_path
                trainer.best_models[alg_name]["reward"] = eval_metrics["reward"]

            print(f"Episode {episode + 1}/{trainer.episodes}")
            print(f"  Training reward (last {trainer.eval_interval}): {np.mean(episode_rewards[-trainer.eval_interval:]):.2f}")
            print(f"  Evaluation reward: {eval_metrics['reward']:.2f}")
            print(f"  Eval penalized objective: {eval_metrics['penalized']:.2f}")
            print(f"  Eval violation count: {eval_metrics['constraint_violations']:.2f}")
            print(f"  Best eval reward: {best_eval_reward:.2f}")

    final_path = os.path.join(trainer.save_dir, f"{alg_name.lower()}_final_model.pth")
    agent.save(final_path)

    print(f"\n{alg_name} training completed!")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    return agent


def train_existing_decentralized_algorithm(trainer, multi_agent, algorithm: str, alg_name: str) -> Any:
    print(f"\n{'='*60}")
    print(f"Training {alg_name} (Decentralized {algorithm})")
    print(f"{'='*60}")
    print(f"Number of agents: {trainer.num_followers}")
    trainer._reseed_for_algorithm(alg_name)

    episode_rewards = []
    best_eval_reward = -float("inf")

    for episode in range(trainer.episodes):
        trainer.env.reset()
        episode_metrics = trainer._new_episode_metrics()
        done = False

        if algorithm == "DDPG" and hasattr(multi_agent, "reset_noise"):
            multi_agent.reset_noise()

        local_states = [trainer.env.get_local_observation(i) for i in range(trainer.num_followers)]

        while not done:
            acting_follower_indices = list(trainer.env.follower_indices)

            if algorithm == "DQN":
                action, _ = multi_agent.select_actions(local_states, explore=True)
            elif algorithm == "DDPG":
                action = multi_agent.select_actions(local_states, explore=True)
            elif algorithm == "SAC":
                action = multi_agent.select_actions(local_states, evaluate=False)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            _, reward, done, info = trainer.env.step(action)
            trainer._accumulate_episode_metrics(episode_metrics, reward, info)

            next_local_states = [trainer.env.get_local_observation(i) for i in range(trainer.num_followers)]
            transition_next_local_states, agent_dones = trainer._aligned_decentralized_next_states(
                acting_follower_indices, done
            )

            multi_agent.step(
                local_states,
                action,
                reward,
                transition_next_local_states,
                agent_dones,
            )

            if algorithm == "DQN" and hasattr(multi_agent, "shared_agent"):
                if multi_agent.shared_agent.step_count % multi_agent.shared_agent.target_update_every == 0:
                    multi_agent.update_target_network()

            local_states = next_local_states

        if trainer._warmup_complete(multi_agent) and hasattr(multi_agent, "update_exploration"):
            multi_agent.update_exploration()

        episode_rewards.append(episode_metrics["reward"])
        trainer._append_training_episode_metrics(alg_name, episode_metrics)

        if (episode + 1) % trainer.eval_interval == 0:
            eval_metrics = trainer.evaluate_decentralized_agent_metrics(
                multi_agent,
                algorithm,
                num_episodes=trainer.checkpoint_eval_episodes,
                stream="checkpoint",
            )
            trainer._append_eval_metrics(alg_name, eval_metrics)

            checkpoint_dir = trainer._checkpoint_dir(alg_name, episode + 1)
            multi_agent.save(checkpoint_dir)
            trainer._write_checkpoint_meta(checkpoint_dir, alg_name, episode + 1, eval_metrics)

            if eval_metrics["reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["reward"]
                model_dir = os.path.join(trainer.save_dir, f"{alg_name.lower()}_best_model")
                multi_agent.save(model_dir)
                trainer.best_models[alg_name]["path"] = model_dir
                trainer.best_models[alg_name]["reward"] = eval_metrics["reward"]

            print(f"Episode {episode + 1}/{trainer.episodes}")
            print(f"  Training reward (last {trainer.eval_interval}): {np.mean(episode_rewards[-trainer.eval_interval:]):.2f}")
            print(f"  Evaluation reward: {eval_metrics['reward']:.2f}")
            print(f"  Eval penalized objective: {eval_metrics['penalized']:.2f}")
            print(f"  Eval violation count: {eval_metrics['constraint_violations']:.2f}")
            print(f"  Best eval reward: {best_eval_reward:.2f}")
            if algorithm == "DQN" and hasattr(multi_agent, "shared_agent"):
                print(f"  Shared epsilon: {multi_agent.shared_agent.epsilon:.3f}")

    final_dir = os.path.join(trainer.save_dir, f"{alg_name.lower()}_final_model")
    multi_agent.save(final_dir)

    print(f"\n{alg_name} training completed!")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    return multi_agent


def save_subset_results(
    trainer,
    selected_algorithms: List[str],
    baseline_results: Dict[str, Any],
    rl_results: Dict[str, Any],
    source_run_dir: str,
    mode: str,
    source_best_models: Dict[str, Dict[str, Any]] | None,
) -> None:
    summary = {
        alg: {
            "final_reward": float(trainer.training_metrics[alg]["rewards"][-1]) if trainer.training_metrics[alg]["rewards"] else 0.0,
            "best_eval_reward": float(max(trainer.training_metrics[alg]["eval_rewards"])) if trainer.training_metrics[alg]["eval_rewards"] else 0.0,
        }
        for alg in selected_algorithms
    }
    payload = {
        "mode": mode,
        "selected_algorithms": selected_algorithms,
        "source_run_dir": source_run_dir,
        "optimization_target": trainer.optimization_target,
        "episodes": trainer.episodes,
        "eval_interval": trainer.eval_interval,
        "seed": trainer.seed,
        "device": trainer.device,
        "timestamp": datetime.now().isoformat(),
        "config_snapshot": convert_to_native(trainer.config),
        "baseline_results": convert_to_native(baseline_results),
        "rl_results": convert_to_native(rl_results),
        "best_models": convert_to_native({alg: trainer.best_models[alg] for alg in selected_algorithms}),
        "source_best_models": convert_to_native(source_best_models) if source_best_models is not None else None,
        "training_summary": summary,
    }
    with open(os.path.join(trainer.save_dir, "combined_comprehensive_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(os.path.join(trainer.save_dir, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump(convert_to_native(trainer.config), handle, indent=2)

    npz_payload = {
        f"{alg}_{metric}": np.array(data)
        for alg in selected_algorithms
        for metric, data in trainer.training_metrics[alg].items()
    }
    np.savez(os.path.join(trainer.save_dir, "combined_training_data.npz"), **npz_payload)

    lines = [
        f"# U-turn {mode.title()} Results",
        "",
        f"Source run: `{source_run_dir}`",
        "",
        "| Algorithm | Objective | Time | Energy | Violations |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for alg in selected_algorithms:
        row = rl_results[alg]
        lines.append(
            f"| `{alg}` | `{row['objective_cost'][0]:.3f}` | `{row['time'][0]:.2f}` | `{row['energy'][0]:.3f}` | `{row['constraint_violations'][0]:.2f}` |"
        )
    with open(os.path.join(trainer.save_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def save_training_rewards_csv(trainer, selected_algorithms: List[str]) -> None:
    max_len = max((len(trainer.training_metrics[alg]["rewards"]) for alg in selected_algorithms), default=0)
    with open(os.path.join(trainer.save_dir, "training_rewards.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", *selected_algorithms])
        for i in range(max_len):
            row = [i]
            for alg in selected_algorithms:
                rewards = trainer.training_metrics[alg]["rewards"]
                row.append(rewards[i] if i < len(rewards) else "")
            writer.writerow(row)


def plot_training_rewards_subset(trainer, selected_algorithms: List[str]) -> None:
    colors = {
        "C-DQN": "blue",
        "C-DDPG": "green",
        "C-SAC": "red",
        "D-DQN": "cyan",
        "D-DDPG": "lime",
        "D-SAC": "orange",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.set_title("Training Rewards (Raw)", fontsize=14)
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Reward", fontsize=12)
    ax1.grid(True, alpha=0.3)

    window_size = 25
    ax2.set_title(f"Training Rewards (Smoothed, window={window_size})", fontsize=14)
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Reward", fontsize=12)
    ax2.grid(True, alpha=0.3)

    for alg in selected_algorithms:
        rewards = trainer.training_metrics[alg]["rewards"]
        if not rewards:
            continue
        linestyle = "-" if alg.startswith("C-") else "--"
        episodes = range(len(rewards))
        ax1.plot(episodes, rewards, color=colors.get(alg, "gray"), label=alg, linewidth=1.5, linestyle=linestyle, alpha=0.7)
        if len(rewards) >= window_size:
            smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
            ax2.plot(range(window_size - 1, len(rewards)), smoothed, color=colors.get(alg, "gray"), label=alg, linewidth=2, linestyle=linestyle)

    if ax1.lines:
        ax1.legend(loc="lower right")
    if ax2.lines:
        ax2.legend(loc="lower right")
    plt.suptitle(f"Training Progress - {trainer._objective_display_name()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(trainer.save_dir, "training_rewards.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(trainer.save_dir, "training_rewards.pdf"), bbox_inches="tight")
    plt.close()


def save_algorithm_comparison_csv(trainer, baseline_results: Dict[str, Any], rl_results: Dict[str, Any]) -> None:
    metric_key = trainer._result_metric_key()
    all_results = {**baseline_results, **rl_results}
    with open(os.path.join(trainer.save_dir, "algorithm_comparison.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "paradigm", "objective_mean", "objective_std", "time_mean", "time_std", "energy_mean", "energy_std"])
        for name, results in all_results.items():
            if name in baseline_results:
                paradigm = "baseline"
            elif name.startswith("C-"):
                paradigm = "centralized"
            else:
                paradigm = "decentralized"
            writer.writerow([
                name,
                paradigm,
                results[metric_key][0],
                results[metric_key][1],
                results["time"][0],
                results["time"][1],
                results["energy"][0],
                results["energy"][1],
            ])


def save_paradigm_comparison_csv_to_dir(save_dir: str, rl_results: Dict[str, Any]) -> None:
    with open(os.path.join(save_dir, "paradigm_comparison.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm_base", "cent_objective", "cent_time", "cent_energy", "decent_objective", "decent_time", "decent_energy"])
        for base in ["DQN", "DDPG", "SAC"]:
            cent_key = f"C-{base}"
            decent_key = f"D-{base}"
            if cent_key not in rl_results or decent_key not in rl_results:
                continue
            cent = rl_results[cent_key]
            decent = rl_results[decent_key]
            writer.writerow([
                base,
                cent["objective_cost"][0],
                cent["time"][0],
                cent["energy"][0],
                decent["objective_cost"][0],
                decent["time"][0],
                decent["energy"][0],
            ])


def save_action_summary_csv(save_dir: str, baseline_results: Dict[str, Any], rl_results: Dict[str, Any]) -> None:
    all_results = {**baseline_results, **rl_results}
    with open(os.path.join(save_dir, "action_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "algorithm",
            "objective_mean",
            "objective_std",
            "time_mean",
            "time_std",
            "energy_mean",
            "energy_std",
            "viol_mean",
            "viol_std",
            "delta_mean",
            "delta_std",
            "p_v2v_mean",
            "p_v2v_std",
            "p_v2i_mean",
            "p_v2i_std",
        ])
        for name, results in all_results.items():
            analysis = results.get("action_analysis")
            if analysis is None:
                if name == "All-Leader":
                    delta_mean, delta_std = 1.0, 0.0
                    pv2v_mean, pv2v_std = 1.0, 0.0
                    pv2i_mean, pv2i_std = 0.0, 0.0
                elif name == "All-Base":
                    delta_mean, delta_std = 0.0, 0.0
                    pv2v_mean, pv2v_std = 0.0, 0.0
                    pv2i_mean, pv2i_std = 1.0, 0.0
                elif name == "Balanced":
                    delta_mean, delta_std = 0.5, 0.0
                    pv2v_mean, pv2v_std = 1.0, 0.0
                    pv2i_mean, pv2i_std = 1.0, 0.0
                else:
                    delta_mean = delta_std = pv2v_mean = pv2v_std = pv2i_mean = pv2i_std = ""
            else:
                delta_mean = analysis["v2v_fraction"]["mean"]
                delta_std = analysis["v2v_fraction"]["std"]
                pv2v_mean = analysis["v2v_power"]["mean"]
                pv2v_std = analysis["v2v_power"]["std"]
                pv2i_mean = analysis["v2i_power"]["mean"]
                pv2i_std = analysis["v2i_power"]["std"]
            writer.writerow([
                name,
                results["objective_cost"][0],
                results["objective_cost"][1],
                results["time"][0],
                results["time"][1],
                results["energy"][0],
                results["energy"][1],
                results["constraint_violations"][0],
                results["constraint_violations"][1],
                delta_mean,
                delta_std,
                pv2v_mean,
                pv2v_std,
                pv2i_mean,
                pv2i_std,
            ])


def save_subset_plot_data(trainer, selected_algorithms: List[str], baseline_results: Dict[str, Any], rl_results: Dict[str, Any]) -> None:
    metric_key = trainer._result_metric_key()
    all_results = {**baseline_results, **rl_results}
    npz_data = {
        **{
            f"training_{alg}_{metric}": np.array(data)
            for alg in selected_algorithms
            for metric, data in trainer.training_metrics[alg].items()
        },
        **{
            f"final_{name}_{metric}_mean": results[metric][0]
            for name, results in all_results.items()
            for metric in [metric_key, "time", "energy"]
        },
        **{
            f"final_{name}_{metric}_std": results[metric][1]
            for name, results in all_results.items()
            for metric in [metric_key, "time", "energy"]
        },
    }
    if hasattr(trainer, "timestep_data") and trainer.timestep_data:
        for alg_name, alg_data in trainer.timestep_data.items():
            for data_type in ["times", "energies", "v2v_fractions", "v2v_powers", "v2i_powers"]:
                if alg_data[data_type]:
                    means = np.array([x[0] for x in alg_data[data_type]])
                    stds = np.array([x[1] for x in alg_data[data_type]])
                    npz_data[f"timestep_{alg_name}_{data_type}_mean"] = means
                    npz_data[f"timestep_{alg_name}_{data_type}_std"] = stds
    np.savez_compressed(os.path.join(trainer.save_dir, "combined_all_plot_data.npz"), **npz_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or fine-tune selected DRL algorithms on the U-turn scenario.")
    parser.add_argument("--mode", choices=["scratch", "finetune"], required=True)
    parser.add_argument("--source-run-dir", type=str, required=True, help="Straight-road run directory used for eta/config and optional fine-tune weights")
    parser.add_argument("--algorithm-set", choices=["paper3", "all6"], default="paper3")
    parser.add_argument("--algorithms", nargs="+", choices=ORDERED_ALGORITHMS, default=None)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=25)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--results-group", type=str, required=True)
    parser.add_argument("--de-mode", choices=["paper", "screen", "sweep"], default="screen")
    parser.add_argument("--uturn-radius", type=float, default=10.0)
    parser.add_argument("--uturn-leg-length", type=float, default=178.04203673205103)
    parser.add_argument("--base-station-x", type=float, default=0.0)
    parser.add_argument("--base-station-y", type=float, default=10.0)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    selected_algorithms = parse_algorithms(args)
    source_run_dir = os.path.abspath(args.source_run_dir)
    source_manifest, source_config = load_source_config(source_run_dir)
    eta = canonical_eta(source_manifest["eta"])
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

    source_best_models = None
    if args.mode == "finetune":
        source_best_models = recover_source_best_models(trainer, source_run_dir, selected_algorithms)

    print("=" * 80)
    print(f"U-TURN {args.mode.upper()} TRAINING")
    print("=" * 80)
    print(f"Algorithms: {', '.join(selected_algorithms)}")
    print(f"Eta: {eta:.2f}")
    print(f"Episodes: {args.episodes}")
    print(f"Geometry: R={args.uturn_radius}, L={args.uturn_leg_length}, BS=({args.base_station_x}, {args.base_station_y})")
    print(f"Source run: {source_run_dir}")

    start_time = time.time()
    for alg_name in selected_algorithms:
        agent = instantiate_and_optionally_load(trainer, alg_name, source_best_models if args.mode == "finetune" else None)
        try:
            family = algorithm_family(alg_name)
            if is_centralized(alg_name):
                trainer.agents[alg_name] = train_existing_centralized_algorithm(trainer, agent, family, alg_name)
            else:
                trainer.agents[alg_name] = train_existing_decentralized_algorithm(trainer, agent, family, alg_name)
        finally:
            clear_algorithm_agent(trainer, alg_name)

    baseline_results = trainer.evaluate_baselines(
        num_episodes=args.final_eval_episodes,
        **get_de_settings(args.de_mode, int(source_manifest["seed"])),
    )
    rl_results = evaluate_selected_algorithms(
        trainer,
        selected_algorithms=selected_algorithms,
        num_episodes=args.final_eval_episodes,
    )
    trainer.timestep_data = trainer.plot_timestep_analysis(baseline_results, rl_results)
    trainer.plot_summary_table(baseline_results, rl_results)
    trainer.plot_action_heatmaps(rl_results)
    save_subset_results(
        trainer,
        selected_algorithms=selected_algorithms,
        baseline_results=baseline_results,
        rl_results=rl_results,
        source_run_dir=source_run_dir,
        mode=args.mode,
        source_best_models=source_best_models,
    )
    save_training_rewards_csv(trainer, selected_algorithms)
    plot_training_rewards_subset(trainer, selected_algorithms)
    save_algorithm_comparison_csv(trainer, baseline_results, rl_results)
    save_paradigm_comparison_csv_to_dir(trainer.save_dir, rl_results)
    save_action_summary_csv(trainer.save_dir, baseline_results, rl_results)
    save_subset_plot_data(trainer, selected_algorithms, baseline_results, rl_results)

    manifest = {
        "mode": args.mode,
        "source_run_dir": source_run_dir,
        "eta": eta,
        "selected_algorithms": selected_algorithms,
        "episodes": args.episodes,
        "eval_interval": args.eval_interval,
        "checkpoint_eval_episodes": args.checkpoint_eval_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "seed": int(source_manifest["seed"]),
        "device": args.device,
        "results_dir": run_dir,
        "de_mode": args.de_mode,
        "uturn_radius": args.uturn_radius,
        "uturn_leg_length": args.uturn_leg_length,
        "uturn_open_ended": not args.closed_loop,
        "base_station_position": [args.base_station_x, args.base_station_y],
        "elapsed_minutes": (time.time() - start_time) / 60.0,
    }
    write_manifest(os.path.join(run_dir, "uturn_training_manifest.json"), manifest)
    print(f"U-turn training suite complete. Results saved in: {run_dir}")


if __name__ == "__main__":
    main()
