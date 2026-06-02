#!/usr/bin/env python3
"""
Train and evaluate all methods for a single eta value.

Trains the six DRL agents, evaluates the fixed baselines and DE-Search,
and saves the training and evaluation outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import torch

from train_evaluate_all import CombinedTrainEvaluate
from centralized.dqn_centralized import CentralizedDQNAgent
from centralized.ddpg_centralized import CentralizedDDPGAgent
from centralized.sac_centralized import CentralizedSACAgent
from decentralized.dqn_decentralized import MultiAgentDQN
from decentralized.ddpg_decentralized import MultiAgentDDPG
from decentralized.sac_decentralized import MultiAgentSAC


PAPER_LAMBDAS = {
    0.0: 0.1,
    0.5: 0.1,
    1.0: 0.1,
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


def build_trainer(
    eta: float,
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
        constraint_penalty_value=PAPER_LAMBDAS[eta],
        base_station_position=(225.0, 0.0),
        leader_selection_mode="episode_start_scored",
        leader_reselection_interval_slots=10,
        initial_position_mode="random_uniform",
        initial_position_x_range=(0.0, 50.0),
        initial_position_jitter=0.0,
        use_fixed_eval_seeds=True,
        fixed_eval_seed_base=100000,
    )


def retarget_save_dir(trainer: CombinedTrainEvaluate, eta: float, results_group: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    eta_tag = f"eta_{int(round(eta * 100)):03d}"
    run_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        results_group,
        eta_tag,
        f"run_{timestamp}_{os.getpid()}",
    )
    os.makedirs(run_dir, exist_ok=True)
    trainer.save_dir = run_dir
    return run_dir


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def instantiate_agents(trainer: CombinedTrainEvaluate) -> None:
    trainer.agents["C-DQN"] = CentralizedDQNAgent(
        trainer.state_dim,
        trainer.num_followers,
        trainer.device,
        optimization_mode=trainer.optimization_target,
    )
    trainer.agents["C-DDPG"] = CentralizedDDPGAgent(
        trainer.state_dim,
        trainer.action_dim,
        trainer.device,
        optimization_mode=trainer.optimization_target,
        mixed_time_weight=trainer.mixed_time_weight,
    )
    trainer.agents["C-SAC"] = CentralizedSACAgent(
        trainer.state_dim,
        trainer.action_dim,
        trainer.device,
        optimization_mode=trainer.optimization_target,
        mixed_time_weight=trainer.mixed_time_weight,
    )
    trainer.agents["D-DQN"] = MultiAgentDQN(
        trainer.local_state_size,
        trainer.num_followers,
        trainer.device,
        optimization_mode=trainer.optimization_target,
    )
    trainer.agents["D-DDPG"] = MultiAgentDDPG(
        trainer.local_state_size,
        trainer.num_followers,
        trainer.device,
        optimization_mode=trainer.optimization_target,
        mixed_time_weight=trainer.mixed_time_weight,
    )
    trainer.agents["D-SAC"] = MultiAgentSAC(
        trainer.local_state_size,
        trainer.num_followers,
        trainer.device,
        optimization_mode=trainer.optimization_target,
        mixed_time_weight=trainer.mixed_time_weight,
    )


def instantiate_single_agent(trainer: CombinedTrainEvaluate, alg_name: str) -> None:
    if alg_name == "C-DQN":
        trainer.agents[alg_name] = CentralizedDQNAgent(
            trainer.state_dim,
            trainer.num_followers,
            trainer.device,
            optimization_mode=trainer.optimization_target,
        )
    elif alg_name == "C-DDPG":
        trainer.agents[alg_name] = CentralizedDDPGAgent(
            trainer.state_dim,
            trainer.action_dim,
            trainer.device,
            optimization_mode=trainer.optimization_target,
            mixed_time_weight=trainer.mixed_time_weight,
        )
    elif alg_name == "C-SAC":
        trainer.agents[alg_name] = CentralizedSACAgent(
            trainer.state_dim,
            trainer.action_dim,
            trainer.device,
            optimization_mode=trainer.optimization_target,
            mixed_time_weight=trainer.mixed_time_weight,
        )
    elif alg_name == "D-DQN":
        trainer.agents[alg_name] = MultiAgentDQN(
            trainer.local_state_size,
            trainer.num_followers,
            trainer.device,
            optimization_mode=trainer.optimization_target,
        )
    elif alg_name == "D-DDPG":
        trainer.agents[alg_name] = MultiAgentDDPG(
            trainer.local_state_size,
            trainer.num_followers,
            trainer.device,
            optimization_mode=trainer.optimization_target,
            mixed_time_weight=trainer.mixed_time_weight,
        )
    elif alg_name == "D-SAC":
        trainer.agents[alg_name] = MultiAgentSAC(
            trainer.local_state_size,
            trainer.num_followers,
            trainer.device,
            optimization_mode=trainer.optimization_target,
            mixed_time_weight=trainer.mixed_time_weight,
        )
    else:
        raise ValueError(f"Unsupported algorithm {alg_name}")


def recover_best_model_info(trainer: CombinedTrainEvaluate, run_dir: str) -> None:
    for alg_name in trainer.best_models:
        checkpoint_root = os.path.join(run_dir, "checkpoints", alg_name.lower())
        best_reward = -float("inf")
        if os.path.isdir(checkpoint_root):
            for episode_dir in sorted(os.listdir(checkpoint_root)):
                meta_path = os.path.join(checkpoint_root, episode_dir, "checkpoint_meta.json")
                if not os.path.isfile(meta_path):
                    continue
                with open(meta_path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                best_reward = max(best_reward, float(meta.get("eval_reward", -float("inf"))))

        if alg_name.startswith("C-"):
            best_path = os.path.join(run_dir, f"{alg_name.lower()}_best_model.pth")
        else:
            best_path = os.path.join(run_dir, f"{alg_name.lower()}_best_model")

        if os.path.exists(best_path):
            trainer.best_models[alg_name]["path"] = best_path
            trainer.best_models[alg_name]["reward"] = best_reward if best_reward > -float("inf") else 0.0


def recover_checkpoint_eval_curves(trainer: CombinedTrainEvaluate, run_dir: str) -> None:
    mapping = {
        "reward": "eval_rewards",
        "penalized": "eval_penalizeds",
        "objective_cost": "eval_objective_costs",
        "time": "eval_times",
        "energy": "eval_energies",
        "mixed": "eval_mixeds",
        "normalized_time": "eval_normalized_times",
        "normalized_energy": "eval_normalized_energies",
        "constraint_violations": "eval_constraint_violations",
        "constraint_penalty": "eval_constraint_penalties",
        "any_constraint_violations": "eval_any_constraint_violations",
        "follower_time_violations": "eval_follower_time_violations",
        "leader_time_violations": "eval_leader_time_violations",
        "follower_energy_violations": "eval_follower_energy_violations",
        "leader_energy_violations": "eval_leader_energy_violations",
    }
    for alg_name in trainer.training_metrics:
        checkpoint_root = os.path.join(run_dir, "checkpoints", alg_name.lower())
        if not os.path.isdir(checkpoint_root):
            continue
        checkpoint_records = []
        for episode_dir in sorted(os.listdir(checkpoint_root)):
            meta_path = os.path.join(checkpoint_root, episode_dir, "checkpoint_meta.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as handle:
                checkpoint_records.append(json.load(handle))
        checkpoint_records.sort(key=lambda item: int(item.get("episode", 0)))
        for record in checkpoint_records:
            for src_key, dst_key in mapping.items():
                trainer.training_metrics[alg_name][dst_key].append(float(record[f"eval_{src_key}"]))


def evaluate_and_save(
    trainer: CombinedTrainEvaluate,
    eta: float,
    episodes: int,
    eval_interval: int,
    checkpoint_eval_episodes: int,
    run_dir: str,
    final_eval_episodes: int,
    seed: int,
    de_mode: str,
    num_vehicles: int,
    beta: float,
    resumed_from: str | None = None,
) -> str:
    if not trainer.agents:
        raise RuntimeError("No initialized RL agents available for evaluation")

    baseline_results = trainer.evaluate_baselines(
        num_episodes=final_eval_episodes,
        **get_de_settings(de_mode, seed),
    )
    rl_results = trainer.final_evaluation(num_episodes=final_eval_episodes)
    trainer.plot_results(baseline_results, rl_results)
    trainer.save_results(baseline_results, rl_results)

    manifest = {
        "eta": eta,
        "lambda": PAPER_LAMBDAS[eta],
        "episodes": episodes,
        "eval_interval": eval_interval,
        "checkpoint_eval_episodes": checkpoint_eval_episodes,
        "final_eval_episodes": final_eval_episodes,
        "seed": seed,
        "device": trainer.device,
        "num_vehicles": num_vehicles,
        "beta": beta,
        "save_dir": run_dir,
        "resumed_from": resumed_from,
        "de_mode": de_mode,
        "de_settings": get_de_settings(de_mode, seed),
        "config_snapshot_path": os.path.join(run_dir, "experiment_config.json"),
        "results_json_path": os.path.join(run_dir, "combined_comprehensive_results.json"),
        "training_npz_path": os.path.join(run_dir, "combined_training_data.npz"),
        "summary_plot_path": os.path.join(run_dir, "combined_comprehensive_results.png"),
        "timestep_plot_path": os.path.join(run_dir, "combined_timestep_analysis.png"),
    }
    write_manifest(os.path.join(run_dir, "paper_run_manifest.json"), manifest)
    return run_dir


def run_single_eta(
    eta: float,
    episodes: int,
    eval_interval: int,
    checkpoint_eval_episodes: int,
    final_eval_episodes: int,
    seed: int,
    device: str,
    chunk_size_mb: float,
    follower_time_budget: float,
    leader_time_budget: float,
    num_vehicles: int,
    beta: float,
    results_group: str,
    de_mode: str,
    only_algorithm: str | None = None,
) -> str:
    trainer = build_trainer(
        eta,
        episodes,
        eval_interval,
        checkpoint_eval_episodes,
        seed,
        device,
        chunk_size_mb,
        follower_time_budget,
        leader_time_budget,
        num_vehicles,
        beta,
    )
    run_dir = retarget_save_dir(trainer, eta, results_group)

    start_time = time.time()

    if only_algorithm is None:
        for algorithm in ["DQN", "DDPG", "SAC"]:
            alg_name = f"C-{algorithm}"
            trainer.agents[alg_name] = trainer.train_centralized_algorithm(algorithm, alg_name)

        for algorithm in ["DQN", "DDPG", "SAC"]:
            alg_name = f"D-{algorithm}"
            trainer.agents[alg_name] = trainer.train_decentralized_algorithm(algorithm, alg_name)
    else:
        family, algorithm = only_algorithm.split("-", 1)
        if family == "C":
            trainer.agents[only_algorithm] = trainer.train_centralized_algorithm(algorithm, only_algorithm)
        else:
            trainer.agents[only_algorithm] = trainer.train_decentralized_algorithm(algorithm, only_algorithm)

    evaluate_and_save(
        trainer,
        eta,
        episodes,
        eval_interval,
        checkpoint_eval_episodes,
        run_dir,
        final_eval_episodes,
        seed,
        de_mode,
        num_vehicles,
        beta,
    )

    manifest_path = os.path.join(run_dir, "paper_run_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["elapsed_minutes"] = (time.time() - start_time) / 60.0
    write_manifest(manifest_path, manifest)
    return run_dir


def resume_evaluation(
    eta: float,
    run_dir: str,
    episodes: int,
    eval_interval: int,
    checkpoint_eval_episodes: int,
    final_eval_episodes: int,
    seed: int,
    device: str,
    chunk_size_mb: float,
    follower_time_budget: float,
    leader_time_budget: float,
    num_vehicles: int,
    beta: float,
    de_mode: str,
    only_algorithm: str | None = None,
) -> str:
    trainer = build_trainer(
        eta,
        episodes,
        eval_interval,
        checkpoint_eval_episodes,
        seed,
        device,
        chunk_size_mb,
        follower_time_budget,
        leader_time_budget,
        num_vehicles,
        beta,
    )
    trainer.save_dir = run_dir
    if only_algorithm is None:
        instantiate_agents(trainer)
    else:
        instantiate_single_agent(trainer, only_algorithm)
    recover_best_model_info(trainer, run_dir)
    recover_checkpoint_eval_curves(trainer, run_dir)
    for alg_name, agent in trainer.agents.items():
        path = trainer.best_models[alg_name]["path"]
        if not path:
            raise FileNotFoundError(f"Could not recover best model path for {alg_name} in {run_dir}")
        agent.load(path)
    return evaluate_and_save(
        trainer,
        eta,
        episodes,
        eval_interval,
        checkpoint_eval_episodes,
        run_dir,
        final_eval_episodes,
        seed,
        de_mode,
        num_vehicles,
        beta,
        resumed_from=run_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate all methods for one eta value.")
    parser.add_argument("--eta", type=float, required=True, help="Eta in {0, 0.5, 1.0}")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    parser.add_argument("--eval-interval", type=int, default=100, help="Checkpoint eval interval")
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=25, help="Checkpoint eval episodes")
    parser.add_argument("--final-eval-episodes", type=int, default=100, help="Final evaluation episodes")
    parser.add_argument("--chunk-size-mb", type=float, default=20.0, help="Chunk size in Mb")
    parser.add_argument("--follower-time-budget", type=float, default=1.0, help="Follower time budget in seconds")
    parser.add_argument("--leader-time-budget", type=float, default=2.0, help="Leader time budget in seconds")
    parser.add_argument("--num-vehicles", type=int, default=5, help="Total vehicles in the cluster")
    parser.add_argument("--beta", type=float, default=0.5, help="Fixed redundancy ratio")
    parser.add_argument("--results-group", type=str, default="paper_main_runs", help="Results subdirectory under results/")
    parser.add_argument("--de-mode", choices=sorted(DE_PRESETS), default="paper", help="DE preset: 'paper' for final tables, lighter presets for quick runs")
    parser.add_argument(
        "--only-algorithm",
        choices=["C-DQN", "C-DDPG", "C-SAC", "D-DQN", "D-DDPG", "D-SAC"],
        default=None,
        help="Train and evaluate only one RL algorithm instead of all six",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume-run-dir", type=str, default=None, help="Existing run directory to resume for evaluation only")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    args = parser.parse_args()

    eta = canonical_eta(args.eta)
    if args.resume_run_dir:
        run_dir = resume_evaluation(
            eta=eta,
            run_dir=os.path.abspath(args.resume_run_dir),
            episodes=args.episodes,
            eval_interval=args.eval_interval,
            checkpoint_eval_episodes=args.checkpoint_eval_episodes,
            final_eval_episodes=args.final_eval_episodes,
            seed=args.seed,
            device=args.device,
            chunk_size_mb=args.chunk_size_mb,
            follower_time_budget=args.follower_time_budget,
            leader_time_budget=args.leader_time_budget,
            num_vehicles=args.num_vehicles,
            beta=args.beta,
            de_mode=args.de_mode,
            only_algorithm=args.only_algorithm,
        )
    else:
        run_dir = run_single_eta(
            eta=eta,
            episodes=args.episodes,
            eval_interval=args.eval_interval,
            checkpoint_eval_episodes=args.checkpoint_eval_episodes,
            final_eval_episodes=args.final_eval_episodes,
            seed=args.seed,
            device=args.device,
            chunk_size_mb=args.chunk_size_mb,
            follower_time_budget=args.follower_time_budget,
            leader_time_budget=args.leader_time_budget,
            num_vehicles=args.num_vehicles,
            beta=args.beta,
            results_group=args.results_group,
            de_mode=args.de_mode,
            only_algorithm=args.only_algorithm,
        )
    print(f"Run complete. Outputs saved in: {run_dir}")


if __name__ == "__main__":
    main()
