# Deep Reinforcement Learning-Aided Strategies for Big Data Offloading in Vehicular Networks

Code for the paper "Deep Reinforcement Learning-Aided Strategies for Big Data Offloading in Vehicular Networks" (IEEE Transactions on Machine Learning in Communications and Networking, 2026) by Talha Akyıldız and Hessam Mahdavifar.

The environment models chunk-based data offloading in a vehicular cluster: each follower splits its data between a V2V link to a leader (which deduplicates and uploads) and a direct V2I link, and sets its transmit powers. Centralized and decentralized DQN, DDPG, and SAC agents are trained to minimize a combined energy–time cost controlled by a single knob `eta` (`0` = energy, `1` = time, `0.5` = balanced).

## Citation

```bibtex
@article{akyildiz2026drl,
  title   = {Deep Reinforcement Learning-Aided Strategies for Big Data Offloading in Vehicular Networks},
  author  = {Aky{\i}ld{\i}z, Talha and Mahdavifar, Hessam},
  journal = {IEEE Transactions on Machine Learning in Communications and Networking},
  year    = {2026}
}
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.9, NumPy 2.1, SciPy 1.15, and Matplotlib 3.10.

## Layout

```
environment.py            vehicular offloading environment
baselines.py              fixed baselines (All-Leader, All-Base, Balanced)
oracle_baselines.py       DE-Search benchmark
train_evaluate_all.py     training and evaluation for all agents
centralized/              centralized DQN, DDPG, SAC
decentralized/            decentralized (shared-policy) DQN, DDPG, SAC
configs/                  environment and algorithm settings
utils/                    replay buffers
run_*.py                  paper experiment entry points
curved_road_analysis/     U-turn geometry environment and transfer runs
```

## Running the experiments

Each run writes a timestamped folder under `results/` with `algorithm_comparison.csv` (the per-method objective/time/energy values) and the training and offloading data used for the figures.

Main results, one per objective setting:

```bash
python run_paper_main_single_eta.py --eta 0.0
python run_paper_main_single_eta.py --eta 0.5
python run_paper_main_single_eta.py --eta 1.0
```

Cluster-size and redundancy sweeps:

```bash
python run_paper_full_sensitivity_sweep.py --sweep num_vehicles --etas 0,1.0 --values 3,4,5,6,7
python run_paper_full_sensitivity_sweep.py --sweep beta --etas 0,1.0 --values 0.3,0.4,0.5,0.6,0.7
```

Constraint-penalty sensitivity:

```bash
python run_lambda_sweep_all_drl.py --eta 0.0
```

Zero-shot transfer to other cluster sizes, starting from a trained `N=5` run:

```bash
python run_straight_cross_n_transfer_eval.py --source-run-dir results/<eta0_N5_run> --num-vehicles 3
```

U-turn geometry — zero-shot, fine-tune, scratch, and the balanced baseline, starting from a trained straight-road run:

```bash
python curved_road_analysis/run_uturn_transfer_eval.py  --source-run-dir results/<straight_run>
python curved_road_analysis/run_uturn_training_suite.py --mode finetune --source-run-dir results/<straight_run> --results-group uturn_finetune
python curved_road_analysis/run_uturn_training_suite.py --mode scratch  --source-run-dir results/<straight_run> --results-group uturn_scratch
python curved_road_analysis/run_uturn_balanced_eval.py  --source-run-dir results/<straight_run>
```

The DRL agents are trained from scratch, so retrained results may differ slightly from the reported values while preserving the same trends.

## License

MIT — see [LICENSE](LICENSE).
