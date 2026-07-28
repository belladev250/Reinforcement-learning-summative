"""
Runs the hyperparameter sweeps required by the assignment: >=10 runs per
algorithm (DQN, REINFORCE, PPO, A2C), varying real hyperparameters, and
writes one CSV per algorithm to logs/sweeps/. Paste these CSVs straight
into the report tables (Section: Implementation).

NOTE on timesteps: --timesteps controls how long EACH of the 40 runs
trains for. 100_000 gives meaningful results but 40 runs x 100k steps
will take a while on CPU. If you are close to the deadline, use
--timesteps 20000 to get the sweep done fast, note in the report that
runs were shortened for time, and optionally retrain just your single
best config per algorithm for longer afterward (that's what main.py /
the demo video should use).

Usage:
    uv run training/hyperparam_sweep.py --timesteps 20000
"""

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from training.dqn_training import train_dqn
from training.pg_training import train_ppo, train_a2c, train_reinforce

OUT_DIR = "logs/sweeps"


def run_dqn_sweep(timesteps):
    configs = [
        dict(learning_rate=1e-2, gamma=0.90, buffer_size=10_000, batch_size=32, exploration_final_eps=0.10),
        dict(learning_rate=1e-3, gamma=0.95, buffer_size=20_000, batch_size=32, exploration_final_eps=0.10),
        dict(learning_rate=1e-3, gamma=0.99, buffer_size=50_000, batch_size=64, exploration_final_eps=0.05),
        dict(learning_rate=5e-4, gamma=0.99, buffer_size=50_000, batch_size=64, exploration_final_eps=0.05),
        dict(learning_rate=5e-4, gamma=0.995, buffer_size=100_000, batch_size=128, exploration_final_eps=0.02),
        dict(learning_rate=1e-4, gamma=0.99, buffer_size=50_000, batch_size=64, exploration_final_eps=0.05),
        dict(learning_rate=3e-4, gamma=0.98, buffer_size=30_000, batch_size=32, exploration_final_eps=0.10),
        dict(learning_rate=3e-4, gamma=0.99, buffer_size=100_000, batch_size=256, exploration_final_eps=0.05),
        dict(learning_rate=1e-3, gamma=0.99, buffer_size=100_000, batch_size=64, exploration_final_eps=0.01),
        dict(learning_rate=2.5e-4, gamma=0.995, buffer_size=50_000, batch_size=128, exploration_final_eps=0.02),
    ]
    rows = []
    for i, cfg in enumerate(configs):
        print(f"\n=== DQN run {i+1}/{len(configs)}: {cfg} ===")
        t0 = time.time()
        _, result = train_dqn(
            total_timesteps=timesteps, seed=i,
            model_path=f"models/dqn/sweep_{i}.zip", log_dir=f"logs/dqn_sweep_{i}",
            **cfg,
        )
        rows.append({**cfg, "run": i, "mean_reward": result["mean_reward"],
                     "std_reward": result["std_reward"], "mean_unmet_demand": result["mean_unmet_demand"],
                     "train_seconds": round(time.time() - t0, 1)})
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/dqn_sweep.csv", index=False)
    print(df)
    return df


def run_reinforce_sweep(timesteps):
    configs = [
        dict(learning_rate=1e-2, gamma=0.90, hidden_size=32, entropy_coef=0.0),
        dict(learning_rate=1e-3, gamma=0.95, hidden_size=32, entropy_coef=0.0),
        dict(learning_rate=1e-3, gamma=0.99, hidden_size=64, entropy_coef=0.0),
        dict(learning_rate=5e-4, gamma=0.99, hidden_size=64, entropy_coef=0.01),
        dict(learning_rate=5e-4, gamma=0.995, hidden_size=128, entropy_coef=0.01),
        dict(learning_rate=1e-4, gamma=0.99, hidden_size=64, entropy_coef=0.0),
        dict(learning_rate=3e-4, gamma=0.98, hidden_size=64, entropy_coef=0.02),
        dict(learning_rate=3e-4, gamma=0.99, hidden_size=128, entropy_coef=0.0),
        dict(learning_rate=1e-3, gamma=0.99, hidden_size=128, entropy_coef=0.01),
        dict(learning_rate=2.5e-4, gamma=0.995, hidden_size=64, entropy_coef=0.02),
    ]
    rows = []
    for i, cfg in enumerate(configs):
        print(f"\n=== REINFORCE run {i+1}/{len(configs)}: {cfg} ===")
        t0 = time.time()
        _, result = train_reinforce(
            total_timesteps=timesteps, seed=i,
            model_path=f"models/pg/reinforce_sweep_{i}.pt", log_dir=f"logs/reinforce_sweep_{i}",
            **cfg,
        )
        rows.append({**cfg, "run": i, "mean_reward": result["mean_reward"],
                     "std_reward": result["std_reward"], "mean_unmet_demand": result["mean_unmet_demand"],
                     "train_seconds": round(time.time() - t0, 1)})
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/reinforce_sweep.csv", index=False)
    print(df)
    return df


def run_ppo_sweep(timesteps):
    configs = [
        dict(learning_rate=1e-2, gamma=0.90, n_steps=64, ent_coef=0.0, clip_range=0.1),
        dict(learning_rate=1e-3, gamma=0.95, n_steps=128, ent_coef=0.0, clip_range=0.2),
        dict(learning_rate=3e-4, gamma=0.99, n_steps=256, ent_coef=0.01, clip_range=0.2),
        dict(learning_rate=3e-4, gamma=0.99, n_steps=512, ent_coef=0.01, clip_range=0.2),
        dict(learning_rate=1e-4, gamma=0.995, n_steps=256, ent_coef=0.005, clip_range=0.3),
        dict(learning_rate=5e-4, gamma=0.99, n_steps=128, ent_coef=0.02, clip_range=0.2),
        dict(learning_rate=3e-4, gamma=0.98, n_steps=256, ent_coef=0.0, clip_range=0.1),
        dict(learning_rate=1e-3, gamma=0.99, n_steps=256, ent_coef=0.01, clip_range=0.3),
        dict(learning_rate=2.5e-4, gamma=0.99, n_steps=1024, ent_coef=0.01, clip_range=0.2),
        dict(learning_rate=3e-4, gamma=0.995, n_steps=256, ent_coef=0.03, clip_range=0.2),
    ]
    rows = []
    for i, cfg in enumerate(configs):
        print(f"\n=== PPO run {i+1}/{len(configs)}: {cfg} ===")
        t0 = time.time()
        _, result = train_ppo(
            total_timesteps=timesteps, seed=i,
            model_path=f"models/pg/ppo_sweep_{i}.zip", log_dir=f"logs/ppo_sweep_{i}",
            **cfg,
        )
        rows.append({**cfg, "run": i, "mean_reward": result["mean_reward"],
                     "std_reward": result["std_reward"], "mean_unmet_demand": result["mean_unmet_demand"],
                     "train_seconds": round(time.time() - t0, 1)})
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/ppo_sweep.csv", index=False)
    print(df)
    return df


def run_a2c_sweep(timesteps):
    configs = [
        dict(learning_rate=1e-2, gamma=0.90, n_steps=5, ent_coef=0.0),
        dict(learning_rate=7e-3, gamma=0.95, n_steps=5, ent_coef=0.0),
        dict(learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.0),
        dict(learning_rate=7e-4, gamma=0.99, n_steps=10, ent_coef=0.01),
        dict(learning_rate=3e-4, gamma=0.995, n_steps=20, ent_coef=0.01),
        dict(learning_rate=1e-4, gamma=0.99, n_steps=10, ent_coef=0.0),
        dict(learning_rate=5e-4, gamma=0.98, n_steps=8, ent_coef=0.02),
        dict(learning_rate=7e-4, gamma=0.99, n_steps=16, ent_coef=0.01),
        dict(learning_rate=2.5e-4, gamma=0.99, n_steps=32, ent_coef=0.01),
        dict(learning_rate=7e-4, gamma=0.995, n_steps=5, ent_coef=0.03),
    ]
    rows = []
    for i, cfg in enumerate(configs):
        print(f"\n=== A2C run {i+1}/{len(configs)}: {cfg} ===")
        t0 = time.time()
        _, result = train_a2c(
            total_timesteps=timesteps, seed=i,
            model_path=f"models/pg/a2c_sweep_{i}.zip", log_dir=f"logs/a2c_sweep_{i}",
            **cfg,
        )
        rows.append({**cfg, "run": i, "mean_reward": result["mean_reward"],
                     "std_reward": result["std_reward"], "mean_unmet_demand": result["mean_unmet_demand"],
                     "train_seconds": round(time.time() - t0, 1)})
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/a2c_sweep.csv", index=False)
    print(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20_000,
                         help="Timesteps PER RUN. 40 runs total across all 4 algos.")
    parser.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c", "all"], default="all")
    args = parser.parse_args()

    if args.algo in ("dqn", "all"):
        run_dqn_sweep(args.timesteps)
    if args.algo in ("reinforce", "all"):
        run_reinforce_sweep(args.timesteps)
    if args.algo in ("ppo", "all"):
        run_ppo_sweep(args.timesteps)
    if args.algo in ("a2c", "all"):
        run_a2c_sweep(args.timesteps)
