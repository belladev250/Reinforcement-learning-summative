"""
Generates the visualizations required by the "Results Discussion" section
of the report:
  1. cumulative_rewards.png   - reward-over-episodes subplot for all 4 methods
  2. dqn_training_curve.png   - DQN loss / objective curve
  3. pg_entropy.png           - policy entropy curves for REINFORCE, PPO, A2C
  4. convergence.png          - episodes-to-converge comparison (rolling mean)
  5. generalization.png       - performance on unseen start conditions

Run this AFTER training each algorithm's main model (via main.py or the
individual training scripts), since it reads from logs/<algo>/.

Usage:
    uv run training/plotting.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment.custom_env import ClinicRestockEnv
from training.evaluate import evaluate_policy_fn

OUT_DIR = "assets"
os.makedirs(OUT_DIR, exist_ok=True)


def _rolling(x, window=10):
    return pd.Series(x).rolling(window, min_periods=1).mean()


def load_monitor_csv(path):
    """SB3 Monitor CSVs have a 1-line JSON header comment; skip it."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, skiprows=1)


def plot_cumulative_rewards():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    algo_paths = {
        "DQN": "logs/dqn/monitor.csv",
        "REINFORCE": "logs/reinforce/training_history.csv",
        "PPO": "logs/ppo/monitor.csv",
        "A2C": "logs/a2c/monitor.csv",
    }
    for ax, (name, path) in zip(axes.flat, algo_paths.items()):
        if name == "REINFORCE":
            df = pd.read_csv(path) if os.path.exists(path) else None
            reward_col = "reward"
        else:
            df = load_monitor_csv(path)
            reward_col = "r"
        if df is None:
            ax.set_title(f"{name} (no log found)")
            continue
        ax.plot(df[reward_col], alpha=0.3, color="steelblue", label="raw")
        ax.plot(_rolling(df[reward_col], 10), color="darkblue", label="rolling mean (10)")
        ax.set_title(f"{name}: Cumulative Reward per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cumulative_rewards.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/cumulative_rewards.png")


def plot_dqn_training_curve():
    path = "logs/dqn/progress.csv"
    if not os.path.exists(path):
        print("no DQN progress.csv found, skipping")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    if "train/loss" in df.columns:
        ax.plot(df["time/total_timesteps"], df["train/loss"], color="firebrick")
    ax.set_title("DQN Training Objective (Loss) Curve")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dqn_training_curve.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/dqn_training_curve.png")


def plot_pg_entropy():
    fig, ax = plt.subplots(figsize=(8, 5))
    # REINFORCE: from custom training history
    if os.path.exists("logs/reinforce/training_history.csv"):
        df = pd.read_csv("logs/reinforce/training_history.csv")
        ax.plot(df["episode"], _rolling(df["entropy"], 10), label="REINFORCE")
    # PPO / A2C: from SB3 progress.csv (train/entropy_loss is -entropy)
    for name, path in [("PPO", "logs/ppo/progress.csv"), ("A2C", "logs/a2c/progress.csv")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "train/entropy_loss" in df.columns:
                ax.plot(df["time/total_timesteps"], -df["train/entropy_loss"], label=name)
    ax.set_title("Policy Entropy over Training (Policy-Gradient Methods)")
    ax.set_xlabel("Timesteps / Episode")
    ax.set_ylabel("Entropy (higher = more exploration)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pg_entropy.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/pg_entropy.png")


def plot_convergence(window=10, threshold_frac=0.9):
    """Episodes-to-converge: first episode index where the rolling mean
    reward reaches within threshold_frac of its own best rolling value."""
    algo_paths = {
        "DQN": ("logs/dqn/monitor.csv", "r", True),
        "REINFORCE": ("logs/reinforce/training_history.csv", "reward", False),
        "PPO": ("logs/ppo/monitor.csv", "r", True),
        "A2C": ("logs/a2c/monitor.csv", "r", True),
    }
    names, episodes_to_converge = [], []
    for name, (path, col, is_monitor) in algo_paths.items():
        df = load_monitor_csv(path) if is_monitor else (pd.read_csv(path) if os.path.exists(path) else None)
        if df is None:
            continue
        roll = _rolling(df[col], window)
        best = roll.max()
        target = best - threshold_frac * abs(best) if best < 0 else threshold_frac * best
        converged_idx = next((i for i, v in enumerate(roll) if v >= target), len(roll) - 1)
        names.append(name)
        episodes_to_converge.append(converged_idx)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(names, episodes_to_converge, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(names)])
    ax.set_title("Episodes to Reach ~Stable Performance")
    ax.set_ylabel("Episode #")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/convergence.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/convergence.png", dict(zip(names, episodes_to_converge)))


def plot_generalization(model_loaders, n_episodes=10):
    """model_loaders: dict of {name: predict_fn(obs)->action}.
    Tests each on UNSEEN initial conditions (different starting stock
    levels and starting week/season) to see how well policies generalize."""
    results = {}
    for name, predict_fn in model_loaders.items():
        rewards = []
        for start_frac, start_week in [(0.2, 0), (0.8, 0), (0.5, 20), (0.1, 40), (0.9, 10)]:
            env = ClinicRestockEnv(episode_length=52, seed=999)
            obs, _ = env.reset()
            env.stock = env.stock * 0 + start_frac * np.array([300.0, 200.0, 150.0])
            env.week = start_week
            total = 0.0
            done = False
            while not done:
                action = predict_fn(obs)
                obs, reward, term, trunc, info = env.step(action)
                total += reward
                done = term or trunc
            rewards.append(total)
        results[name] = rewards

    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(v) for v in results.values()]
    stds = [np.std(v) for v in results.values()]
    ax.bar(list(results.keys()), means, yerr=stds, capsize=5,
           color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(results)])
    ax.set_title("Generalization to Unseen Start Conditions")
    ax.set_ylabel("Mean Episode Reward (5 unseen scenarios)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/generalization.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/generalization.png", results)


if __name__ == "__main__":
    plot_cumulative_rewards()
    plot_dqn_training_curve()
    plot_pg_entropy()
    plot_convergence()
    print("Run training/generalization_test.py separately once all 4 models are trained "
          "(it loads saved models and calls plot_generalization()).")
