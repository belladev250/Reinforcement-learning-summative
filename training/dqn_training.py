"""Train a DQN agent on ClinicRestockEnv.

Usage:
    uv run training/dqn_training.py --timesteps 100000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from environment.custom_env import ClinicRestockEnv
from training.evaluate import evaluate_policy_fn


def train_dqn(
    total_timesteps=100_000,
    learning_rate=1e-3,
    gamma=0.99,
    buffer_size=50_000,
    batch_size=64,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    target_update_interval=1000,
    seed=0,
    log_dir="logs/dqn",
    model_path="models/dqn/dqn_clinic.zip",
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    env = Monitor(ClinicRestockEnv(episode_length=52, seed=seed), filename=os.path.join(log_dir, "monitor.csv"))

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        seed=seed,
        verbose=1,
    )
    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)

    eval_env = ClinicRestockEnv(episode_length=52, seed=seed + 500)
    result = evaluate_policy_fn(
        lambda obs: model.predict(obs, deterministic=True)[0], eval_env, n_episodes=10
    )
    print("DQN eval result:", result)
    return model, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_dqn(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
