"""Train REINFORCE, PPO, or A2C on ClinicRestockEnv.

Usage:
    uv run training/pg_training.py --algo ppo --timesteps 100000
    uv run training/pg_training.py --algo a2c --timesteps 100000
    uv run training/pg_training.py --algo reinforce --timesteps 100000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from environment.custom_env import ClinicRestockEnv
from training.evaluate import evaluate_policy_fn
from training.reinforce import REINFORCEAgent


def train_ppo(total_timesteps=100_000, learning_rate=3e-4, gamma=0.99, n_steps=256,
              ent_coef=0.01, clip_range=0.2, seed=0,
              log_dir="logs/ppo", model_path="models/pg/ppo_clinic.zip"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    env = Monitor(ClinicRestockEnv(episode_length=52, seed=seed), filename=os.path.join(log_dir, "monitor.csv"))
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps,
                ent_coef=ent_coef, clip_range=clip_range, seed=seed, verbose=1)
    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)

    eval_env = ClinicRestockEnv(episode_length=52, seed=seed + 500)
    result = evaluate_policy_fn(lambda obs: model.predict(obs, deterministic=True)[0], eval_env, n_episodes=10)
    print("PPO eval result:", result)
    return model, result


def train_a2c(total_timesteps=100_000, learning_rate=7e-4, gamma=0.99, n_steps=5,
              ent_coef=0.0, seed=0,
              log_dir="logs/a2c", model_path="models/pg/a2c_clinic.zip"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    env = Monitor(ClinicRestockEnv(episode_length=52, seed=seed), filename=os.path.join(log_dir, "monitor.csv"))
    model = A2C("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps,
                ent_coef=ent_coef, seed=seed, verbose=1)
    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)

    eval_env = ClinicRestockEnv(episode_length=52, seed=seed + 500)
    result = evaluate_policy_fn(lambda obs: model.predict(obs, deterministic=True)[0], eval_env, n_episodes=10)
    print("A2C eval result:", result)
    return model, result


def train_reinforce(total_timesteps=100_000, learning_rate=3e-4, gamma=0.99,
                     hidden_size=64, entropy_coef=0.0, seed=0,
                     log_dir="logs/reinforce", model_path="models/pg/reinforce_clinic.pt"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    env = ClinicRestockEnv(episode_length=52, seed=seed)

    agent = REINFORCEAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        hidden_size=hidden_size,
        entropy_coef=entropy_coef,
        seed=seed,
    )
    agent.learn(env, total_timesteps=total_timesteps)
    agent.save(model_path)

    # write reward/entropy history to CSV so it can be plotted like the SB3 logs
    pd.DataFrame({
        "episode": range(len(agent.reward_history)),
        "reward": agent.reward_history,
        "entropy": agent.entropy_history,
    }).to_csv(os.path.join(log_dir, "training_history.csv"), index=False)

    eval_env = ClinicRestockEnv(episode_length=52, seed=seed + 500)
    result = evaluate_policy_fn(lambda obs: agent.predict(obs, deterministic=True)[0], eval_env, n_episodes=10)
    print("REINFORCE eval result:", result)
    return agent, result


ALGO_FN = {"ppo": train_ppo, "a2c": train_a2c, "reinforce": train_reinforce}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c", "reinforce"], required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    kwargs = {"total_timesteps": args.timesteps, "gamma": args.gamma, "seed": args.seed}
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr

    ALGO_FN[args.algo](**kwargs)
