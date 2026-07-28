"""Loads the 4 trained models and runs the generalization comparison plot.
Run this AFTER main.py has trained (or after you've trained all 4 manually).

Usage:
    uv run training/generalization_test.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import ClinicRestockEnv
from training.reinforce import REINFORCEAgent
from training.plotting import plot_generalization


def main():
    loaders = {}

    if os.path.exists("models/dqn/dqn_clinic.zip"):
        dqn = DQN.load("models/dqn/dqn_clinic.zip")
        loaders["DQN"] = lambda obs: dqn.predict(obs, deterministic=True)[0]

    if os.path.exists("models/pg/ppo_clinic.zip"):
        ppo = PPO.load("models/pg/ppo_clinic.zip")
        loaders["PPO"] = lambda obs: ppo.predict(obs, deterministic=True)[0]

    if os.path.exists("models/pg/a2c_clinic.zip"):
        a2c = A2C.load("models/pg/a2c_clinic.zip")
        loaders["A2C"] = lambda obs: a2c.predict(obs, deterministic=True)[0]

    if os.path.exists("models/pg/reinforce_clinic.pt"):
        env = ClinicRestockEnv()
        agent = REINFORCEAgent(env.observation_space.shape[0], env.action_space.n)
        agent.load("models/pg/reinforce_clinic.pt", env.observation_space.shape[0], env.action_space.n)
        loaders["REINFORCE"] = lambda obs: agent.predict(obs, deterministic=True)[0]

    if not loaders:
        print("No trained models found. Train models first (uv run main.py).")
        return

    plot_generalization(loaders)


if __name__ == "__main__":
    main()
