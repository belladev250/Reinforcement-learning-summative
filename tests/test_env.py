"""Basic sanity tests for ClinicRestockEnv. Run with: uv run pytest"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from environment.custom_env import ClinicRestockEnv


def test_reset_returns_valid_obs():
    env = ClinicRestockEnv(seed=0)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_returns_valid_types():
    env = ClinicRestockEnv(seed=0)
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_ends_at_episode_length():
    env = ClinicRestockEnv(episode_length=10, seed=0)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert steps == 10


def test_stock_never_exceeds_capacity():
    env = ClinicRestockEnv(seed=1)
    env.reset()
    for _ in range(52):
        env.step(4)  # always order max amount, should still clip at capacity
    from environment.custom_env import MAX_CAPACITY
    assert np.all(env.stock <= MAX_CAPACITY + 1e-6)


def test_action_space_is_discrete_5():
    env = ClinicRestockEnv()
    assert env.action_space.n == 5


if __name__ == "__main__":
    test_reset_returns_valid_obs()
    test_step_returns_valid_types()
    test_episode_ends_at_episode_length()
    test_stock_never_exceeds_capacity()
    test_action_space_is_discrete_5()
    print("All tests passed.")
