"""Shared evaluation utility so DQN, REINFORCE, PPO and A2C are all scored
the same way -- mean episodic reward over N held-out episodes."""

import numpy as np


def evaluate_policy_fn(predict_fn, env, n_episodes: int = 10, seed_offset: int = 1000):
    """predict_fn(obs) -> action (int). Works for SB3 models (model.predict)
    and for the custom REINFORCE policy alike."""
    episode_rewards = []
    episode_unmet = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        total_reward = 0.0
        total_unmet = 0.0
        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_unmet += info.get("unmet_demand", 0.0)
            done = terminated or truncated
        episode_rewards.append(total_reward)
        episode_unmet.append(total_unmet)
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_unmet_demand": float(np.mean(episode_unmet)),
    }
