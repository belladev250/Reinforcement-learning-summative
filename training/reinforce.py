"""
Custom REINFORCE (vanilla Monte-Carlo policy gradient) implementation.

Stable-Baselines3 does not ship REINFORCE (only PPO/A2C/DQN/SAC/TD3/DDPG),
so it's implemented directly in PyTorch here, following the same
`.predict(obs)` / `.save(path)` interface style as SB3 models so the rest
of the training + evaluation pipeline can treat all four algorithms
uniformly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class REINFORCEAgent:
    def __init__(self, obs_dim, n_actions, learning_rate=3e-4, gamma=0.99,
                 hidden_size=64, entropy_coef=0.0, seed=0):
        torch.manual_seed(seed)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.policy = PolicyNet(obs_dim, n_actions, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reward_history = []  # mean episodic reward per training episode
        self.entropy_history = []  # mean policy entropy per training episode

    def _select_action(self, obs, greedy=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.policy(obs_t)
        dist = Categorical(logits=logits)
        if greedy:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def predict(self, obs, deterministic=True):
        action, _, _ = self._select_action(obs, greedy=deterministic)
        return action, None

    def learn(self, env, total_timesteps: int, log_every: int = 20):
        steps_done = 0
        episode = 0
        while steps_done < total_timesteps:
            obs, _ = env.reset()
            log_probs, rewards, entropies = [], [], []
            done = False
            while not done:
                action, log_prob, entropy = self._select_action(obs, greedy=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                steps_done += 1

            # discounted returns, normalized for stability
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            if returns.std() > 1e-6:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)
            loss = -(log_probs_t * returns).sum() - self.entropy_coef * entropies_t.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ep_reward = float(np.sum(rewards))
            ep_entropy = float(entropies_t.mean().item())
            self.reward_history.append(ep_reward)
            self.entropy_history.append(ep_entropy)
            episode += 1
            if episode % log_every == 0:
                recent = np.mean(self.reward_history[-log_every:])
                print(f"[REINFORCE] episode {episode} steps {steps_done}/{total_timesteps} "
                      f"mean_reward(last {log_every})={recent:.2f} entropy={ep_entropy:.3f}")
        return self

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path, obs_dim, n_actions, hidden_size=64):
        self.policy = PolicyNet(obs_dim, n_actions, hidden_size)
        self.policy.load_state_dict(torch.load(path))
        return self
