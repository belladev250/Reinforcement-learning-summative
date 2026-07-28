"""
Loads your BEST-performing trained agent and runs it in the rendered
environment -- this is the script to run on camera for the demo video.

Usage:
    uv run play.py --algo ppo
    uv run play.py --algo ppo --seconds-per-week 4   # slower, for a longer video
"""

import argparse
import os
import time

from stable_baselines3 import DQN, PPO, A2C

from environment.custom_env import ClinicRestockEnv
from training.reinforce import REINFORCEAgent

MODEL_PATHS = {
    "dqn": "models/dqn/dqn_clinic.zip",
    "ppo": "models/pg/ppo_clinic.zip",
    "a2c": "models/pg/a2c_clinic.zip",
    "reinforce": "models/pg/reinforce_clinic.pt",
}


def load_agent(algo):
    path = MODEL_PATHS[algo]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model at {path}. Train it first, e.g.\n"
            f"  uv run training/dqn_training.py        (for dqn)\n"
            f"  uv run training/pg_training.py --algo {algo}"
        )
    if algo == "dqn":
        model = DQN.load(path)
        return lambda obs: model.predict(obs, deterministic=True)[0]
    if algo == "ppo":
        model = PPO.load(path)
        return lambda obs: model.predict(obs, deterministic=True)[0]
    if algo == "a2c":
        model = A2C.load(path)
        return lambda obs: model.predict(obs, deterministic=True)[0]
    if algo == "reinforce":
        env = ClinicRestockEnv()
        agent = REINFORCEAgent(env.observation_space.shape[0], env.action_space.n)
        agent.load(path, env.observation_space.shape[0], env.action_space.n)
        return lambda obs: agent.predict(obs, deterministic=True)[0]
    raise ValueError(algo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(MODEL_PATHS.keys()), default="ppo")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--seconds-per-week", type=float, default=4.0,
        help="Pause after each simulated week, in seconds. 52 weeks x 4s = ~3.5 "
             "minutes per episode, which fills the required 3/4-of-video execution time.",
    )
    parser.add_argument(
        "--hold-at-end", type=float, default=15.0,
        help="Seconds to keep the final frame on screen after the episode ends "
             "(so the window doesn't vanish mid-recording).",
    )
    args = parser.parse_args()

    predict_fn = load_agent(args.algo)
    env = ClinicRestockEnv(render_mode="human", episode_length=52)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        print(f"\n=== Episode {ep + 1} ({args.algo.upper()}) ===")
        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(
                f"week={info['week']:>2}  action={int(action)}  reward={reward:7.2f}  "
                f"unmet_demand={info['unmet_demand']:6.1f}  stock_total={info['stock_total']:6.1f}"
            )
            done = terminated or truncated
            time.sleep(args.seconds_per_week)
        print(f"Episode {ep + 1} total reward: {total_reward:.2f}")

    print(f"\nEpisode(s) finished. Holding final frame for {args.hold_at_end:.0f}s...")
    time.sleep(args.hold_at_end)
    env.close()


if __name__ == "__main__":
    main()