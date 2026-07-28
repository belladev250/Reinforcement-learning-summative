"""
Main entry point. Running `uv run main.py` with no arguments trains all
four algorithms (DQN, REINFORCE, PPO, A2C) on ClinicRestockEnv with sane
default hyperparameters, evaluates each, and prints a comparison table.

For the full hyperparameter sweep (required for the report tables), use
training/hyperparam_sweep.py instead -- that's a separate, longer-running
step. This script is for getting one solid model per algorithm quickly.

Usage:
    uv run main.py                     # train all 4, default timesteps
    uv run main.py --timesteps 200000  # longer training for final models
    uv run main.py --algo ppo          # train just one
"""

import argparse

from training.dqn_training import train_dqn
from training.pg_training import train_ppo, train_a2c, train_reinforce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c", "all"], default="all")
    args = parser.parse_args()

    results = {}

    if args.algo in ("dqn", "all"):
        print("\n" + "=" * 60 + "\nTraining DQN\n" + "=" * 60)
        _, results["DQN"] = train_dqn(total_timesteps=args.timesteps)

    if args.algo in ("reinforce", "all"):
        print("\n" + "=" * 60 + "\nTraining REINFORCE\n" + "=" * 60)
        _, results["REINFORCE"] = train_reinforce(total_timesteps=args.timesteps)

    if args.algo in ("ppo", "all"):
        print("\n" + "=" * 60 + "\nTraining PPO\n" + "=" * 60)
        _, results["PPO"] = train_ppo(total_timesteps=args.timesteps)

    if args.algo in ("a2c", "all"):
        print("\n" + "=" * 60 + "\nTraining A2C\n" + "=" * 60)
        _, results["A2C"] = train_a2c(total_timesteps=args.timesteps)

    print("\n" + "=" * 60 + "\nFINAL COMPARISON\n" + "=" * 60)
    for name, r in results.items():
        print(f"{name:12s}  mean_reward={r['mean_reward']:9.2f}  "
              f"std_reward={r['std_reward']:7.2f}  mean_unmet_demand={r['mean_unmet_demand']:8.2f}")

    print("\nModels saved under models/. Next steps:")
    print("  1. uv run training/plotting.py            -> generates plots for the report")
    print("  2. uv run training/generalization_test.py  -> generalization.png")
    print("  3. uv run play.py --algo <best_algo>       -> for your demo video")


if __name__ == "__main__":
    main()
