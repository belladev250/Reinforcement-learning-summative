# Rural Clinic Medicine Restocking Agent — RL Summative

Trains and compares four reinforcement learning algorithms (DQN, REINFORCE,
PPO, A2C) on a custom Gymnasium environment simulating weekly medicine
restocking decisions at a rural health clinic in Rwanda, based on real
patterns in Rwanda's HMIS medicine-supply challenge (seasonal malaria
demand, stochastic patient load, order lead times).

## Setup (uv-only, no manual installs)

```bash
uv sync
```

> Note: `torch` is a large dependency (~2-5GB depending on platform/GPU
> wheel). Make sure you have a few GB of free disk space and a stable
> connection the first time you run `uv sync`.

## Quick start

```bash
# 1. Train all four algorithms with default hyperparameters
uv run main.py --timesteps 100000

# 2. Generate the report plots (reward curves, DQN loss, PG entropy, convergence)
uv run training/plotting.py

# 3. Generate the generalization-test plot 
uv run training/generalization_test.py

# 4. Run your best agent with the rendered GUI
uv run play.py --algo ppo
```

## Full hyperparameter sweep (required for the report's 4 tables, 10 runs each)

```bash
uv run training/hyperparam_sweep.py --timesteps 20000
```

Produces `logs/sweeps/{dqn,reinforce,ppo,a2c}_sweep.csv` — paste these
straight into the report's Implementation tables.

## Project structure

```
environment/custom_env.py   # ClinicRestockEnv (Gymnasium API)
environment/rendering.py    # pygame visualization
training/dqn_training.py    # DQN (Stable-Baselines3)
training/pg_training.py     # PPO, A2C (Stable-Baselines3) + REINFORCE wrapper
training/reinforce.py       # custom REINFORCE (not in SB3)
training/hyperparam_sweep.py
training/plotting.py
training/generalization_test.py
main.py                     # trains all 4 algos end-to-end
play.py                     # renders best agent for the demo video
tests/test_env.py
```

## Environment summary

- **Action space:** `Discrete(5)` — order nothing / 0.5x / 1.0x / 1.5x / 2.0x
  the estimated reorder quantity (kept discrete so DQN can be compared
  fairly against the policy-gradient methods on the identical environment).
- **Observation space:** `Box(10,)` — normalized stock per drug (3),
  weeks since last delivery, in-transit flag, weeks until delivery,
  cyclic week-of-year encoding (2), malaria-season flag, patient load.
- **Reward:** `-(stockout_penalty * unmet_demand) - (holding_cost * stock) - order_cost`
- **Episode:** 52 simulated weeks (1 year), no early termination.

See `environment/custom_env.py` docstring for full details.
