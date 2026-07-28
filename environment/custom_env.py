"""
ClinicRestockEnv: a custom Gymnasium environment simulating weekly medicine
restocking decisions at a rural health clinic in Rwanda.

The agent decides, once per week, how aggressively to reorder a basket of
essential medicines (antimalarials, antibiotics, ORS) for a single clinic.
Demand is stochastic, seasonal (malaria season spikes antimalarial demand),
and orders arrive after a lead time -- so the agent must anticipate need
rather than react to it, which is what makes this a genuine sequential
decision problem rather than a lookup table.

Action space: Discrete(5)
    0 -> order nothing this week
    1 -> order at 0.5x the estimated reorder quantity
    2 -> order at 1.0x the estimated reorder quantity
    3 -> order at 1.5x the estimated reorder quantity
    4 -> order at 2.0x the estimated reorder quantity
    (Discrete so the SAME environment can be used by DQN, REINFORCE, PPO,
    and A2C for a fair, objective comparison.)

Observation space: Box(10,) float32, all normalized to roughly [0, 1]
    [0:3]  current stock level per drug (normalized by max capacity)
    [3]    days since the last delivery arrived (normalized)
    [4]    whether an order is currently in transit (0/1)
    [5]    weeks remaining until an in-transit order arrives (normalized)
    [6]    sin(2*pi*week/52)  -- cyclic encoding of week-of-year (season)
    [7]    cos(2*pi*week/52)
    [8]    malaria season flag (1 during Mar-May & Oct-Nov, else 0)
    [9]    current patient load multiplier (normalized)

Reward:
    - stockout_penalty  * unmet_demand (summed across drugs, per unit)
    - holding_cost      * total stock held (encourages not over-ordering)
    - order_cost        * 1 if an order was placed this week (flat cost)

Terminal condition: episode ends after `episode_length` weeks (default 52,
i.e. one simulated year). No early termination -- clinics don't "die".
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


DRUG_NAMES = ["Antimalarial (ACT)", "Amoxicillin", "ORS"]
N_DRUGS = len(DRUG_NAMES)

# Base weekly demand (units) per drug, before seasonal/noise adjustment
BASE_DEMAND = np.array([40.0, 25.0, 20.0])

# Max on-shelf capacity per drug (storage constraint at a small clinic)
MAX_CAPACITY = np.array([300.0, 200.0, 150.0])

ORDER_MULTIPLIERS = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

LEAD_TIME_WEEKS = 2
HOLDING_COST_COEF = 0.01
STOCKOUT_PENALTY_COEF = 2.0
ORDER_FLAT_COST = 1.0


def _malaria_season_flag(week_of_year: int) -> float:
    """Rwanda has two malaria peaks: Mar-May and Oct-Nov (roughly weeks
    9-22 and weeks 40-48 of the year)."""
    w = week_of_year % 52
    return 1.0 if (9 <= w <= 22 or 40 <= w <= 48) else 0.0


class ClinicRestockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, episode_length: int = 52, seed: int | None = None):
        super().__init__()
        self.episode_length = episode_length
        self.render_mode = render_mode
        self._np_random = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(len(ORDER_MULTIPLIERS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3 + 1 + 1 + 1 + 2 + 1 + 1,), dtype=np.float32
        )

        self._renderer = None  # lazily created (pygame) only if render() is called

        # episode state, set in reset()
        self.week = 0
        self.stock = None
        self.pending_orders = []  # list of (arrival_week, amounts[N_DRUGS])
        self.weeks_since_delivery = 0
        self.patient_load = 1.0
        self.total_stockouts = 0.0
        self.total_units_ordered = 0.0
        self.history = []  # for plotting/rendering after an episode

    def _get_obs(self):
        stock_norm = np.clip(self.stock / MAX_CAPACITY, 0.0, 1.0)
        in_transit = 1.0 if len(self.pending_orders) > 0 else 0.0
        weeks_until_arrival = 0.0
        if self.pending_orders:
            next_arrival = min(a for a, _ in self.pending_orders)
            weeks_until_arrival = np.clip((next_arrival - self.week) / LEAD_TIME_WEEKS, 0, 1)
        week_norm_sin = (np.sin(2 * np.pi * (self.week % 52) / 52) + 1) / 2
        week_norm_cos = (np.cos(2 * np.pi * (self.week % 52) / 52) + 1) / 2
        season_flag = _malaria_season_flag(self.week)
        obs = np.array(
            [
                *stock_norm,
                np.clip(self.weeks_since_delivery / 8.0, 0, 1),
                in_transit,
                weeks_until_arrival,
                week_norm_sin,
                week_norm_cos,
                season_flag,
                np.clip(self.patient_load / 2.0, 0, 1),
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.week = 0
        self.stock = MAX_CAPACITY * 0.5
        self.pending_orders = []
        self.weeks_since_delivery = 0
        self.patient_load = 1.0
        self.total_stockouts = 0.0
        self.total_units_ordered = 0.0
        self.history = []

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action {action}"
        multiplier = ORDER_MULTIPLIERS[action]

        season = _malaria_season_flag(self.week)
        seasonal_boost = np.array([1.0 + 0.8 * season, 1.0 + 0.15 * season, 1.0 + 0.2 * season])

        # place an order this week if multiplier > 0
        order_cost = 0.0
        if multiplier > 0:
            reorder_qty = multiplier * BASE_DEMAND * seasonal_boost * LEAD_TIME_WEEKS
            self.pending_orders.append((self.week + LEAD_TIME_WEEKS, reorder_qty))
            self.total_units_ordered += reorder_qty.sum()
            order_cost = ORDER_FLAT_COST

        # deliver any orders arriving this week
        delivered = np.zeros(N_DRUGS)
        still_pending = []
        for arrival_week, amounts in self.pending_orders:
            if arrival_week <= self.week:
                delivered += amounts
                self.weeks_since_delivery = 0
            else:
                still_pending.append((arrival_week, amounts))
        self.pending_orders = still_pending
        self.stock = np.clip(self.stock + delivered, 0, MAX_CAPACITY)
        if delivered.sum() == 0:
            self.weeks_since_delivery += 1

        # patient load drifts slowly (busier clinics -> more demand)
        self.patient_load = float(
            np.clip(self.patient_load + self._np_random.normal(0, 0.05), 0.7, 1.6)
        )

        # stochastic weekly demand
        noise = self._np_random.normal(1.0, 0.15, size=N_DRUGS)
        demand = BASE_DEMAND * seasonal_boost * self.patient_load * np.clip(noise, 0.4, 2.0)

        consumption = np.minimum(self.stock, demand)
        unmet = np.maximum(0.0, demand - self.stock)
        self.stock = np.clip(self.stock - consumption, 0, MAX_CAPACITY)
        self.total_stockouts += unmet.sum()

        holding_cost = HOLDING_COST_COEF * self.stock.sum()
        stockout_cost = STOCKOUT_PENALTY_COEF * unmet.sum()
        reward = -(stockout_cost + holding_cost + order_cost)

        self.history.append(
            {
                "week": self.week,
                "stock": self.stock.copy(),
                "demand": demand.copy(),
                "unmet": unmet.copy(),
                "action": int(action),
                "reward": reward,
            }
        )

        self.week += 1
        terminated = False
        truncated = self.week >= self.episode_length

        obs = self._get_obs()
        info = {
            "unmet_demand": float(unmet.sum()),
            "stock_total": float(self.stock.sum()),
            "week": self.week,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return
        from environment.rendering import ClinicRenderer

        if self._renderer is None:
            self._renderer = ClinicRenderer(DRUG_NAMES, MAX_CAPACITY, self.render_mode)
        return self._renderer.render(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
