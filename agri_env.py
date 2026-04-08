"""
AgriEnv — A Gymnasium-compatible Reinforcement Learning Environment
for Precision Agriculture in Greenhouse Systems.

The agent controls irrigation, nutrient injection, CO2, and pesticide
usage to maximise crop growth while minimising operational costs.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────
#  Constants & Configuration
# ──────────────────────────────────────────────

MAX_STEPS        = 140          # Standard tomato growth cycle from seedling to final harvest
ALPHA            = 0.5          # Weighting factor to balance yield maximization with energy conservation

# Observation bounds  [soil_moist, N, P, K, temp, humidity, pest, energy_price]
# Temp range reflects actual sensor data from 7.5°C to 45.2°C
# Energy price reflects 2025 industrial/commercial tariffs (approx. 1–17 units)
OBS_LOW  = np.array([0.0, 0.0, 0.0, 0.0, 7.5,  0.0, 0.0, 1.0], dtype=np.float32)
OBS_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 45.0, 100.0, 1.0, 17.0], dtype=np.float32)

# Action bounds  [irrigation, N_inj, P_inj, K_inj, CO2, pesticide]
# Irrigation: Mature plants require up to 2,000–3,000 mL/day
# CO2: Enrichment is effective up to 1,200 ppm; higher levels can be phytotoxic
ACT_LOW  = np.array([0.0, 0.0, 0.0, 0.0, 300.0, 0.0], dtype=np.float32)
ACT_HIGH = np.array([3000.0, 0.5, 0.5, 0.5, 1200.0, 1.0], dtype=np.float32)

# Optimal moisture for crop growth (target 70% of field capacity)
OPTIMAL_MOISTURE = 0.7


# ──────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────

def _moisture_score(soil_moisture: float) -> float:
    """Gaussian-shaped score centred on the optimal moisture level."""
    return np.exp(-((soil_moisture - OPTIMAL_MOISTURE) ** 2) / 0.05)


def _nutrient_score(nitrogen: float, phosphorus: float, potassium: float) -> float:
    """Average nutrient availability — higher is better, capped at 1."""
    return np.clip((nitrogen + phosphorus + potassium) / 3.0, 0.0, 1.0)


def _compute_growth(soil_moisture: float, nitrogen: float,
                    phosphorus: float, potassium: float,
                    pest_density: float) -> float:
    """
    Crop growth for this timestep.

    growth = moisture_score × nutrient_score × (1 − pest_density)

    All components are in [0, 1], so growth ∈ [0, 1].
    """
    m_score = _moisture_score(soil_moisture)
    n_score = _nutrient_score(nitrogen, phosphorus, potassium)
    return m_score * n_score * (1.0 - pest_density)


def _compute_cost(irrigation: float, co2_level: float,
                  energy_price: float) -> float:
    """
    Operational cost for this timestep.

    • Water cost  — proportional to irrigation volume (normalised by max 500)
    • Energy cost — proportional to CO2 enrichment above ambient (300 ppm)
                    scaled by current energy price
    """
    water_cost  = irrigation / 500.0                          # normalise to [0, 1]
    co2_above_ambient = max(0.0, co2_level - 300.0)
    energy_cost = (co2_above_ambient / 500.0) * energy_price
    return water_cost + energy_cost


# ──────────────────────────────────────────────
#  AgriEnv
# ──────────────────────────────────────────────

class AgriEnv(gym.Env):
    """
    Precision-agriculture greenhouse environment.

    Observation (8-dim continuous):
        soil_moisture, nitrogen, phosphorus, potassium,
        temperature, humidity, pest_density, energy_price

    Action (6-dim continuous):
        irrigation, nitrogen_injection, phosphorus_injection,
        potassium_injection, co2_level, pesticide_usage
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space      = spaces.Box(low=ACT_LOW, high=ACT_HIGH, dtype=np.float32)

        # Internal bookkeeping
        self.state: np.ndarray | None = None
        self.current_step: int = 0
        self.cumulative_yield: float = 0.0   # bonus: tracks total growth across episode

    # ── Reset ────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Sample a plausible initial state (not extreme)
        self.state = np.array([
            self.np_random.uniform(0.3, 0.7),    # soil_moisture
            self.np_random.uniform(0.2, 0.5),    # nitrogen
            self.np_random.uniform(0.2, 0.5),    # phosphorus
            self.np_random.uniform(0.2, 0.5),    # potassium
            self.np_random.uniform(20.0, 30.0),  # temperature (°C)
            self.np_random.uniform(40.0, 70.0),  # humidity (%)
            self.np_random.uniform(0.0, 0.2),    # pest_density
            self.np_random.uniform(3.0, 7.0),    # energy_price
        ], dtype=np.float32)

        self.current_step    = 0
        self.cumulative_yield = 0.0

        info = {"cumulative_yield": self.cumulative_yield}
        return self.state.copy(), info

    # ── Step ─────────────────────────────────
    def step(self, action: np.ndarray):
        """
        Advance the environment by one timestep.

        Args:
            action (np.ndarray): 6-dim vector
                [irrigation, n_inject, p_inject, k_inject, co2_level, pesticide]

        Returns:
            obs      (np.ndarray): Updated 8-dim observation.
            reward   (float):      growth − α·cost, with shaping penalties.
            terminated (bool):     True when MAX_STEPS reached.
            truncated  (bool):     Always False (no early truncation).
            info     (dict):       Diagnostic data (growth, cost, action, etc.).

        Transition logic:
            • Soil moisture increases with irrigation, decreases via
              temperature-dependent evaporation.
            • Nutrients (N, P, K) increase with injection and decay each step.
            • Pest density rises when humidity > 70 and falls with pesticide.
            • Temperature, humidity, and energy price undergo stochastic drift.
            • All state values are clipped to their valid bounds after update.
        """
        assert self.state is not None, "Call reset() before step()."

        # Clip action to valid range (safety net)
        action = np.clip(action, ACT_LOW, ACT_HIGH)

        # Unpack current state
        soil_moisture, nitrogen, phosphorus, potassium, \
            temperature, humidity, pest_density, energy_price = self.state

        # Unpack action
        irrigation   = action[0]
        n_inject     = action[1]
        p_inject     = action[2]
        k_inject     = action[3]
        co2_level    = action[4]
        pesticide    = action[5]

        # ── State transitions ──────────────
        # Soil moisture: irrigation adds, evaporation removes
        evaporation_rate = 0.02 + 0.001 * (temperature - 20.0)  # hotter → more evaporation
        soil_moisture += (irrigation / 500.0) * 0.3 - evaporation_rate
        # Add small stochastic noise
        soil_moisture += self.np_random.normal(0.0, 0.01)

        # Nutrients: injection adds, natural decay removes
        nutrient_decay = 0.02
        nitrogen   += n_inject * 0.4 - nutrient_decay + self.np_random.normal(0.0, 0.005)
        phosphorus += p_inject * 0.4 - nutrient_decay + self.np_random.normal(0.0, 0.005)
        potassium  += k_inject * 0.4 - nutrient_decay + self.np_random.normal(0.0, 0.005)

        # Pest dynamics: increase when humid, decrease with pesticide
        pest_growth = 0.02 if humidity > 70.0 else 0.0
        pest_density += pest_growth - pesticide * 0.15
        pest_density += self.np_random.normal(0.0, 0.005)

        # Weather stochasticity (bonus)
        temperature += self.np_random.normal(0.0, 0.5)
        humidity    += self.np_random.normal(0.0, 1.0)

        # Energy price: random fluctuation
        energy_price += self.np_random.uniform(-0.5, 0.5)

        # ── Clip everything to valid bounds ──
        soil_moisture = np.clip(soil_moisture, 0.0, 1.0)
        nitrogen      = np.clip(nitrogen,      0.0, 1.0)
        phosphorus    = np.clip(phosphorus,    0.0, 1.0)
        potassium     = np.clip(potassium,     0.0, 1.0)
        temperature   = np.clip(temperature,  10.0, 50.0)
        humidity      = np.clip(humidity,       0.0, 100.0)
        pest_density  = np.clip(pest_density,  0.0, 1.0)
        energy_price  = np.clip(energy_price,  1.0, 10.0)

        # Write back state
        self.state = np.array([
            soil_moisture, nitrogen, phosphorus, potassium,
            temperature, humidity, pest_density, energy_price
        ], dtype=np.float32)

        # ── Growth & Cost ──────────────────
        growth = _compute_growth(soil_moisture, nitrogen, phosphorus,
                                 potassium, pest_density)
        cost   = _compute_cost(irrigation, co2_level, energy_price)

        # ── Reward ─────────────────────────
        reward = float(growth - ALPHA * cost)

        # Reward shaping: penalise excessive resource usage
        if irrigation > 400.0:
            reward -= 0.05
        if pesticide > 0.8:
            reward -= 0.03

        # ── Bookkeeping ───────────────────
        self.cumulative_yield += growth
        self.current_step += 1

        terminated = self.current_step >= MAX_STEPS
        truncated  = False

        info = {
            "growth":           growth,
            "cost":             cost,
            "cumulative_yield": self.cumulative_yield,
            "step":             self.current_step,
            "action":           action.copy(),
        }

        return self.state.copy(), reward, terminated, truncated, info

    # ── Render (minimal) ─────────────────────
    def render(self):
        if self.render_mode == "human" and self.state is not None:
            labels = [
                "soil_moisture", "nitrogen", "phosphorus", "potassium",
                "temperature", "humidity", "pest_density", "energy_price",
            ]
            print(f"Step {self.current_step:>3d} | "
                  + " | ".join(f"{l}: {v:.2f}" for l, v in zip(labels, self.state))
                  + f" | yield: {self.cumulative_yield:.2f}")


# ──────────────────────────────────────────────
#  Example Usage — Random Agent Loop
# ──────────────────────────────────────────────

if __name__ == "__main__":
    env = AgriEnv(render_mode="human")
    obs, info = env.reset(seed=42)

    total_reward = 0.0

    for step in range(MAX_STEPS):
        action = env.action_space.sample()       # random policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print every 50 steps to keep output readable
        if (step + 1) % 50 == 0:
            env.render()

        if terminated or truncated:
            break

    print(f"\n{'='*60}")
    print(f"Episode finished after {info['step']} steps")
    print(f"  Total reward      : {total_reward:.4f}")
    print(f"  Cumulative yield  : {info['cumulative_yield']:.4f}")
    print(f"{'='*60}")
