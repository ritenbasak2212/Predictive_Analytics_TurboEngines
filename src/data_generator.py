"""Synthetic turbofan-fleet simulator.

Produces data in the same shape as NASA's C-MAPSS benchmark (unit, cycle, 3
operating settings, 14 sensors) so learners can later swap in the real dataset.

Physics of the simulation
-------------------------
* Each engine has a random life T (cycles until failure) and a random amount of
  initial wear.
* Hidden degradation d(t) grows exponentially from initial wear to 1.0 at failure.
* Every cycle the engine flies in one of 6 operating regimes. The regime shifts
  sensor levels far more than degradation does, which is why regime-aware
  normalisation is a key feature-engineering step.
* 10 sensors respond to degradation (with different signs/strengths); 4 are pure
  noise, so feature selection matters.

Outputs (in data/)
------------------
train.csv        run-to-failure histories
test.csv         histories truncated at a random point
test_rul.csv     true remaining life at the truncation point (answer key)
drift_batch.csv  later batch with sensor calibration drift + different regime mix
sim_fleet.csv    full histories used only for the maintenance-policy simulation
"""
import numpy as np
import pandas as pd

from config import (DATA_DIR, N_DRIFT_ENGINES, N_REGIMES, N_SIM_ENGINES,
                    N_TEST_ENGINES, N_TRAIN_ENGINES, OP_SETTINGS, SEED, SENSORS)

# Operating regimes: (altitude in kft, Mach, throttle resolver angle)
REGIMES = np.array([[0, 0.00, 100], [10, 0.25, 100], [20, 0.70, 100],
                     [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]], dtype=float)


def make_physics(seed: int = 7) -> dict:
    """Fixed sensor physics shared by every fleet (train, test, drift...)."""
    rng = np.random.default_rng(seed)
    n = len(SENSORS)
    base = rng.uniform(50, 1500, n)                         # nominal sensor level
    regime_effect = rng.normal(0, 1, (N_REGIMES, n)) * base[None, :] * 0.08
    sens = rng.uniform(0.010, 0.035, n) * base * rng.choice([-1, 1], n)
    sens[[0, 4, 9, 13]] = 0.0                               # 4 uninformative sensors
    noise = base * rng.uniform(0.006, 0.014, n)
    return {"base": base, "regime_effect": regime_effect, "sens": sens, "noise": noise}


def simulate_engine(unit, rng, phys, truncate=False, drift=None, regime_p=None):
    """Simulate one engine's history. Returns (DataFrame, true RUL at the last row)."""
    life = int(rng.integers(150, 360))
    w0 = rng.uniform(0.0, 0.12)                       # initial wear (manufacturing variation)
    b = rng.uniform(3.0, 6.0)                         # how sharply degradation accelerates
    t = np.arange(1, life + 1)
    d = w0 + (1 - w0) * (np.exp(b * t / life) - 1) / (np.exp(b) - 1)

    regime = rng.choice(N_REGIMES, size=life, p=regime_p)
    ops = REGIMES[regime] + rng.normal(0, [0.05, 0.002, 0.05], (life, 3))

    unit_bias = rng.normal(0, 0.4, len(SENSORS)) * phys["noise"]   # engine-to-engine variation
    sensors = (phys["base"][None, :] + phys["regime_effect"][regime] + unit_bias[None, :]
               + phys["sens"][None, :] * d[:, None] * 2.5
               + rng.normal(0, 1, (life, len(SENSORS))) * phys["noise"][None, :])
    if drift is not None:                             # calibration offset on some sensors
        sensors = sensors + drift[None, :]

    df = pd.DataFrame(sensors, columns=SENSORS)
    df.insert(0, "cycle", t)
    df.insert(0, "unit", unit)
    for i, c in enumerate(OP_SETTINGS):
        df.insert(2 + i, c, ops[:, i])

    rul_at_end = 0
    if truncate:                                      # cut history at 30-95% of life
        cut = int(life * rng.uniform(0.30, 0.95))
        rul_at_end = life - cut
        df = df.iloc[:cut]
    return df, rul_at_end


def simulate_fleet(n, rng, phys, start_id=1, **kw):
    frames, ruls = [], []
    for u in range(start_id, start_id + n):
        df, rul = simulate_engine(u, rng, phys, **kw)
        frames.append(df)
        ruls.append({"unit": u, "true_rul": rul})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(ruls)


def main():
    rng = np.random.default_rng(SEED)
    phys = make_physics()

    train, _ = simulate_fleet(N_TRAIN_ENGINES, rng, phys)
    test, test_rul = simulate_fleet(N_TEST_ENGINES, rng, phys, truncate=True)

    # Drift batch: 3 sensors drift by ~4% of their level (sensor recalibration) and the fleet flies more high-altitude routes
    drift = np.zeros(len(SENSORS))
    drift[[5, 8, 12]] = phys["base"][[5, 8, 12]] * 0.03 * np.sign(phys["sens"][[5, 8, 12]]) * -1   # s6, s9, s13 read "healthier" than reality
    drift_p = np.array([0.05, 0.05, 0.10, 0.10, 0.35, 0.35])
    drift_df, drift_rul = simulate_fleet(N_DRIFT_ENGINES, rng, phys, truncate=True,
                                         drift=drift, regime_p=drift_p)
    sim, _ = simulate_fleet(N_SIM_ENGINES, rng, phys)

    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    test_rul.to_csv(DATA_DIR / "test_rul.csv", index=False)
    drift_df.to_csv(DATA_DIR / "drift_batch.csv", index=False)
    drift_rul.to_csv(DATA_DIR / "drift_rul.csv", index=False)
    sim.to_csv(DATA_DIR / "sim_fleet.csv", index=False)
    print(f"train: {train.shape}, engines={train.unit.nunique()}, "
          f"mean life={train.groupby('unit').cycle.max().mean():.0f} cycles")
    print(f"test: {test.shape}, drift batch: {drift_df.shape}, sim fleet: {sim.shape}")


if __name__ == "__main__":
    main()
