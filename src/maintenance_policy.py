"""Business impact: simulate maintenance policies on a fleet the model has never seen.

Policies compared (cost per engine, INR)
  1. Reactive        - run every engine to failure.
  2. Fixed interval  - shop visit at a fixed cycle K (K optimised on TRAINING lifetimes).
  3. Predictive      - trigger a shop visit when predicted RUL <= lead time + safety margin.
       3a uses the LightGBM point prediction
       3b uses the conservative lower bound of the 80% interval (q10)
  4. Oracle          - perfect knowledge (planned visit exactly at end of life): lower bound.

Mechanics of the predictive policy: at cycle c the model triggers; the visit
happens MAINT_LEAD_TIME cycles later. If the engine fails before the visit, it
is an unplanned failure; otherwise it is a planned visit and the unused life is
charged at COST_PER_LOST_CYCLE.

To stay honest, the simulation fleet is split in half: safety margins are tuned on
the first half and all reported costs come from the second half.
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

from config import (COST_PER_LOST_CYCLE, COST_PLANNED_MAINT, COST_UNPLANNED_FAILURE, DATA_DIR,
                    FIG_DIR, MAINT_LEAD_TIME, MODEL_DIR, OUT_DIR)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def visit_cost(life, visit_cycle):
    """Cost of one engine given its true life and the cycle of the planned visit."""
    if visit_cycle is None or visit_cycle >= life:
        return COST_UNPLANNED_FAILURE, 1
    return COST_PLANNED_MAINT + (life - visit_cycle) * COST_PER_LOST_CYCLE, 0


def predictive_policy(df, col, margin):
    total, failures = 0, 0
    for _, g in df.groupby("unit"):
        life = int(g["cycle"].max())
        trig = g.loc[g[col] <= MAINT_LEAD_TIME + margin, "cycle"]
        visit = int(trig.iloc[0]) + MAINT_LEAD_TIME if len(trig) else None
        c, f = visit_cost(life, visit)
        total, failures = total + c, failures + f
    return total, failures


def main():
    b = joblib.load(MODEL_DIR / "pdm_bundle.joblib")
    fp, feats = b["pipeline"], b["features"]
    sim = pd.read_csv(DATA_DIR / "sim_fleet.csv")
    X = fp.transform(sim)
    sim = sim.loc[X.index, ["unit", "cycle"]].copy()
    sim["pred"] = b["regressor"].predict(X[feats])
    sim["q10"] = b["intervals"].predict(sim["pred"].clip(0, None))[0]   # lower bound of 80% interval

    units = np.sort(sim["unit"].unique())
    tune, evalu = sim[sim["unit"].isin(units[::2])], sim[sim["unit"].isin(units[1::2])]
    n_eval = evalu["unit"].nunique()
    lives_eval = evalu.groupby("unit")["cycle"].max().to_numpy()

    # Fixed interval K optimised on training-fleet lifetimes
    train_lives = pd.read_csv(DATA_DIR / "train.csv").groupby("unit")["cycle"].max().to_numpy()
    ks = np.arange(80, 360, 5)
    k_cost = [sum(visit_cost(L, k)[0] for L in train_lives) for k in ks]
    best_k = int(ks[int(np.argmin(k_cost))])

    # Safety margins tuned on the tuning half
    margins = np.arange(0, 61, 2)
    best_m = {col: int(margins[int(np.argmin([predictive_policy(tune, col, m)[0] for m in margins]))])
              for col in ("pred", "q10")}

    rows = []
    reactive = n_eval * COST_UNPLANNED_FAILURE
    rows.append(("Reactive (run to failure)", reactive, n_eval))
    fixed = [visit_cost(L, best_k) for L in lives_eval]
    rows.append((f"Fixed interval (every {best_k} cycles)", sum(c for c, _ in fixed), sum(f for _, f in fixed)))
    for col, label in (("pred", "Predictive: LightGBM point"), ("q10", "Predictive: LightGBM lower bound (q10)")):
        c, f = predictive_policy(evalu, col, best_m[col])
        rows.append((f"{label}, margin {best_m[col]}", c, f))
    rows.append(("Oracle (perfect knowledge)", n_eval * COST_PLANNED_MAINT, 0))

    tbl = pd.DataFrame(rows, columns=["policy", "total_cost_inr", "unplanned_failures"])
    tbl["cost_per_engine_inr"] = (tbl["total_cost_inr"] / n_eval).round(0)
    tbl["saving_vs_reactive_pct"] = (100 * (1 - tbl["total_cost_inr"] / reactive)).round(1)
    tbl["saving_vs_fixed_pct"] = (100 * (1 - tbl["total_cost_inr"] / tbl.loc[1, "total_cost_inr"])).round(1)
    tbl.to_csv(OUT_DIR / "maintenance_policy_results.csv", index=False)
    (OUT_DIR / "metrics_policy.json").write_text(json.dumps(
        {"engines_evaluated": n_eval, "fixed_interval_K": best_k, "tuned_margins": best_m,
         "results": tbl.to_dict(orient="records")}, indent=2))
    print(tbl.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4))
    names = [r.split(",")[0].replace("Predictive: ", "Predictive\n") for r in tbl["policy"]]
    bars = ax.bar(names, tbl["cost_per_engine_inr"] / 1e5,
                  color=["#de2d26", "#fd8d3c", "#6baed6", "#2171b5", "#74c476"])
    for bar, f in zip(bars, tbl["unplanned_failures"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{f} failures",
                ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="Cost per engine (INR lakh)", title=f"Maintenance policy comparison ({n_eval} unseen engines)")
    ax.tick_params(axis="x", labelsize=8); fig.tight_layout()
    fig.savefig(FIG_DIR / "policy_costs.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
