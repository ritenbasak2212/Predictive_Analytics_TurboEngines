"""Unsupervised health monitoring with an autoencoder.

Idea: train an autoencoder ONLY on healthy behaviour (early-life cycles). When an
engine degrades, its sensor pattern drifts away from "healthy" and the
reconstruction error rises. The smoothed error is a Health Index that needs no
failure labels, which is useful for new equipment types with no failure history yet.

We measure the early-warning lead time: how many cycles before failure the health
index first crosses an alarm threshold (99th percentile of healthy error).
"""
import json

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn

from config import DATA_DIR, FIG_DIR, MODEL_DIR, OUT_DIR, SEED
from src.features import FeaturePipeline, add_targets

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

torch.manual_seed(SEED)


class AutoEncoder(nn.Module):
    def __init__(self, n_in, latent=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, n_in))

    def forward(self, x):
        return self.dec(self.enc(x))


def recon_error(model, X):
    with torch.no_grad():
        X = torch.from_numpy(np.ascontiguousarray(X).copy())
        return ((model(X) - X) ** 2).mean(dim=1).numpy()


def main():
    train = add_targets(pd.read_csv(DATA_DIR / "train.csv"))
    fp = FeaturePipeline().fit(train)
    norm = fp.normalised_sensors(train)
    sensors = fp.selected_sensors
    norm["rul"] = train.loc[norm.index, "rul"]

    units = norm["unit"].unique()
    rng = np.random.default_rng(SEED)
    hold = set(rng.choice(units, 40, replace=False))           # engines kept for evaluation
    fit_df = norm[~norm["unit"].isin(hold)]
    healthy = fit_df[fit_df["rul"] > 150][sensors].to_numpy(np.float32).copy()   # clearly healthy cycles

    model = AutoEncoder(len(sensors))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xh = torch.from_numpy(healthy)
    for epoch in range(40):
        perm = torch.randperm(len(Xh))
        for i in range(0, len(Xh), 256):
            xb = Xh[perm[i:i + 256]]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), xb)
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), MODEL_DIR / "autoencoder.pt")

    # Health index = reconstruction error smoothed with a 10-cycle rolling mean
    ev = norm[norm["unit"].isin(hold)].copy()
    ev["err"] = recon_error(model, ev[sensors].to_numpy(np.float32))
    ev["health_index"] = ev.groupby("unit")["err"].transform(lambda v: v.rolling(10, min_periods=1).mean())
    fit_df = fit_df.copy()
    fit_df["err"] = recon_error(model, fit_df[sensors].to_numpy(np.float32))
    fit_df["hi"] = fit_df.groupby("unit")["err"].transform(lambda v: v.rolling(10, min_periods=1).mean())
    threshold = float(np.percentile(fit_df.loc[fit_df["rul"] > 150, "hi"], 99))

    # Lead time: first cycle (after 3 consecutive alarms) where the health index exceeds the threshold
    leads, false_early = [], 0
    for u, g in ev.groupby("unit"):
        alarm = (g["health_index"] > threshold).astype(int).rolling(3).sum() >= 3
        if alarm.any():
            lead = int(g.loc[alarm.idxmax(), "rul"])
            leads.append(lead)
            false_early += lead > 150
    leads = np.array(leads)
    res = {"alarm_threshold": round(threshold, 4), "engines_evaluated": len(hold),
           "engines_alarmed": int(len(leads)),
           "median_lead_time_cycles": float(np.median(leads)),
           "p10_lead_time_cycles": float(np.percentile(leads, 10)),
           "alarms_while_clearly_healthy(RUL>150)": int(false_early)}
    (OUT_DIR / "metrics_anomaly.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(9, 4))
    for u in sorted(hold)[:6]:
        g = ev[ev["unit"] == u]
        ax.plot(-g["rul"], g["health_index"], lw=1.2, label=f"engine {u}")
    ax.axhline(threshold, ls="--", color="red", label="alarm threshold")
    ax.set(xlabel="Cycles relative to failure (0 = failure)", ylabel="Health index (recon. error)",
           title="Autoencoder health index rises as engines approach failure", yscale="log")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(FIG_DIR / "health_index.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
