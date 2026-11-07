"""Deep-learning sequence models: LSTM, GRU and Transformer encoder (PyTorch).

Input  : the last WINDOW cycles of regime-normalised sensors  -> shape (WINDOW, n_sensors)
Target : capped RUL at the last cycle of the window
Split  : 80% / 20% of training ENGINES (never split cycles of one engine across sets)

After training, the script builds a weighted ensemble of Ridge, LightGBM and the
best deep model, with weights chosen on the validation engines only.
"""
import json
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_DIR, FIG_DIR, MODEL_DIR, OUT_DIR, RUL_CAP, SEED, WINDOW
from src.features import FeaturePipeline, add_targets
from src.metrics import regression_report

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, torch.get_num_threads()))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ windowing
def make_windows(norm: pd.DataFrame, sensors, target=None, last_only=False):
    """Slide a WINDOW-cycle window over every engine. Early cycles are padded with
    the engine's first reading so that every cycle gets a prediction."""
    X, y, units = [], [], []
    for u, g in norm.groupby("unit", sort=False):
        a = g[sensors].to_numpy(np.float32)
        pad = np.repeat(a[:1], WINDOW - 1, axis=0)
        a = np.vstack([pad, a])
        ends = [len(g) - 1] if last_only else range(len(g))
        for e in ends:
            X.append(a[e:e + WINDOW])
            units.append(u)
            if target is not None:
                y.append(target.loc[g.index[e]])
    X = np.stack(X)
    y = np.asarray(y, np.float32) if target is not None else None
    return X, y, np.asarray(units)


# ------------------------------------------------------------------ models
class RNNRegressor(nn.Module):
    def __init__(self, n_in, cell="lstm", hidden=64, layers=2, dropout=0.2):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)                 # out: (batch, time, hidden)
        return self.head(out[:, -1]).squeeze(-1)   # use the last time step


class TransformerRegressor(nn.Module):
    def __init__(self, n_in, d_model=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, d_model))     # learned positional encoding
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=128,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.encoder(self.proj(x) + self.pos)
        return self.head(h.mean(dim=1)).squeeze(-1)   # average-pool over time


def train_model(model, Xtr, ytr, Xva, yva, epochs=25, lr=2e-3, patience=5):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr / RUL_CAP)),
                    batch_size=256, shuffle=True)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guards against exploding gradients
            opt.step()
        va_rmse = np.sqrt(np.mean((predict(model, Xva) - yva) ** 2))   # in cycles
        sched.step(va_rmse)
        hist.append(va_rmse)
        if va_rmse < best - 1e-3:
            best, bad = va_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + 2048]).to(DEVICE)).cpu().numpy()
           for i in range(0, len(X), 2048)]
    return np.clip(np.concatenate(out) * RUL_CAP, 0, RUL_CAP)


def main():
    train = add_targets(pd.read_csv(DATA_DIR / "train.csv"))
    test = add_targets(pd.read_csv(DATA_DIR / "test.csv"), pd.read_csv(DATA_DIR / "test_rul.csv"))
    fp = FeaturePipeline().fit(train)
    sensors = fp.selected_sensors
    ntr, nte = fp.normalised_sensors(train), fp.normalised_sensors(test)

    units = ntr["unit"].unique()
    rng = np.random.default_rng(SEED)
    va_units = set(rng.choice(units, size=int(0.2 * len(units)), replace=False))
    tr_mask = ~ntr["unit"].isin(va_units)

    Xtr, ytr, _ = make_windows(ntr[tr_mask], sensors, train["rul_capped"])
    Xva, yva, _ = make_windows(ntr[~tr_mask], sensors, train["rul_capped"])
    Xte, _, te_units = make_windows(nte, sensors, last_only=True)
    y_te = test.groupby("unit")["rul"].last().clip(upper=RUL_CAP).loc[te_units].to_numpy()
    print(f"windows: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}, device={DEVICE}")

    builders = {"LSTM": lambda: RNNRegressor(len(sensors), "lstm"),
                "GRU": lambda: RNNRegressor(len(sensors), "gru"),
                "Transformer": lambda: TransformerRegressor(len(sensors))}
    results, histories, val_preds, test_preds = {}, {}, {}, {}
    for name, build in builders.items():
        t0 = time.time()
        lr = 5e-4 if name == "Transformer" else 2e-3     # attention models prefer a smaller learning rate
        model, hist = train_model(build(), Xtr, ytr, Xva, yva, lr=lr, epochs=30, patience=6)
        n_params = sum(p.numel() for p in model.parameters())
        test_preds[name] = predict(model, Xte)
        val_preds[name] = predict(model, Xva)
        results[name] = {**regression_report(y_te, test_preds[name]),
                         "val_RMSE": round(float(np.sqrt(np.mean((val_preds[name] - yva) ** 2))), 2),
                         "params": n_params, "epochs": len(hist),
                         "train_time_s": round(time.time() - t0, 1)}
        histories[name] = hist
        torch.save(model.state_dict(), MODEL_DIR / f"{name.lower()}.pt")
        print(name, results[name])

    # ---------------------------------------------------- ensemble (weights fitted on validation engines)
    ml = pd.read_csv(OUT_DIR / "test_predictions_ml.csv").set_index("unit").loc[te_units]
    best_dl = min(results, key=lambda k: results[k]["val_RMSE"])
    # ML validation predictions: refit Ridge + LightGBM on the 80% training engines only
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.train_ml import LGB_PARAMS
    feat = fp.transform(train)
    F, yF = feat[fp.feature_names], train.loc[feat.index, "rul_capped"]
    fmask = ~feat["unit"].isin(va_units)
    ridge_v = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(F[fmask], yF[fmask])
    # n_jobs=1: LightGBM's multithreaded OpenMP clashes with PyTorch's in the same process
    # (segfault on macOS), so this one validation model is fitted single-threaded.
    lgb_v = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_jobs": 1}).fit(F[fmask], yF[fmask])
    # validation rows are in the same (unit, cycle) order as the windows
    va_ridge = np.clip(ridge_v.predict(F[~fmask]), 0, RUL_CAP)
    va_lgb = np.clip(lgb_v.predict(F[~fmask]), 0, RUL_CAP)

    best_w, best_rmse = None, np.inf
    for w1 in np.arange(0, 1.01, 0.1):
        for w2 in np.arange(0, 1.01 - w1, 0.1):
            w3 = 1 - w1 - w2
            p = w1 * va_ridge + w2 * va_lgb + w3 * val_preds[best_dl]
            r = np.sqrt(np.mean((p - yva) ** 2))
            if r < best_rmse:
                best_rmse, best_w = r, (round(w1, 1), round(w2, 1), round(w3, 1))
    ens = best_w[0] * ml["pred_ridge"].to_numpy() + best_w[1] * ml["pred_lgbm"].to_numpy() + best_w[2] * test_preds[best_dl]
    results[f"Ensemble (Ridge {best_w[0]}, LightGBM {best_w[1]}, {best_dl} {best_w[2]})"] = regression_report(y_te, ens)

    out = pd.DataFrame({"unit": te_units, "true_rul": y_te, **{f"pred_{k}": v for k, v in test_preds.items()},
                        "pred_ensemble": ens})
    out.to_csv(OUT_DIR / "test_predictions_dl.csv", index=False)
    (OUT_DIR / "metrics_dl.json").write_text(json.dumps(
        {"models": results, "best_deep_model": best_dl, "ensemble_weights": best_w}, indent=2))
    print(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, h in histories.items():
        ax.plot(h, marker="o", ms=3, label=k)
    ax.set(xlabel="Epoch", ylabel="Validation RMSE (cycles)", title="Deep models: validation RMSE per epoch")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "dl_training_curves.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
