"""Explainability (SHAP) and production monitoring (data drift + performance).

Explainability
  * Global: mean |SHAP| per feature -> what drives RUL predictions overall.
  * Local : SHAP waterfall-style breakdown for one at-risk test engine -> why THIS
    engine was flagged (what a maintenance engineer needs to see).

Monitoring
  * Population Stability Index (PSI), training data vs a later "drift batch":
      - operating settings (raw)  -> detects a change in route / usage mix
      - sensors AFTER regime normalisation, early-life cycles only (cycle <= 100)
        -> detects calibration drift without being fooled by a different regime mix
           or by engines simply being at a different stage of wear.
      PSI < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 significant shift
  * Model performance on the drift batch vs the clean test set.
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

from config import DATA_DIR, FIG_DIR, MODEL_DIR, OP_SETTINGS, OUT_DIR, RUL_CAP, SENSORS
from src.features import add_targets
from src.metrics import regression_report

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using quantile bins of the reference data."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def main():
    b = joblib.load(MODEL_DIR / "pdm_bundle.joblib")
    fp, reg, feats = b["pipeline"], b["regressor"], b["features"]
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = add_targets(pd.read_csv(DATA_DIR / "test.csv"), pd.read_csv(DATA_DIR / "test_rul.csv"))
    drift = add_targets(pd.read_csv(DATA_DIR / "drift_batch.csv"), pd.read_csv(DATA_DIR / "drift_rul.csv"))

    # ------------------------------------------------------------- SHAP
    Xte_full = fp.transform(test)
    last_idx = Xte_full.groupby("unit")["cycle"].idxmax()
    X_last = Xte_full.loc[last_idx, feats]
    explainer = shap.TreeExplainer(reg)
    sv = explainer.shap_values(X_last)
    glob = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    glob.head(15)[::-1].plot.barh(ax=ax, color="#31a354")
    ax.set(title="Global explanation: mean |SHAP| (cycles of RUL)", xlabel="mean |SHAP value|")
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_global.png", dpi=150); plt.close(fig)

    preds = reg.predict(X_last)
    i = int(np.argmin(preds))                       # the most at-risk engine
    unit = int(last_idx.index[i])                   # last_idx is indexed by engine id
    contrib = pd.Series(sv[i], index=feats)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
    base = float(np.ravel(explainer.expected_value)[0])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index, top.values, color=["#de2d26" if v < 0 else "#3182bd" for v in top.values])
    ax.set(title=f"Local explanation, engine {unit}: base {base:.0f} -> predicted {preds[i]:.0f} cycles",
           xlabel="SHAP contribution to RUL (cycles)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_local.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------- drift (PSI)
    def monitor_frame(df):
        early = df[df["cycle"] <= 100]
        norm = fp._normalise(early, fp._regime(early))            # regime-normalised sensors
        return pd.concat([norm[SENSORS], early[OP_SETTINGS]], axis=1)

    ref, cur, clean = monitor_frame(train), monitor_frame(drift), monitor_frame(test)
    rows = [{"feature": c, "PSI": round(psi(ref[c].to_numpy(), cur[c].to_numpy()), 3)}
            for c in SENSORS + OP_SETTINGS]
    drift_tbl = pd.DataFrame(rows).sort_values("PSI", ascending=False)
    drift_tbl["status"] = pd.cut(drift_tbl["PSI"], [-1, 0.10, 0.25, 99],
                                 labels=["stable", "moderate", "significant"]).astype(str)
    drift_tbl.to_csv(OUT_DIR / "drift_report.csv", index=False)

    # Same check on the clean test set (should be stable) -> shows the monitor is not over-alarming
    psi_clean = {c: round(psi(ref[c].to_numpy(), clean[c].to_numpy()), 3) for c in SENSORS + OP_SETTINGS}

    # ------------------------------------------------------------- performance on drifted data
    Xd_full = fp.transform(drift)
    d_last = Xd_full.groupby("unit")["cycle"].idxmax()
    yd = drift.loc[d_last, "rul"].clip(upper=RUL_CAP).to_numpy()
    pd_last = np.clip(reg.predict(Xd_full.loc[d_last, feats]), 0, RUL_CAP)
    y_clean = test.loc[last_idx, "rul"].clip(upper=RUL_CAP).to_numpy()

    res = {"shap_top10": glob.head(10).round(2).to_dict(),
           "local_explanation_engine": unit,
           "drift_psi": drift_tbl.set_index("feature")["PSI"].to_dict(),
           "drift_features_significant": drift_tbl.query("status == 'significant'")["feature"].tolist(),
           "max_psi_clean_test": max(psi_clean.values()),
           "lgbm_clean_test": regression_report(y_clean, np.clip(preds, 0, RUL_CAP)),
           "lgbm_drift_batch": regression_report(yd, pd_last)}
    (OUT_DIR / "metrics_monitoring.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = drift_tbl["status"].map({"stable": "#74c476", "moderate": "#fd8d3c", "significant": "#de2d26"})
    ax.bar(drift_tbl["feature"], drift_tbl["PSI"], color=colors)
    ax.axhline(0.10, ls=":", color="grey"); ax.axhline(0.25, ls="--", color="grey")
    ax.set(title="Data drift (PSI): training reference vs new batch", ylabel="PSI")
    ax.tick_params(axis="x", rotation=60); fig.tight_layout()
    fig.savefig(FIG_DIR / "drift_psi.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
