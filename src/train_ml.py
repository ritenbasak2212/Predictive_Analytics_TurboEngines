"""Classical / gradient-boosting models.

1. Baselines: constant mean RUL, Ridge regression on engineered features.
2. LightGBM RUL regressor with engine-grouped 5-fold cross-validation.
3. 80% prediction intervals from out-of-fold residuals (see src/intervals.py).
4. LightGBM classifier: "fails within 30 cycles", with a cost-optimal threshold
   chosen on out-of-fold predictions.
5. Saves a single deployable bundle (feature pipeline + all models) for the API.
"""
import json

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (COST_PLANNED_MAINT, COST_UNPLANNED_FAILURE, DATA_DIR, FIG_DIR,
                    MODEL_DIR, OUT_DIR, RUL_CAP, SEED)
from src.features import FeaturePipeline, add_targets
from src.intervals import ResidualIntervals
from src.metrics import regression_report

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)


def load():
    train = add_targets(pd.read_csv(DATA_DIR / "train.csv"))
    test_rul = pd.read_csv(DATA_DIR / "test_rul.csv")
    test = add_targets(pd.read_csv(DATA_DIR / "test.csv"), test_rul)
    return train, test


def main():
    train, test = load()
    fp = FeaturePipeline().fit(train)
    Xtr_full, Xte_full = fp.transform(train), fp.transform(test)
    feats = fp.feature_names
    Xtr, Xte = Xtr_full[feats], Xte_full[feats]
    ytr = train.loc[Xtr.index, "rul_capped"]
    groups = train.loc[Xtr.index, "unit"]

    # Evaluation rows: the LAST observed cycle of each test engine (the real-world question)
    last_idx = Xte_full.groupby("unit")["cycle"].idxmax()
    X_last = Xte.loc[last_idx]
    y_last = test.loc[last_idx, "rul"].clip(upper=RUL_CAP).to_numpy()

    results, cv_results = {}, {}
    gkf = GroupKFold(n_splits=5)

    # ---------------------------------------------------- 1. baselines
    results["Baseline: mean RUL"] = regression_report(y_last, np.full(len(y_last), ytr.mean()))
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr)
    p_ridge = np.clip(ridge.predict(X_last), 0, RUL_CAP)
    results["Ridge regression"] = regression_report(y_last, p_ridge)

    # ---------------------------------------------------- 2. LightGBM with grouped CV
    oof = np.zeros(len(Xtr))
    for k, (a, b) in enumerate(gkf.split(Xtr, ytr, groups)):
        m = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr.iloc[a], ytr.iloc[a])
        oof[b] = m.predict(Xtr.iloc[b])
    r = regression_report(ytr, np.clip(oof, 0, RUL_CAP)); r.pop("NASA_score")  # NASA score is a sum, only comparable on the 100-engine test set
    cv_results["LightGBM 5-fold grouped CV (all training cycles)"] = r

    reg = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr, ytr)
    p_last = np.clip(reg.predict(X_last), 0, RUL_CAP)
    results["LightGBM"] = regression_report(y_last, p_last)
    all_rows = {"LightGBM (all test rows)": regression_report(test.loc[Xte.index, "rul_capped"],
                                                             np.clip(reg.predict(Xte), 0, RUL_CAP))}
    all_rows["LightGBM (all test rows)"].pop("NASA_score")

    # ---------------------------------------------------- 3. residual-based prediction intervals
    # Calibrate only on the part of life where the model is actually queried in service
    # (30-95% of each engine's life, matching the test protocol), not on every cycle.
    life = train.loc[Xtr.index].groupby("unit")["cycle"].transform("max")
    frac = train.loc[Xtr.index, "cycle"] / life
    cal = ((frac >= 0.30) & (frac <= 0.95)).to_numpy()
    intervals = ResidualIntervals.fit(ytr[cal], np.clip(oof[cal], 0, RUL_CAP), coverage=0.80)
    lo, hi = intervals.predict(p_last)
    coverage = float(np.mean((y_last >= lo) & (y_last <= hi)))
    p_all = np.clip(reg.predict(Xte), 0, RUL_CAP)                  # every test row, not just the last cycle
    lo_all, hi_all = intervals.predict(p_all)
    y_all = test.loc[Xte.index, "rul_capped"].to_numpy()
    coverage_all = float(np.mean((y_all >= lo_all) & (y_all <= hi_all)))
    interval = {"nominal_coverage": 0.80, "coverage_last_cycle_100_engines": round(coverage, 3),
                "coverage_all_test_rows": round(coverage_all, 3),
                "mean_interval_width": round(float(np.mean(hi - lo)), 1)}

    # ---------------------------------------------------- 4. failure classifier + cost threshold
    yc = train.loc[Xtr.index, "fail_soon"]
    oof_p = np.zeros(len(Xtr))
    clf_params = {**LGB_PARAMS, "n_estimators": 400}
    for a, b in gkf.split(Xtr, yc, groups):
        oof_p[b] = lgb.LGBMClassifier(**clf_params).fit(Xtr.iloc[a], yc.iloc[a]).predict_proba(Xtr.iloc[b])[:, 1]
    thresholds = np.linspace(0.02, 0.9, 89)
    # cost of a missed failure vs cost of an unnecessary inspection/shop visit
    # Row-level proxy cost: each missed "fail-soon" cycle ~ unplanned failure risk, each false alarm ~ needless visit
    costs = [((oof_p < t) & (yc == 1)).sum() * COST_UNPLANNED_FAILURE
             + ((oof_p >= t) & (yc == 0)).sum() * COST_PLANNED_MAINT for t in thresholds]
    best_t = float(thresholds[int(np.argmin(costs))])
    clf = lgb.LGBMClassifier(**clf_params).fit(Xtr, yc)

    # classifier is evaluated on every test row (each row has a known true RUL)
    p_te = clf.predict_proba(Xte)[:, 1]
    y_te_c = test.loc[Xte.index, "fail_soon"]
    tn, fp_, fn, tp = confusion_matrix(y_te_c, p_te >= best_t).ravel()
    classifier = {"ROC_AUC": round(roc_auc_score(y_te_c, p_te), 4),
                  "PR_AUC": round(average_precision_score(y_te_c, p_te), 4),
                  "cost_optimal_threshold": round(best_t, 2),
                  "precision": round(tp / (tp + fp_), 3), "recall": round(tp / (tp + fn), 3),
                  "confusion_matrix": {"TP": int(tp), "FP": int(fp_), "FN": int(fn), "TN": int(tn)}}

    # ---------------------------------------------------- 5. save bundle + metrics + predictions
    bundle = {"pipeline": fp, "regressor": reg, "ridge": ridge, "intervals": intervals,
              "classifier": clf, "threshold": best_t, "features": feats}
    joblib.dump(bundle, MODEL_DIR / "pdm_bundle.joblib")
    pd.DataFrame({"unit": Xte_full.loc[last_idx, "unit"].to_numpy(), "true_rul": y_last,
                  "pred_ridge": p_ridge, "pred_lgbm": p_last, "pred_q10": lo, "pred_q90": hi}).to_csv(
        OUT_DIR / "test_predictions_ml.csv", index=False)
    metrics = {"test_last_cycle": results, "test_all_rows": all_rows, "cross_validation": cv_results,
               "prediction_interval": interval, "failure_classifier": classifier,
               "selected_sensors": fp.selected_sensors, "n_features": len(feats),
               "sensor_correlation_with_rul": fp.sensor_corr.round(3).to_dict()}
    (OUT_DIR / "metrics_ml.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # ---------------------------------------------------- figures
    order = np.argsort(y_last)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(range(len(order)), lo[order], hi[order], color="#9ecae1", alpha=0.6, label="80% prediction interval")
    ax.plot(y_last[order], "k.-", lw=1, label="True RUL")
    ax.plot(p_last[order], "o", color="#d94801", ms=3.5, label="LightGBM prediction")
    ax.set(xlabel="Test engines (sorted by true RUL)", ylabel="RUL (cycles)",
           title="Test set: predicted vs true remaining useful life")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "rul_pred_vs_true.png", dpi=150); plt.close(fig)

    imp = pd.Series(reg.booster_.feature_importance("gain"), index=feats).nlargest(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#3182bd"); ax.set(title="LightGBM top-15 features (gain)", xlabel="Gain")
    fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, np.array(costs) / 1e7, color="#756bb1")
    ax.axvline(best_t, ls="--", color="k"); ax.text(best_t + 0.01, max(costs) / 1e7 * 0.9, f"optimal = {best_t:.2f}")
    ax.set(xlabel="Probability threshold", ylabel="Proxy cost on training folds (INR crore)",
           title="Cost-based threshold selection (out-of-fold)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "threshold_cost_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
