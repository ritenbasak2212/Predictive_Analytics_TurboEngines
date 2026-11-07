"""Feature engineering pipeline.

Steps
-----
1. Operating-regime detection: KMeans on the 3 operating settings (unsupervised).
2. Regime-wise normalisation: z-score every sensor using the mean/std of its own
   regime, removing the huge regime effect and exposing the small degradation signal.
3. Sensor selection: keep sensors whose normalised value correlates with RUL.
4. Temporal features per engine: rolling mean / std, rolling slope (trend),
   exponentially weighted mean, and cycle count.
5. Targets: capped RUL (regression) and fail-within-horizon (classification).

All statistics are learned from TRAINING data only and stored in a
`FeaturePipeline` object, so exactly the same transformation is applied at
test time and inside the API (no leakage, no training/serving skew).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import FAIL_HORIZON, N_REGIMES, OP_SETTINGS, ROLL_WINDOWS, RUL_CAP, SEED, SENSORS


def add_targets(df: pd.DataFrame, rul_at_end: pd.DataFrame | None = None) -> pd.DataFrame:
    """RUL for every row. For truncated histories, pass the true RUL at the last row."""
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    extra = 0
    if rul_at_end is not None:
        extra = df["unit"].map(rul_at_end.set_index("unit")["true_rul"])
    df["rul"] = (last - df["cycle"]) + extra
    df["rul_capped"] = df["rul"].clip(upper=RUL_CAP)
    df["fail_soon"] = (df["rul"] <= FAIL_HORIZON).astype(int)
    return df


@dataclass
class FeaturePipeline:
    selected_sensors: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)

    # ------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "FeaturePipeline":
        self.op_scaler = StandardScaler().fit(train[OP_SETTINGS])
        self.kmeans = KMeans(n_clusters=N_REGIMES, n_init=10, random_state=SEED)
        self.kmeans.fit(self.op_scaler.transform(train[OP_SETTINGS]))
        regime = self._regime(train)

        # Per-regime sensor statistics (learned on training data only)
        grp = train[SENSORS].groupby(regime)
        self.reg_mean, self.reg_std = grp.mean(), grp.std().replace(0, 1)

        # Sensor selection: |Spearman correlation| between normalised sensor and RUL
        norm = self._normalise(train, regime)
        rul = add_targets(train)["rul_capped"]
        corr = norm.corrwith(rul, method="spearman").abs()
        self.sensor_corr = corr.sort_values(ascending=False)
        self.selected_sensors = [s for s in SENSORS if corr[s] > 0.10]
        return self

    # ------------------------------------------------------------ transform
    def _regime(self, df):
        return self.kmeans.predict(self.op_scaler.transform(df[OP_SETTINGS]))

    def _normalise(self, df, regime):
        mu = self.reg_mean.loc[regime].to_numpy()
        sd = self.reg_std.loc[regime].to_numpy()
        return pd.DataFrame((df[SENSORS].to_numpy() - mu) / sd, columns=SENSORS, index=df.index)

    def normalised_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regime-normalised selected sensors (input for the deep-learning models)."""
        df = df.sort_values(["unit", "cycle"])
        norm = self._normalise(df, self._regime(df))[self.selected_sensors]
        return pd.concat([df[["unit", "cycle"]], norm], axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full tabular feature matrix (input for the gradient-boosting models)."""
        base = self.normalised_sensors(df)
        g = base.groupby("unit")
        feats = {"cycle": base["cycle"]}
        t = base["cycle"].astype(float)
        for s in self.selected_sensors:
            x = base[s]
            for w in ROLL_WINDOWS:
                feats[f"{s}_mean{w}"] = g[s].transform(lambda v, w=w: v.rolling(w, min_periods=1).mean())
                feats[f"{s}_std{w}"] = g[s].transform(lambda v, w=w: v.rolling(w, min_periods=2).std()).fillna(0)
            # rolling slope over 15 cycles = cov(x, t) / var(t)  -> local degradation trend
            xt = (x * t).groupby(base["unit"]).transform(lambda v: v.rolling(15, min_periods=3).mean())
            xm = g[s].transform(lambda v: v.rolling(15, min_periods=3).mean())
            tm = t.groupby(base["unit"]).transform(lambda v: v.rolling(15, min_periods=3).mean())
            tt = (t * t).groupby(base["unit"]).transform(lambda v: v.rolling(15, min_periods=3).mean())
            feats[f"{s}_slope15"] = ((xt - xm * tm) / (tt - tm ** 2).replace(0, np.nan)).fillna(0)
            feats[f"{s}_ewm"] = g[s].transform(lambda v: v.ewm(alpha=0.1).mean())
        out = pd.DataFrame(feats, index=base.index)
        self.feature_names = list(out.columns)
        out.insert(0, "unit", base["unit"])
        return out
