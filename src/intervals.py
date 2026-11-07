"""Prediction intervals from out-of-fold residuals (split-conformal style).

Why not quantile regression? With a capped target (many rows equal to RUL_CAP),
gradient-boosted quantile models for the upper quantile tend to stick at the cap.
Residual-based intervals are simpler, model-agnostic and well calibrated:

1. Get out-of-fold predictions for every training row (grouped CV by engine).
2. Bin rows by their predicted RUL (uncertainty is larger for healthier engines).
3. In each bin, store the 10th and 90th percentile of (true - predicted).
4. At prediction time: interval = prediction + stored offsets of its bin.
"""
from dataclasses import dataclass

import numpy as np

from config import RUL_CAP


@dataclass
class ResidualIntervals:
    edges: np.ndarray
    low: np.ndarray
    high: np.ndarray
    coverage: float = 0.80

    @classmethod
    def fit(cls, y_true, y_pred_oof, coverage=0.80, edges=(0, 20, 40, 60, 80, 100, 115, RUL_CAP + 1)):
        y_true, y_pred_oof = np.asarray(y_true), np.asarray(y_pred_oof)
        edges = np.asarray(edges, float)
        a = (1 - coverage) / 2
        low, high = [], []
        bins = np.clip(np.digitize(y_pred_oof, edges) - 1, 0, len(edges) - 2)
        for k in range(len(edges) - 1):
            r = (y_true - y_pred_oof)[bins == k]
            low.append(np.quantile(r, a) if len(r) else 0.0)
            high.append(np.quantile(r, 1 - a) if len(r) else 0.0)
        return cls(edges, np.array(low), np.array(high), coverage)

    def predict(self, y_pred):
        y_pred = np.asarray(y_pred, float)
        k = np.clip(np.digitize(y_pred, self.edges) - 1, 0, len(self.edges) - 2)
        lo = np.clip(y_pred + self.low[k], 0, RUL_CAP)
        hi = np.clip(y_pred + self.high[k], 0, RUL_CAP)
        return lo, hi
