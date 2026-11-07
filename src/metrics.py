"""Evaluation metrics for RUL prediction."""
import numpy as np


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def nasa_score(y, p):
    """Asymmetric PHM08 score (lower is better).

    d = predicted - true. Late predictions (d > 0: model thinks the engine has more
    life than it really has) are penalised more heavily than early ones, because a
    late prediction can mean an in-service failure.
    """
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def regression_report(y, p) -> dict:
    return {"RMSE": round(rmse(y, p), 2), "MAE": round(mae(y, p), 2),
            "NASA_score": round(nasa_score(y, p), 1)}
