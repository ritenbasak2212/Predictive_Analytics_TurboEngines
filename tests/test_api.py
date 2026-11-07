"""Smoke tests for the API and the feature pipeline. Run:  python -m pytest -q"""
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.app import app  # noqa: E402
from config import DATA_DIR  # noqa: E402

client = TestClient(app)


def _payload(unit: int, n_cycles: int | None = None):
    df = pd.read_csv(DATA_DIR / "test.csv")
    g = df[df["unit"] == unit].drop(columns="unit")
    if n_cycles:
        g = g.head(n_cycles)
    return {"engine_id": unit, "records": g.to_dict(orient="records")}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_valid_response():
    r = client.post("/predict", json=_payload(1))
    assert r.status_code == 200
    body = r.json()
    lo, hi = body["rul_interval_80pct"]
    assert 0 <= lo <= hi
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert len(body["top_drivers"]) == 3


def test_new_engine_is_low_risk():
    r = client.post("/predict", json=_payload(2, n_cycles=20)).json()
    assert r["risk_level"] == "LOW"


def test_rejects_duplicate_cycles():
    p = _payload(1, n_cycles=5)
    p["records"][1]["cycle"] = p["records"][0]["cycle"]
    assert client.post("/predict", json=p).status_code == 422
