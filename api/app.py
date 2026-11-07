"""FastAPI service that scores an engine's sensor history in real time.

Run:   uvicorn api.app:app --reload        (from the project root)
Docs:  http://127.0.0.1:8000/docs          (interactive Swagger UI)

POST /predict  body = engine id + list of cycle records (oldest first)
returns predicted RUL, 80% interval, probability of failure within 30 cycles,
a risk level, a recommended action and the top 3 drivers (SHAP) of the prediction.
"""
from typing import List

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import FAIL_HORIZON, MAINT_LEAD_TIME, MODEL_DIR, OP_SETTINGS, RUL_CAP, SENSORS

BUNDLE = joblib.load(MODEL_DIR / "pdm_bundle.joblib")
EXPLAINER = shap.TreeExplainer(BUNDLE["regressor"])
app = FastAPI(title="Predictive Maintenance API", version="1.0.0",
              description="Remaining-useful-life and failure-risk scoring for turbofan engines")


class CycleRecord(BaseModel):
    cycle: int = Field(..., ge=1)
    op1: float
    op2: float
    op3: float
    s1: float; s2: float; s3: float; s4: float; s5: float; s6: float; s7: float  # noqa: E702
    s8: float; s9: float; s10: float; s11: float; s12: float; s13: float; s14: float  # noqa: E702


class EngineHistory(BaseModel):
    engine_id: int
    records: List[CycleRecord] = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {"selected_sensors": BUNDLE["pipeline"].selected_sensors,
            "n_features": len(BUNDLE["features"]), "rul_cap": RUL_CAP,
            "failure_horizon_cycles": FAIL_HORIZON,
            "decision_threshold": BUNDLE["threshold"]}


@app.post("/predict")
def predict(history: EngineHistory):
    df = pd.DataFrame([r.model_dump() for r in history.records]).sort_values("cycle")
    if df["cycle"].duplicated().any():
        raise HTTPException(422, "duplicate cycle numbers in history")
    df.insert(0, "unit", history.engine_id)
    df = df[["unit", "cycle"] + OP_SETTINGS + SENSORS]

    feats = BUNDLE["pipeline"].transform(df)
    x = feats[BUNDLE["features"]].iloc[[-1]]            # features at the latest cycle
    rul = float(np.clip(BUNDLE["regressor"].predict(x)[0], 0, RUL_CAP))
    lo, hi = (float(v[0]) for v in BUNDLE["intervals"].predict([rul]))
    p_fail = float(BUNDLE["classifier"].predict_proba(x)[0, 1])

    if p_fail >= BUNDLE["threshold"] or lo <= MAINT_LEAD_TIME:
        risk, action = "HIGH", "Schedule shop visit now"
    elif lo <= MAINT_LEAD_TIME + 30:
        risk, action = "MEDIUM", "Plan maintenance in the next planning cycle; increase inspection frequency"
    else:
        risk, action = "LOW", "Continue normal operation"

    sv = pd.Series(EXPLAINER.shap_values(x)[0], index=BUNDLE["features"])
    drivers = [{"feature": f, "effect_on_rul_cycles": round(float(sv[f]), 2)}
               for f in sv.abs().sort_values(ascending=False).index[:3]]

    return {"engine_id": history.engine_id, "last_cycle": int(df["cycle"].iloc[-1]),
            "predicted_rul": round(rul, 1),
            "rul_interval_80pct": [round(lo, 1), round(hi, 1)],
            "rul_note": f"RUL is capped at {RUL_CAP}; {RUL_CAP} means 'healthy, no degradation detected'",
            "failure_probability_30_cycles": round(p_fail, 3),
            "risk_level": risk, "recommended_action": action, "top_drivers": drivers}
