# Predictive Maintenance: Remaining Useful Life (RUL) Prediction for a Turbofan Engine Fleet

An end-to-end, production-style predictive analytics project. It predicts **how many
cycles each engine has left before failure**, flags engines likely to fail within 30
cycles, explains every prediction, watches for data drift, and shows the
**business value in rupees** compared with reactive and fixed-interval maintenance.

## What the project covers

| Area | Techniques |
|---|---|
| Data | Fleet simulator in the C-MAPSS format: 6 operating regimes, 14 sensors, run-to-failure histories |
| Feature engineering | KMeans regime detection, regime-wise normalisation, sensor selection, rolling mean/std/slope, EWM |
| Classical ML | Mean baseline, Ridge, LightGBM with engine-grouped cross-validation |
| Uncertainty | 80% prediction intervals from out-of-fold residuals (conformal style) |
| Classification | Failure within 30 cycles, cost-optimal decision threshold |
| Deep learning | LSTM, GRU, Transformer encoder (PyTorch) on 30-cycle windows, weighted ensemble |
| Unsupervised | Autoencoder health index and early-warning lead time |
| Explainability | SHAP global and local explanations |
| MLOps | PSI drift monitoring, performance under drift, FastAPI service, Docker, tests |
| Business | Simulation of maintenance policies with cost per engine |

## Project structure

```
predictive-maintenance/
├── config.py                 paths, constants, business cost assumptions
├── run_all.py                runs every stage end to end
├── src/
│   ├── data_generator.py     synthetic fleet (train / test / drift batch / simulation fleet)
│   ├── features.py           FeaturePipeline (fit on train only, reused at serving time)
│   ├── intervals.py          residual-based prediction intervals
│   ├── metrics.py            RMSE, MAE, NASA asymmetric score
│   ├── train_ml.py           baselines, LightGBM, intervals, failure classifier, model bundle
│   ├── train_dl.py           LSTM / GRU / Transformer + ensemble
│   ├── anomaly.py            autoencoder health index
│   ├── explain_monitor.py    SHAP + PSI drift + performance under drift
│   └── maintenance_policy.py cost simulation of maintenance strategies
├── api/app.py                FastAPI scoring service
├── tests/test_api.py         API smoke tests
├── data/                     generated CSVs
└── outputs/                  metrics (JSON/CSV), figures, trained models
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                     # about 5 minutes on a laptop CPU
python -m pytest -q                   # 4 tests
uvicorn api.app:app --reload          # open http://127.0.0.1:8000/docs
```

Run a single stage: `python -m src.train_ml` (stages must run in the order shown in `run_all.py`).

## Using the real NASA C-MAPSS dataset

The simulator writes the same columns as C-MAPSS (`unit, cycle, op1-op3, sensors`). To switch:
download C-MAPSS FD002 or FD004 (6 operating regimes), rename its 21 sensor columns, set
`SENSORS` in `config.py` to the 21 names, and write `train.csv`, `test.csv` and `test_rul.csv`
into `data/`. Everything downstream is unchanged.

## Example API call

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "engine_id": 7,
  "records": [{"cycle": 1, "op1": 20.0, "op2": 0.70, "op3": 100.0,
               "s1": 512.3, "s2": 640.1, "...": "...all 14 sensors..."}]
}'
```

The response contains the predicted RUL, an 80% interval, the probability of failure
within 30 cycles, a risk level (LOW / MEDIUM / HIGH), a recommended action and the top 3
SHAP drivers.

## Key results

See `outputs/*.json`, `outputs/figures/` and the project report for full results.
