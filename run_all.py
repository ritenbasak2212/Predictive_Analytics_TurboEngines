"""Run the complete project end to end:  python run_all.py

Stages (≈5 minutes on a 2-core laptop CPU):
  1. data_generator     synthetic fleet data
  2. train_ml           baselines, LightGBM, intervals, failure classifier
  3. train_dl           LSTM, GRU, Transformer + ensemble
  4. anomaly            autoencoder health index
  5. explain_monitor    SHAP + drift monitoring
  6. maintenance_policy business-impact simulation
"""
import subprocess
import sys
import time

STAGES = ["data_generator", "train_ml", "train_dl", "anomaly", "explain_monitor", "maintenance_policy"]

if __name__ == "__main__":
    t_all = time.time()
    for name in STAGES:
        t0 = time.time()
        print(f"\n{'=' * 70}\n>>> {name}\n{'=' * 70}", flush=True)
        # One process per stage: LightGBM and PyTorch each bundle an OpenMP runtime, and
        # loading both in the same process can deadlock on macOS.
        subprocess.run([sys.executable, "-m", f"src.{name}"], check=True)
        print(f"[{name} done in {time.time() - t0:.0f}s]", flush=True)
    print(f"\nAll stages finished in {time.time() - t_all:.0f}s. Results are in outputs/.")
