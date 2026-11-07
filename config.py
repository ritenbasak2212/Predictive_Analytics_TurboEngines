"""Central configuration for the Predictive Maintenance project.

Every script imports its paths, constants and business assumptions from here,
so the whole project can be re-tuned from one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"
for _d in (DATA_DIR, OUT_DIR, MODEL_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data
SEED = 42
N_TRAIN_ENGINES = 200          # run-to-failure histories used for training
N_TEST_ENGINES = 100           # histories cut off at a random point (RUL unknown to model)
N_DRIFT_ENGINES = 60           # a later batch with sensor-calibration drift (for monitoring)
N_SIM_ENGINES = 100            # full histories used only for the maintenance-cost simulation
SENSORS = [f"s{i}" for i in range(1, 15)]   # 14 raw sensor channels
OP_SETTINGS = ["op1", "op2", "op3"]         # altitude, Mach number, throttle angle
N_REGIMES = 6                               # distinct flight operating conditions

# ---------------------------------------------------------------- modelling
RUL_CAP = 125        # piecewise-linear target: early-life RUL is capped (engine is "healthy")
WINDOW = 30          # sequence length (cycles) for LSTM / GRU / Transformer
FAIL_HORIZON = 30    # classification label: "will fail within the next 30 cycles"
ROLL_WINDOWS = (5, 15, 30)

# ---------------------------------------------------------------- business assumptions (INR)
COST_UNPLANNED_FAILURE = 2_500_000   # in-service failure: repair + downtime + penalties
COST_PLANNED_MAINT = 600_000         # scheduled shop visit
COST_PER_LOST_CYCLE = 4_000          # value of useful life thrown away by early maintenance
MAINT_LEAD_TIME = 15                 # cycles needed to schedule a shop visit
