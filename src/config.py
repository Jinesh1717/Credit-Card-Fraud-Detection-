from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.2
FALSE_ALARM_COST = 5.0
MISSED_FRAUD_COST = 100.0
np.random.seed(RANDOM_STATE)
