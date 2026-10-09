"""Central configuration for StudentGuard AI."""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ann_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
DB_PATH = os.path.join(BASE_DIR, "database", "studentguard.db")

TARGET_COL = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]

# Risk level is derived from the ANN's Dropout probability (0-1).
RISK_HIGH = 0.60      # >= 60 %  -> HIGH
RISK_MEDIUM = 0.30    # >= 30 %  -> MEDIUM, otherwise LOW

# ANN training settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
HIDDEN_LAYERS = (64, 32)

SECRET_KEY = os.environ.get("SECRET_KEY", "studentguard-dev-key")