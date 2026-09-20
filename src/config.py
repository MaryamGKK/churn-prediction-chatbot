import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
TREE_MODEL_PATH = MODELS_DIR / "decision_tree.joblib"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
ONNX_MODEL_PATH = MODELS_DIR / "xgb_model.onnx"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "qwen2.5:1.5b")
API_KEY = os.getenv("API_KEY", "")

TEST_SIZE = 0.2
RANDOM_STATE = 42
OPTUNA_TRIALS = 50
CV_FOLDS = 5
MIN_RECALL = 0.80

MAX_SESSIONS = 100
SESSION_TTL_MINUTES = 60

DROP_COLS = ["gender", "Dual"]

BINARY_COLS = [
    "Is_Married", "Dependents", "Phone_Service", "Paperless_Billing",
    "Senior_Citizen",
]
MULTI_CAT_COLS = [
    "Internet_Service", "Online_Security", "Online_Backup",
    "Device_Protection", "Tech_Support", "Streaming_TV", "Streaming_Movies",
    "Contract", "Payment_Method", "tenure_bins",
]
NUMERIC_COLS = [
    "tenure", "Monthly_Charges", "Total_Charges",
    "service_count", "avg_monthly_charge", "charge_per_service",
]

INTERNET_DEPENDENT_COLS = [
    "Online_Security", "Online_Backup", "Device_Protection",
    "Tech_Support", "Streaming_TV", "Streaming_Movies",
]
SERVICE_COLS = INTERNET_DEPENDENT_COLS
