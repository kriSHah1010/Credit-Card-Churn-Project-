from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"


def make_dirs():
    for path in [DATA_PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_data(filename: str) -> pd.DataFrame:
    file_path = DATA_RAW_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def load_processed_data(filename: str) -> pd.DataFrame:
    file_path = DATA_PROCESSED_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def save_data(df: pd.DataFrame, filename: str):
    file_path = DATA_PROCESSED_DIR / filename
    df.to_csv(file_path, index=False)


def save_model(model, filename: str = "churn_model.pkl"):
    file_path = MODELS_DIR / filename
    joblib.dump(model, file_path)


def load_model(filename: str = "churn_model.pkl"):
    file_path = MODELS_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Model not found: {file_path}")
    return joblib.load(file_path)