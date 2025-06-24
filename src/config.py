# src/config.py
"""
Central configuration file for the Bostadsvärdering project.

This file holds all static parameters, file paths, and model settings
to make the code easier to manage and modify.
"""
import os
import pandas as pd
from pathlib import Path

# --- Project Structure (for local development) ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- Local Data File Paths ---
RAW_DATA_PATH = RAW_DATA_DIR / "hemnet_sold_villas_final.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "hemnet_sold_villas_processed.csv"

# --- Azure Blob Storage Configuration ---
# Check if running in the cloud (where this env var will be set)
IS_CLOUD = os.environ.get('AZURE_STORAGE_CONNECTION_STRING') is not None

AZURE_RAW_DATA_CONTAINER = "raw-data"
AZURE_PROCESSED_DATA_CONTAINER = "processed-data"
AZURE_MODELS_CONTAINER = "models"

RAW_DATA_BLOB_NAME = RAW_DATA_PATH.name
PROCESSED_DATA_BLOB_NAME = PROCESSED_DATA_PATH.name

# --- Scraper Configuration ---
HEMNET_BASE_URL = "https://www.hemnet.se/salda/bostader?item_types%5B%5D=villa&location_ids%5B%5D=946677"
SCRAPER_MAX_PAGES = 500

# --- Model Training Configuration ---
TARGET_VARIABLE = 'final_price'

QUANTILES_AND_NAMES = {
    'lower': 0.05,
    'median': 0.50,
    'upper': 0.95,
}

FEATURE_ENGINEERING_EPOCH = pd.Timestamp("2000-01-01")

BASE_FEATURES = [
    'living_area_m2', 'rooms', 'plot_area_m2', 'non_living_area_m2',
    'location_area', 'total_area_m2', 'plot_to_living_ratio',
    'sale_days_since_epoch', 'log_living_area', 'log_plot_area',
]

# --- Model Artifacts Configuration ---
MODEL_FILE_PREFIX = "xgb_model"
MODEL_COLUMNS_FILE = "model_columns.joblib"
LOCATION_OPTIONS_FILE = "location_area_columns.json"
LOCATION_PRICE_MAP_FILE = "location_price_map.json"
SHAP_EXPLAINER_FILE = "shap_explainer.joblib"

# --- Full paths to model artifacts (for local development) ---
MODEL_COLUMNS_PATH = MODELS_DIR / MODEL_COLUMNS_FILE
LOCATION_OPTIONS_PATH = MODELS_DIR / LOCATION_OPTIONS_FILE
LOCATION_PRICE_MAP_PATH = MODELS_DIR / LOCATION_PRICE_MAP_FILE
SHAP_EXPLAINER_PATH = MODELS_DIR / SHAP_EXPLAINER_FILE

MODEL_PATHS = {
    name: MODELS_DIR / f"{MODEL_FILE_PREFIX}_q{int(q * 100)}.joblib"
    for name, q in QUANTILES_AND_NAMES.items()
}