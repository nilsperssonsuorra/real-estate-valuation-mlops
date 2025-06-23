# src/train.py
import pandas as pd
import xgboost as xgb
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# --- Configuration ---
CURRENT_DIR = os.path.dirname(__file__)
PROCESSED_DATA_PATH = os.path.join(CURRENT_DIR, '..', 'data', 'processed', 'hemnet_sold_villas_processed.csv')
MODEL_OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', 'models')

# --- Model Artifact Filenames ---
MODEL_FILE_PREFIX = "xgb_model"
COLUMNS_FILE = "model_columns.json"
LOCATION_COLS_FILE = "location_area_columns.json"

# --- Model Parameters ---
FEATURES = [
    'living_area_m2',
    'rooms',
    'plot_area_m2',
    'non_living_area_m2',
    'location_area',
    'total_area_m2',
    'plot_to_living_ratio',
    'sale_days_since_epoch'
]
TARGET = 'final_price'
# --- MODIFIED: We are now aiming for a 90% interval to widen the predictions ---
QUANTILES = [0.05, 0.5, 0.95] 
EPOCH = pd.Timestamp("2000-01-01")

def train_model():
    """
    Loads processed data, engineers features, trains XGBoost quantile regression models,
    and saves them along with all necessary artifacts for the prediction app.
    """
    print("--- Starting Model Training ---")

    # --- 1. Load Data (no changes) ---
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"Successfully loaded processed data with {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: Processed data file not found at '{PROCESSED_DATA_PATH}'.")
        print("Please run the cleaning script (src/clean.py) first.")
        return

    # --- 2. Feature Engineering & Preparation (no changes) ---
    print("\n--- Engineering New Features & Preparing Data ---")
    df['plot_area_m2'] = df['plot_area_m2'].fillna(df['plot_area_m2'].median())
    df['non_living_area_m2'] = df['non_living_area_m2'].fillna(0)
    df['sold_date'] = pd.to_datetime(df['sold_date'], errors='coerce')
    critical_cols = ['living_area_m2', 'final_price', 'sold_date']
    initial_rows = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows due to missing critical data in {critical_cols}.")
    if df.empty:
        print("\nERROR: DataFrame is empty after cleaning. Cannot proceed with training.")
        return
    df['total_area_m2'] = df['living_area_m2'] + df['non_living_area_m2']
    df['plot_to_living_ratio'] = df['plot_area_m2'] / (df['living_area_m2'] + 1e-6)
    df['sale_days_since_epoch'] = (df['sold_date'] - EPOCH).dt.days
    print("Successfully engineered new features: 'total_area_m2', 'plot_to_living_ratio', 'sale_days_since_epoch'.")
    location_counts = df['location_area'].value_counts()
    rare_locations = location_counts[location_counts < 10].index.tolist()
    if rare_locations:
        df['location_area'] = df['location_area'].replace(rare_locations, 'Other')
        print(f"Consolidated {len(rare_locations)} rare locations (count < 10) into 'Other' category.")
    location_area_list = sorted(list(df['location_area'].unique()))
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    location_file_path = os.path.join(MODEL_OUTPUT_DIR, LOCATION_COLS_FILE)
    with open(location_file_path, 'w', encoding='utf-8') as f:
        json.dump(location_area_list, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(location_area_list)} unique location areas to '{location_file_path}'.")
    X = df[FEATURES]
    y = df[TARGET]
    X = pd.get_dummies(X, columns=['location_area'], dummy_na=False)
    model_columns = X.columns.tolist()
    # Using joblib is fine, but json is also a good choice
    joblib.dump(model_columns, os.path.join(MODEL_OUTPUT_DIR, COLUMNS_FILE))
    print(f"Saved model columns to '{COLUMNS_FILE}'. Total features: {len(model_columns)}")

    # --- 3. Split Data (no changes) ---
    print("\n--- Splitting data into training and testing sets (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # --- 4. Train Quantile Models ---
    print("\n--- Training XGBoost models for each quantile ---")
    models = {}
    for q in QUANTILES:
        print(f"Training model for quantile: {q}...")
        
        # --- MODIFIED: Reverting some hyperparameters to be more conservative ---
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            n_estimators=2000,       # Back to original value
            learning_rate=0.03,      # Back to original value
            max_depth=4,             # Reverted to 4 to prevent overfitting
            subsample=0.7,           # Reverted to 0.7 for more regularization
            colsample_bytree=0.7,    # Reverted to 0.7 for more regularization
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50 # Back to original value
        )
        
        model.fit(X_train, y_train, 
                  eval_set=[(X_test, y_test)], 
                  verbose=False)
        
        # NOTE: This will now create files like 'xgb_model_q5.joblib' and 'xgb_model_q95.joblib'
        model_filename = f"{MODEL_FILE_PREFIX}_q{int(q*100)}.joblib"
        model_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"  > Model saved to '{model_path}'")
        models[q] = model

    # --- 5. Evaluate Models on Test Set ---
    print("\n--- Evaluating Model Performance on Test Set ---")
    y_pred_lower = models[QUANTILES[0]].predict(X_test)
    y_pred_median = models[0.5].predict(X_test)
    y_pred_upper = models[QUANTILES[2]].predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred_median)
    mape = np.mean(np.abs((y_test - y_pred_median) / y_test)) * 100
    within_interval = ((y_test >= y_pred_lower) & (y_test <= y_pred_upper)).mean()
    within_15_percent = (abs(y_pred_median - y_test) / y_test <= 0.15).mean()
    # --- MODIFIED: The target is now 90% ---
    target_coverage = QUANTILES[2] - QUANTILES[0] 
    
    print(f"Median Prediction MAE (Mean Absolute Error): {mae:,.0f} kr".replace(',', ' '))
    print(f"Median Prediction MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"Coverage (actual price within prediction interval): {within_interval:.2%} (Target: {target_coverage:.0%})")
    print(f"Accuracy (predictions within ±15% of actual price): {within_15_percent:.2%}")

    print("\n--- Training complete! All artifacts saved to 'models/' directory. ---")
    
if __name__ == '__main__':
    train_model()