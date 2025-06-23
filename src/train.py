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
MODEL_FILE_PREFIX = "xgb_model"
COLUMNS_FILE = "model_columns.joblib"
LOCATION_COLS_FILE = "location_area_columns.json"
LOCATION_PRICE_MAP_FILE = "location_price_map.json"

# --- Model Parameters ---
# Define the initial features we can get from the raw dataframe
BASE_FEATURES = [
    'living_area_m2',
    'rooms',
    'plot_area_m2',
    'non_living_area_m2',
    'location_area',
    'total_area_m2',
    'plot_to_living_ratio',
    'sale_days_since_epoch',
    'log_living_area',
    'log_plot_area',
]
# The final list will include the target-encoded feature
FINAL_FEATURES = BASE_FEATURES + ['location_median_price_per_m2']
TARGET = 'final_price'
QUANTILES = [0.05, 0.5, 0.95]
EPOCH = pd.Timestamp("2000-01-01")

def train_model():
    """
    Loads data, engineers basic and advanced features (including target encoding
    and log transforms), trains XGBoost quantile models, and saves all artifacts.
    """
    print("--- Starting Model Training ---")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"Successfully loaded processed data with {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: Processed data file not found at '{PROCESSED_DATA_PATH}'.")
        print("Please run the cleaning script (src/clean.py) first.")
        return

    # --- 2. Feature Engineering & Preparation ---
    print("\n--- Engineering New Features & Preparing Data ---")
    
    # Impute missing values
    df['plot_area_m2'] = df['plot_area_m2'].fillna(df['plot_area_m2'].median())
    df['non_living_area_m2'] = df['non_living_area_m2'].fillna(0)
    
    # Convert 'sold_date' and drop invalid rows
    df['sold_date'] = pd.to_datetime(df['sold_date'], errors='coerce')
    critical_cols = ['living_area_m2', 'final_price', 'sold_date']
    df.dropna(subset=critical_cols, inplace=True)

    # --- Basic Feature Engineering ---
    df['total_area_m2'] = df['living_area_m2'] + df['non_living_area_m2']
    df['plot_to_living_ratio'] = df['plot_area_m2'] / (df['living_area_m2'] + 1e-6)
    df['sale_days_since_epoch'] = (df['sold_date'] - EPOCH).dt.days

    # --- Advanced Feature Engineering ---
    # B. Log transformations
    df['log_living_area'] = np.log1p(df['living_area_m2'])
    df['log_plot_area'] = np.log1p(df['plot_area_m2'])
    print("Successfully engineered log-transformed features.")
    
    # Consolidate rare locations before target encoding
    location_counts = df['location_area'].value_counts()
    rare_locations = location_counts[location_counts < 10].index.tolist()
    if rare_locations:
        df['location_area'] = df['location_area'].replace(rare_locations, 'Other')
        print(f"Consolidated {len(rare_locations)} rare locations into 'Other'.")

    # --- Corrected Data Splitting and Feature Selection Logic ---
    
    # Select only the base features and the target to create our full dataset
    X_full = df[BASE_FEATURES + [TARGET]] 
    y_full = df[TARGET]

    # Split data BEFORE target encoding to prevent data leakage
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    # A. Target Encoding - calculated ONLY on the training set
    print("Calculating target encoding on training set to prevent data leakage...")
    # Important: Create a copy to avoid SettingWithCopyWarning
    X_train_full = X_train_full.copy()
    X_test_full = X_test_full.copy()
    
    price_per_m2_map = X_train_full.groupby('location_area')['final_price'].median() / X_train_full.groupby('location_area')['living_area_m2'].median()
    
    # Now, add the new feature to our training and test sets
    X_train_full['location_median_price_per_m2'] = X_train_full['location_area'].map(price_per_m2_map)
    X_test_full['location_median_price_per_m2'] = X_test_full['location_area'].map(price_per_m2_map)

    # Handle any locations in test set that weren't in training set
    global_median_price = X_train_full['location_median_price_per_m2'].median()
    X_train_full['location_median_price_per_m2'] = X_train_full['location_median_price_per_m2'].fillna(global_median_price)
    X_test_full['location_median_price_per_m2'] = X_test_full['location_median_price_per_m2'].fillna(global_median_price)
    print("Successfully engineered target-encoded feature.")
    
    # --- 3. Save Artifacts & Prepare Final Data ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # --- FIX: Save the price map using json.dump for consistency ---
    price_map_path = os.path.join(MODEL_OUTPUT_DIR, LOCATION_PRICE_MAP_FILE)
    # Convert the pandas Series to a dictionary before saving
    price_map_dict = price_per_m2_map.to_dict()
    with open(price_map_path, 'w', encoding='utf-8') as f:
        json.dump(price_map_dict, f, ensure_ascii=False, indent=4)
    print(f"Saved location price map to '{LOCATION_PRICE_MAP_FILE}'.")
    
    # Save the list of valid locations for the dropdown
    location_area_list = sorted(list(df['location_area'].unique()))
    location_file_path = os.path.join(MODEL_OUTPUT_DIR, LOCATION_COLS_FILE)
    with open(location_file_path, 'w', encoding='utf-8') as f:
        json.dump(location_area_list, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(location_area_list)} unique location areas.")
        
    # Finalize X for training, using the FINAL_FEATURES list now
    X_train = X_train_full[FINAL_FEATURES]
    X_test = X_test_full[FINAL_FEATURES]
    
    # One-hot encode and align columns
    X_train = pd.get_dummies(X_train, columns=['location_area'], dummy_na=False)
    X_test = pd.get_dummies(X_test, columns=['location_area'], dummy_na=False)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Save the final column list
    model_columns = X_train.columns.tolist()
    joblib.dump(model_columns, os.path.join(MODEL_OUTPUT_DIR, COLUMNS_FILE))
    print(f"Saved model columns to '{COLUMNS_FILE}'. Total features: {len(model_columns)}")
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # --- 4. Train Quantile Models ---
    print("\n--- Training XGBoost models for each quantile ---")
    models = {}
    for q in QUANTILES:
        print(f"Training model for quantile: {q}...")
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
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
    target_coverage = QUANTILES[2] - QUANTILES[0] 
    
    print(f"Median Prediction MAE (Mean Absolute Error): {mae:,.0f} kr".replace(',', ' '))
    print(f"Median Prediction MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"Coverage (actual price within prediction interval): {within_interval:.2%} (Target: {target_coverage:.0%})")
    print(f"Accuracy (predictions within ±15% of actual price): {within_15_percent:.2%}")

    print("\n--- Training complete! All artifacts saved to 'models/' directory. ---")
    
if __name__ == '__main__':
    train_model()