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
# These filenames are used for saving and loading the trained models and artifacts.
MODEL_FILE_PREFIX = "xgb_model"
COLUMNS_FILE = "model_columns.json" # List of all feature columns after one-hot encoding.
LOCATION_COLS_FILE = "location_area_columns.json" # List of unique location areas for the app dropdown.

# --- Model Parameters ---
# Define the features to be used for training the model.
FEATURES = [
    'living_area_m2', 
    'rooms', 
    'plot_area_m2', 
    'non_living_area_m2',
    'location_area', # This will be one-hot encoded.
    # Date-based features are crucial for capturing market trends over time.
    'sale_year',
    'sale_month',
    'sale_dayofyear',
]
TARGET = 'final_price'
# We train models for three quantiles to create a prediction interval.
QUANTILES = [0.10, 0.5, 0.90] 

def train_model():
    """
    Loads processed data, engineers features, trains XGBoost quantile regression models,
    and saves them along with all necessary artifacts for the prediction app.
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
    
    # Impute missing values for numeric features.
    df['plot_area_m2'] = df['plot_area_m2'].fillna(df['plot_area_m2'].median())
    df['non_living_area_m2'] = df['non_living_area_m2'].fillna(0) # Assume 0 if not specified.
    
    # Convert 'sold_date' to datetime and drop rows where it's invalid.
    df['sold_date'] = pd.to_datetime(df['sold_date'], errors='coerce')
    
    # Drop rows with missing data in critical columns needed for training and target.
    critical_cols = ['living_area_m2', 'final_price', 'sold_date']
    initial_rows = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows due to missing critical data in {critical_cols}.")

    if df.empty:
        print("\nERROR: DataFrame is empty after cleaning. Cannot proceed with training.")
        return

    # Engineer date-based features from the 'sold_date' column.
    df['sale_year'] = df['sold_date'].dt.year
    df['sale_month'] = df['sold_date'].dt.month
    df['sale_dayofyear'] = df['sold_date'].dt.dayofyear
    print("Successfully engineered date features.")
    
    # --- Data Consolidation & Artifact Saving ---
    # Consolidate rare 'location_area' categories into an 'Other' group to prevent overfitting
    # and improve model generalization. A threshold of 10 occurrences is used.
    location_counts = df['location_area'].value_counts()
    rare_locations = location_counts[location_counts < 10].index.tolist()
    
    if rare_locations:
        df['location_area'] = df['location_area'].replace(rare_locations, 'Other')
        print(f"Consolidated {len(rare_locations)} rare locations (count < 10) into 'Other' category.")

    # Save the final list of location areas for the app's dropdown menu.
    # This is saved as a human-readable JSON file with UTF-8 encoding.
    location_area_list = sorted(list(df['location_area'].unique()))
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    location_file_path = os.path.join(MODEL_OUTPUT_DIR, LOCATION_COLS_FILE)
    with open(location_file_path, 'w', encoding='utf-8') as f:
        json.dump(location_area_list, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(location_area_list)} unique location areas to '{location_file_path}'.")

    # --- Prepare Final DataFrame for Training ---
    X = df[FEATURES]
    y = df[TARGET]

    # One-hot encode the final set of location categories.
    X = pd.get_dummies(X, columns=['location_area'], dummy_na=False)
    
    # Save the complete list of feature columns (including one-hot encoded ones).
    # This is crucial for the prediction app to create a DataFrame with the exact same structure.
    model_columns = X.columns.tolist()
    joblib.dump(model_columns, os.path.join(MODEL_OUTPUT_DIR, COLUMNS_FILE))
    print(f"Saved model columns to '{COLUMNS_FILE}'. Total features: {len(model_columns)}")

    # --- 3. Split Data ---
    print("\n--- Splitting data into training and testing sets (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # --- 4. Train Quantile Models ---
    print("\n--- Training XGBoost models for each quantile ---")
    models = {}
    for q in QUANTILES:
        print(f"Training model for quantile: {q}...")
        
        # Initialize the XGBoost regressor with the quantile objective.
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
            early_stopping_rounds=50 # Stops training if performance on validation set doesn't improve.
        )
        
        # Train the model.
        model.fit(X_train, y_train, 
                  eval_set=[(X_test, y_test)], 
                  verbose=False)
        
        # Save the trained model to disk.
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
    
    # Calculate key performance metrics.
    mae = mean_absolute_error(y_test, y_pred_median)
    mape = np.mean(np.abs((y_test - y_pred_median) / y_test)) * 100
    # Check how often the true price falls within our predicted 10%-90% interval.
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