import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import numpy as np # Import numpy for calculations

# --- Configuration ---
CURRENT_DIR = os.path.dirname(__file__)
PROCESSED_DATA_PATH = os.path.join(CURRENT_DIR, '..', 'data', 'processed', 'hemnet_sold_villas_processed.csv')
MODEL_OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', 'models')
MODEL_FILE_PREFIX = "xgb_model"
COLUMNS_FILE = "model_columns.json"

# --- Model Parameters ---
## --- IMPROVEMENT 1: Add new engineered date features to the list --- ##
FEATURES = [
    'living_area_m2', 
    'rooms', 
    'plot_area_m2', 
    'non_living_area_m2',
    'location_area',
    # New date-based features
    'sale_year',
    'sale_month',
    'sale_dayofyear',
]
TARGET = 'final_price'
QUANTILES = [0.10, 0.5, 0.90] 

def train_model():
    """
    Loads processed data, engineers features, trains XGBoost quantile regression models,
    and saves them along with the feature columns.
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

    # --- 2. Feature Engineering & Preparation (with Debugging) ---
    print("\n--- Engineering New Features & Preparing Data ---")
    print(f"Initial rows: {len(df)}")
    
    # Let's see the initial state of missing values
    print("\n--- Initial Missing Value Counts ---")
    print(df.isnull().sum())
    print("-" * 35)

    # Handle numeric missing values
    df['plot_area_m2'] = df['plot_area_m2'].fillna(df['plot_area_m2'].median())
    df['non_living_area_m2'] = df['non_living_area_m2'].fillna(0)
    print(f"Rows after filling plot/non-living area: {len(df)}")


    df['sold_date'] = pd.to_datetime(df['sold_date'], errors='coerce')
    print(f"Rows after converting sold_date to datetime: {len(df)}")
    
    print(f"Number of rows with invalid dates (NaT): {df['sold_date'].isnull().sum()}")

    critical_cols = ['living_area_m2', 'final_price', 'sold_date']
    df.dropna(subset=critical_cols, inplace=True)
    print(f"Rows after dropping NaNs in {critical_cols}: {len(df)}")
    
    # If we have 0 rows here, the script will fail. Let's add a check.
    if df.empty:
        print("\nERROR: DataFrame is empty after cleaning. This means all rows had missing critical data.")
        print("Please check the 'Initial Missing Value Counts' above. A large number of NaNs in")
        print("'living_area_m2', 'final_price', or especially 'sold_date' is the likely cause.")
        return # Exit gracefully

    # --- End of critical section ---

    # Engineer new features from the clean data
    df['price_per_m2'] = (df['final_price'] / df['living_area_m2']).round(2)
    df['sale_year'] = df['sold_date'].dt.year
    df['sale_month'] = df['sold_date'].dt.month
    df['sale_dayofyear'] = df['sold_date'].dt.dayofyear
    print("Successfully engineered date and price features.")
    
    # Consolidate rare categorical features ('location_area')
    location_counts = df['location_area'].value_counts()
    rare_locations = location_counts[location_counts < 10].index.tolist()
    
    if rare_locations:
        df['location_area'] = df['location_area'].replace(rare_locations, 'Other')
        print(f"Consolidated {len(rare_locations)} rare locations into 'Other' category.")

    # Define X and y
    X = df[FEATURES]
    y = df[TARGET]

    # One-hot encode
    X = pd.get_dummies(X, columns=['location_area'], dummy_na=False)
    
    model_columns = X.columns.tolist()
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
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
        
        model.fit(X_train, y_train, 
                  eval_set=[(X_test, y_test)], 
                  verbose=False)
        
        model_filename = f"{MODEL_FILE_PREFIX}_q{int(q*100)}.joblib"
        model_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"  > Model saved to '{model_path}'")
        models[q] = model

    # --- 5. Evaluate Models ---
    print("\n--- Evaluating Model Performance on Test Set ---")
    median_model = models[0.5]
    y_pred = median_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f"Median Prediction MAE: {mae:,.0f} kr".replace(',', ' '))
    print(f"Median Prediction MAPE: {mape:.2f}%")

    lower_pred = models[QUANTILES[0]].predict(X_test)
    upper_pred = models[QUANTILES[2]].predict(X_test)

    within_interval = ((y_test >= lower_pred) & (y_test <= upper_pred)).mean()
    target_coverage = QUANTILES[2] - QUANTILES[0] 
    print(f"Coverage ({QUANTILES[0]}-{QUANTILES[2]} interval): {within_interval:.2%} (Target: {target_coverage:.0%})")

    within_15_percent = (abs(y_pred - y_test) / y_test <= 0.15).mean()
    print(f"Predictions within ±15% of actual price: {within_15_percent:.2%}")

    print("\n--- Training complete! ---")
    
if __name__ == '__main__':
    train_model()