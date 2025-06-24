# src/train.py
import pandas as pd
import xgboost as xgb
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import config
import shap
import azure_utils

def run_training_pipeline():
    """
    Loads processed data, trains models, and saves all artifacts
    either locally or to Azure Blob Storage depending on the environment.
    """
    print("--- Starting Model Training ---")

    # --- 1. Load Data ---
    df = pd.DataFrame()
    if config.IS_CLOUD:
        print("--- CLOUD MODE: Loading processed data from Azure Blob Storage ---")
        df = azure_utils.download_df_from_blob(
            config.AZURE_PROCESSED_DATA_CONTAINER, config.PROCESSED_DATA_BLOB_NAME
        )
    else:
        print(f"--- LOCAL MODE: Loading processed data from '{config.PROCESSED_DATA_PATH}' ---")
        try:
            df = pd.read_csv(config.PROCESSED_DATA_PATH)
        except FileNotFoundError:
            print(f"ERROR: Processed data file not found at '{config.PROCESSED_DATA_PATH}'.")
            print("Please run the cleaning script (src/clean.py) first.")
            return

    if df.empty:
        print("Processed data is empty. Exiting training.")
        return
    print(f"Successfully loaded processed data with {len(df)} rows.")

    # --- 2. Feature Engineering & Preparation ---
    print("\n--- Engineering New Features & Preparing Data ---")
    df['plot_area_m2'] = df['plot_area_m2'].fillna(df['plot_area_m2'].median())
    df['non_living_area_m2'] = df['non_living_area_m2'].fillna(0)
    df['sold_date'] = pd.to_datetime(df['sold_date'], errors='coerce')
    critical_cols = ['living_area_m2', 'final_price', 'sold_date']
    df.dropna(subset=critical_cols, inplace=True)
    df['total_area_m2'] = df['living_area_m2'] + df['non_living_area_m2']
    df['plot_to_living_ratio'] = df['plot_area_m2'] / (df['living_area_m2'] + 1e-6)
    df['sale_days_since_epoch'] = (df['sold_date'] - config.FEATURE_ENGINEERING_EPOCH).dt.days
    df['log_living_area'] = np.log1p(df['living_area_m2'])
    df['log_plot_area'] = np.log1p(df['plot_area_m2'])
    location_counts = df['location_area'].value_counts()
    rare_locations = location_counts[location_counts < 10].index.tolist()
    if rare_locations:
        df['location_area'] = df['location_area'].replace(rare_locations, 'Other')
        print(f"Consolidated {len(rare_locations)} rare locations into 'Other'.")
    X_full = df[config.BASE_FEATURES + [config.TARGET_VARIABLE]]
    y_full = df[config.TARGET_VARIABLE]
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )
    X_train_full = X_train_full.copy()
    X_test_full = X_test_full.copy()
    price_per_m2_map = X_train_full.groupby('location_area')['final_price'].median() / X_train_full.groupby('location_area')['living_area_m2'].median()
    X_train_full['location_median_price_per_m2'] = X_train_full['location_area'].map(price_per_m2_map)
    X_test_full['location_median_price_per_m2'] = X_test_full['location_area'].map(price_per_m2_map)
    global_median_price = X_train_full['location_median_price_per_m2'].median()
    X_train_full['location_median_price_per_m2'] = X_train_full['location_median_price_per_m2'].fillna(global_median_price)
    X_test_full['location_median_price_per_m2'] = X_test_full['location_median_price_per_m2'].fillna(global_median_price)

    # --- 3. Prepare Final Data & Artifacts ---
    price_map_dict = price_per_m2_map.to_dict()
    location_area_list = sorted(list(df['location_area'].unique()))
    final_features = config.BASE_FEATURES + ['location_median_price_per_m2']
    X_train = X_train_full[final_features]
    X_test = X_test_full[final_features]
    X_train = pd.get_dummies(X_train, columns=['location_area'], dummy_na=False)
    X_test = pd.get_dummies(X_test, columns=['location_area'], dummy_na=False)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    model_columns = X_train.columns.tolist()
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # --- 4. Train Quantile Models ---
    print("\n--- Training XGBoost models for each quantile ---")
    models = {}
    for name, q in config.QUANTILES_AND_NAMES.items():
        print(f"Training model for quantile: {q} (name: {name})...")
        model = xgb.XGBRegressor(
            objective='reg:quantileerror', quantile_alpha=q, n_estimators=2000,
            learning_rate=0.03, max_depth=4, subsample=0.7, colsample_bytree=0.7,
            random_state=42, n_jobs=-1, early_stopping_rounds=50
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        models[name] = model

    # --- CREATE SHAP EXPLAINER ---
    print("\n--- Creating SHAP explainer ---")
    median_model = models['median']
    explainer = shap.TreeExplainer(median_model)
    
    # --- 5. Save All Artifacts ---
    if config.IS_CLOUD:
        print("\n--- CLOUD MODE: Uploading all artifacts to Azure Blob Storage ---")
        azure_utils.upload_json_to_blob(price_map_dict, config.AZURE_MODELS_CONTAINER, config.LOCATION_PRICE_MAP_FILE)
        azure_utils.upload_json_to_blob(location_area_list, config.AZURE_MODELS_CONTAINER, config.LOCATION_OPTIONS_FILE)
        azure_utils.upload_joblib_to_blob(model_columns, config.AZURE_MODELS_CONTAINER, config.MODEL_COLUMNS_FILE)
        azure_utils.upload_joblib_to_blob(explainer, config.AZURE_MODELS_CONTAINER, config.SHAP_EXPLAINER_FILE)
        for name, model in models.items():
            model_filename = config.MODEL_PATHS[name].name
            azure_utils.upload_joblib_to_blob(model, config.AZURE_MODELS_CONTAINER, model_filename)
    else:
        print(f"\n--- LOCAL MODE: Saving all artifacts to '{config.MODELS_DIR}' ---")
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        with open(config.LOCATION_PRICE_MAP_PATH, 'w', encoding='utf-8') as f:
            json.dump(price_map_dict, f, ensure_ascii=False, indent=4)
        with open(config.LOCATION_OPTIONS_PATH, 'w', encoding='utf-8') as f:
            json.dump(location_area_list, f, ensure_ascii=False, indent=4)
        joblib.dump(model_columns, config.MODEL_COLUMNS_PATH)
        joblib.dump(explainer, config.SHAP_EXPLAINER_PATH)
        for name, model in models.items():
            joblib.dump(model, config.MODEL_PATHS[name])
        print("All artifacts saved locally.")

    # --- 5. Evaluate Models on Test Set ---
    print("\n--- Evaluating Model Performance on Test Set ---")
    y_pred_lower = models['lower'].predict(X_test)
    y_pred_median = models['median'].predict(X_test)
    y_pred_upper = models['upper'].predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_median)
    mape = np.mean(np.abs((y_test - y_pred_median) / y_test)) * 100
    within_interval = ((y_test >= y_pred_lower) & (y_test <= y_pred_upper)).mean()
    within_15_percent = (abs(y_pred_median - y_test) / y_test <= 0.15).mean()
    target_coverage = config.QUANTILES_AND_NAMES['upper'] - config.QUANTILES_AND_NAMES['lower']
    print(f"Median Prediction MAE (Mean Absolute Error): {mae:,.0f} kr".replace(',', ' '))
    print(f"Median Prediction MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"Coverage (actual price within prediction interval): {within_interval:.2%} (Target: {target_coverage:.0%})")
    print(f"Accuracy (predictions within ±15% of actual price): {within_15_percent:.2%}")
    print("\n--- Training complete! ---")

if __name__ == '__main__':
    run_training_pipeline()