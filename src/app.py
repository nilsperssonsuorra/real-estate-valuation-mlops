# src/app.py
import streamlit as st
import pandas as pd
import joblib
import os
import json
from datetime import datetime

# --- Page Configuration ---
# Sets the title, icon, and layout for the Streamlit page. This should be the first Streamlit command.
st.set_page_config(
    page_title="Bostadsvärdering",
    page_icon="🏠",
    layout="centered"
)

# --- Path Configuration ---
# Get the absolute path of the directory where this script is located (e.g., .../project/src).
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'models' directory by going up one level from 'src'.
# This makes the script runnable from any location.
MODEL_DIR = os.path.join(APP_DIR, '..', 'models')

# Define paths to all required model and artifact files.
MODEL_PATHS = {
    'lower': os.path.join(MODEL_DIR, 'xgb_model_q10.joblib'),   # Predicts the 10th percentile price
    'median': os.path.join(MODEL_DIR, 'xgb_model_q50.joblib'), # Predicts the 50th percentile (median) price
    'upper': os.path.join(MODEL_DIR, 'xgb_model_q90.joblib')    # Predicts the 90th percentile price
}
# Path to the list of feature columns the model was trained on. Saved with joblib.
COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.json')
# Path to the list of valid location areas for the dropdown. Saved as a standard JSON file.
LOCATION_COLUMNS_PATH = os.path.join(MODEL_DIR, 'location_area_columns.json')


# --- Helper Functions ---

@st.cache_resource
def load_models_and_columns():
    """
    Loads the trained XGBoost models and the list of feature columns from disk.
    Uses @st.cache_resource to ensure these large objects are loaded only once.
    """
    try:
        # Load the dictionary of models.
        models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
        # The list of model columns was also saved using joblib, so we load it the same way.
        model_columns = joblib.load(COLUMNS_PATH)
        return models, model_columns
    except FileNotFoundError as e:
        st.error(
            "Ett fel uppstod: En modell- eller kolumnfil kunde inte hittas. "
            f"Kontrollera att filerna finns i mappen '{MODEL_DIR}'. "
            "Se till att du har kört 'src/train.py' för att skapa alla nödvändiga filer. "
            f"Specifikt fel: {e}"
        )
        return None, None
    except Exception as e:
        st.error(f"Ett oväntat fel uppstod vid laddning av modellfiler: {e}")
        return None, None

@st.cache_data
def load_location_options():
    """
    Loads the list of unique 'location_area' values from its JSON file.
    Uses @st.cache_data for efficient caching of serializable data (like a list).
    """
    try:
        # This file is a standard JSON, so we load it with the json library.
        with open(LOCATION_COLUMNS_PATH, 'r', encoding='utf-8') as f:
            locations = json.load(f)
        return locations
    except FileNotFoundError:
        st.error(
            f"Ett fel uppstod: Filen med områden ('{os.path.basename(LOCATION_COLUMNS_PATH)}') hittades inte. "
            f"Sökväg: '{LOCATION_COLUMNS_PATH}'. Kör 'src/train.py' för att skapa den."
        )
        return []
    except json.JSONDecodeError:
        st.error(
            f"Ett fel uppstod: Filen '{os.path.basename(LOCATION_COLUMNS_PATH)}' är inte en giltig JSON-fil. "
            "Kör 'src/train.py' igen för att återskapa den korrekt."
        )
        return []

def make_prediction(input_data: dict, models: dict, model_columns: list) -> dict:
    """
    Prepares user input, creates predictions using the loaded models, and returns them.

    Args:
        input_data: A dictionary containing the user's input from the sidebar.
        models: The dictionary of loaded XGBoost models.
        model_columns: The list of feature names the models expect.

    Returns:
        A dictionary with 'lower', 'median', and 'upper' price predictions.
    """
    # 1. Create a DataFrame from the single input dictionary.
    df = pd.DataFrame([input_data])
    
    # 2. Engineer date features, same as in the training script.
    df['sold_date'] = pd.to_datetime(df['sale_date'])
    df['sale_year'] = df['sold_date'].dt.year
    df['sale_month'] = df['sold_date'].dt.month
    df['sale_dayofyear'] = df['sold_date'].dt.dayofyear
    df = df.drop(columns=['sale_date', 'sold_date'])
    
    # 3. One-hot encode the categorical 'location_area' feature.
    df = pd.get_dummies(df, columns=['location_area'])
    
    # 4. Align the DataFrame columns with the original model's columns.
    # This is a crucial step: it adds any missing one-hot encoded columns (with a value of 0)
    # and ensures the column order matches exactly what the model was trained on.
    df_aligned = df.reindex(columns=model_columns, fill_value=0)
    
    # 5. Make predictions for each quantile model.
    predictions = {}
    for name, model in models.items():
        pred_value = model.predict(df_aligned)[0]
        predictions[name] = int(pred_value) # Convert to integer for clean display
    return predictions

# --- Streamlit UI ---

st.title("🏠 Automatisk Bostadsvärdering")
st.markdown(
    """
    Skriv in egenskaperna för en villa för att få en prisuppskattning.
    Modellen är tränad på 1600+ försäljningar från Hemnet i Uppsala och använder kvantilregression
    för att ge ett troligt prisintervall.
    """
)

# Load all necessary artifacts before building the UI.
models, model_columns = load_models_and_columns()
location_options = load_location_options()

# Only proceed to build the main UI if all artifacts were loaded successfully.
if models and model_columns and location_options:
    st.sidebar.header("Ange bostadens egenskaper")

    # --- User Input Fields ---
    living_area = st.sidebar.number_input("Boarea (m²)", min_value=30, max_value=500, value=120, step=5)
    plot_area = st.sidebar.number_input("Tomtarea (m²)", min_value=100, max_value=10000, value=800, step=50)
    rooms = st.sidebar.number_input("Antal rum", min_value=1, max_value=20, value=5, step=1)
    
    # Set a default location for a better user experience. Fallback to the first item if not found.
    try:
        default_location_index = location_options.index('Täby')
    except (ValueError, IndexError):
        default_location_index = 0
    
    location_area = st.sidebar.selectbox("Område", options=location_options, index=default_location_index)
    
    non_living_area = st.sidebar.number_input("Biarea (m²)", min_value=0, max_value=300, value=20, step=5)
    sale_date = st.sidebar.date_input("Uppskattat försäljningsdatum", value=datetime.today())

    # --- Prediction Trigger ---
    if st.sidebar.button("Värdera Bostad", type="primary", use_container_width=True):
        input_data = {
            'living_area_m2': living_area,
            'rooms': rooms,
            'plot_area_m2': plot_area,
            'non_living_area_m2': non_living_area,
            'location_area': location_area,
            'sale_date': sale_date
        }
        predictions = make_prediction(input_data, models, model_columns)
        
        st.subheader("Beräknad Värdering")
        
        # Display predictions in a 3-column layout.
        col1, col2, col3 = st.columns(3)
        # Format numbers with spaces as thousand separators for Swedish locale.
        median_price_str = f"{predictions['median']:,}".replace(",", " ")
        lower_price_str = f"{predictions['lower']:,}".replace(",", " ")
        upper_price_str = f"{predictions['upper']:,}".replace(",", " ")

        col1.metric("Lägre estimat (10%)", f"{lower_price_str} kr")
        col2.metric("Median-värdering (50%)", f"{median_price_str} kr")
        col3.metric("Högre estimat (90%)", f"{upper_price_str} kr")

# Display warnings if artifacts are missing, guiding the user to run the training script.
elif not models or not model_columns:
    st.warning("Väntar på att modellfilerna ska skapas. Kör `src/train.py` för att generera dem.")
elif not location_options:
     st.warning("Väntar på att områdesfilen ska skapas. Kör `src/train.py` för att generera den.")