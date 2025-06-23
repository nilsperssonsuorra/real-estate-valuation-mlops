# src/app.py
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from datetime import datetime
import config  # Import the new config file

# --- Page Configuration ---
st.set_page_config(
    page_title="Bostadsvärdering",
    page_icon="🏠",
    layout="centered"
)

# --- Helper Functions ---

@st.cache_resource
def load_models_and_columns():
    """
    Loads the trained XGBoost models and the list of feature columns from disk.
    Uses @st.cache_resource to ensure these large objects are loaded only once.
    """
    try:
        # Use the paths from the config file
        models = {name: joblib.load(path) for name, path in config.MODEL_PATHS.items()}
        model_columns = joblib.load(config.MODEL_COLUMNS_PATH)
        return models, model_columns
    except FileNotFoundError as e:
        st.error(
            "Ett fel uppstod: En modell- eller kolumnfil kunde inte hittas. "
            f"Kontrollera att filerna finns i mappen '{config.MODELS_DIR}'. "
            "Se till att du har kört den senaste versionen av 'src/train.py' för att skapa alla nödvändiga filer. "
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
    """
    try:
        # Use the path from the config file
        with open(config.LOCATION_OPTIONS_PATH, 'r', encoding='utf-8') as f:
            locations = json.load(f)
        return locations
    except FileNotFoundError:
        st.error(
            f"Ett fel uppstod: Filen med områden ('{config.LOCATION_OPTIONS_FILE}') hittades inte. "
            "Kör 'src/train.py' för att skapa den."
        )
        return []
    except json.JSONDecodeError:
        st.error(f"Filen '{config.LOCATION_OPTIONS_FILE}' är inte en giltig JSON-fil.")
        return []

@st.cache_data
def load_location_price_map():
    """
    Loads the location-to-price-per-m2 map.
    Also calculates a global median to use as a fallback for any unseen locations.
    """
    try:
        # Use the path from the config file
        with open(config.LOCATION_PRICE_MAP_PATH, 'r', encoding='utf-8') as f:
            price_map_dict = json.load(f)
        
        # Convert to a Pandas Series for easier mapping
        price_map = pd.Series(price_map_dict)
        # Calculate a robust fallback value (median) for any location not in the map
        fallback_price = price_map.median()
        
        return price_map, fallback_price
    except FileNotFoundError:
        st.error(
            f"Ett fel uppstod: Filen med områdespriser ('{config.LOCATION_PRICE_MAP_FILE}') hittades inte. "
            "Kör 'src/train.py' för att skapa den."
        )
        return None, None
    except Exception as e:
        st.error(f"Ett oväntat fel uppstod vid laddning av pris-mappningsfilen: {e}")
        return None, None


def make_prediction(input_data: dict, models: dict, model_columns: list, price_map: pd.Series, fallback_price: float) -> dict:
    """
    Prepares user input using the exact same feature engineering pipeline from train3.py,
    creates predictions using the loaded models, and returns them.
    """
    # 1. Create a DataFrame from the single input dictionary.
    df = pd.DataFrame([input_data])
    
    # 2. Engineer features EXACTLY as in train3.py
    df['sold_date'] = pd.to_datetime(df['sale_date'])
    
    # Basic features
    df['total_area_m2'] = df['living_area_m2'] + df['non_living_area_m2']
    df['plot_to_living_ratio'] = df['plot_area_m2'] / (df['living_area_m2'] + 1e-6)
    # Use the epoch from the config file
    df['sale_days_since_epoch'] = (df['sold_date'] - config.FEATURE_ENGINEERING_EPOCH).dt.days

    # Log-transformed features
    df['log_living_area'] = np.log1p(df['living_area_m2'])
    df['log_plot_area'] = np.log1p(df['plot_area_m2'])
    
    # Target-encoded feature
    df['location_median_price_per_m2'] = df['location_area'].map(price_map).fillna(fallback_price)
    
    # Drop columns that are no longer needed
    df = df.drop(columns=['sale_date', 'sold_date'])
    
    # 3. One-hot encode the categorical 'location_area' feature.
    df = pd.get_dummies(df, columns=['location_area'])
    
    # 4. Align the DataFrame columns with the original model's columns.
    # This is the most crucial step: it ensures the column order and presence
    # matches exactly what the model was trained on.
    df_aligned = df.reindex(columns=model_columns, fill_value=0)
    
    # 5. Make predictions for each quantile model.
    predictions = {}
    for name, model in models.items():
        pred_value = model.predict(df_aligned)[0]
        predictions[name] = int(pred_value) # Convert to integer for clean display
    return predictions

# --- Streamlit UI ---

def main():
    """
    Defines and runs the Streamlit user interface for the application.
    This function orchestrates the loading of artifacts and the rendering of UI components.
    """
    st.title("🏠 Automatisk Bostadsvärdering")
    st.markdown(
        """
        Skriv in egenskaperna för en villa för att få en prisuppskattning.
        Modellen är tränad på 1600+ försäljningar från Hemnet i Uppsala och använder avancerad
        feature engineering och kvantilregression för att ge ett troligt prisintervall.
        """
    )

    # Load all necessary artifacts before building the UI.
    models, model_columns = load_models_and_columns()
    location_options = load_location_options()
    location_price_map, fallback_price = load_location_price_map()

    # Only proceed to build the main UI if all artifacts were loaded successfully.
    if all([models, model_columns, location_options, location_price_map is not None]):
        st.sidebar.header("Ange bostadens egenskaper")

        # --- User Input Fields ---
        living_area = st.sidebar.number_input("Boarea (m²)", min_value=30, max_value=500, value=120, step=5)
        plot_area = st.sidebar.number_input("Tomtarea (m²)", min_value=100, max_value=10000, value=800, step=50)
        rooms = st.sidebar.number_input("Antal rum", min_value=1, max_value=20, value=5, step=1)
        
        try:
            # 'Other' is a good default as it will use the fallback median price.
            default_location_index = location_options.index('Other')
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
            predictions = make_prediction(input_data, models, model_columns, location_price_map, fallback_price)
            
            st.subheader("Beräknad Värdering")
            
            col1, col2, col3 = st.columns(3)
            median_price_str = f"{predictions['median']:,}".replace(",", " ")
            lower_price_str = f"{predictions['lower']:,}".replace(",", " ")
            upper_price_str = f"{predictions['upper']:,}".replace(",", " ")

            # UPDATED labels to reflect 5% and 95% quantiles
            col1.metric("Lägre estimat (5%)", f"{lower_price_str} kr")
            col2.metric("Median-värdering (50%)", f"{median_price_str} kr")
            col3.metric("Högre estimat (95%)", f"{upper_price_str} kr")

    # Display warnings if artifacts are missing, guiding the user.
    else:
        st.warning(
            "Vissa modellfiler saknas eller kunde inte laddas. "
            "Kör `src/train.py` (den senaste versionen) för att generera alla nödvändiga artefakter i mappen `models/`. "
            "Kontrollera eventuella felmeddelanden ovan."
        )

if __name__ == '__main__':
    main()