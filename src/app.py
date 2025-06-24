# src/app.py
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from datetime import datetime
import config
import os
import tempfile
import logging

# Import Azure utilities if in cloud
if config.IS_CLOUD:
    import azure_utils
    from opencensus.ext.azure.log_exporter import AzureLogHandler

# --- Page Configuration ---
st.set_page_config(page_title="Bostadsvärdering", page_icon="🏠", layout="centered")

# --- Setup Application Insights Logger ---
logger = logging.getLogger(__name__)
if config.IS_CLOUD and os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    handler = AzureLogHandler(connection_string=os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Azure Application Insights logger configured for Streamlit app.")
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Local logger configured. Azure logging is disabled.")


# --- Helper Functions ---
@st.cache_resource
def load_models_and_columns():
    """
    Loads artifacts from Azure Blob Storage (if in cloud) or
    local disk (if running locally).
    """
    if config.IS_CLOUD:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                print("CLOUD MODE: Downloading models and artifacts from Azure.")
                # Download and load models
                models = {}
                for name, _ in config.MODEL_PATHS.items():
                    model_filename = config.MODEL_PATHS[name].name
                    local_path = os.path.join(temp_dir, model_filename)
                    azure_utils.download_file_from_blob(config.AZURE_MODELS_CONTAINER, model_filename, local_path)
                    models[name] = joblib.load(local_path)
                
                # Download and load other artifacts
                columns_path = os.path.join(temp_dir, config.MODEL_COLUMNS_FILE)
                azure_utils.download_file_from_blob(config.AZURE_MODELS_CONTAINER, config.MODEL_COLUMNS_FILE, columns_path)
                model_columns = joblib.load(columns_path)

                explainer_path = os.path.join(temp_dir, config.SHAP_EXPLAINER_FILE)
                azure_utils.download_file_from_blob(config.AZURE_MODELS_CONTAINER, config.SHAP_EXPLAINER_FILE, explainer_path)
                explainer = joblib.load(explainer_path)
                
                print("All artifacts successfully downloaded and loaded from Azure.")
                return models, model_columns, explainer
        except Exception as e:
            st.error(f"Ett fel uppstod vid laddning av filer från Azure: {e}")
            logger.error(f"Failed to load artifacts from Azure: {e}", exc_info=True)
            return None, None, None
    else: # Local mode
        try:
            print("LOCAL MODE: Loading models and artifacts from local disk.")
            models = {name: joblib.load(path) for name, path in config.MODEL_PATHS.items()}
            model_columns = joblib.load(config.MODEL_COLUMNS_PATH)
            explainer = joblib.load(config.SHAP_EXPLAINER_PATH)
            return models, model_columns, explainer
        except FileNotFoundError as e:
            st.error(f"En modell-, kolumn- eller SHAP-fil kunde inte hittas. Kör 'src/train.py'. Fel: {e}")
            return None, None, None
        except Exception as e:
            st.error(f"Ett oväntat fel uppstod vid laddning av modellfiler: {e}")
            return None, None, None

@st.cache_data
def load_location_options():
    """Loads location options from Azure or local disk."""
    if config.IS_CLOUD:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = os.path.join(temp_dir, config.LOCATION_OPTIONS_FILE)
                azure_utils.download_file_from_blob(config.AZURE_MODELS_CONTAINER, config.LOCATION_OPTIONS_FILE, local_path)
                with open(local_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Kunde inte ladda områdesalternativ från Azure: {e}")
            return []
    else: # Local mode
        try:
            with open(config.LOCATION_OPTIONS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            st.error("Filen med områden hittades inte. Kör 'src/train.py'.")
            return []
        except json.JSONDecodeError:
            st.error(f"Filen '{config.LOCATION_OPTIONS_FILE}' är korrupt eller inte en giltig JSON-fil.")
            return []

@st.cache_data
def load_location_price_map():
    """Loads location price map from Azure or local disk."""
    if config.IS_CLOUD:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = os.path.join(temp_dir, config.LOCATION_PRICE_MAP_FILE)
                azure_utils.download_file_from_blob(config.AZURE_MODELS_CONTAINER, config.LOCATION_PRICE_MAP_FILE, local_path)
                with open(local_path, 'r', encoding='utf-8') as f:
                    price_map_dict = json.load(f)
        except Exception as e:
            st.error(f"Kunde inte ladda pris-mappning från Azure: {e}")
            return None, None
    else: # Local mode
        try:
            with open(config.LOCATION_PRICE_MAP_PATH, 'r', encoding='utf-8') as f:
                price_map_dict = json.load(f)
        except FileNotFoundError:
            st.error("Filen med pris-mappning hittades inte. Kör 'src/train.py'.")
            return None, None
        except json.JSONDecodeError:
            st.error(f"Filen '{config.LOCATION_PRICE_MAP_FILE}' är korrupt eller inte en giltig JSON-fil.")
            return None, None
        except Exception as e:
            st.error(f"Ett oväntat fel uppstod vid laddning av pris-mappning: {e}")
            return None, None
            
    price_map = pd.Series(price_map_dict)
    fallback_price = price_map.median()
    return price_map, fallback_price


def make_prediction(input_data: dict, models: dict, model_columns: list, price_map: pd.Series, fallback_price: float) -> tuple[dict, pd.DataFrame]:
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
    return predictions, df_aligned 

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
    models, model_columns, explainer = load_models_and_columns()
    location_options = load_location_options()
    location_price_map, fallback_price = load_location_price_map()

    if all([models, model_columns, explainer, location_options, location_price_map is not None]):
        st.sidebar.header("Ange bostadens egenskaper")
        living_area = st.sidebar.number_input("Boarea (m²)", min_value=30, max_value=500, value=120, step=5)
        plot_area = st.sidebar.number_input("Tomtarea (m²)", min_value=100, max_value=10000, value=800, step=50)
        rooms = st.sidebar.number_input("Antal rum", min_value=1, max_value=20, value=5, step=1)
        try:
            default_location_index = location_options.index('Other')
        except (ValueError, IndexError):
            default_location_index = 0
        location_area = st.sidebar.selectbox("Område", options=location_options, index=default_location_index)
        non_living_area = st.sidebar.number_input("Biarea (m²)", min_value=0, max_value=300, value=20, step=5)
        sale_date = st.sidebar.date_input("Uppskattat försäljningsdatum", value=datetime.today())

        if st.sidebar.button("Värdera Bostad", type="primary", use_container_width=True):
            input_data = {
                'living_area_m2': living_area, 'rooms': rooms, 'plot_area_m2': plot_area,
                'non_living_area_m2': non_living_area, 'location_area': location_area,
                'sale_date': sale_date
            }
            # Log the prediction event
            log_payload = {'input_features': {k: str(v) for k, v in input_data.items()}}
            logger.info("PredictionRequested", extra={'custom_dimensions': log_payload})

            predictions, df_aligned = make_prediction(
                input_data, models, model_columns, location_price_map, fallback_price
            )

            # Log the result
            logger.info("PredictionMade", extra={'custom_dimensions': {'predictions': predictions}})

            st.subheader("Beräknad Värdering")
            
            col1, col2, col3 = st.columns(3)
            median_price_str = f"{predictions['median']:,}".replace(",", " ")
            lower_price_str = f"{predictions['lower']:,}".replace(",", " ")
            upper_price_str = f"{predictions['upper']:,}".replace(",", " ")

            col1.metric("Lägre estimat (5%)", f"{lower_price_str} kr")
            col2.metric("Median-värdering (50%)", f"{median_price_str} kr")
            col3.metric("Högre estimat (95%)", f"{upper_price_str} kr")

            with st.expander("Varför detta pris? Klicka för att se detaljerna."):
                base_price = int(explainer.expected_value)

                st.markdown(f"""
                Nedan ser du hur de olika egenskaperna du angett påverkar den slutgiltiga värderingen.
                Utgångspunkten är en genomsnittlig bostad i datamängden, som värderas till **{base_price:,} kr**.
                """.replace(",", " "))

                # Calculate SHAP values for the single prediction
                shap_values_raw = explainer.shap_values(df_aligned)[0]
                shap_df = pd.DataFrame(
                    data=shap_values_raw,
                    index=df_aligned.columns,
                    columns=['shap_value']
                )

                # --- Aggregate SHAP values for user-friendly features ---
                aggregated_shaps = {}

                # 1. Location (combines one-hot encoding and target encoding effects)
                location_cols = [c for c in shap_df.index if c.startswith('location_area_')]
                location_shap = shap_df.loc[location_cols].sum().shap_value
                if 'location_median_price_per_m2' in shap_df.index:
                    location_shap += shap_df.loc['location_median_price_per_m2'].shap_value
                aggregated_shaps['Område'] = location_shap

                # 2. Living Area (combines raw and log-transformed effects)
                living_area_shap = 0
                if 'living_area_m2' in shap_df.index:
                    living_area_shap += shap_df.loc['living_area_m2'].shap_value
                if 'log_living_area' in shap_df.index:
                    living_area_shap += shap_df.loc['log_living_area'].shap_value
                aggregated_shaps['Boarea'] = living_area_shap

                # 3. Plot Area (combines raw and log-transformed effects)
                plot_area_shap = 0
                if 'plot_area_m2' in shap_df.index:
                    plot_area_shap += shap_df.loc['plot_area_m2'].shap_value
                if 'log_plot_area' in shap_df.index:
                    plot_area_shap += shap_df.loc['log_plot_area'].shap_value
                aggregated_shaps['Tomtarea'] = plot_area_shap

                # 4. Simple passthrough features
                simple_features_map = {
                    'rooms': 'Antal rum',
                    'non_living_area_m2': 'Biarea',
                    'sale_days_since_epoch': 'Försäljningsdatum'
                }
                for feature, name in simple_features_map.items():
                    if feature in shap_df.index:
                        aggregated_shaps[name] = shap_df.loc[feature].shap_value

                # 5. Other engineered features (shown if they have a significant impact)
                other_features_map = {
                    'total_area_m2': 'Totalarea (Boarea + Biarea)',
                    'plot_to_living_ratio': 'Förhållande Tomt / Boarea'
                }
                for feature, name in other_features_map.items():
                    if feature in shap_df.index:
                        aggregated_shaps[name] = shap_df.loc[feature].shap_value

                # --- Display the results in a clear, two-column layout ---
                shap_summary = pd.Series(aggregated_shaps)
                # Filter out insignificant values (e.g., under 1,000 SEK) to keep the display clean
                shap_summary = shap_summary[abs(shap_summary) > 1000]

                positive_shaps = shap_summary[shap_summary > 0].sort_values(ascending=False)
                negative_shaps = shap_summary[shap_summary < 0].sort_values(ascending=True)

                # Helper function to format the output with colors and currency
                def format_shap_value(value):
                    prefix = '+' if value > 0 else ''
                    color = 'green' if value > 0 else 'red'
                    # Use a non-breaking space for thousands separator for better HTML rendering
                    formatted_value = f"{int(value):,}".replace(",", " ")
                    return f"<span style='font-weight:bold; color:{color};'>{prefix}{formatted_value} kr</span>"

                col1_exp, col2_exp = st.columns(2)

                with col1_exp:
                    col1_exp.markdown("<h5>Faktorer som höjer priset <span style='color:green;'>▲</span></h5>", unsafe_allow_html=True)
                    if not positive_shaps.empty:
                        for feature, value in positive_shaps.items():
                            col1_exp.markdown(f"**{feature}:** {format_shap_value(value)}", unsafe_allow_html=True)
                    else:
                        col1_exp.info("Inga betydande faktorer höjde priset.", icon="ℹ️")

                with col2_exp:
                    col2_exp.markdown("<h5>Faktorer som sänker priset <span style='color:red;'>▼</span></h5>", unsafe_allow_html=True)
                    if not negative_shaps.empty:
                        for feature, value in negative_shaps.items():
                            col2_exp.markdown(f"**{feature}:** {format_shap_value(value)}", unsafe_allow_html=True)
                    else:
                        col2_exp.info("Inga betydande faktorer sänkte priset.", icon="ℹ️")

                st.markdown("---")

                # Final summary calculation to show transparency
                final_price_check = base_price + shap_summary.sum()
                total_adjustment = shap_summary.sum()
                adjustment_sign = '+' if total_adjustment >= 0 else ''

                st.markdown(
                    f"""
                    <div style="text-align: center; font-size: 1.1em; background-color: #f0f2f6; padding: 15px; border-radius: 7px; color: #31333F;">
                        {base_price:,} kr <span style="color: #606770;">(Baspris)</span>
                        {adjustment_sign} {int(total_adjustment):,} kr <span style="color: #606770;">(Justering)</span>
                        = <strong>{int(final_price_check):,} kr</strong>
                    </div>
                    """.replace(",", " "), unsafe_allow_html=True)

                st.caption(
                    "Summan ovan kan skilja sig något från median-värderingen p.g.a. avrundning och att "
                    "faktorer med mycket liten påverkan (under 1 000 kr) ignorerats för tydlighetens skull."
                )

    # Display warnings if artifacts are missing, guiding the user.
    else:
        st.warning(
            "Vissa modellfiler saknas eller kunde inte laddas. "
            "Kör `src/train.py` (den senaste versionen) för att generera alla nödvändiga artefakter i mappen `models/`. "
            "Kontrollera eventuella felmeddelanden ovan."
        )

if __name__ == '__main__':
    main()