# tests/test_app.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from datetime import date
import json
import importlib
import uuid
import logging # <-- Import logging to use its constants

# Import the functions to test from the Streamlit app
# We need to import the original function to access its __wrapped__ attribute for testing
from app import (
    main, make_prediction, load_models_and_columns, load_location_options,
    load_location_price_map
)
import config  # Import config for path constants

# A dictionary representing a single user input from the Streamlit UI
@pytest.fixture
def sample_input_data():
    return {
        'living_area_m2': 120,
        'rooms': 5,
        'plot_area_m2': 800,
        'non_living_area_m2': 20,
        'location_area': 'Uppsala_Centrum',
        'sale_date': date(2023, 10, 27)
    }

# Mocked model columns that the model was "trained" on
@pytest.fixture
def mock_model_columns():
    return [
        'living_area_m2', 'rooms', 'plot_area_m2', 'non_living_area_m2',
        'total_area_m2', 'plot_to_living_ratio', 'sale_days_since_epoch',
        'log_living_area', 'log_plot_area', 'location_median_price_per_m2',
        'location_area_Uppsala_Centrum', 'location_area_Annat_Omrade'
    ]

# Mocked location-to-price map
@pytest.fixture
def mock_price_map():
    return pd.Series({'Uppsala_Centrum': 50000, 'Annat_Omrade': 40000})

# This is the most important test for the app
def test_make_prediction(sample_input_data, mock_model_columns, mock_price_map):
    """
    Tests the full prediction pipeline from user input to final prediction.
    """
    # --- Mock Setup ---
    mock_model_lower = MagicMock()
    mock_model_lower.predict.return_value = np.array([4800000])
    mock_model_median = MagicMock()
    mock_model_median.predict.return_value = np.array([5000000])
    mock_model_upper = MagicMock()
    mock_model_upper.predict.return_value = np.array([5200000])
    mock_models = {
        'lower': mock_model_lower,
        'median': mock_model_median,
        'upper': mock_model_upper
    }
    fallback_price = 45000.0

    # --- Run the function ---
    predictions, df_aligned = make_prediction(
        sample_input_data, mock_models, mock_model_columns, mock_price_map, fallback_price
    )

    # --- Assertions ---
    assert isinstance(predictions, dict)
    assert predictions['lower'] == 4800000
    assert predictions['median'] == 5000000
    assert predictions['upper'] == 5200000
    df_passed_to_predict = mock_model_median.predict.call_args.args[0]
    pd.testing.assert_frame_equal(df_aligned, df_passed_to_predict)
    assert isinstance(df_aligned, pd.DataFrame)
    assert len(df_aligned) == 1
    assert list(df_aligned.columns) == mock_model_columns
    assert df_aligned['location_area_Uppsala_Centrum'].iloc[0] == 1

def test_make_prediction_unseen_location(mock_model_columns, mock_price_map):
    """
    Tests that an unseen location correctly uses the fallback price.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1000000])
    mock_models = {'lower': mock_model, 'median': mock_model, 'upper': mock_model}
    unseen_input_data = {
        'living_area_m2': 100, 'rooms': 4, 'plot_area_m2': 500,
        'non_living_area_m2': 10, 'location_area': 'Helt_Nytt_Område',
        'sale_date': date(2023, 10, 27)
    }
    fallback_price = 45000.0
    _predictions, df_aligned = make_prediction(
        unseen_input_data, mock_models, mock_model_columns, mock_price_map, fallback_price
    )
    assert df_aligned['location_median_price_per_m2'].iloc[0] == fallback_price

@patch('app.st')
@patch('app.joblib.load')
def test_load_models_and_columns_success(mock_joblib_load, mock_st):
    mock_joblib_load.side_effect = ["model_lower", "model_median", "model_upper", "columns_list", "explainer_obj"]
    models, columns, explainer = load_models_and_columns.__wrapped__()
    assert mock_joblib_load.call_count == 5
    assert 'median' in models
    assert columns == "columns_list"
    assert explainer == "explainer_obj"
    mock_st.error.assert_not_called()

@patch('app.st')
@patch('app.joblib.load', side_effect=FileNotFoundError("File not found!"))
def test_load_models_and_columns_file_not_found(mock_joblib_load, mock_st):
    models, columns, explainer = load_models_and_columns.__wrapped__()
    assert models is None
    assert columns is None
    assert explainer is None
    mock_st.error.assert_called_once()
    assert "En modell-, kolumn- eller SHAP-fil kunde inte hittas" in mock_st.error.call_args[0][0]

@patch('app.st')
@patch('app.joblib.load', side_effect=Exception("Unexpected error!"))
def test_load_models_and_columns_generic_exception(mock_joblib_load, mock_st):
    """Tests the failure case with a generic exception."""
    models, columns, explainer = load_models_and_columns.__wrapped__()
    assert models is None
    assert columns is None
    assert explainer is None
    mock_st.error.assert_called_once()
    assert "Ett oväntat fel uppstod" in mock_st.error.call_args[0][0]

@patch('app.st')
@patch("builtins.open", new_callable=mock_open, read_data='["Area1", "Area2"]')
@patch("app.json.load", return_value=["Area1", "Area2"])
def test_load_location_options_success(mock_json_load, mock_file_open, mock_st):
    locations = load_location_options.__wrapped__()
    assert locations == ["Area1", "Area2"]
    mock_file_open.assert_called_once_with(config.LOCATION_OPTIONS_PATH, 'r', encoding='utf-8')
    mock_st.error.assert_not_called()

@patch('app.st')
@patch("builtins.open", side_effect=FileNotFoundError)
def test_load_location_options_file_not_found(mock_file_open, mock_st):
    locations = load_location_options.__wrapped__()
    assert locations == []
    mock_st.error.assert_called_once()
    assert "hittades inte" in mock_st.error.call_args[0][0]

@patch('app.st')
@patch("builtins.open", new_callable=mock_open, read_data='invalid json')
@patch("app.json.load", side_effect=json.JSONDecodeError("msg", "doc", 0))
def test_load_location_options_json_error(mock_json_load, mock_file_open, mock_st):
    locations = load_location_options.__wrapped__()
    assert locations == []
    mock_st.error.assert_called_once()
    assert "inte en giltig JSON-fil" in mock_st.error.call_args[0][0]

@patch('app.st')
@patch("builtins.open", new_callable=mock_open, read_data='{"Area1": 50000, "Area2": 40000}')
@patch("app.json.load", return_value={"Area1": 50000, "Area2": 40000})
def test_load_location_price_map_success(mock_json_load, mock_file_open, mock_st):
    price_map, fallback = load_location_price_map.__wrapped__()
    assert isinstance(price_map, pd.Series)
    assert price_map['Area1'] == 50000
    assert fallback == 45000.0
    mock_st.error.assert_not_called()

@patch('app.st')
@patch("builtins.open", side_effect=FileNotFoundError)
def test_load_location_price_map_file_not_found(mock_file_open, mock_st):
    price_map, fallback = load_location_price_map.__wrapped__()
    assert price_map is None
    assert fallback is None
    mock_st.error.assert_called_once()
    assert "hittades inte" in mock_st.error.call_args[0][0]

@patch('app.st')
@patch("builtins.open", new_callable=mock_open, read_data='invalid json')
@patch("app.json.load", side_effect=json.JSONDecodeError("msg", "doc", 0))
def test_load_location_price_map_json_error(mock_json_load, mock_file_open, mock_st):
    price_map, fallback = load_location_price_map.__wrapped__()
    assert price_map is None
    assert fallback is None
    mock_st.error.assert_called_once()
    assert "inte en giltig JSON-fil" in mock_st.error.call_args[0][0]

@patch('app.st')
@patch("builtins.open", side_effect=Exception("Unexpected error!"))
def test_load_location_price_map_generic_exception(mock_file_open, mock_st):
    price_map, fallback = load_location_price_map.__wrapped__()
    assert price_map is None
    assert fallback is None
    mock_st.error.assert_called_once()
    assert "Ett oväntat fel uppstod" in mock_st.error.call_args[0][0]

@patch('app.st')
@patch('app.make_prediction')
@patch('app.load_location_price_map')
@patch('app.load_location_options')
@patch('app.load_models_and_columns')
def test_main_success_flow(mock_load_models, mock_load_loc_options, mock_load_price_map, mock_make_prediction, mock_st):
    """Tests the main UI flow on a successful run, covering all SHAP logic."""
    # --- Arrange Mocks ---
    test_model_columns = [
        'living_area_m2', 'log_living_area', 'plot_area_m2', 'log_plot_area',
        'rooms', 'non_living_area_m2', 'sale_days_since_epoch',
        'total_area_m2', 'plot_to_living_ratio',
        'location_median_price_per_m2', 'location_area_Uppsala_Centrum'
    ]
    test_shap_values = np.array([
        50000, 25000, -30000, -15000, 40000, -10000, 80000,
        5000, -2000, 150000, 75000
    ])

    mock_explainer = MagicMock()
    mock_explainer.expected_value = 3000000
    mock_explainer.shap_values.return_value = [test_shap_values]
    mock_load_models.return_value = ({"mock": "models"}, test_model_columns, mock_explainer)

    mock_load_loc_options.return_value = ["Uppsala_Centrum", "Other"]
    mock_load_price_map.return_value = (pd.Series(dtype='float64'), 45000.0)

    mock_df_aligned = pd.DataFrame([np.zeros(len(test_model_columns))], columns=test_model_columns)
    mock_make_prediction.return_value = ({'lower': 4800000, 'median': 5000000, 'upper': 5200000}, mock_df_aligned)

    mock_cols_metrics = (MagicMock(), MagicMock(), MagicMock())
    mock_cols_expander = (MagicMock(), MagicMock())
    mock_st.columns.side_effect = [mock_cols_metrics, mock_cols_expander]
    mock_st.sidebar.button.return_value = True

    # --- Act ---
    main()

    # --- Assert ---
    mock_make_prediction.assert_called_once()
    mock_st.expander.assert_called_once()
    mock_explainer.shap_values.assert_called_once_with(mock_df_aligned)
    mock_cols_expander[0].info.assert_not_called()
    mock_cols_expander[1].info.assert_not_called()
    assert mock_cols_expander[0].markdown.called
    assert mock_cols_expander[1].markdown.called

@patch('app.st')
@patch('app.make_prediction')
@patch('app.load_location_price_map')
@patch('app.load_location_options')
@patch('app.load_models_and_columns')
def test_main_shap_display_no_positives(mock_load_models, mock_load_loc_options, mock_load_price_map, mock_make_prediction, mock_st):
    """Tests the SHAP display when there are no significant positive factors."""
    test_model_columns = ['rooms']
    test_shap_values = np.array([-50000])

    mock_explainer = MagicMock()
    mock_explainer.expected_value = 3000000
    mock_explainer.shap_values.return_value = [test_shap_values]
    mock_load_models.return_value = ({"mock": "models"}, test_model_columns, mock_explainer)
    mock_load_loc_options.return_value = ["Other"]
    mock_load_price_map.return_value = (pd.Series(dtype='float64'), 45000.0)
    mock_df_aligned = pd.DataFrame([[0]], columns=test_model_columns)
    mock_make_prediction.return_value = ({'lower': 1, 'median': 2, 'upper': 3}, mock_df_aligned)
    mock_st.sidebar.button.return_value = True

    mock_cols_metrics = (MagicMock(), MagicMock(), MagicMock())
    mock_cols_expander = (MagicMock(), MagicMock())
    mock_st.columns.side_effect = [mock_cols_metrics, mock_cols_expander]

    main()

    mock_cols_expander[0].info.assert_called_once_with("Inga betydande faktorer höjde priset.", icon="ℹ️")

@patch('app.st')
@patch('app.make_prediction')
@patch('app.load_location_price_map')
@patch('app.load_location_options')
@patch('app.load_models_and_columns')
def test_main_shap_display_no_negatives(mock_load_models, mock_load_loc_options, mock_load_price_map, mock_make_prediction, mock_st):
    """Tests the SHAP display when there are no significant negative factors."""
    test_model_columns = ['rooms']
    test_shap_values = np.array([50000])

    mock_explainer = MagicMock()
    mock_explainer.expected_value = 3000000
    mock_explainer.shap_values.return_value = [test_shap_values]
    mock_load_models.return_value = ({"mock": "models"}, test_model_columns, mock_explainer)
    mock_load_loc_options.return_value = ["Other"]
    mock_load_price_map.return_value = (pd.Series(dtype='float64'), 45000.0)
    mock_df_aligned = pd.DataFrame([[0]], columns=test_model_columns)
    mock_make_prediction.return_value = ({'lower': 1, 'median': 2, 'upper': 3}, mock_df_aligned)
    mock_st.sidebar.button.return_value = True

    mock_cols_metrics = (MagicMock(), MagicMock(), MagicMock())
    mock_cols_expander = (MagicMock(), MagicMock())
    mock_st.columns.side_effect = [mock_cols_metrics, mock_cols_expander]

    main()

    mock_cols_expander[1].info.assert_called_once_with("Inga betydande faktorer sänkte priset.", icon="ℹ️")

@patch('app.st')
@patch('app.load_models_and_columns', return_value=(None, None, None))
@patch('app.load_location_options', return_value=[])
@patch('app.load_location_price_map', return_value=(None, None))
def test_main_load_failure_flow(mock_load_price_map, mock_load_loc_options, mock_load_models, mock_st):
    """Tests the main UI flow when artifact loading fails."""
    main()
    mock_load_models.assert_called_once()
    mock_st.sidebar.button.assert_not_called()
    mock_st.warning.assert_called_once()
    assert "Vissa modellfiler saknas" in mock_st.warning.call_args[0][0]

@patch('app.st')
@patch('app.load_models_and_columns', return_value=(True, True, True))
@patch('app.load_location_price_map', return_value=(True, True))
@patch('app.load_location_options')
def test_main_default_location_index_not_found(mock_load_loc_options, mock_load_price_map, mock_load_models, mock_st):
    """Tests the try-except block for finding the default location index."""
    mock_load_loc_options.return_value = ["Area1", "Area2"] # 'Other' is not in the list
    mock_st.sidebar.button.return_value = False
    main()
    # Should fall back to index 0
    mock_st.sidebar.selectbox.assert_called_once_with("Område", options=["Area1", "Area2"], index=0)
    mock_st.metric.assert_not_called()

# --- Tests for Cloud (Azure) Logic ---
@pytest.fixture
def mock_azure_env(monkeypatch):
    """Fixture to simulate a cloud environment and reload the app module."""
    monkeypatch.setattr(config, 'IS_CLOUD', True)
    dummy_instrumentation_key = str(uuid.uuid4())
    conn_str = f"InstrumentationKey={dummy_instrumentation_key}"
    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", conn_str)
    # This import is necessary to make the app module see the patched config
    import app
    importlib.reload(app)
    # Return the connection string so tests can use it for assertions
    return conn_str

@patch('app.azure_utils.download_file_from_blob')
@patch('app.st')
@patch('app.joblib.load')
def test_load_models_and_columns_cloud_success(mock_joblib_load, mock_st, mock_azure_download, mock_azure_env):
    mock_joblib_load.side_effect = ["model_lower", "model_median", "model_upper", "cols", "explainer"]
    models, columns, explainer = load_models_and_columns.__wrapped__()
    assert mock_azure_download.call_count == 5
    assert mock_joblib_load.call_count == 5
    assert 'median' in models
    mock_st.error.assert_not_called()

@patch('app.azure_utils.download_file_from_blob', side_effect=Exception("Azure Blob Error"))
@patch('app.st')
def test_load_models_and_columns_cloud_fail(mock_st, mock_azure_download, mock_azure_env):
    models, columns, explainer = load_models_and_columns.__wrapped__()
    assert models is None
    mock_st.error.assert_called_once()
    assert "Ett fel uppstod vid laddning av filer från Azure" in mock_st.error.call_args[0][0]

@patch('app.azure_utils.download_file_from_blob')
@patch("builtins.open", new_callable=mock_open, read_data='[]')
@patch("app.json.load")
@patch('app.st')
def test_load_location_options_cloud_success(mock_st, mock_json_load, mock_file_open, mock_azure_download, mock_azure_env):
    load_location_options.__wrapped__()
    mock_azure_download.assert_called_once()
    mock_st.error.assert_not_called()
    mock_json_load.assert_called_once()

@patch('app.azure_utils.download_file_from_blob', side_effect=Exception("Azure Blob Error"))
@patch('app.st')
def test_load_location_options_cloud_fail(mock_st, mock_azure_download, mock_azure_env):
    locations = load_location_options.__wrapped__()
    assert locations == []
    mock_st.error.assert_called_once()

@patch('app.azure_utils.download_file_from_blob')
@patch("builtins.open", new_callable=mock_open, read_data='{}')
@patch("app.json.load", return_value={})
@patch('app.st')
def test_load_location_price_map_cloud_success(mock_st, mock_json_load, mock_file_open, mock_azure_download, mock_azure_env):
    load_location_price_map.__wrapped__()
    mock_azure_download.assert_called_once()
    mock_st.error.assert_not_called()
    mock_json_load.assert_called_once()

@patch('app.azure_utils.download_file_from_blob', side_effect=Exception("Azure Blob Error"))
@patch('app.st')
def test_load_location_price_map_cloud_fail(mock_st, mock_azure_download, mock_azure_env):
    price_map, fallback = load_location_price_map.__wrapped__()
    assert price_map is None
    assert fallback is None
    mock_st.error.assert_called_once()

@patch('opencensus.ext.azure.log_exporter.AzureLogHandler')
@patch('app.logging.getLogger')
def test_azure_logger_setup(mock_get_logger, mock_azure_handler, monkeypatch):
    """
    Tests the Azure logger setup by controlling the environment and module
    reload to ensure patches are active when the module-level code runs.
    """
    # 1. Manually set up the same "cloud" environment as the mock_azure_env fixture
    monkeypatch.setattr(config, 'IS_CLOUD', True)
    dummy_instrumentation_key = str(uuid.uuid4())
    conn_str = f"InstrumentationKey={dummy_instrumentation_key}"
    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", conn_str)

    # 2. Now that the test's patches are active, reload the app module.
    #    This will trigger the top-level logger setup code which will now
    #    use the mocked getLogger and AzureLogHandler.
    import app
    importlib.reload(app)

    # 3. Perform assertions on the mocks
    logger_instance = mock_get_logger.return_value
    mock_get_logger.assert_called_with('app')
    mock_azure_handler.assert_called_once_with(connection_string=conn_str)
    logger_instance.addHandler.assert_called_once_with(mock_azure_handler.return_value)
    logger_instance.setLevel.assert_called_once_with(logging.INFO)
    logger_instance.info.assert_called_once_with("Azure Application Insights logger configured for Streamlit app.")