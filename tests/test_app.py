# tests/test_app.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from datetime import date
import json

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
    predictions = make_prediction(
        sample_input_data, mock_models, mock_model_columns, mock_price_map, fallback_price
    )

    # --- Assertions ---
    assert isinstance(predictions, dict)
    assert predictions['lower'] == 4800000
    assert predictions['median'] == 5000000
    assert predictions['upper'] == 5200000
    df_aligned = mock_model_median.predict.call_args.args[0]
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
    make_prediction(unseen_input_data, mock_models, mock_model_columns, mock_price_map, fallback_price)
    df_passed_to_model = mock_model.predict.call_args.args[0]
    assert df_passed_to_model['location_median_price_per_m2'].iloc[0] == fallback_price

@patch('app.st')
@patch('app.joblib.load')
def test_load_models_and_columns_success(mock_joblib_load, mock_st):
    mock_joblib_load.side_effect = ["model_lower", "model_median", "model_upper", "columns_list"]
    models, columns = load_models_and_columns.__wrapped__()
    assert mock_joblib_load.call_count == 4
    assert 'median' in models
    assert columns == "columns_list"
    mock_st.error.assert_not_called()

@patch('app.st')
@patch('app.joblib.load', side_effect=FileNotFoundError("File not found!"))
def test_load_models_and_columns_file_not_found(mock_joblib_load, mock_st):
    models, columns = load_models_and_columns.__wrapped__()
    assert models is None
    assert columns is None
    mock_st.error.assert_called_once()
    assert "En modell- eller kolumnfil kunde inte hittas" in mock_st.error.call_args[0][0]

@patch('app.st')
@patch('app.joblib.load', side_effect=Exception("Unexpected error!"))
def test_load_models_and_columns_generic_exception(mock_joblib_load, mock_st):
    """Tests the failure case with a generic exception."""
    models, columns = load_models_and_columns.__wrapped__()
    assert models is None
    assert columns is None
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
    """Tests the main UI flow on a successful run."""
    mock_load_models.return_value = ("mock_models", "mock_cols")
    mock_load_loc_options.return_value = ["Uppsala_Centrum", "Other"]
    mock_load_price_map.return_value = (pd.Series(dtype=float), 45000.0)
    mock_make_prediction.return_value = {'lower': 1, 'median': 2, 'upper': 3}

    # FIX: Create explicit mocks for the columns so we can check calls on them.
    mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
    mock_st.columns.return_value = (mock_col1, mock_col2, mock_col3)
    mock_st.sidebar.button.return_value = True

    main()

    # Assertions
    mock_load_models.assert_called_once()
    mock_load_loc_options.assert_called_once()
    mock_load_price_map.assert_called_once()
    mock_st.sidebar.button.assert_called_once()
    mock_make_prediction.assert_called_once()

    # FIX: Assert that `metric` was called on each individual column mock.
    mock_col1.metric.assert_called_once()
    mock_col2.metric.assert_called_once()
    mock_col3.metric.assert_called_once()


@patch('app.st')
@patch('app.load_models_and_columns', return_value=(None, None))
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
@patch('app.load_models_and_columns', return_value=(True, True))
@patch('app.load_location_price_map', return_value=(True, True))
@patch('app.load_location_options')
def test_main_default_location_index_not_found(mock_load_loc_options, mock_load_price_map, mock_load_models, mock_st):
    """Tests the try-except block for finding the default location index."""
    mock_load_loc_options.return_value = ["Area1", "Area2"]
    mock_st.sidebar.button.return_value = False
    main()
    mock_st.sidebar.selectbox.assert_called_once_with("Område", options=["Area1", "Area2"], index=0)
    mock_st.metric.assert_not_called()