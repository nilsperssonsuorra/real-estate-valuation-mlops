# tests/test_train.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

import train
import config
import xgboost as xgb # Keep this import to reference the real class if needed, but not for spec

@pytest.fixture
def sample_processed_df():
    """
    Provides a DataFrame that mimics the processed data file.
    This data is designed so that 'Area_A' has a count of 10 (not rare)
    and 'Rare_Area' has a count of 2 (rare, will be consolidated to 'Other').
    """
    data = {
        'living_area_m2':     [100, 150, 120, 80, 200, 110, 90, 130, 140, 160, 105, 115],
        'rooms':              [4, 5, 5, 3, 6, 4, 3, 5, 5, 6, 4, 5],
        'plot_area_m2':       [800, 1000, 900, 600, 1200, np.nan, 700, 950, 1100, 1300, 850, 880],
        'non_living_area_m2': [20, 30, 25, np.nan, 40, 22, 18, 28, 32, 35, 21, 23],
        'location_area':      ['Area_A', 'Area_A', 'Area_A', 'Area_A', 'Area_A',
                               'Area_A', 'Area_A', 'Area_A', 'Area_A', 'Area_A',
                               'Rare_Area', 'Rare_Area'],
        'sold_date': pd.to_datetime(['2023-01-10', '2023-02-15', '2023-03-20', '2023-04-25',
                                     '2023-05-10', '2023-06-15', '2023-07-20', '2023-08-25',
                                     '2023-09-10', '2023-10-15', '2023-11-20', '2023-12-25']),
        'final_price':        [5000000, 7000000, 6000000, 4000000, 9000000, 5500000,
                               4500000, 6500000, 6800000, 7500000, 5200000, 5300000]
    }
    return pd.DataFrame(data)


@patch('builtins.open', new_callable=mock_open)
@patch('train.os.makedirs')
@patch('train.json.dump')
@patch('train.joblib.dump')
@patch('train.shap.TreeExplainer')
@patch('train.xgb.XGBRegressor') # This decorator provides mock_xgb_class
@patch('train.pd.read_csv')
def test_run_training_pipeline(
    mock_read_csv,          # from @patch('train.pd.read_csv')
    mock_xgb_class,         # from @patch('train.xgb.XGBRegressor')
    mock_shap_explainer,    # from @patch('train.shap.TreeExplainer')
    mock_joblib_dump,
    mock_json_dump,
    mock_os_makedirs,
    mock_file_open,
    sample_processed_df
):
    """
    Tests the entire training pipeline function, mocking I/O and model training.
    """
    mock_read_csv.return_value = sample_processed_df

    # We will use this instance to check calls to .fit() and .predict().
    mock_model_instance = mock_xgb_class.return_value
    # The test set has 3 samples (12 rows * 0.2 test_size, rounded up by train_test_split).
    # The mock prediction array must also have 3 values to match y_test.
    mock_model_instance.predict.return_value = np.array([5000000, 6000000, 7000000])

    train.run_training_pipeline()

    # --- Assertions ---
    mock_read_csv.assert_called_once()
    mock_os_makedirs.assert_called_once()

    # Assert that the XGBRegressor class was instantiated 3 times
    assert mock_xgb_class.call_count == 3
    # Assert that the .fit() method on the instance was called 3 times
    assert mock_model_instance.fit.call_count == 3

    # Assert that joblib.dump was called 5 times (3 models, 1 columns, 1 explainer)
    assert mock_joblib_dump.call_count == 5
    assert mock_json_dump.call_count == 2

    # Assert that shap.TreeExplainer was called with the mock of the *median model instance*.
    # The median model is the second one created. We can get it from the call list.
    median_model_instance = mock_xgb_class.return_value
    mock_shap_explainer.assert_called_once_with(median_model_instance)


    # Verify saved columns and location consolidation logic
    joblib_calls = mock_joblib_dump.call_args_list
    column_save_call = next(c for c in joblib_calls if config.MODEL_COLUMNS_FILE in str(c.args[1]))
    saved_cols = column_save_call.args[0]

    assert 'log_living_area' in saved_cols
    assert 'location_median_price_per_m2' in saved_cols
    # The sample data has counts of Area_A=10 (not rare) and Rare_Area=2 (rare).
    # Only Rare_Area should be consolidated into 'Other'.
    assert 'location_area_Other' in saved_cols
    assert 'location_area_Area_A' in saved_cols
    assert 'location_area_Rare_Area' not in saved_cols

    # Verify saved location list for the UI dropdown
    json_calls = mock_json_dump.call_args_list
    location_save_call = next(c for c in json_calls if isinstance(c.args[0], list))
    saved_locations = location_save_call.args[0]
    # Check that 'Other' and the common area are in the final list, but the original rare one is not.
    assert 'Other' in saved_locations
    assert 'Area_A' in saved_locations
    assert 'Rare_Area' not in saved_locations # This one was consolidated


@patch('train.pd.read_csv', side_effect=FileNotFoundError)
@patch('builtins.print')
def test_run_training_pipeline_file_not_found(mock_print, mock_read_csv):
    """
    Tests the pipeline's behavior when the processed data file is not found.
    """
    train.run_training_pipeline()

    mock_read_csv.assert_called_once_with(config.PROCESSED_DATA_PATH)

    any_call_with_error = any("Processed data file not found" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_error