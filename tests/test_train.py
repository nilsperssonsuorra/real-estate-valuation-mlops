# tests/test_train.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

from train import run_training_pipeline
import config

@pytest.fixture
def sample_processed_df():
    """Provides a DataFrame that mimics the processed data file."""
    data = {
        'living_area_m2': [100, 150, 120, 80, 200, 110, 90, 130, 140, 160],
        'rooms': [4, 5, 5, 3, 6, 4, 3, 5, 5, 6],
        'plot_area_m2': [800, 1000, 900, 600, 1200, np.nan, 700, 950, 1100, 1300],
        'non_living_area_m2': [20, 30, 25, np.nan, 40, 22, 18, 28, 32, 35],
        'location_area': ['Area_A', 'Area_A', 'Area_B', 'Area_B', 'Area_C', 
                          'Area_C', 'Area_D', 'Area_D', 'Rare_Area', 'Area_A'],
        'sold_date': pd.to_datetime(['2023-01-10', '2023-02-15', '2023-03-20', '2023-04-25',
                                     '2023-05-10', '2023-06-15', '2023-07-20', '2023-08-25',
                                     '2023-09-10', '2023-10-15']),
        'final_price': [5000000, 7000000, 6000000, 4000000, 9000000, 5500000, 
                        4500000, 6500000, 6800000, 7500000]
    }
    return pd.DataFrame(data)

@patch('train.pd.read_csv')
@patch('train.xgb.XGBRegressor')
@patch('train.joblib.dump')
@patch('train.json.dump')
@patch('train.os.makedirs')
@patch('builtins.open', new_callable=mock_open)
def test_run_training_pipeline(
    mock_file_open, mock_os_makedirs, mock_json_dump, mock_joblib_dump,
    mock_xgb_class, mock_read_csv, sample_processed_df
):
    """
    Tests the entire training pipeline function, mocking I/O and model training.
    """
    mock_read_csv.return_value = sample_processed_df
    mock_model_instance = MagicMock()
    mock_model_instance.predict.return_value = np.array([5000000, 6000000])
    mock_xgb_class.return_value = mock_model_instance

    run_training_pipeline()

    mock_read_csv.assert_called_once()
    mock_os_makedirs.assert_called_once()
    assert mock_xgb_class.call_count == 3
    assert mock_model_instance.fit.call_count == 3
    assert mock_joblib_dump.call_count == 4
    assert mock_json_dump.call_count == 2
    
    # Verify saved columns and location consolidation logic
    joblib_calls = mock_joblib_dump.call_args_list
    column_save_call = next(c for c in joblib_calls if isinstance(c.args[0], list))
    saved_cols = column_save_call.args[0]
    
    assert 'log_living_area' in saved_cols
    assert 'location_median_price_per_m2' in saved_cols
    # All locations in the sample data are rare (count < 10) and become 'Other'
    assert 'location_area_Other' in saved_cols
    assert 'location_area_Area_A' not in saved_cols
    
    # Verify saved location list for the UI dropdown
    json_calls = mock_json_dump.call_args_list
    location_save_call = next(c for c in json_calls if isinstance(c.args[0], list))
    saved_locations = location_save_call.args[0]
    # Check that 'Other' is in the final list, but the original rare ones are not
    assert 'Other' in saved_locations
    assert 'Area_A' not in saved_locations
    assert 'Rare_Area' not in saved_locations

@patch('train.pd.read_csv', side_effect=FileNotFoundError)
@patch('builtins.print')
def test_run_training_pipeline_file_not_found(mock_print, mock_read_csv):
    """
    Tests the pipeline's behavior when the processed data file is not found.
    """
    run_training_pipeline()
    
    mock_read_csv.assert_called_once_with(config.PROCESSED_DATA_PATH)
    
    any_call_with_error = any("Processed data file not found" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_error