# tests/test_train.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open

import train
import config

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
@patch('train.xgb.XGBRegressor')
@patch('train.pd.read_csv')
def test_run_training_pipeline(
    mock_read_csv, mock_xgb_class, mock_shap_explainer, mock_joblib_dump,
    mock_json_dump, mock_os_makedirs, mock_file_open, sample_processed_df
):
    """ Tests the entire training pipeline function in local mode. """
    mock_read_csv.return_value = sample_processed_df
    mock_model_instance = mock_xgb_class.return_value
    mock_model_instance.predict.return_value = np.array([5000000, 6000000, 7000000])
    train.run_training_pipeline()

    mock_read_csv.assert_called_once()
    mock_os_makedirs.assert_called_once()
    assert mock_xgb_class.call_count == 3
    assert mock_model_instance.fit.call_count == 3
    assert mock_joblib_dump.call_count == 5
    assert mock_json_dump.call_count == 2
    mock_shap_explainer.assert_called_once_with(mock_xgb_class.return_value)

    # Verify saved columns and location consolidation logic
    joblib_calls = mock_joblib_dump.call_args_list
    column_save_call = next(c for c in joblib_calls if config.MODEL_COLUMNS_FILE in str(c.args[1]))
    saved_cols = column_save_call.args[0]
    assert 'location_area_Other' in saved_cols
    assert 'location_area_Rare_Area' not in saved_cols

@patch('train.pd.read_csv', side_effect=FileNotFoundError)
@patch('builtins.print')
def test_run_training_pipeline_file_not_found(mock_print, mock_read_csv):
    """ Tests behavior when the processed data file is not found. """
    train.run_training_pipeline()
    mock_read_csv.assert_called_once_with(config.PROCESSED_DATA_PATH)
    any_call_with_error = any("Processed data file not found" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_error

@patch('train.pd.read_csv', return_value=pd.DataFrame())
@patch('builtins.print')
def test_run_training_pipeline_empty_input(mock_print, mock_read_csv):
    """ Tests the pipeline when the loaded dataframe is empty. """
    train.run_training_pipeline()
    mock_read_csv.assert_called_once()
    any_call_with_msg = any("Processed data is empty. Exiting training." in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_msg

@patch('train.azure_utils.upload_joblib_to_blob')
@patch('train.azure_utils.upload_json_to_blob')
@patch('train.azure_utils.download_df_from_blob')
@patch('train.shap.TreeExplainer')
@patch('train.xgb.XGBRegressor')
def test_run_training_pipeline_cloud(
    mock_xgb_class, mock_shap_explainer, mock_download,
    mock_upload_json, mock_upload_joblib, sample_processed_df, monkeypatch
):
    """ Tests the entire training pipeline function in cloud mode. """
    monkeypatch.setattr(config, 'IS_CLOUD', True)
    mock_download.return_value = sample_processed_df
    
    mock_model_instance = mock_xgb_class.return_value
    mock_model_instance.predict.return_value = np.array([5000000, 6000000, 7000000])

    train.run_training_pipeline()

    # Assert I/O
    mock_download.assert_called_once_with(config.AZURE_PROCESSED_DATA_CONTAINER, config.PROCESSED_DATA_BLOB_NAME)
    assert mock_upload_json.call_count == 2
    assert mock_upload_joblib.call_count == 5 # 3 models + columns + explainer
    
    # Assert ML logic
    assert mock_xgb_class.call_count == 3
    mock_shap_explainer.assert_called_once()