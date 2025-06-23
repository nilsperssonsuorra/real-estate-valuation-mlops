# tests/test_clean.py
import pytest
import pandas as pd
from unittest.mock import patch

from clean import (
    clean_location, parse_swedish_date, perform_data_cleaning_and_engineering,
    analyze_data, run_cleaning_pipeline
)
import config

@pytest.fixture
def sample_raw_df():
    """Provides a sample DataFrame mimicking raw scraped data."""
    data = {
        'location': ['Gamla Uppsala / Nyby, Uppsala', 'Svartbäcken, Uppsala', 'Invalid Location'],
        'sold_date': ['15 januari 2023', '2 feb. 2022', 'Ogiltigt Datum'],
        'final_price': [5000000, 6000000, 7000000],
        'living_area_m2': [120, 150, 160],
        'extra_col': [1, 2, 3] # A column that should be dropped
    }
    return pd.DataFrame(data)

def test_clean_location(sample_raw_df):
    """Tests the splitting and cleaning of the location column."""
    df = clean_location(sample_raw_df)
    assert 'location_area' in df.columns
    assert 'municipality' in df.columns
    assert 'location' not in df.columns
    assert df['location_area'].iloc[0] == 'Gamla Uppsala'
    assert df['municipality'].iloc[0] == 'Uppsala'
    assert df['location_area'].iloc[1] == 'Svartbäcken'

def test_clean_location_no_location_column():
    """Tests that the function handles a missing 'location' column gracefully."""
    df_no_loc = pd.DataFrame({'other_col': [1, 2]})
    result_df = clean_location(df_no_loc.copy())
    pd.testing.assert_frame_equal(df_no_loc, result_df)

@pytest.mark.parametrize("date_str, expected_month, expected_day", [
    ("15 januari 2023", 1, 15),
    ("5 feb. 2022", 2, 5),
    ("3 MAR 2021", 3, 3),
    ("29 Maj 2020", 5, 29),
    ("1 December 2023", 12, 1),
])
def test_parse_swedish_date_valid(date_str, expected_month, expected_day):
    parsed_date = parse_swedish_date(date_str)
    assert parsed_date.month == expected_month
    assert parsed_date.day == expected_day

def test_parse_swedish_date_invalid():
    assert pd.isna(parse_swedish_date("not a date"))
    assert pd.isna(parse_swedish_date(None))
    assert pd.isna(parse_swedish_date(12345))

def test_perform_data_cleaning_and_engineering(sample_raw_df):
    """Tests the main data processing pipeline function."""
    df_with_location = clean_location(sample_raw_df.copy())
    df_with_location['sold_date'] = df_with_location['sold_date'].apply(parse_swedish_date)
    df = perform_data_cleaning_and_engineering(df_with_location)
    assert 'price_per_m2' in df.columns
    assert df['price_per_m2'].iloc[0] == round(5000000 / 120, 2)
    assert 'extra_col' not in df.columns
    assert 'location_area' in df.columns
    assert 'final_price' in df.columns

@patch('builtins.print')
def test_analyze_data(mock_print, sample_raw_df):
    """Tests that the analysis function runs and calls print."""
    df = clean_location(sample_raw_df)
    analyze_data(df)
    mock_print.assert_called()
    any_call_with_info = any("DataFrame Info" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_info

@patch('builtins.print')
def test_analyze_data_with_missing_values(mock_print):
    """NEW: Tests the analysis function when data has missing values to improve coverage."""
    df_missing = pd.DataFrame({'A': [1, 2, None], 'location_area': ['X', 'Y', 'Z']})
    analyze_data(df_missing)
    any_call_with_missing_header = any("Missing Value Counts" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_missing_header
    # Check that the block for printing missing values was entered and printed column A
    any_call_with_col_A = any("A    1" in str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert any_call_with_col_A

@patch('clean.pd.read_csv')
@patch('clean.clean_location')
@patch('clean.perform_data_cleaning_and_engineering')
@patch('clean.analyze_data')
@patch('clean.os.makedirs')
@patch('pandas.DataFrame.to_csv')
def test_run_cleaning_pipeline_success(mock_to_csv, mock_makedirs, mock_analyze, mock_perform, mock_clean, mock_read_csv):
    """Tests the successful, end-to-end flow of the cleaning pipeline."""
    mock_read_csv.return_value = pd.DataFrame({'location': ['Test Area, Uppsala']})
    mock_clean.return_value = pd.DataFrame({'location_area': ['Test']})
    mock_perform.return_value = pd.DataFrame({'location_area': ['Test'], 'head': lambda: 'mock'})
    run_cleaning_pipeline()
    mock_read_csv.assert_called_once_with(config.RAW_DATA_PATH)
    mock_to_csv.assert_called_once()

@patch('clean.pd.read_csv', side_effect=FileNotFoundError)
@patch('builtins.print')
def test_run_cleaning_pipeline_file_not_found(mock_print, mock_read_csv):
    run_cleaning_pipeline()
    mock_read_csv.assert_called_once()
    any_call_with_error = any("Raw data file not found" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_error

@patch('clean.pd.read_csv', side_effect=pd.errors.EmptyDataError)
@patch('builtins.print')
def test_run_cleaning_pipeline_empty_data_error(mock_print, mock_read_csv):
    run_cleaning_pipeline()
    mock_read_csv.assert_called_once()
    any_call_with_error = any("raw data file is empty" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_error

@patch('clean.pd.read_csv')
@patch('pandas.DataFrame.to_csv', side_effect=IOError("Permission denied"))
@patch('builtins.print')
def test_run_cleaning_pipeline_save_error(mock_print, mock_to_csv, mock_read_csv):
    # FIX: Provide a more complete mock DataFrame that includes all columns
    # needed by the cleaning pipeline to prevent the KeyError.
    mock_data = {
        'location': ['Test Area, Uppsala'],
        'sold_date': ['15 januari 2023'],
        'final_price': [5000000],
        'living_area_m2': [100]
    }
    mock_read_csv.return_value = pd.DataFrame(mock_data)

    run_cleaning_pipeline()

    mock_to_csv.assert_called_once()
    any_call_with_error = any("Failed to save processed data" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_error