# tests/test_scrape.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock
import json
from bs4 import BeautifulSoup

# Selenium-specific imports for testing exceptions
from selenium.common.exceptions import TimeoutException

# Import functions from the script we are testing
from scrape import parse_from_next_data, clean_dataframe, run_scraper
import config

# Sample __NEXT_DATA__ JSON to simulate what we get from Hemnet
SAMPLE_NEXT_DATA = {
    "props": {
        "pageProps": {
            "__APOLLO_STATE__": {
                "SaleCard:12345": {
                    "location": {"__ref": "Location:1"},
                    "streetAddress": "Testgatan",
                    "locationDescription": "Villaområde",
                    "finalPrice": "5 000 000 kr",
                    "priceChange": "+5%",
                    "soldAtLabel": "Såld 15 januari 2023",
                    "livingArea": "120+20 m²",
                    "rooms": 5,
                    "landArea": "850 m²",
                    "slug": "/salda/villa-5rum-testgatan-uppsala-kommun-12345"
                },
                "SaleCard:67890": {
                    "location": {"__ref": "Location:2"},
                    "streetAddress": "Annan Väg",
                    "locationDescription": "Nära stan",
                    "finalPrice": "3 250 000 kr",
                    "priceChange": None,
                    "soldAtLabel": "Såld 1 februrari 2023",
                    "livingArea": "100 m²",
                    "rooms": "4,5",
                    "landArea": "1 200 m²",
                    "slug": "/salda/villa-4,5rum-annan-vag-uppsala-kommun-67890"
                },
                "Location:1": {"parentFullName": "Testområde, Uppsala"},
                "Location:2": {"parentFullName": "Annat Område, Uppsala"}
            }
        }
    }
}

@pytest.fixture
def sample_soup():
    """Creates a BeautifulSoup object with our sample __NEXT_DATA__."""
    json_str = json.dumps(SAMPLE_NEXT_DATA)
    html = f'<html><body><script id="__NEXT_DATA__" type="application/json">{json_str}</script></body></html>'
    return BeautifulSoup(html, 'html.parser')

def test_parse_from_next_data(sample_soup):
    data = parse_from_next_data(sample_soup)
    assert len(data) == 2
    prop1 = data[0]
    assert prop1['street_address'] == 'Testgatan'
    assert prop1['location'] == 'Villaområde, Testområde, Uppsala'
    prop2 = data[1]
    assert prop2['price_change'] is None

def test_clean_dataframe():
    raw_data = [
        {'final_price_str': '5 000 000 kr', 'rooms_str': '5', 'plot_area_m2_str': '850 m²', 'price_change': '+5%', 'living_area_str': '120+20 m²'},
        {'final_price_str': '3 250 000 kr', 'rooms_str': '4,5', 'plot_area_m2_str': '1 200 m²', 'price_change': '-10%', 'living_area_str': '100 m²'}
    ]
    raw_df = pd.DataFrame(raw_data)
    cleaned_df = clean_dataframe(raw_df)
    assert cleaned_df['final_price'].iloc[0] == 5000000
    assert cleaned_df['rooms'].iloc[1] == 4.5
    assert cleaned_df['plot_area_m2'].iloc[1] == 1200
    assert cleaned_df['price_change_percent'].iloc[0] == 5
    assert cleaned_df['living_area_m2'].iloc[0] == 120
    assert pd.isna(cleaned_df['non_living_area_m2'].iloc[1])

def test_parse_from_next_data_no_script_tag():
    soup = BeautifulSoup("<html><body><p>No data here</p></body></html>", 'html.parser')
    data = parse_from_next_data(soup)
    assert data == []

def test_parse_from_next_data_malformed_json():
    html = '<html><body><script id="__NEXT_DATA__" type="application/json">This is not JSON</script></body></html>'
    soup = BeautifulSoup(html, 'html.parser')
    data = parse_from_next_data(soup)
    assert data == []

def test_parse_from_next_data_key_error():
    # Simulate JSON that is valid but missing a required key ('pageProps')
    malformed_next_data = {"props": {}}
    json_str = json.dumps(malformed_next_data)
    html = f'<html><body><script id="__NEXT_DATA__" type="application/json">{json_str}</script></body></html>'
    soup = BeautifulSoup(html, 'html.parser')
    data = parse_from_next_data(soup)
    assert data == [] # The function should gracefully handle the KeyError and return an empty list

def test_clean_dataframe_empty():
    df = pd.DataFrame()
    cleaned_df = clean_dataframe(df)
    assert cleaned_df.empty

@patch('scrape.webdriver.Chrome')
@patch('scrape.pd.read_csv', side_effect=pd.errors.EmptyDataError)
@patch('scrape.os.path.exists', return_value=True)
@patch('builtins.print')
def test_run_scraper_existing_file_is_empty(mock_print, mock_exists, mock_read_csv, mock_driver_class):
    mock_driver = MagicMock()
    type(mock_driver).page_source = PropertyMock(return_value="<html></html>")
    mock_driver_class.return_value = mock_driver
    with patch('scrape.Service'), patch('scrape.ChromeDriverManager'):
        run_scraper()
    any_call_with_warning = any("Existing CSV file is empty" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_warning
    mock_driver.quit.assert_called_once()

@patch('scrape.webdriver.Chrome')
@patch('scrape.pd.read_csv', side_effect=Exception("Generic read error"))
@patch('scrape.os.path.exists', return_value=True)
@patch('builtins.print')
def test_run_scraper_existing_file_read_error(mock_print, mock_exists, mock_read_csv, mock_driver_class):
    mock_driver = MagicMock()
    type(mock_driver).page_source = PropertyMock(return_value="<html></html>")
    mock_driver_class.return_value = mock_driver
    with patch('scrape.Service'), patch('scrape.ChromeDriverManager'):
        run_scraper()
    any_call_with_error = any("Could not read existing CSV file" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_error
    mock_driver.quit.assert_called_once()

@patch('scrape.webdriver.Chrome')
@patch('scrape.WebDriverWait')
def test_run_scraper_timeout_on_page_load(mock_wait, mock_driver_class):
    mock_driver = MagicMock()
    mock_driver_class.return_value = mock_driver
    mock_wait.return_value.until.side_effect = TimeoutException()
    with patch('scrape.Service'), patch('scrape.ChromeDriverManager'):
        run_scraper()
    mock_driver.get.assert_called_once()
    mock_driver.quit.assert_called_once()

@patch('scrape.webdriver.Chrome')
@patch('scrape.parse_from_next_data', return_value=[])
def test_run_scraper_no_listings_found_on_page(mock_parse, mock_driver_class):
    mock_driver = MagicMock()
    type(mock_driver).page_source = PropertyMock(return_value="<html></html>")
    mock_driver_class.return_value = mock_driver
    with patch('scrape.Service'), patch('scrape.ChromeDriverManager'):
        run_scraper()
    mock_driver.get.assert_called_once()
    mock_parse.assert_called_once()
    mock_driver.quit.assert_called_once()

@patch('scrape.webdriver.Chrome')
@patch('scrape.os.path.exists', return_value=True)
@patch('scrape.pd.read_csv')
@patch('pandas.DataFrame.to_csv')
@patch('builtins.print')
def test_run_scraper_no_new_data_found(mock_print, mock_to_csv, mock_read_csv, mock_exists, mock_driver_class):
    existing_df = pd.DataFrame({'url': ['https://www.hemnet.se/salda/villa-5rum-testgatan-uppsala-kommun-12345', 'https://www.hemnet.se/salda/villa-4,5rum-annan-vag-uppsala-kommun-67890']})
    mock_read_csv.return_value = existing_df
    mock_driver = MagicMock()
    type(mock_driver).page_source = PropertyMock(return_value=f'<html><body><script id="__NEXT_DATA__" type="application/json">{json.dumps(SAMPLE_NEXT_DATA)}</script></body></html>')
    mock_driver_class.return_value = mock_driver
    with patch('scrape.Service'), patch('scrape.ChromeDriverManager'):
        run_scraper()
    mock_driver.get.assert_called_once()
    mock_driver.quit.assert_called_once()
    mock_to_csv.assert_not_called()
    any_call_with_message = any("No new data was scraped" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_message

@patch('scrape.Service')
@patch('scrape.ChromeDriverManager')
@patch('scrape.webdriver.Chrome')
@patch('scrape.os.path.exists', return_value=False)
@patch('pandas.DataFrame.to_csv')
@patch('pandas.DataFrame.sort_values', side_effect=Exception("Mocked sorting error"))
@patch('builtins.print')
def test_run_scraper_sort_by_date_exception(mock_print, mock_sort_values, mock_to_csv, mock_exists, mock_driver_class, mock_cdm, mock_service):
    mock_driver = MagicMock()
    type(mock_driver).page_source = PropertyMock(side_effect=[
        f'<html><body><script id="__NEXT_DATA__" type="application/json">{json.dumps(SAMPLE_NEXT_DATA)}</script></body></html>',
        "<html></html>"
    ])
    mock_driver_class.return_value = mock_driver
    run_scraper()
    mock_sort_values.assert_called_once()
    any_call_with_error = any("Could not sort by date" in call.args[0] for call in mock_print.call_args_list if call.args)
    assert any_call_with_error
    mock_to_csv.assert_called_once()
    
def test_parse_from_next_data_slug_edge_cases():
    next_data = {"props": {"pageProps": {"__APOLLO_STATE__": {"SaleCard:1": {"slug": "test-slug"}}}}}
    html = f'<html><body><script id="__NEXT_DATA__" type="application/json">{json.dumps(next_data)}</script></body></html>'
    soup = BeautifulSoup(html, 'html.parser')
    data = parse_from_next_data(soup)
    assert data[0]['url'] == "https://www.hemnet.se/test-slug"

    next_data2 = {"props": {"pageProps": {"__APOLLO_STATE__": {"SaleCard:2": {}}}}}
    html2 = f'<html><body><script id="__NEXT_DATA__" type="application/json">{json.dumps(next_data2)}</script></body></html>'
    soup2 = BeautifulSoup(html2, 'html.parser')
    data2 = parse_from_next_data(soup2)
    assert data2[0]['url'] is None

@patch('scrape.webdriver.Chrome')
@patch('scrape.pd.read_csv')
@patch('scrape.os.path.exists', return_value=True)
@patch('builtins.print')
def test_run_scraper_missing_url_column(mock_print, mock_exists, mock_read_csv, mock_driver_class):
    existing_df = pd.DataFrame({'foo': [1, 2]})
    mock_read_csv.return_value = existing_df
    mock_driver = MagicMock()
    type(mock_driver).page_source = PropertyMock(return_value="<html></html>")
    mock_driver_class.return_value = mock_driver
    with patch('scrape.Service'), patch('scrape.ChromeDriverManager'):
        run_scraper()
    assert any("WARNING: 'url' column not found" in call.args[0] for call in mock_print.call_args_list if call.args)
    mock_driver.quit.assert_called_once()

@patch('scrape.Service')
@patch('scrape.ChromeDriverManager')
@patch('scrape.webdriver.Chrome')
@patch('scrape.os.path.exists', return_value=False)
@patch('pandas.DataFrame.to_csv')
@patch('pandas.DataFrame.sort_values')
@patch('pandas.DataFrame.drop')
@patch('pandas.to_datetime')
def test_run_scraper_drops_temp_date_column_after_sort(mock_to_datetime, mock_drop, mock_sort, mock_to_csv, mock_exists, mock_driver_class, mock_cdm, mock_service):
    mock_driver = MagicMock()
    type(mock_driver).page_source = PropertyMock(side_effect=[
        f'<html><body><script id="__NEXT_DATA__" type="application/json">{json.dumps(SAMPLE_NEXT_DATA)}</script></body></html>',
        "<html></html>"
    ])
    mock_driver_class.return_value = mock_driver
    mock_to_datetime.return_value = pd.Series([pd.Timestamp('2023-01-15'), pd.Timestamp('2023-02-01')])
    run_scraper()
    mock_sort.assert_called_once_with(by='sold_date_dt', ascending=False, inplace=True)
    mock_drop.assert_called_once_with(columns=['sold_date_dt'], inplace=True)
    mock_to_csv.assert_called_once()

@patch('scrape.azure_utils.upload_df_to_blob')
@patch('scrape.azure_utils.download_df_from_blob')
@patch('scrape.webdriver.Chrome')
def test_run_scraper_cloud(mock_driver_class, mock_download, mock_upload, monkeypatch):
    """Tests the scraper's cloud execution path."""
    monkeypatch.setattr(config, 'IS_CLOUD', True)
    mock_download.return_value = pd.DataFrame()
    mock_driver = MagicMock()
    type(mock_driver).page_source = PropertyMock(side_effect=[
        f'<html><body><script id="__NEXT_DATA__" type="application/json">{json.dumps(SAMPLE_NEXT_DATA)}</script></body></html>',
        "<html></html>"
    ])
    mock_driver_class.return_value = mock_driver

    with patch('scrape.Service'), patch('scrape.ChromeDriverManager'):
        run_scraper()

    mock_download.assert_called_once_with(config.AZURE_RAW_DATA_CONTAINER, config.RAW_DATA_BLOB_NAME)
    mock_upload.assert_called_once()
    uploaded_df = mock_upload.call_args.args[0]
    assert isinstance(uploaded_df, pd.DataFrame)
    assert len(uploaded_df) == 2