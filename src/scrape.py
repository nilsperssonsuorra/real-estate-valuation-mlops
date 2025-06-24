# src/scrape.py
import pandas as pd
import time
import random
from bs4 import BeautifulSoup
import json
import os
import config
import azure_utils

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException

# --- Parsing and Cleaning Functions ---

def parse_from_next_data(soup):
    """
    Parses property data directly from the __NEXT_DATA__ JSON object.
    This version is robust and handles potentially missing keys safely.
    """
    all_properties_data = []
    
    try:
        next_data_script = soup.find('script', {'id': '__NEXT_DATA__'})
        if not next_data_script:
            print("  > ERROR: __NEXT_DATA__ script tag not found.")
            return []
            
        json_data = json.loads(next_data_script.string)
        apollo_state = json_data['props']['pageProps']['__APOLLO_STATE__']
        
        for key, value in apollo_state.items():
            if key.startswith("SaleCard:"):
                # --- Safely get location ---
                location_dict = value.get('location', {})
                location_ref = location_dict.get('__ref') if location_dict else None
                
                parent_name = ''
                if location_ref:
                    parent_name = apollo_state.get(location_ref, {}).get('parentFullName', '')
                
                location_parts = [value.get('locationDescription', ''), parent_name]
                location = ', '.join(filter(None, location_parts))

                # --- Safely get other data using .get() ---
                price_change = value.get('priceChange')
                slug = value.get('slug')
                url = "https://www.hemnet.se" + slug if slug and slug.startswith('/') else "https://www.hemnet.se/" + slug if slug else None


                prop_data = {
                    'street_address': value.get('streetAddress'),
                    'location': location,
                    'final_price_str': value.get('finalPrice'),
                    'price_change': price_change,
                    'sold_date': value.get('soldAtLabel', '').replace('Såld ', ''),
                    'living_area_str': value.get('livingArea'),
                    'rooms_str': str(value.get('rooms', '')),  # Ensure it's a string for cleaning
                    'plot_area_m2_str': value.get('landArea'),
                    'url': url
                }
                all_properties_data.append(prop_data)

    except (KeyError, AttributeError, json.JSONDecodeError) as e:
        print(f"  > ERROR: Could not parse __NEXT_DATA__ JSON. Error: {e}")
        return []
        
    return all_properties_data

def clean_dataframe(df):
    """
    Cleans and processes the raw DataFrame columns into numeric types.
    Correctly handles various string formats from the JSON data.
    """
    if df.empty:
        return df

    if 'final_price_str' in df.columns:
        df['final_price'] = pd.to_numeric(df['final_price_str'].astype(str).str.extract(r'(\d[\d\s]*\d)')[0].str.replace(r'\s+', '', regex=True), errors='coerce')
    
    if 'rooms_str' in df.columns:
        df['rooms'] = pd.to_numeric(df['rooms_str'].astype(str).str.replace(',', '.').str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    
    if 'plot_area_m2_str' in df.columns and df['plot_area_m2_str'].notna().any():
        df['plot_area_m2'] = pd.to_numeric(df['plot_area_m2_str'].astype(str).str.replace(',', '.').str.replace(r'\s+', '', regex=True).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    
    if 'price_change' in df.columns and df['price_change'].notna().any():
        df['price_change_percent'] = pd.to_numeric(df['price_change'].astype(str).str.extract(r'([+\-]?\d+)')[0], errors='coerce')

    if 'living_area_str' in df.columns and df['living_area_str'].notna().any():
        area_parts = df['living_area_str'].astype(str).str.extract(r'(\d+)(?:\+(\d+))?', expand=True)
        area_parts.columns = ['living_area_m2', 'non_living_area_m2']
        df['living_area_m2'] = pd.to_numeric(area_parts['living_area_m2'], errors='coerce')
        df['non_living_area_m2'] = pd.to_numeric(area_parts['non_living_area_m2'], errors='coerce')

    return df

# --- Main Scraper Logic ---

def run_scraper():
    """
    Main function to orchestrate the scraping process. It checks for existing
    data (locally or in Azure), scrapes new listings, cleans them, and merges
    before saving back to the source.
    """
    existing_df = pd.DataFrame()
    existing_urls = set()

    if config.IS_CLOUD:
        print("--- CLOUD MODE: Loading existing data from Azure Blob Storage ---")
        existing_df = azure_utils.download_df_from_blob(
            config.AZURE_RAW_DATA_CONTAINER, config.RAW_DATA_BLOB_NAME
        )
    else:
        print(f"--- LOCAL MODE: Loading existing data from '{config.RAW_DATA_PATH}' ---")
        if os.path.exists(config.RAW_DATA_PATH):
            try:
                existing_df = pd.read_csv(config.RAW_DATA_PATH)
            except pd.errors.EmptyDataError:
                print("WARNING: Existing local CSV file is empty. Starting fresh.")
            except Exception as e:
                print(f"ERROR: Could not read existing local CSV. Error: {e}. Starting fresh.")
    
    if not existing_df.empty and 'url' in existing_df.columns:
        existing_urls = set(existing_df['url'].dropna())
        print(f"Loaded {len(existing_urls)} unique listings to check against.")
    else:
        print("No existing data found or 'url' column missing. Scraping all pages.")

    newly_scraped_data = []
    print("\n--- Setting up Selenium WebDriver ---")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        print("--- Starting Hemnet Scraper ---")
        stop_scraping = False
        for page_num in range(1, config.SCRAPER_MAX_PAGES + 1):
            if stop_scraping:
                break
            page_url = f"{config.HEMNET_BASE_URL}&page={page_num}"
            print(f"Scraping page {page_num} of {config.SCRAPER_MAX_PAGES} from {page_url}")
            driver.get(page_url)
            try:
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "__NEXT_DATA__")))
                time.sleep(random.uniform(0.5, 1.5))
            except TimeoutException:
                print(f"  > Timed out waiting for page content on page {page_num}. This might be the final page.")
                break
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            page_data = parse_from_next_data(soup)
            if not page_data:
                print(f"  > No listings found on page {page_num}. Ending scrape.")
                break
            new_listings_on_page = []
            for prop_data in page_data:
                if prop_data['url'] in existing_urls:
                    print(f"  > Found previously saved listing: {prop_data['url']}. Halting scrape.")
                    stop_scraping = True
                    break
                else:
                    new_listings_on_page.append(prop_data)
            if new_listings_on_page:
                newly_scraped_data.extend(new_listings_on_page)
                print(f"  > Found and added {len(new_listings_on_page)} new listings from this page.")
            time.sleep(random.uniform(1.0, 2.0))
    finally:
        driver.quit()
        print("\n--- Scraping process completed ---")

    if not newly_scraped_data:
        print("No new data was scraped. The existing data is up-to-date.")
        return

    print(f"Total new properties scraped: {len(newly_scraped_data)}")
    new_df = pd.DataFrame(newly_scraped_data)
    new_df_clean = clean_dataframe(new_df)

    print("--- Merging new data with existing data ---")
    combined_df = pd.concat([new_df_clean, existing_df], ignore_index=True)

    final_columns = [
        'street_address', 'location', 'final_price', 'price_change_percent',
        'sold_date', 'living_area_m2', 'non_living_area_m2', 'rooms', 'plot_area_m2', 'url'
    ]
    existing_final_columns = [col for col in final_columns if col in combined_df.columns]
    combined_df = combined_df[existing_final_columns]
    combined_df.drop_duplicates(subset=['url'], keep='first', inplace=True)
    
    try:
        combined_df['sold_date_dt'] = pd.to_datetime(combined_df['sold_date'], format='%d %B %Y', errors='coerce')
        combined_df.sort_values(by='sold_date_dt', ascending=False, inplace=True)
        combined_df.drop(columns=['sold_date_dt'], inplace=True)
    except Exception as e:
        print(f"Could not sort by date: {e}. Data will be unsorted.")

    # --- Save the final combined data ---
    if config.IS_CLOUD:
        print(f"--- CLOUD MODE: Uploading {len(combined_df)} listings to Azure Blob Storage ---")
        azure_utils.upload_df_to_blob(
            combined_df, config.AZURE_RAW_DATA_CONTAINER, config.RAW_DATA_BLOB_NAME
        )
    else:
        print(f"--- LOCAL MODE: Saving {len(combined_df)} total listings to '{config.RAW_DATA_PATH}' ---")
        combined_df.to_csv(config.RAW_DATA_PATH, index=False, encoding='utf-8-sig')

    print("\nPreview of newly added data:")
    print(new_df_clean.head())


if __name__ == '__main__':
    run_scraper()