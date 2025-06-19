import pandas as pd
import time
import random
from bs4 import BeautifulSoup
import json

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException

# --- Configuration ---
BASE_URL = "https://www.hemnet.se/salda/bostader?item_types%5B%5D=villa&location_ids%5B%5D=946677"
TOTAL_PAGES = 1
OUTPUT_CSV_FILE = 'hemnet_sold_villas_final.csv'

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
                url = "https://www.hemnet.se" + slug if slug else None

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

def main():
    all_properties_data = []
    print("--- Setting up Selenium WebDriver ---")
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
        
        for page_num in range(1, TOTAL_PAGES + 1):
            page_url = f"{BASE_URL}&page={page_num}"
            print(f"Scraping page {page_num} of {TOTAL_PAGES} from {page_url}")
            
            driver.get(page_url)

            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "__NEXT_DATA__"))
                )
                time.sleep(random.uniform(0.5, 1.5)) # Small delay for stability
            except TimeoutException:
                print(f"Could not find __NEXT_DATA__ on page {page_num}. This might be the final page.")
                break

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            page_data = parse_from_next_data(soup)
            
            if not page_data:
                print(f"  > No listings found in __NEXT_DATA__ on page {page_num}. Ending scrape.")
                break
                
            all_properties_data.extend(page_data)
            print(f"  > Found and parsed {len(page_data)} listings on this page.")
            time.sleep(random.uniform(1.0, 2.0))

    finally:
        driver.quit()
        print("\n--- Scraping process completed ---")

    if not all_properties_data:
        print("No data was scraped. Please check for website changes or anti-bot measures.")
        return

    print(f"Total properties scraped: {len(all_properties_data)}")
    df = pd.DataFrame(all_properties_data)
    df_clean = clean_dataframe(df)

    final_columns = [
        'street_address', 'location', 'final_price', 'price_change_percent',
        'sold_date', 'living_area_m2', 'non_living_area_m2', 'rooms', 'plot_area_m2', 'url'
    ]
    existing_columns = [col for col in final_columns if col in df_clean.columns]

    df_clean.to_csv(OUTPUT_CSV_FILE, index=False, columns=existing_columns, encoding='utf-8-sig')

    print(f"\nSuccessfully saved data to '{OUTPUT_CSV_FILE}'")
    print("\nData preview:")
    print(df_clean[existing_columns].head())


if __name__ == '__main__':
    main()