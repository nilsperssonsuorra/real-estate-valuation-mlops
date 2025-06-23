# src/clean.py
import pandas as pd
import os
from dateutil import parser
import config  # Import the new config file

def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'location' column.
    """
    print("--- Cleaning 'location' column ---")
    if 'location' not in df.columns:
        print("WARNING: 'location' column not found. Skipping location cleaning.")
        return df

    loc_parts = df['location'].str.split(',', n=1, expand=True)
    loc_parts.columns = ['area', 'municipality']

    df['location_area'] = loc_parts['area'].str.split('/').str.get(0).str.strip().str.title()
    df['municipality'] = loc_parts['municipality'].str.strip()
    df.drop(columns=['location'], inplace=True)
    
    print("Location successfully split into 'location_area' and 'municipality'.")
    return df

def parse_swedish_date(date_str):
    """
    Robustly parses a Swedish date string by MANUALLY translating month names.
    This avoids any dependency on system locales, making it work on any machine.
    """
    if not isinstance(date_str, str):
        return pd.NaT

    original_date_str = date_str 
    
    month_map = {
        'januari': 'January', 'jan.': 'January', 'jan': 'January',
        'februari': 'February', 'feb.': 'February', 'feb': 'February',
        'mars': 'March', 'mar.': 'March', 'mar': 'March',
        'april': 'April', 'apr.': 'April', 'apr': 'April',
        'maj': 'May', 'maj.': 'May',
        'juni': 'June', 'jun.': 'June', 'jun': 'June',
        'juli': 'July', 'jul.': 'July', 'jul': 'July',
        'augusti': 'August', 'aug.': 'August', 'aug': 'August',
        'september': 'September', 'sep.': 'September', 'sep': 'September',
        'oktober': 'October', 'okt.': 'October', 'okt': 'October',
        'november': 'November', 'nov.': 'November', 'nov': 'November',
        'december': 'December', 'dec.': 'December', 'dec': 'December'
    }

    date_str_lower = date_str.lower()
    
    for swedish_month, english_month in month_map.items():
        if swedish_month in date_str_lower:
            date_str_lower = date_str_lower.replace(swedish_month, english_month)
            break 

    try:
        return parser.parse(date_str_lower, dayfirst=True)
    except Exception:
        print(f"DEBUG: Failed to parse date string even after manual translation: '{original_date_str}'")
        return pd.NaT

def perform_data_cleaning_and_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to orchestrate all cleaning and feature engineering steps.
    """
    print("\n--- Converting 'sold_date' to datetime objects (locale-independent) ---")
    df['sold_date'] = df['sold_date'].astype(str).apply(parse_swedish_date)
    print("'sold_date' converted.")

    print("\n--- Performing Feature Engineering ---")
    if 'final_price' in df.columns and 'living_area_m2' in df.columns:
        df['price_per_m2'] = (df['final_price'] / df['living_area_m2']).round(2)
        print("Created 'price_per_m2' column.")

    desired_cols = [
        'street_address', 'location_area', 'municipality',
        'final_price', 'price_per_m2', 'price_change_percent',
        'sold_date',
        'living_area_m2', 'non_living_area_m2', 'rooms', 'plot_area_m2',
        'url'
    ]
    
    existing_cols = [col for col in desired_cols if col in df.columns]
    df = df[existing_cols]
    print("\nColumns reordered for clarity.")

    return df

def analyze_data(df: pd.DataFrame):
    """
    Prints a basic analysis of the cleaned DataFrame.
    """
    print("\n" + "="*50)
    print("          DATA ANALYSIS SUMMARY")
    print("="*50)

    print("\n--- DataFrame Info ---")
    df.info()

    print("\n--- Descriptive Statistics for Numerical Columns ---")
    print(df.describe().to_string())

    print("\n--- Missing Value Counts ---")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found in the final dataset.")
    
    print("\n--- Top 10 Location Areas by Count ---")
    print(df['location_area'].value_counts().head(10).to_string())
    print("\n" + "="*50)


def run_cleaning_pipeline():
    """
    Loads raw data, runs the full cleaning and processing pipeline,
    analyzes the result, and saves the cleaned data to a new CSV file.
    This is the main entry point for the cleaning process.
    """
    print(f"--- Starting Data Cleaning Process ---")
    
    try:
        print(f"Loading raw data from: '{config.RAW_DATA_PATH}'")
        df = pd.read_csv(config.RAW_DATA_PATH)
        print(f"Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at '{config.RAW_DATA_PATH}'.")
        print("Please run the scraper script (src/scrape.py) first.")
        return
    except pd.errors.EmptyDataError:
        print(f"ERROR: The raw data file is empty. No data to process.")
        return

    df_cleaned = clean_location(df.copy())
    df_processed = perform_data_cleaning_and_engineering(df_cleaned)

    analyze_data(df_processed)

    try:
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        
        df_processed.to_csv(config.PROCESSED_DATA_PATH, index=False, encoding='utf-8-sig')
        print(f"\n--- Success! ---")
        print(f"Processed data saved to: '{config.PROCESSED_DATA_PATH}'")
        
        print("\n--- Preview of Processed Data ---")
        print(df_processed.head().to_string())

    except Exception as e:
        print(f"\nERROR: Failed to save processed data. Reason: {e}")


if __name__ == '__main__':
    run_cleaning_pipeline()