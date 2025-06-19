import pandas as pd
import os
import locale

# --- Configuration ---
# Define paths relative to the script's location
CURRENT_DIR = os.path.dirname(__file__)
RAW_DATA_PATH = os.path.join(CURRENT_DIR, '..', 'data', 'raw', 'hemnet_sold_villas_final.csv')
PROCESSED_DATA_PATH = os.path.join(CURRENT_DIR, '..', 'data', 'processed', 'hemnet_sold_villas_processed.csv')

def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'location' column.
    - Splits 'Area, Municipality'
    - Handles multi-part areas like 'Sunnersta/Graneberg' by taking only the first part ('Sunnersta').
    - Converts the 'Area' part to title case (e.g., 'FYRISLUND' -> 'Fyrislund')
    - Recombines the parts.
    """
    print("--- Cleaning 'location' column ---")
    if 'location' not in df.columns:
        print("WARNING: 'location' column not found. Skipping location cleaning.")
        return df

    # Split the location into 'area' and 'municipality' parts
    # n=1 ensures we only split on the first comma, expand=True creates new columns
    loc_parts = df['location'].str.split(',', n=1, expand=True)
    loc_parts.columns = ['area', 'municipality']

    # Clean the 'area' part
    # .str.split('/') -> Splits 'Sunnersta/Graneberg' into ['Sunnersta', 'Graneberg']
    # .str.get(0)    -> Selects the first element, 'Sunnersta'
    # .str.strip()   -> Removes leading/trailing whitespace
    # .str.title()   -> Converts "FYRISLUND" to "Fyrislund" or "sunnersta" to "Sunnersta"
    df['location_area'] = loc_parts['area'].str.split('/').str.get(0).str.strip().str.title() # <-- MODIFIED LINE

    # Clean and store the municipality part
    df['municipality'] = loc_parts['municipality'].str.strip()

    # We can now drop the original 'location' column as it's been split and cleaned
    df.drop(columns=['location'], inplace=True)
    
    print("Location successfully split into 'location_area' and 'municipality'.")
    print("Multi-part areas like 'Sunnersta/Graneberg' are simplified to the first area.")
    return df

def perform_data_cleaning_and_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to orchestrate all cleaning and feature engineering steps.
    """
    # --- Date Conversion ---
    # To handle Swedish month names like 'december', we set the locale.
    # This might require the language pack to be installed on your system.
    # Common locales: 'sv_SE.UTF-8' on Linux/macOS, 'Swedish' on Windows.
    print("\n--- Converting 'sold_date' to datetime objects ---")
    try:
        locale.setlocale(locale.LC_TIME, 'sv_SE.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_TIME, 'Swedish')
        except locale.Error:
            print("WARNING: Could not set locale to Swedish. Date parsing may fail for some months.")
    
    # Convert to datetime, coercing errors to NaT (Not a Time)
    df['sold_date'] = pd.to_datetime(df['sold_date'], format='%d %B %Y', errors='coerce')
    print("'sold_date' converted.")

    # --- Feature Engineering: Price per Square Meter ---
    print("\n--- Performing Feature Engineering ---")
    if 'final_price' in df.columns and 'living_area_m2' in df.columns:
        # Calculate price per square meter, a very useful metric for analysis
        df['price_per_m2'] = (df['final_price'] / df['living_area_m2']).round(2)
        print("Created 'price_per_m2' column.")

    # Reorder columns for better readability
    # Put the new location columns at the front and engineered columns near their sources.
    desired_cols = [
        'street_address', 'location_area', 'municipality',
        'final_price', 'price_per_m2', 'price_change_percent',
        'sold_date',
        'living_area_m2', 'non_living_area_m2', 'rooms', 'plot_area_m2',
        'url'
    ]
    
    # Filter to only include columns that actually exist in the DataFrame
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
    # Use .to_string() to ensure all columns are displayed
    print(df.describe().to_string())

    print("\n--- Missing Value Counts ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Let's also check the unique values in our new location_area column
    print("\n--- Top 10 Location Areas by Count ---")
    print(df['location_area'].value_counts().head(10).to_string())


    print("\n" + "="*50)


def main():
    """
    Main function to load, clean, analyze, and save the data.
    """
    print(f"--- Starting Data Cleaning Process ---")
    
    # --- Load Data ---
    try:
        print(f"Loading raw data from: '{RAW_DATA_PATH}'")
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at '{RAW_DATA_PATH}'.")
        print("Please run the scraper script (src/scrape.py) first.")
        return
    except pd.errors.EmptyDataError:
        print(f"ERROR: The raw data file is empty. No data to process.")
        return

    # --- Clean Data ---
    df_cleaned = clean_location(df.copy()) # Use a copy to avoid SettingWithCopyWarning
    df_processed = perform_data_cleaning_and_engineering(df_cleaned)

    # --- Analyze Data ---
    analyze_data(df_processed)

    # --- Save Data ---
    try:
        # Ensure the 'processed' directory exists
        output_dir = os.path.dirname(PROCESSED_DATA_PATH)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the cleaned DataFrame to a new CSV file
        df_processed.to_csv(PROCESSED_DATA_PATH, index=False, encoding='utf-8-sig')
        print(f"\n--- Success! ---")
        print(f"Processed data saved to: '{PROCESSED_DATA_PATH}'")
        
        print("\n--- Preview of Processed Data ---")
        print(df_processed.head().to_string())

    except Exception as e:
        print(f"\nERROR: Failed to save processed data. Reason: {e}")


if __name__ == '__main__':
    main()